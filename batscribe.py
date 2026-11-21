"""Batscribe: batch-caption every video under a folder with faster-whisper on the GPU.

Usage: python batscribe.py <folder-or-video> [more ...] [--model NAME] [--force]
Writes <video>.srt next to each video. Existing .srt files are skipped (resumable).
"""
import argparse
import datetime as dt
import os
import shutil
import site
import subprocess
import sys
import time
import traceback
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
MAX_LINE = 42          # chars per subtitle line
MAX_LINES = 2
MAX_CUE_SECONDS = 6.0
MIN_CUE_CHARS_AT_SENTENCE_END = 20
# Don't open a console window per ffmpeg/ffprobe call when run from the windowless GUI.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def add_cuda_dll_dirs():
    """Make pip-installed nvidia cuBLAS/cuDNN DLLs visible to CTranslate2 on Windows."""
    if os.name != "nt":
        return
    for sp in site.getsitepackages():
        nv = Path(sp) / "nvidia"
        if not nv.is_dir():
            continue
        for bin_dir in nv.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def fmt_ts(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_dur(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def wrap_lines(text: str) -> str:
    """Split into at most 2 lines, breaking at the space closest to the middle."""
    if len(text) <= MAX_LINE:
        return text
    mid = len(text) // 2
    spaces = [i for i, c in enumerate(text) if c == " "]
    if not spaces:
        return text
    cut = min(spaces, key=lambda i: abs(i - mid))
    return text[:cut].strip() + "\n" + text[cut + 1:].strip()


def build_cues(words):
    """Group word timestamps into readable subtitle cues: (start, end, text)."""
    cues = []
    cur, start, end = [], None, None
    max_chars = MAX_LINE * MAX_LINES

    def flush():
        nonlocal cur, start, end
        text = " ".join(cur).strip()
        if text:
            cues.append((start, end, text))
        cur, start, end = [], None, None

    for w in words:
        token = w.word.strip()
        if not token:
            continue
        if cur:
            candidate = len(" ".join(cur)) + 1 + len(token)
            gap = w.start - end
            if candidate > max_chars or (w.end - start) > MAX_CUE_SECONDS or gap > 1.0:
                flush()
        if not cur:
            start = w.start
        cur.append(token)
        end = w.end
        if token[-1] in ".?!" and len(" ".join(cur)) >= MIN_CUE_CHARS_AT_SENTENCE_END:
            flush()
    flush()
    return cues


def write_srt(cues, out_path: Path):
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for i, (s, e, text) in enumerate(cues, 1):
            if e <= s:
                e = s + 0.5
            f.write(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{wrap_lines(text)}\n\n")
    os.replace(tmp, out_path)


def video_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def load_audio(path: Path):
    """Decode to 16 kHz mono float32 with ffmpeg (avoids PyAV version breakage)."""
    import numpy as np
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "f32le", "-"],
        capture_output=True, creationflags=NO_WINDOW,
    )
    if proc.returncode != 0 or not proc.stdout:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("ffmpeg could not decode audio: " + (msg[-1] if msg else "no audio stream"))
    return np.frombuffer(proc.stdout, dtype=np.float32)


def collect(inputs):
    vids = []
    for raw in inputs:
        p = Path(raw.strip().strip('"'))
        if p.is_dir():
            vids += [v for v in p.rglob("*") if v.is_file() and v.suffix.lower() in VIDEO_EXTS]
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
        else:
            print(f"!! Not a folder or video: {p}")
    return sorted(set(vids), key=lambda v: str(v).lower())


def error_log_path(first_input) -> Path:
    root = Path(str(first_input).strip('"'))
    return (root if root.is_dir() else root.parent) / "batscribe_errors.log"


HISTORY_LOG = Path(__file__).with_name("batscribe_history.log")


def log_history(line: str):
    """Append a timestamped line to the history log shared by the CLI and the GUI."""
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {line}\n")


def check_prerequisites():
    """Return a list of human-readable problems that would make every video fail."""
    problems = [f"{tool} was not found. Install ffmpeg and make sure it is on PATH."
                for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() == 0:
            problems.append("No NVIDIA GPU (CUDA) was found. Update the NVIDIA driver or check the GPU is enabled.")
    except Exception as e:
        problems.append(f"The transcription engine could not be loaded: {e}")
    return problems


def is_gpu_error(e: Exception) -> bool:
    """A GPU failure (out of memory, driver crash) would fail every remaining video too."""
    return any(k in str(e) for k in ("CUDA", "cuBLAS", "cuDNN", "out of memory"))


def keep_awake(on: bool):
    """Stop Windows from sleeping while this thread runs a batch (the screen may still turn off)."""
    if os.name == "nt":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def run_batch(todo, durations, log_path, model_name="large-v3-turbo", compute_type="int8_float16",
              on_event=lambda kind, **info: None, should_stop=lambda: False, skip_existing=True):
    """Transcribe each video in `todo`, reporting through on_event(kind, **info).

    Events: loading, start, progress, done, skipped, failed, finished. A stop request abandons
    the current file (no .srt is written for it); a GPU error stops the batch.
    Returns the number of failures.
    """
    durations = dict(durations)  # corrected below with each file's decoded length
    total_audio = sum(durations.values())
    keep_awake(True)
    log_history(f"BATCH START  {len(todo)} videos, {fmt_dur(total_audio)} of audio, "
                f"model {model_name}, in {log_path.parent}")
    try:
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel
        on_event("loading", model=model_name)
        model = WhisperModel(model_name, device="cuda", compute_type=compute_type)

        done_audio, ok, failures, stopped, gpu_error = 0.0, 0, 0, False, None
        t0 = time.time()
        for n, vid in enumerate(todo, 1):
            if should_stop():
                stopped = True
                break
            if skip_existing and vid.with_suffix(".srt").exists():  # e.g. made by another window meanwhile
                log_history(f"  SKIPPED  {vid}  (already has .srt)")
                on_event("skipped", n=n, video=vid)
                done_audio += durations[vid]
                continue
            dur = durations[vid]
            on_event("start", n=n, total=len(todo), video=vid, duration=dur)
            t_file = time.time()
            try:
                audio = load_audio(vid)
                segments, info = model.transcribe(
                    audio, language="en", vad_filter=True, word_timestamps=True,
                    condition_on_previous_text=False, beam_size=5,
                )
                dur = info.duration or dur
                total_audio += dur - durations[vid]  # the scan's ffprobe value may be off (0 if it failed)
                durations[vid] = dur
                words = []
                for seg in segments:
                    if should_stop():
                        stopped = True
                        break
                    words.extend(seg.words or [])
                    elapsed = time.time() - t0
                    processed = done_audio + seg.end
                    speed = processed / elapsed if elapsed > 0 else 0
                    on_event("progress", n=n, video=vid,
                             file_pct=min(100, 100 * seg.end / dur) if dur else 0,
                             overall_pct=min(100, 100 * processed / max(total_audio, 1)),
                             speed=speed,
                             eta=max(0, total_audio - processed) / speed if speed > 0 else 0)
                if stopped:
                    log_history(f"  STOPPED  {vid}  (not saved)")
                    break
                srt = vid.with_suffix(".srt")
                write_srt(build_cues(words), srt)
                ok += 1
                took = time.time() - t_file
                log_history(f"  OK       {vid}  ({fmt_dur(dur)} of audio in {fmt_dur(took)}, "
                            f"{dur / max(took, 0.001):.1f}x realtime)")
                on_event("done", n=n, video=vid, srt=srt)
            except Exception as e:
                failures += 1
                on_event("failed", n=n, video=vid, error=str(e))
                log_history(f"  FAILED   {vid}  ({e}; details in {log_path})")
                with open(log_path, "a", encoding="utf-8") as log:
                    log.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M}] {vid}\n{traceback.format_exc()}\n")
                if is_gpu_error(e):
                    stopped, gpu_error = True, str(e)
                    break
            done_audio += durations[vid]

        log_history(f"BATCH {'STOPPED' if stopped else 'END'}  {ok} ok, {failures} failed, "
                    f"took {fmt_dur(time.time() - t0)}" + (f"  (GPU error: {gpu_error})" if gpu_error else ""))
        on_event("finished", ok=ok, failed=failures, stopped=stopped, gpu_error=gpu_error,
                 elapsed=time.time() - t0, log_path=log_path)
        return failures
    except BaseException as e:  # model failed to load, Ctrl+C, ...
        log_history(f"BATCH ABORTED  {e!r}")
        raise
    finally:
        keep_awake(False)


def print_event(kind, **e):
    if kind == "loading":
        print(f"Loading model {e['model']} on GPU ...")
    elif kind == "start":
        print(f"\n[{e['n']}/{e['total']}] {e['video'].name}  ({fmt_dur(e['duration'])})")
    elif kind == "progress":
        print(f"\r   {e['file_pct']:5.1f}%  | overall {e['overall_pct']:5.1f}%"
              f"  | {e['speed']:4.1f}x realtime | ETA {fmt_dur(e['eta'])}   ", end="", flush=True)
    elif kind == "done":
        print(f"\r   done -> {e['srt'].name}" + " " * 40)
    elif kind == "skipped":
        print("   skipped (an .srt appeared meanwhile)")
    elif kind == "failed":
        print(f"\n   FAILED: {e['error']}")
    elif kind == "finished":
        if e["gpu_error"]:
            print("\nStopped: the GPU failed, so the remaining videos were not attempted.\n"
                  "Close other GPU-heavy apps (or restart the PC), then run again to continue.")
        print(f"\nFinished in {fmt_dur(e['elapsed'])}. {e['ok']} ok, {e['failed']} failed.")
        if e["failed"]:
            print(f"See {e['log_path']}")
        if os.name == "nt":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Batch-caption videos to .srt with faster-whisper.")
    ap.add_argument("inputs", nargs="+", help="folders and/or video files")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--force", action="store_true", help="redo videos that already have .srt")
    args = ap.parse_args()

    problems = check_prerequisites()
    if problems:
        print("Cannot start:\n" + "\n".join(f"  - {p}" for p in problems))
        return 1

    videos = collect(args.inputs)
    todo = [v for v in videos if args.force or not v.with_suffix(".srt").exists()]
    print(f"Found {len(videos)} videos, {len(videos) - len(todo)} already captioned, {len(todo)} to do.")
    if not todo:
        return 0

    durations = {v: video_duration(v) for v in todo}
    print(f"Audio to transcribe: {fmt_dur(sum(durations.values()))}")

    try:
        failures = run_batch(todo, durations, error_log_path(args.inputs[0]), args.model, args.compute_type,
                             on_event=print_event, skip_existing=not args.force)
    except KeyboardInterrupt:
        print("\nStopped. Rerun to continue where you left off.")
        return 130
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
