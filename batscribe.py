"""Batscribe: batch-caption every video under a folder with faster-whisper on the GPU.

Usage: python batscribe.py <folder-or-video> [more ...] [--model NAME] [--force]
Writes <video>.srt next to each video. Existing .srt files are skipped (resumable).
"""
import argparse
import datetime as dt
import os
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
            capture_output=True, text=True, timeout=60,
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
        capture_output=True,
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


def main():
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Batch-caption videos to .srt with faster-whisper.")
    ap.add_argument("inputs", nargs="+", help="folders and/or video files")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--compute-type", default="int8_float16")
    ap.add_argument("--force", action="store_true", help="redo videos that already have .srt")
    args = ap.parse_args()

    videos = collect(args.inputs)
    todo = [v for v in videos if args.force or not v.with_suffix(".srt").exists()]
    print(f"Found {len(videos)} videos, {len(videos) - len(todo)} already captioned, {len(todo)} to do.")
    if not todo:
        return 0

    durations = {v: video_duration(v) for v in todo}
    total_audio = sum(durations.values())
    print(f"Audio to transcribe: {fmt_dur(total_audio)}")

    add_cuda_dll_dirs()
    from faster_whisper import WhisperModel
    print(f"Loading model {args.model} on GPU ...")
    model = WhisperModel(args.model, device="cuda", compute_type=args.compute_type)

    root = Path(args.inputs[0].strip('"'))
    log_path = (root if root.is_dir() else root.parent) / "batscribe_errors.log"
    done_audio, failures = 0.0, 0
    t0 = time.time()

    for n, vid in enumerate(todo, 1):
        dur = durations[vid]
        print(f"\n[{n}/{len(todo)}] {vid.name}  ({fmt_dur(dur)})")
        try:
            audio = load_audio(vid)
            segments, info = model.transcribe(
                audio, language="en", vad_filter=True, word_timestamps=True,
                condition_on_previous_text=False, beam_size=5,
            )
            dur = info.duration or dur
            words = []
            for seg in segments:
                words.extend(seg.words or [])
                elapsed = time.time() - t0
                processed = done_audio + seg.end
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total_audio - processed) / speed if speed > 0 else 0
                pct = 100 * seg.end / dur if dur else 0
                print(f"\r   {pct:5.1f}%  | overall {100 * processed / max(total_audio, 1):5.1f}%"
                      f"  | {speed:4.1f}x realtime | ETA {fmt_dur(eta)}   ", end="", flush=True)
            write_srt(build_cues(words), vid.with_suffix(".srt"))
            print(f"\r   done -> {vid.with_suffix('.srt').name}" + " " * 40)
        except KeyboardInterrupt:
            print("\nStopped. Rerun to continue where you left off.")
            return 130
        except Exception as e:
            failures += 1
            print(f"\n   FAILED: {e}")
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M}] {vid}\n{traceback.format_exc()}\n")
        done_audio += durations[vid]

    print(f"\nFinished in {fmt_dur(time.time() - t0)}. {len(todo) - failures} ok, {failures} failed.")
    if failures:
        print(f"See {log_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
