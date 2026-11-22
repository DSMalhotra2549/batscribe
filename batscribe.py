"""Batscribe: batch-caption every video under a folder with faster-whisper on the GPU.

Usage: python batscribe.py <folder-or-video> [more ...] [--language CODE] [--fast] [--force] [--device auto|cuda|cpu]
Writes <video>.srt next to each video. Existing .srt files are skipped (resumable).
"""
import argparse
import datetime as dt
import os
import re
import shutil
import site
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
CAPTION_EXTS = {".srt", ".vtt", ".ass", ".ssa", ".sub"}
SR = 16000             # audio sample rate the model expects
# faster-whisper needs ~60 MB of RAM per minute of audio it is given at once, so long videos
# are transcribed in pieces of at most this length, cut at a quiet moment.
CHUNK_SECONDS = 600
MAX_LINE = 42          # chars per subtitle line
MAX_LINES = 2
MAX_CUE_SECONDS = 6.0
MIN_CUE_CHARS_AT_SENTENCE_END = 20
# Don't open a console window per ffmpeg/ffprobe call when run from the windowless GUI.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
# Languages Whisper can transcribe (code -> name). None/"auto" means detect per video.
LANGUAGES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fo": "Faroese",
    "fr": "French", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian", "id": "Indonesian",
    "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh",
    "km": "Khmer", "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish", "ln": "Lingala",
    "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian",
    "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese",
    "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian", "oc": "Occitan", "pa": "Punjabi",
    "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian", "sa": "Sanskrit",
    "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona", "so": "Somali",
    "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish", "yo": "Yoruba",
    "yue": "Cantonese", "zh": "Chinese",
}
COMPUTE_TYPES = {"cuda": "int8_float16", "cpu": "int8"}  # int8_float16 fits large-v3-turbo in 6 GB VRAM


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
    """Decode to 16 kHz mono float32 with ffmpeg (avoids PyAV version breakage).

    ffmpeg writes to a temp file rather than a pipe so the audio is held in memory once, not twice.
    """
    import numpy as np
    fd, tmp = tempfile.mkstemp(suffix=".f32", prefix="batscribe_")
    os.close(fd)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-protocol_whitelist", "file", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(SR), "-f", "f32le", tmp],
            capture_output=True, creationflags=NO_WINDOW,
        )
        audio = np.fromfile(tmp, dtype=np.float32) if proc.returncode == 0 else None
    finally:
        os.remove(tmp)
    if audio is None or not audio.size:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise RuntimeError("ffmpeg could not decode audio: " + (msg[-1] if msg else "no audio stream"))
    return audio


def chunk_bounds(audio):
    """Split audio into pieces of at most CHUNK_SECONDS, each cut at the quietest 0.1 s of the
    last 30 s before the limit (so a cut doesn't land mid-word). Returns [(start, end)] in samples."""
    size, frame = CHUNK_SECONDS * SR, SR // 10
    search = int(min(30, CHUNK_SECONDS / 2) * SR) // frame * frame
    bounds, start = [], 0
    while len(audio) - start > size:
        lo = start + size - search
        energy = (audio[lo:start + size].reshape(-1, frame) ** 2).mean(axis=1)
        quietest = len(energy) - 1 - int(energy[::-1].argmin())  # the latest one if tied (digital silence)
        cut = lo + quietest * frame + frame // 2
        bounds.append((start, cut))
        start = cut
    bounds.append((start, len(audio)))
    return bounds


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


def name_clashes(videos):
    """Videos that would write the same .srt as an earlier one (e.g. 'a.mp4' and 'a.mkv'): {video: earlier}."""
    owner, clashes = {}, {}
    for v in videos:
        srt = v.with_suffix(".srt")
        if srt in owner:
            clashes[v] = owner[srt]
        else:
            owner[srt] = v
    return clashes


def other_captions(video):
    """Caption files next to the video that aren't its <video>.srt, e.g. 'a.en.srt' or 'a.vtt'."""
    prefix, own = video.stem.lower() + ".", video.with_suffix(".srt")
    return sorted(f.name for f in video.parent.iterdir()
                  if f.suffix.lower() in CAPTION_EXTS and f.name.lower().startswith(prefix) and f != own)


# Logs live in a per-user folder, so the app also works from a read-only install folder.
DATA_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "state") / "Batscribe"
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    pass  # logging then fails quietly; captioning still works
ERROR_LOG = DATA_DIR / "batscribe_errors.log"
HISTORY_LOG = DATA_DIR / "batscribe_history.log"
CRASH_LOG = DATA_DIR / "batscribe_crash.log"


def log_history(line: str):
    """Append a timestamped line to the history log shared by the CLI and the GUI."""
    try:
        with open(HISTORY_LOG, "a", encoding="utf-8") as f:
            f.write(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {line}\n")
    except OSError:
        pass  # a log problem (e.g. disk full) must not stop captioning


def log_batch(text: str):
    """BATCH lines carry the process id so a batch that never logged its end can be detected later."""
    log_history(f"BATCH {text}  [pid {os.getpid()}]")


def pid_alive(pid: int) -> bool:
    if os.name != "nt":
        return True  # only checked on Windows; assume alive so nothing is misreported
    import ctypes
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = ctypes.c_void_p
    handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(code))
    k32.CloseHandle(ctypes.c_void_p(handle))
    return code.value == 259  # STILL_ACTIVE


def unfinished_batches():
    """START lines of batches whose process ended without logging an end (hard crash, power loss,
    console closed). Each is marked CRASHED in the log so it is reported only once."""
    try:
        lines = HISTORY_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    open_batches = {}
    for line in lines:
        m = re.search(r"BATCH (\w+) .*\[pid (\d+)\]$", line)
        if m and m[1] == "START":
            open_batches[m[2]] = line
        elif m:
            open_batches.pop(m[2], None)
    lost = {pid: line for pid, line in open_batches.items() if not pid_alive(int(pid))}
    for pid in lost:
        log_history(f"BATCH CRASHED  ended without finishing (noticed at next launch)  [pid {pid}]")
    return list(lost.values())


def check_prerequisites(device="auto"):
    """Return a list of human-readable problems that would make every video fail."""
    problems = [f"{tool} was not found. Install ffmpeg and make sure it is on PATH."
                for tool in ("ffmpeg", "ffprobe") if not shutil.which(tool)]
    try:
        import ctranslate2  # noqa: F401
    except Exception as e:
        problems.append(f"The transcription engine could not be loaded: {e}")
        return problems
    if device == "cuda" and not cuda_available():
        problems.append("No NVIDIA GPU (CUDA) was found. Update the NVIDIA driver or check the GPU is enabled.")
    return problems


def cuda_available() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def resolve_device(device: str) -> str:
    """'auto' means the NVIDIA GPU if there is one, otherwise the (much slower) CPU."""
    if device != "auto":
        return device
    return "cuda" if cuda_available() else "cpu"


def is_gpu_error(e: Exception) -> bool:
    """A GPU failure (out of memory, driver crash) would fail every remaining video too."""
    return any(k in str(e) for k in ("CUDA", "cuBLAS", "cuDNN", "out of memory"))


def keep_awake(on: bool):
    """Stop Windows from sleeping while this thread runs a batch (the screen may still turn off)."""
    if os.name == "nt":
        import ctypes
        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | (es_system_required if on else 0))


def run_batch(todo, durations, log_path, model_name="large-v3-turbo", compute_type=None,
              on_event=lambda kind, **info: None, should_stop=lambda: False, skip_existing=True, fast=False,
              language=None, device="auto"):
    """Transcribe each video in `todo`, reporting through on_event(kind, **info).

    fast=True uses faster-whisper's batched mode: about 2.5x faster, slightly less accurate.
    language is a code from LANGUAGES, or None to detect it per video (from its first chunk).
    device is "cuda", "cpu" or "auto"; compute_type defaults to COMPUTE_TYPES[device].

    Events: loading, start, progress, done, skipped, failed, finished. A stop request abandons
    the current file (no .srt is written for it); a GPU error stops the batch.
    Returns the number of failures.
    """
    durations = dict(durations)  # corrected below with each file's decoded length
    total_audio = sum(durations.values())
    keep_awake(True)
    try:
        add_cuda_dll_dirs()
        device = resolve_device(device)
        compute_type = compute_type or COMPUTE_TYPES[device]
        first = todo[0].parent if todo else ""
        log_batch(f"START  {len(todo)} videos, {fmt_dur(total_audio)} of audio, model {model_name} on {device}, "
                  f"language {language or 'auto'}{', fast mode' if fast else ''}, in {first}")
        from faster_whisper import WhisperModel
        on_event("loading", model=model_name, device=device)
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        if fast:
            from faster_whisper import BatchedInferencePipeline
            batched = BatchedInferencePipeline(model=model)

            def transcribe(audio, lang):
                return batched.transcribe(audio, language=lang, word_timestamps=True, batch_size=8, beam_size=5)
        else:
            def transcribe(audio, lang):
                return model.transcribe(audio, language=lang, vad_filter=True, word_timestamps=True,
                                        condition_on_previous_text=False, beam_size=5)

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
            audio = None  # free the previous video's audio before decoding the next
            try:
                audio = load_audio(vid)
                dur = len(audio) / SR
                total_audio += dur - durations[vid]  # the scan's ffprobe value may be off (0 if it failed)
                durations[vid] = dur
                words, lang = [], language
                for a, b in chunk_bounds(audio):
                    offset = a / SR
                    segments, info = transcribe(audio[a:b], lang)
                    lang = lang or info.language  # detected once, then kept for the video's other chunks
                    for seg in segments:
                        if should_stop():
                            stopped = True
                            break
                        words.extend(SimpleNamespace(word=w.word, start=w.start + offset, end=w.end + offset)
                                     for w in seg.words or [])
                        pos = offset + seg.end
                        elapsed = time.time() - t0
                        processed = done_audio + pos
                        speed = processed / elapsed if elapsed > 0 else 0
                        on_event("progress", n=n, video=vid,
                                 file_pct=min(100, 100 * pos / dur) if dur else 0,
                                 overall_pct=min(100, 100 * processed / max(total_audio, 1)),
                                 speed=speed,
                                 eta=max(0, total_audio - processed) / speed if speed > 0 else 0)
                    if stopped:
                        break
                if stopped:
                    log_history(f"  STOPPED  {vid}  (not saved)")
                    break
                srt = vid.with_suffix(".srt")
                write_srt(build_cues(words), srt)
                ok += 1
                took = time.time() - t_file
                log_history(f"  OK       {vid}  ({fmt_dur(dur)} of audio in {fmt_dur(took)}, "
                            f"{dur / max(took, 0.001):.1f}x realtime, language {lang})")
                on_event("done", n=n, video=vid, srt=srt)
            except Exception as e:
                failures += 1
                on_event("failed", n=n, video=vid, error=str(e))
                log_history(f"  FAILED   {vid}  ({e}; details in {log_path})")
                try:
                    with open(log_path, "a", encoding="utf-8") as log:
                        log.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M}] {vid}\n{traceback.format_exc()}\n")
                except OSError:
                    pass  # read-only folder or full disk: carry on with the next video
                if is_gpu_error(e):
                    stopped, gpu_error = True, str(e)
                    break
            done_audio += durations[vid]

        log_batch(f"{'STOPPED' if stopped else 'END'}  {ok} ok, {failures} failed, "
                  f"took {fmt_dur(time.time() - t0)}" + (f"  (GPU error: {gpu_error})" if gpu_error else ""))
        on_event("finished", ok=ok, failed=failures, stopped=stopped, gpu_error=gpu_error,
                 elapsed=time.time() - t0, log_path=log_path)
        return failures
    except BaseException as e:  # model failed to load, Ctrl+C, ...
        log_batch(f"ABORTED  {e!r}")
        raise
    finally:
        keep_awake(False)


def print_event(kind, **e):
    if kind == "loading":
        print(f"Loading model {e['model']} on {'GPU' if e['device'] == 'cuda' else 'CPU'} ...")
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
    ap.add_argument("--language", default="auto", choices=["auto", *LANGUAGES], metavar="CODE",
                    help="spoken language, e.g. en, de, es (default: auto-detect per video)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto = NVIDIA GPU if available, else CPU (much slower)")
    ap.add_argument("--compute-type", default=None, help="CTranslate2 precision (default: int8_float16 on GPU, int8 on CPU)")
    ap.add_argument("--force", action="store_true", help="redo videos that already have .srt")
    ap.add_argument("--fast", action="store_true", help="batched mode: about 2.5x faster, slightly less accurate")
    args = ap.parse_args()

    problems = check_prerequisites(args.device)
    if problems:
        print("Cannot start:\n" + "\n".join(f"  - {p}" for p in problems))
        return 1
    if resolve_device(args.device) == "cpu":
        print("Note: running on the CPU (no NVIDIA GPU found or --device cpu). This is much slower than a GPU.")

    for line in unfinished_batches():
        print(f"Note: a previous run ended without finishing (crash, power loss or window closed):\n  {line}")

    videos = collect(args.inputs)
    clashes = name_clashes(videos)
    for v, first in clashes.items():
        print(f"!! Skipping {v}: same name as {first.name}, so both would use the same .srt. Rename one of them.")
    todo = [v for v in videos if v not in clashes and (args.force or not v.with_suffix(".srt").exists())]
    print(f"Found {len(videos)} videos, {len(videos) - len(todo) - len(clashes)} already captioned, {len(todo)} to do.")
    if not todo:
        return 0

    durations = {v: video_duration(v) for v in todo}
    print(f"Audio to transcribe: {fmt_dur(sum(durations.values()))}")

    try:
        failures = run_batch(todo, durations, ERROR_LOG, args.model, args.compute_type,
                             on_event=print_event, skip_existing=not args.force, fast=args.fast,
                             language=None if args.language == "auto" else args.language, device=args.device)
    except KeyboardInterrupt:
        print("\nStopped. Rerun to continue where you left off.")
        return 130
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
