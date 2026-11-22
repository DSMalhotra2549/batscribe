# Architecture (for agents and contributors)

Read [`CONTEXT.md`](../CONTEXT.md) first for the vocabulary (Video, Caption file, Batch, Redo, Chunk, …) and use those terms in code, comments and messages. Decisions that look odd are explained in [`docs/adr/`](adr/). User-visible changes go in [`CHANGELOG.md`](../CHANGELOG.md).

## Files

| File | Role |
|---|---|
| `batscribe.py` | The engine and the CLI. All captioning logic lives here; it has no GUI imports. |
| `batscribe_gui.py` | Tkinter window (with `tkinterdnd2` for drag and drop). Imports `batscribe as bs` and only adds UI. Windows-only (`winsound`, `ctypes.windll`). |
| `captions.bat` | Drag-and-drop wrapper: runs `batscribe.py` with the venv's Python and pauses at the end. |
| `create_shortcut.ps1` | Creates `Batscribe.lnk` (app folder + Start menu) with the icon and AppUserModelID. Machine-specific, so `*.lnk` is gitignored. |
| `make_icon.py` | Draws `batscribe.png` and `batscribe.ico` (quill on blue tile). Needs Pillow, which is not in `requirements.txt`. |
| `tests/test_batscribe.py` | Unit tests with a fake model and fake audio. No GPU, ffmpeg or media needed. |
| `requirements.txt` | Minimum versions; the default install. |
| `requirements-lock.txt` | Exact pinned versions known to work (Python 3.12, Windows, NVIDIA GPU). Refresh it when you upgrade a dependency and have verified a real GPU run. |
| `.github/workflows/tests.yml` | CI: runs the unit tests on Windows with only numpy installed. |
| `docs/images/screenshot.png` | README screenshot (demo folder, not real data). |

## Flow of one Batch (`batscribe.py`)

1. **Setup check** – `check_prerequisites(device)` returns human-readable Setup problems (ffmpeg/ffprobe on PATH, `ctranslate2` importable, and a CUDA device only if `device="cuda"` was explicitly requested). A missing GPU is not a problem with `device="auto"`: `resolve_device()` falls back to the CPU and the front ends warn that it will be slow (`cuda_available()`).
2. **Crash report** – `unfinished_batches()` finds Crashed batches in the History log (ADR 0006).
3. **Scan** – `collect(inputs)` → sorted, de-duplicated Videos (`VIDEO_EXTS`, recursive). `name_clashes()` drops Name clashes; Captioned Videos are dropped unless Redo. The GUI also calls `other_captions()` to warn. `video_duration()` (ffprobe) gives the estimate for progress; 0.0 on failure.
4. **`run_batch(todo, durations, log_path, ..., on_event, should_stop, skip_existing, fast, language, device, out_format)`** – the shared core used by both front ends (both pass `ERROR_LOG` as `log_path`):
   - keeps Windows awake (`keep_awake`), makes pip-installed CUDA DLLs loadable (`add_cuda_dll_dirs`), resolves the device, logs `BATCH START`, loads `WhisperModel` (`large-v3-turbo`; `COMPUTE_TYPES[device]` unless overridden), optionally wraps it in `BatchedInferencePipeline` (Fast mode).
   - per Video: re-check for an existing Caption file, `caption_path(video, out_format)` (skip) → `load_audio()` (ffmpeg → float32 array, ADR 0002) → `chunk_bounds()` (ADR 0003) → transcribe each Chunk, offsetting word timestamps → `build_cues()` → `write_srt()` (atomic, ADR 0001), or with `out_format="md"` → `build_paragraphs()` → `write_transcript()` (also atomic).
   - language: `None` means auto-detect. Whisper detects it on the Video's first Chunk (`info.language`), and that language is then passed explicitly for the remaining Chunks so one Video can't switch language midway. Codes are the keys of `LANGUAGES` (a test checks they match faster-whisper's list).
   - a stop request is checked between segments; the current Video is abandoned unsaved.
   - exceptions: logged to the Error log and History log; `is_gpu_error()` stops the Batch (ADR 0005).
   - returns the number of Failed videos; logs `BATCH END`/`STOPPED`, or `ABORTED` and re-raises if the model fails to load or on Ctrl+C.

### Event protocol

`run_batch` never prints. It reports through `on_event(kind, **info)`:

| kind | info |
|---|---|
| `loading` | `model, device` |
| `start` | `n, total, video, duration` |
| `progress` | `n, video, file_pct, overall_pct, speed, eta` |
| `done` | `n, video, srt` (the Caption file written: `.srt`, or the `.md` transcript) |
| `skipped` | `n, video` |
| `failed` | `n, video, error` |
| `finished` | `ok, failed, stopped, gpu_error, elapsed, log_path` |

The CLI handles these in `print_event`. The GUI runs `run_batch` on a worker thread, pushes events onto a `queue.Queue`, and `App.pump()` dispatches each to `App.on_<kind>(**info)` on the UI thread every 100 ms. GUI-only events (`checked`, `scanned`, `scan_failed`, `crashed`, `error`) use the same queue. **If you add an event, add a handler to both `print_event` and `App`.**

### Cue formatting

`build_cues` groups words into Cues, starting a new one when the text would exceed `MAX_LINE * MAX_LINES` (42 × 2) characters, the Cue would exceed `MAX_CUE_SECONDS` (6 s), or there is a gap > 1 s; it also ends a Cue at sentence punctuation once it has ≥ 20 characters. `wrap_lines` splits a Cue into two lines at the space nearest the middle. `write_srt` forces a 0.5 s minimum on zero-length Cues.

For `--format md`, `build_paragraphs` groups the same words into paragraphs instead, starting a new one after a pause > `PARAGRAPH_GAP` (2 s) or once it spans `PARAGRAPH_MAX` (90 s). `write_transcript` writes a header (title, source, duration, model, language), a `---` rule, then one `[h:mm:ss] text` paragraph per block.

## Invariants to keep

- A `.srt` (or, for `md`, a `<video> transcript.md`) beside a Video means "done". Never write it non-atomically, never write a partial one, never touch Other caption files.
- `batscribe.py` must stay usable without the GUI and without a console (the GUI runs under `pythonw`): subprocesses use `NO_WINDOW`, and stdout/stderr are reconfigured to UTF-8 only in the CLI `main()`.
- A log write failure must never stop a Batch (`log_history` and the Error log swallow `OSError`).
- Filenames may contain emoji/Unicode and spaces; always pass paths as list arguments, never through a shell string.
- ffmpeg/ffprobe calls keep `-protocol_whitelist file`.
- All logs live in `DATA_DIR` (`%LOCALAPPDATA%\Batscribe`), never in the app folder or the user's video folders, so a read-only install or read-only media still works.
- Nothing personal goes into the repo: screenshots and examples use demo data and neutral paths.

## Setup and testing

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m unittest discover tests
```

Tests replace `bs.load_audio`, `bs.HISTORY_LOG`, `bs.add_cuda_dll_dirs` and `sys.modules["faster_whisper"]`; a fake audio file's content is its length in seconds. To make a video fail on purpose, set `FakeModel.fail_on = {"<stem>": "<message>"}` (include `CUDA` in the message to simulate a GPU error). Set `bs.CHUNK_SECONDS` small to exercise chunking.

`FakeModel` "detects" German when no language is given and records each call's language in `FakeModel.languages` and its constructor arguments in `FakeModel.init_kwargs`.

The GUI and real-GPU behaviour have no automated tests; verify by hand (`.venv\Scripts\pythonw.exe batscribe_gui.py`) when you change them.
