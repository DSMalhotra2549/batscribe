<p align="center"><img src="batscribe.png" width="96" alt="Batscribe icon"></p>

# Batscribe

**Caption a whole folder of videos at once, locally, on your NVIDIA GPU.**

Drop a folder onto Batscribe and every video inside it (including subfolders) gets a matching `.srt` subtitle file right next to it. Players like VLC and MPC pick those up automatically, so you can watch with captions straight away. Nothing is uploaded; transcription runs on your own graphics card using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and the Whisper `large-v3-turbo` model.

```
Course/
├── 01 Intro.mp4
├── 01 Intro.srt        ← created by Batscribe
├── Module 2/
│   ├── Lesson 1.mp4
│   └── Lesson 1.srt    ← created by Batscribe
```

## Features

- **Batch everything**: folders are searched recursively; `.mp4 .mkv .mov .avi .webm .m4v` supported.
- **Resumable**: videos that already have an `.srt` are skipped, so you can stop at any time and pick up later.
- **Two ways to use it**: a desktop window with drag and drop, or a command line / drag-onto-`captions.bat`.
- **Progress you can trust**: per-video and overall progress, speed (e.g. "24x realtime") and time left.
- **Fast mode** (optional): about 2.5x faster, slightly less accurate.
- **Handles long videos and long nights**: long videos are processed in pieces to keep memory low, the PC is kept awake during a run, and you get a sound and message when it finishes.
- **Safe with problems**: a broken video is logged and skipped; a GPU failure stops the run cleanly; a crash is reported the next time you open the app.

## Requirements

- Windows 10/11
- An NVIDIA GPU with a recent driver (developed on an RTX 3060 Laptop, 6 GB VRAM)
- Python 3.12
- [ffmpeg](https://ffmpeg.org/download.html) (`ffmpeg` and `ffprobe` on your PATH)

You don't need to install the CUDA toolkit; the CUDA libraries come from pip.

## Install

```powershell
git clone <this repo> batscribe
cd batscribe
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first run downloads the Whisper model (about 1.6 GB).

Optional: create a Start menu shortcut (with the icon, and pinnable to the taskbar):

```powershell
powershell -ExecutionPolicy Bypass -File create_shortcut.ps1
```

## Use

### Desktop app

Open **Batscribe** from the Start menu (or run `.venv\Scripts\pythonw.exe batscribe_gui.py`).

1. Drop a folder or some videos on the window, or use **Choose folder…** / **Choose videos…**.
2. Check the list: what's to do, what's already captioned, and any warnings.
3. Press **Start**. You can **Stop** at any time; the video in progress is simply redone next time.

Double-click a video to play it. **Logs** opens the history of all runs; **New batch** clears the window.

Options:
- **Fast mode** – about 2.5x faster, slightly less accurate.
- **Redo videos that already have captions** – replaces existing `.srt` files (asks first).

### Drag and drop onto `captions.bat`

Drag a folder (or video files) onto `captions.bat`. A console window shows progress and stays open at the end.

### Command line

```powershell
.venv\Scripts\python.exe batscribe.py "D:\Videos\Course" [more folders or videos] [--fast] [--force]
```

| Option | Meaning |
|---|---|
| `--fast` | Fast mode: about 2.5x faster, slightly less accurate |
| `--force` | Redo videos that already have an `.srt` |
| `--model NAME` | Whisper model (default `large-v3-turbo`) |
| `--compute-type T` | CTranslate2 precision (default `int8_float16`, fits in 6 GB VRAM) |

Exit code is 0 on success, 1 if any video failed (or setup is broken), 130 if interrupted.

## Good to know

- **Speed**: on an RTX 3060 Laptop expect roughly 20–25x realtime (a 1-hour video in ~3 minutes); Fast mode roughly 50–60x.
- **English only**: the language is fixed to English.
- **Same-name videos**: `lesson.mp4` and `lesson.mkv` in one folder would share one `lesson.srt`, so only the first is captioned. Rename one to caption both.
- **Other subtitles**: existing files like `lesson.en.srt` or `lesson.vtt` are left untouched; Batscribe still writes `lesson.srt`.

## Logs

| File | Where | What |
|---|---|---|
| `batscribe_errors.log` | in the folder you captioned | details of videos that failed |
| `batscribe_history.log` | app folder | every run and every video's result |
| `batscribe_crash.log` | app folder | unexpected app errors |

## Troubleshooting

- **"ffmpeg was not found"** – install ffmpeg and make sure `ffmpeg -version` works in a new terminal.
- **"No NVIDIA GPU (CUDA) was found"** – update the NVIDIA driver; on laptops, make sure the GPU isn't disabled.
- **"The GPU failed"** – usually out of video memory. Close games or other GPU-heavy apps (or restart), then press Start again; finished videos are kept.
- **"The last run didn't finish"** – the PC crashed, lost power or the window was closed. Choose the same folder and press Start to continue.

## Development

```powershell
.venv\Scripts\python.exe -m unittest discover tests
```

The tests use a fake model, so no GPU, ffmpeg or videos are needed.

For how the code is organised, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). The project vocabulary is in [`CONTEXT.md`](CONTEXT.md) and design decisions are in [`docs/adr/`](docs/adr/).
