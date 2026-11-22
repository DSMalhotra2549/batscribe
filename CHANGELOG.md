# Changelog

## Unreleased

- Command line: `--format md` writes a Markdown transcript (`<video> transcript.md`: header, then `[h:mm:ss]` paragraphs split at pauses over 2 s) instead of an `.srt`. The default stays `srt`; the desktop app is unchanged.
- Audio files (`.mp3 .m4a .wav .aac .ogg .opus .flac`) are captioned too, from folders, drag and drop, and the file picker.

## 0.1.0 – 2026-11-22

First public release.

- Batch-caption every video in a folder (recursively) to a `.srt` next to it, with faster-whisper `large-v3-turbo`.
- Desktop app with drag and drop, per-video status, progress, speed and time left; command line and `captions.bat` drag-and-drop.
- Resumable: videos that already have an `.srt` are skipped; Redo option to replace them.
- Spoken language auto-detected per video, or chosen (99 languages).
- Optional Fast mode (batched inference, about 2.5x faster).
- Runs on an NVIDIA GPU, or falls back to the CPU when there is none.
- Long videos are transcribed in pieces to keep memory use low.
- Robustness: broken videos are logged and skipped, GPU errors stop the batch cleanly, crashed runs are reported at the next launch, the PC is kept awake while running.
- Logs in `%LOCALAPPDATA%\Batscribe`.
