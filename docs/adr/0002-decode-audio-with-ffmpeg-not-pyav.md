# Decode audio with the ffmpeg CLI, not PyAV

faster-whisper 1.2.1 normally decodes audio through PyAV, but it calls `av.open(metadata_errors=...)`, which PyAV 19 removed, and older PyAV wheels would not install on Python 3.12/Windows. So Batscribe decodes each Video itself with an `ffmpeg` subprocess (16 kHz mono float32) and gets durations from `ffprobe`, passing the raw array to the model. ffmpeg/ffprobe are therefore a required system install on PATH. ffmpeg writes to a temp file rather than a pipe so the audio is held in memory once, not twice; both tools are restricted to the `file` protocol.

`av` is still installed, but only because faster-whisper depends on it.
