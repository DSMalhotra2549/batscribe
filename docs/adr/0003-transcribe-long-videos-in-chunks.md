# Transcribe long Videos in ≤10-minute Chunks cut at the quietest moment

faster-whisper's feature extraction uses about 62 MB of RAM per minute of audio it is given at once; a 157-minute Video peaked at ~11 GB on a 7.7 GB PC. Audio is therefore split into Chunks of at most `CHUNK_SECONDS` (600 s), each cut at the quietest 0.1 s within the last 30 s before the limit so a cut rarely lands mid-word. Word timestamps are shifted by the Chunk's offset and merged into one Caption file. Measured peak for the 157-minute Video afterwards: 1.2 GB.

## Consequences

- A word spoken exactly across a cut may be lost or duplicated; the quiet-point search makes this rare.
- Progress and timestamps must always add the Chunk offset (covered by a unit test).
