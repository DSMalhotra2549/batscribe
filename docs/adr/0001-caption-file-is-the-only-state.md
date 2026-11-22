# The Caption file is the only record of progress

Batscribe keeps no database or progress file: a Video is done exactly when its `<video>.srt` exists. The `.srt` is written to `<video>.srt.tmp` and renamed into place, so a half-written file never counts as done. This makes every Batch resumable for free (rerun the same folder), survives crashes and moving folders, and lets several windows work on one folder (each Video is re-checked right before it starts).

## Consequences

- A stopped or crashed Video is simply redone next time; partial work on one Video is never kept.
- Redo has to be explicit (`--force` / the "Redo" checkbox), since an existing `.srt` otherwise means "skip".
- Two Videos that map to the same `.srt` (Name clash) can't both be tracked, so only the first is captioned.
