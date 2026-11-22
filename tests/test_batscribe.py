"""Tests for batscribe. No GPU, ffmpeg or media files needed: the model and audio are faked.

Run:  .venv\\Scripts\\python.exe -m unittest discover tests
"""
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import batscribe as bs  # noqa: E402

SR = 16000


class FakeModel:
    """Stands in for faster_whisper.WhisperModel: one 4-second segment of two words at a time."""
    fail_on = {}  # file stem -> exception message raised when that video is transcribed
    calls = 0
    languages = []  # the language argument of each transcribe call
    init_kwargs = {}

    def __init__(self, *args, **kwargs):
        FakeModel.init_kwargs = kwargs

    def transcribe(self, audio, **kwargs):
        FakeModel.calls += 1
        FakeModel.languages.append(kwargs.get("language"))
        if audio.name in self.fail_on:
            raise RuntimeError(self.fail_on[audio.name])
        dur = len(audio) / SR

        def segments():
            t, i = 0.0, 0
            while t < dur:
                e = min(t + 4.0, dur)
                yield NS(start=t, end=e, words=[NS(word=f" word{i}", start=t, end=t + 1.5),
                                                NS(word=" ends here.", start=t + 1.6, end=e)])
                t, i = e, i + 1
        return segments(), NS(duration=dur, language=kwargs.get("language") or "de")  # "detects" German


class FakeBatched:
    """Stands in for faster_whisper.BatchedInferencePipeline."""
    calls = 0

    def __init__(self, model):
        self.model = model

    def transcribe(self, audio, **kwargs):
        FakeBatched.calls += 1
        return self.model.transcribe(audio, **kwargs)


class FakeAudio(np.ndarray):
    """Silent audio that remembers which video it came from (so FakeModel can fail on purpose)."""

    def __array_finalize__(self, obj):  # keep the name on slices (long videos are split into chunks)
        self.name = getattr(obj, "name", None)


def fake_load_audio(path):
    audio = np.zeros(int(SR * float(path.read_text())), dtype=np.float32).view(FakeAudio)
    audio.name = path.stem
    return audio


class FormattingTests(unittest.TestCase):
    def test_fmt_ts(self):
        self.assertEqual(bs.fmt_ts(0), "00:00:00,000")
        self.assertEqual(bs.fmt_ts(3725.5), "01:02:05,500")
        self.assertEqual(bs.fmt_ts(-1), "00:00:00,000")

    def test_wrap_lines_keeps_short_text_and_splits_long_text_in_two(self):
        self.assertEqual(bs.wrap_lines("short line"), "short line")
        wrapped = bs.wrap_lines("this subtitle line is definitely longer than forty two characters")
        self.assertEqual(len(wrapped.splitlines()), 2)

    def test_build_cues_respects_length_and_duration_limits(self):
        words = [NS(word=f" w{i}", start=i * 0.4, end=i * 0.4 + 0.3) for i in range(100)]
        cues = bs.build_cues(words)
        self.assertEqual(" ".join(text for _, _, text in cues), " ".join(f"w{i}" for i in range(100)))
        for start, end, text in cues:
            self.assertLessEqual(len(text), bs.MAX_LINE * bs.MAX_LINES)
            self.assertLessEqual(end - start, bs.MAX_CUE_SECONDS)

    def test_write_srt_format(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "v.srt"
            bs.write_srt([(0.0, 1.5, "Hello there."), (2.0, 2.0, "Zero length cue.")], out)
            blocks = out.read_text(encoding="utf-8").strip().split("\n\n")
            self.assertEqual(blocks[0], "1\n00:00:00,000 --> 00:00:01,500\nHello there.")
            self.assertEqual(blocks[1], "2\n00:00:02,000 --> 00:00:02,500\nZero length cue.")
            self.assertEqual(list(Path(d).iterdir()), [out])  # no .tmp left behind


class RunBatchTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.patches = {"load_audio": bs.load_audio, "HISTORY_LOG": bs.HISTORY_LOG,
                        "add_cuda_dll_dirs": bs.add_cuda_dll_dirs, "CHUNK_SECONDS": bs.CHUNK_SECONDS}
        bs.load_audio, bs.HISTORY_LOG = fake_load_audio, self.dir / "history.log"
        bs.add_cuda_dll_dirs = lambda: None
        self.old_fw = sys.modules.get("faster_whisper")
        sys.modules["faster_whisper"] = types.SimpleNamespace(WhisperModel=FakeModel,
                                                              BatchedInferencePipeline=FakeBatched)
        FakeModel.fail_on, FakeModel.calls, FakeBatched.calls, FakeModel.languages = {}, 0, 0, []
        self.events = []

    def tearDown(self):
        for name, value in self.patches.items():
            setattr(bs, name, value)
        if self.old_fw is None:
            sys.modules.pop("faster_whisper", None)
        else:
            sys.modules["faster_whisper"] = self.old_fw
        for f in sorted(self.dir.rglob("*"), reverse=True):
            f.rmdir() if f.is_dir() else f.unlink()
        self.dir.rmdir()

    def videos(self, **seconds):
        """Create fake .mp4 files whose content is their audio length in seconds."""
        paths = []
        for name, sec in seconds.items():
            p = self.dir / f"{name}.mp4"
            p.write_text(str(sec))
            paths.append(p)
        return paths

    def run_batch(self, todo, durations, **kwargs):
        return bs.run_batch(todo, durations, self.dir / "errors.log",
                            on_event=lambda kind, **e: self.events.append((kind, e)), **kwargs)

    def of_kind(self, kind):
        return [e for k, e in self.events if k == kind]

    def test_all_videos_captioned_and_logged(self):
        vids = self.videos(a=30, b=20)
        self.assertEqual(self.run_batch(vids, {v: float(v.read_text()) for v in vids}), 0)
        for v in vids:
            self.assertTrue(v.with_suffix(".srt").exists())
        history = bs.HISTORY_LOG.read_text(encoding="utf-8")
        self.assertIn("BATCH START  2 videos", history)
        self.assertEqual(len(re.findall(r"  OK  ", history)), 2)
        self.assertIn("BATCH END  2 ok, 0 failed", history)

    def test_progress_never_exceeds_100_when_a_duration_probe_failed(self):
        # Regression: one ffprobe failure (duration 0) made overall progress reach 115%.
        vids = self.videos(a=30, b=60)
        self.run_batch(vids, {vids[0]: 30.0, vids[1]: 0.0})
        overall = [e["overall_pct"] for e in self.of_kind("progress")]
        self.assertLessEqual(max(overall), 100)
        self.assertGreater(overall[-1], 99)

    def test_skips_video_whose_srt_appeared_after_scan(self):
        vids = self.videos(a=10, b=10)
        vids[1].with_suffix(".srt").write_text("made by another window", encoding="utf-8")
        self.run_batch(vids, {v: 10.0 for v in vids})
        self.assertEqual([e["video"] for e in self.of_kind("skipped")], [vids[1]])
        self.assertEqual(vids[1].with_suffix(".srt").read_text(encoding="utf-8"), "made by another window")

    def test_force_redoes_existing_srt(self):
        (vid,) = self.videos(a=10)
        vid.with_suffix(".srt").write_text("old", encoding="utf-8")
        self.run_batch([vid], {vid: 10.0}, skip_existing=False)
        self.assertNotEqual(vid.with_suffix(".srt").read_text(encoding="utf-8"), "old")

    def test_fast_mode_uses_batched_pipeline(self):
        vids = self.videos(a=10, b=5)
        self.assertEqual(self.run_batch(vids, {v: 10.0 for v in vids}, fast=True), 0)
        self.assertEqual(FakeBatched.calls, 2)
        self.assertTrue(all(v.with_suffix(".srt").exists() for v in vids))
        self.assertIn("fast mode", bs.HISTORY_LOG.read_text(encoding="utf-8"))

    def test_normal_mode_does_not_use_batched_pipeline(self):
        (vid,) = self.videos(a=10)
        self.run_batch([vid], {vid: 10.0})
        self.assertEqual((FakeBatched.calls, FakeModel.calls), (0, 1))
        self.assertNotIn("fast mode", bs.HISTORY_LOG.read_text(encoding="utf-8"))

    def test_ordinary_failure_is_logged_and_batch_continues(self):
        vids = self.videos(a=10, bad=10, c=10)
        FakeModel.fail_on = {"bad": "Invalid data found when processing input"}
        self.assertEqual(self.run_batch(vids, {v: 10.0 for v in vids}), 1)
        self.assertTrue(vids[2].with_suffix(".srt").exists())
        finished = self.of_kind("finished")[0]
        self.assertEqual((finished["ok"], finished["failed"], finished["gpu_error"]), (2, 1, None))
        self.assertIn("bad.mp4", (self.dir / "errors.log").read_text(encoding="utf-8"))

    def test_gpu_error_stops_the_batch(self):
        vids = self.videos(a=10, gpu=10, c=10)
        FakeModel.fail_on = {"gpu": "CUDA failed with error out of memory"}
        self.run_batch(vids, {v: 10.0 for v in vids})
        self.assertNotIn(vids[2], [e["video"] for e in self.of_kind("start")])  # not attempted
        finished = self.of_kind("finished")[0]
        self.assertTrue(finished["stopped"])
        self.assertIn("out of memory", finished["gpu_error"])
        self.assertIn("GPU error", bs.HISTORY_LOG.read_text(encoding="utf-8"))

    def test_stop_request_abandons_current_video(self):
        vids = self.videos(a=60, b=10)
        self.run_batch(vids, {v: float(v.read_text()) for v in vids},
                       should_stop=lambda: len(self.of_kind("progress")) >= 2)
        self.assertFalse(any(v.with_suffix(".srt").exists() for v in vids))
        self.assertTrue(self.of_kind("finished")[0]["stopped"])

    def test_long_video_is_transcribed_in_chunks_with_continuous_timestamps(self):
        bs.CHUNK_SECONDS = 20
        (vid,) = self.videos(long=50)
        self.run_batch([vid], {vid: 50.0})
        self.assertEqual(FakeModel.calls, 3)
        starts = [line.split(" --> ")[0] for line in vid.with_suffix(".srt").read_text(encoding="utf-8").splitlines()
                  if " --> " in line]
        self.assertEqual(starts, sorted(starts))
        self.assertGreater(starts[-1], "00:00:45")  # the last chunk's cues are offset, not restarted at 0
        self.assertGreater(self.of_kind("progress")[-1]["file_pct"], 99)

    def test_language_detected_on_first_chunk_is_kept_for_the_rest(self):
        bs.CHUNK_SECONDS = 20
        vids = self.videos(a=50, b=10)
        self.run_batch(vids, {v: float(v.read_text()) for v in vids})
        # each video detects afresh on its first chunk, then reuses that language
        self.assertEqual(FakeModel.languages, [None, "de", "de", None])
        self.assertIn("language de", bs.HISTORY_LOG.read_text(encoding="utf-8"))

    def test_chosen_language_is_passed_to_every_chunk(self):
        bs.CHUNK_SECONDS = 20
        (vid,) = self.videos(a=50)
        self.run_batch([vid], {vid: 50.0}, language="en", fast=True)
        self.assertEqual(FakeModel.languages, ["en", "en", "en"])

    def test_cpu_device_uses_cpu_compute_type(self):
        (vid,) = self.videos(a=10)
        self.run_batch([vid], {vid: 10.0}, device="cpu")
        self.assertEqual(FakeModel.init_kwargs, {"device": "cpu", "compute_type": "int8"})
        self.assertEqual(self.of_kind("loading")[0]["device"], "cpu")

    def test_unwritable_error_log_does_not_abort_the_batch(self):
        vids = self.videos(bad=10, c=10)
        FakeModel.fail_on = {"bad": "Invalid data found when processing input"}
        (self.dir / "errors.log").mkdir()  # opening it for writing fails
        self.assertEqual(self.run_batch(vids, {v: 10.0 for v in vids}), 1)
        self.assertTrue(vids[1].with_suffix(".srt").exists())

    def test_unfinished_batch_is_reported_once(self):
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        bs.HISTORY_LOG.write_text(
            f"2026-01-01 10:00:00  BATCH START  3 videos  [pid {dead.pid}]\n"
            "2026-01-01 10:05:00  BATCH START  1 videos  [pid 1]\n"  # finished below
            "2026-01-01 10:06:00  BATCH END  1 ok, 0 failed  [pid 1]\n", encoding="utf-8")
        bs.log_batch("START  running right now")  # this process is alive, so not reported
        lost = bs.unfinished_batches()
        self.assertEqual(len(lost), 1)
        self.assertIn("3 videos", lost[0])
        self.assertEqual(bs.unfinished_batches(), [])

    def test_collect_includes_audio_files_but_not_other_files(self):
        (vid,) = self.videos(a=1)
        for name in ("talk.mp3", "memo.M4A", "notes.txt", "a.srt"):
            (self.dir / name).write_text("1")
        self.assertEqual([p.name for p in bs.collect([str(self.dir)])], ["a.mp4", "memo.M4A", "talk.mp3"])
        self.assertEqual(bs.collect([str(self.dir / "talk.mp3")]), [self.dir / "talk.mp3"])

    def test_name_clashes_and_other_captions(self):
        a_mp4, a_mkv, b = self.videos(a=1, b=1)[0], self.dir / "a.mkv", self.dir / "b.mp4"
        a_mkv.write_text("1")
        for name in ("a.en.srt", "a.vtt", "a.srt", "ab.srt", "b.srt"):
            (self.dir / name).write_text("x")
        self.assertEqual(bs.name_clashes(bs.collect([str(self.dir)])), {a_mp4: a_mkv})
        self.assertEqual(bs.other_captions(a_mp4), ["a.en.srt", "a.vtt"])
        self.assertEqual(bs.other_captions(b), [])


class ChunkTests(unittest.TestCase):
    def test_cuts_at_quiet_moment_and_covers_everything(self):
        original = bs.CHUNK_SECONDS
        bs.CHUNK_SECONDS = 20
        try:
            audio = np.random.default_rng(0).uniform(-0.5, 0.5, 45 * SR).astype(np.float32)
            audio[int(16.0 * SR):int(16.3 * SR)] = 0  # a pause shortly before the 20 s limit
            bounds = bs.chunk_bounds(audio)
        finally:
            bs.CHUNK_SECONDS = original
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], len(audio))
        self.assertTrue(all(b1 == a2 for (_, b1), (a2, _) in zip(bounds, bounds[1:])))
        self.assertTrue(16.0 <= bounds[0][1] / SR <= 16.3)
        self.assertTrue(all(b - a <= 20 * SR for a, b in bounds))


class PrerequisiteTests(unittest.TestCase):
    def test_missing_ffmpeg_is_reported(self):
        original = bs.shutil.which
        bs.shutil.which = lambda tool: None
        try:
            problems = bs.check_prerequisites()
        finally:
            bs.shutil.which = original
        self.assertTrue(any("ffmpeg was not found" in p for p in problems))
        self.assertTrue(any("ffprobe was not found" in p for p in problems))

    def test_no_gpu_falls_back_to_cpu_unless_cuda_is_required(self):
        original = bs.cuda_available
        bs.cuda_available = lambda: False
        try:
            self.assertEqual(bs.resolve_device("auto"), "cpu")
            self.assertEqual(bs.resolve_device("cuda"), "cuda")
            self.assertFalse(any("GPU" in p for p in bs.check_prerequisites("auto")))
            if "could not be loaded" not in " ".join(bs.check_prerequisites("cuda")):  # engine installed
                self.assertTrue(any("No NVIDIA GPU" in p for p in bs.check_prerequisites("cuda")))
        finally:
            bs.cuda_available = original


class LanguageTests(unittest.TestCase):
    def test_language_list_matches_whisper(self):
        try:
            from faster_whisper.tokenizer import _LANGUAGE_CODES
        except ImportError:
            self.skipTest("faster-whisper not installed")
        self.assertEqual(set(bs.LANGUAGES), set(_LANGUAGE_CODES))


if __name__ == "__main__":
    unittest.main()
