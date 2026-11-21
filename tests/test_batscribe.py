"""Tests for batscribe. No GPU, ffmpeg or media files needed: the model and audio are faked.

Run:  .venv\\Scripts\\python.exe -m unittest discover tests
"""
import re
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

    def __init__(self, *args, **kwargs):
        pass

    def transcribe(self, audio, **kwargs):
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
        return segments(), NS(duration=dur)


class FakeAudio(np.ndarray):
    """Silent audio that remembers which video it came from (so FakeModel can fail on purpose)."""


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
                        "add_cuda_dll_dirs": bs.add_cuda_dll_dirs}
        bs.load_audio, bs.HISTORY_LOG = fake_load_audio, self.dir / "history.log"
        bs.add_cuda_dll_dirs = lambda: None
        self.old_fw = sys.modules.get("faster_whisper")
        sys.modules["faster_whisper"] = types.SimpleNamespace(WhisperModel=FakeModel)
        FakeModel.fail_on = {}
        self.events = []

    def tearDown(self):
        for name, value in self.patches.items():
            setattr(bs, name, value)
        if self.old_fw is None:
            sys.modules.pop("faster_whisper", None)
        else:
            sys.modules["faster_whisper"] = self.old_fw
        for f in sorted(self.dir.rglob("*"), reverse=True):
            f.unlink()
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


if __name__ == "__main__":
    unittest.main()
