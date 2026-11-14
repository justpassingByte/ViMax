import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from agents.unattended import BudgetTracker, aggregate_chapter_exceptions, check_clip
from agents.identity_qa import FrameCheck
from pipelines.apply_decisions import apply_decisions


class BudgetTests(unittest.TestCase):
    def test_unlimited_by_default(self):
        b = BudgetTracker(0, 0)
        for _ in range(100):
            self.assertTrue(b.add_image())
            self.assertTrue(b.add_clip())
        self.assertFalse(b.over_budget)

    def test_image_budget_stops(self):
        b = BudgetTracker(max_images=3)
        self.assertTrue(b.add_image())   # 1
        self.assertTrue(b.add_image())   # 2
        self.assertTrue(b.add_image())   # 3
        self.assertFalse(b.add_image())  # 4 -> over
        self.assertTrue(b.over_budget)
        self.assertIn("image budget", b.stopped_reason)


class CheckClipTests(unittest.TestCase):
    def test_clip_fails_when_a_frame_fails(self):
        async def run():
            def extract(video):
                return ["/f0.png", "/f1.png", "/f2.png"]
            def check(frame_path, shot, position):
                passed = position != "mid"
                return FrameCheck(shot_idx=0, passed=passed, reasons=[] if passed else ["mid drift"])
            res = await check_clip("/v.mp4", {"shot_idx": 0}, extract_frames_fn=extract, check_frame_fn=check)
            self.assertFalse(res["passed"])
            self.assertIn("mid drift", res["reasons"])
        asyncio.run(run())


class AggregateTests(unittest.TestCase):
    def test_aggregates_scene_exceptions_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s0 = root / "scene_0"
            s0.mkdir()
            (s0 / "exceptions.json").write_text(json.dumps([{"shot_idx": 1, "frame": "first", "reasons": ["drift"]}]), encoding="utf-8")
            (s0 / "canon_report.json").write_text(json.dumps({"passed": True, "issues": [{"kind": "uncited"}]}), encoding="utf-8")
            (root / "story_memory").mkdir()
            (root / "story_memory" / "conflicts.json").write_text(json.dumps([{"new": {"text": "x"}}]), encoding="utf-8")
            out = aggregate_chapter_exceptions(root, 1)
            self.assertEqual(len(out["keyframe_exceptions"]), 1)
            self.assertEqual(len(out["canon_repairs"]), 1)
            self.assertEqual(len(out["story_conflicts"]), 1)
            self.assertTrue((root / "exceptions" / "ch01.json").is_file())


class ApplyDecisionsTests(unittest.TestCase):
    def test_retake_deletes_only_that_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot0 = root / "scene_0" / "shots" / "0"
            shot1 = root / "scene_0" / "shots" / "1"
            shot0.mkdir(parents=True)
            shot1.mkdir(parents=True)
            (shot0 / "first_frame.png").write_bytes(b"a")
            (shot1 / "first_frame.png").write_bytes(b"b")
            exc = root / "exceptions"
            exc.mkdir()
            (exc / "ch01.decisions.json").write_text(
                json.dumps([{"action": "retake", "scene": "scene_0", "shot_idx": 0, "frame": "first"}]),
                encoding="utf-8",
            )
            res = apply_decisions(root, "ch01")
            self.assertEqual(res["retakes"], 1)
            self.assertFalse((shot0 / "first_frame.png").is_file())   # deleted
            self.assertTrue((shot1 / "first_frame.png").is_file())    # untouched


if __name__ == "__main__":
    unittest.main()
