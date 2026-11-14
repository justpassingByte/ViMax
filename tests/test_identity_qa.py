import asyncio
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from agents.identity_qa import check_frame, qa_and_retake_keyframes
from interfaces.profile import CharacterProfile, SourceCrop
from utils.profile_store import ProfileStore


def _png(path, color=(10, 20, 30), size=(1600, 900)):
    Image.new("RGB", size, color).save(path)
    return path


class FakeEmbedder:
    """Returns a stored vector per path; unknown paths -> fixed 'frame' vector per test."""
    name = "fake"

    def __init__(self, mapping, default=None):
        self.mapping = mapping
        self.default = default if default is not None else [0.0, 1.0]

    def embed_images(self, paths):
        return np.asarray([self.mapping.get(p, self.default) for p in paths], dtype="float32")


def _store_with_profile(tmp, cid, ref_paths):
    store = ProfileStore(tmp)
    prof = CharacterProfile(
        character_id=cid, display_name=cid, aliases=[cid],
        source_crops=[SourceCrop(path=p, beat_id="ch01_b000") for p in ref_paths],
    )
    store.upsert(prof)
    return store


def _vlm_ok(frame_path, expected, outfit_desc):
    return {"detected": [{"character_id": c, "box": None} for c in expected],
            "outfit_ok": {c: True for c in expected}, "has_lettering": False}


class CheckFrameTests(unittest.TestCase):
    def test_passes_when_identity_matches(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r1 = _png(os.path.join(tmp, "r1.png"))
                r2 = _png(os.path.join(tmp, "r2.png"))
                frame = _png(os.path.join(tmp, "frame.png"))
                store = _store_with_profile(tmp, "alex", [r1, r2])
                emb = FakeEmbedder({r1: [1.0, 0.0], r2: [0.98, 0.02], frame: [0.97, 0.05]})
                fc = await check_frame(frame, 0, ["alex"], "ch01_b000", store, emb, _vlm_ok, identity_margin=0.15)
                self.assertTrue(fc.passed, msg=str(fc.reasons))
                self.assertIn("alex", fc.identity_scores)
        asyncio.run(run())

    def test_fails_on_low_identity(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                r1 = _png(os.path.join(tmp, "r1.png"))
                r2 = _png(os.path.join(tmp, "r2.png"))
                frame = _png(os.path.join(tmp, "frame.png"))
                store = _store_with_profile(tmp, "alex", [r1, r2])
                emb = FakeEmbedder({r1: [1.0, 0.0], r2: [0.99, 0.01], frame: [0.0, 1.0]})
                fc = await check_frame(frame, 0, ["alex"], "ch01_b000", store, emb, _vlm_ok, identity_margin=0.15)
                self.assertFalse(fc.passed)
                self.assertTrue(any("identity" in r for r in fc.reasons))
        asyncio.run(run())

    def test_fails_on_lettering_and_outfit(self):
        async def run():
            with tempfile.TemporaryDirectory() as tmp:
                frame = _png(os.path.join(tmp, "frame.png"))
                store = _store_with_profile(tmp, "alex", [])
                def vlm(frame_path, expected, outfit_desc):
                    return {"detected": [{"character_id": "alex"}], "outfit_ok": {"alex": False}, "has_lettering": True}
                fc = await check_frame(frame, 0, ["alex"], "ch01_b000", store, None, vlm)
                self.assertFalse(fc.passed)
                self.assertTrue(any("outfit" in r for r in fc.reasons))
                self.assertTrue(any("lettering" in r for r in fc.reasons))
        asyncio.run(run())


class RetakeLoopTests(unittest.TestCase):
    def test_retakes_then_passes_three_renders(self):
        async def run():
            calls = {"n": 0}
            def render_frame(shot_idx, frame, attempt):
                calls["n"] += 1
                return f"/f/{shot_idx}_{attempt}.png"
            # fail attempts 0 and 1, pass attempt 2
            state = {"attempt": -1}
            def check(frame_path, shot):
                from agents.identity_qa import FrameCheck
                state["attempt"] += 1
                passed = state["attempt"] >= 2
                return FrameCheck(shot_idx=shot["shot_idx"], passed=passed,
                                  reasons=[] if passed else ["low"], identity_scores={"a": 0.5})
            res = await qa_and_retake_keyframes(
                [{"shot_idx": 0, "frame": "first"}], render_frame=render_frame, check=check, max_retakes=2)
            self.assertEqual(res["n_renders"], 3)
            self.assertEqual(res["exceptions"], [])
        asyncio.run(run())

    def test_persistent_failure_records_exception(self):
        async def run():
            def render_frame(shot_idx, frame, attempt):
                return f"/f/{shot_idx}_{attempt}.png"
            def check(frame_path, shot):
                from agents.identity_qa import FrameCheck
                return FrameCheck(shot_idx=shot["shot_idx"], passed=False, reasons=["low"], identity_scores={"a": 0.3})
            res = await qa_and_retake_keyframes(
                [{"shot_idx": 5, "frame": "first"}], render_frame=render_frame, check=check, max_retakes=2)
            self.assertEqual(res["n_renders"], 3)  # initial + 2 retakes
            self.assertEqual(len(res["exceptions"]), 1)
            self.assertEqual(res["exceptions"][0]["shot_idx"], 5)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
