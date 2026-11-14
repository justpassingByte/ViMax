import tempfile
import unittest
from types import SimpleNamespace

from utils.profile_store import ProfileStore
from utils.profile_builder import build_draft_profiles


def _beat(bid, present):
    return SimpleNamespace(beat_id=bid, characters_present=present)


class _Char:
    def __init__(self, ident, static=""):
        self.identifier_in_scene = ident
        self.static_features = static


class ProfileBuilderTests(unittest.TestCase):
    def test_builds_profiles_with_cooccurrence_and_blocks_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(tmp)
            chars = [_Char("Alex", "silver hair"), _Char("Bea", "red coat")]
            seeds = {"Alex": ["/a.png"], "Bea": ["/b.png"]}
            beats = {0: [_beat("ch01_b000", ["Alex"]), _beat("ch01_b004", ["Alex", "Bea"])]}
            touched = build_draft_profiles(chars, seeds, beats, store)
            self.assertEqual(set(touched), {"alex", "bea"})
            alex = store.get("Alex")
            self.assertEqual(alex.permanent_features, "silver hair")
            self.assertIn("ch01_b004", alex.cooccur_beat_ids)
            # default outfit spans from first beat
            self.assertEqual(alex.outfits[0].valid_from_beat, "ch01_b000")
            # a candidate sharing the co-occur beat must not auto-merge into Alex
            _pid, _s, decision = store.match("ghost", candidate_beats=["ch01_b004"], score_fn=lambda p: 0.99)
            self.assertEqual(decision, "new")

    def test_rerun_preserves_locked_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(tmp)
            build_draft_profiles([_Char("Alex")], {"Alex": []}, {}, store)
            store.set_status("alex", "locked")
            build_draft_profiles([_Char("Alex")], {"Alex": []}, {}, store)
            self.assertEqual(store.get("Alex").status, "locked")


if __name__ == "__main__":
    unittest.main()
