import tempfile
import unittest
from pathlib import Path

from interfaces.profile import CharacterProfile, Outfit, SourceCrop, StateEntry
from utils.profile_store import ProfileStore, beat_key


def _prof(cid, aliases=None, outfits=None, states=None, cooccur=None, status="draft"):
    return CharacterProfile(
        character_id=cid, display_name=cid, aliases=aliases or [],
        outfits=outfits or [], states=states or [],
        cooccur_beat_ids=cooccur or [], status=status,
    )


class BeatKeyTests(unittest.TestCase):
    def test_parse_and_order(self):
        self.assertEqual(beat_key("ch01_b012"), (1, 12))
        self.assertEqual(beat_key("ch02_b000"), (2, 0))
        self.assertLess(beat_key("ch01_b099"), beat_key("ch02_b000"))
        self.assertEqual(beat_key("garbage"), (0, 0))


class OutfitAtTests(unittest.TestCase):
    def test_outfit_boundaries_across_chapters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(tmp)
            prof = _prof("alex", outfits=[
                Outfit(outfit_id="o1", label="uniform", valid_from_beat="ch01_b000", valid_to_beat="ch01_b017"),
                Outfit(outfit_id="o2", label="casual", valid_from_beat="ch01_b018", valid_to_beat=None),
            ])
            store.upsert(prof)
            self.assertEqual(store.outfit_at("alex", "ch01_b005").outfit_id, "o1")
            self.assertEqual(store.outfit_at("alex", "ch01_b017").outfit_id, "o1")
            self.assertEqual(store.outfit_at("alex", "ch01_b018").outfit_id, "o2")
            self.assertEqual(store.outfit_at("alex", "ch02_b003").outfit_id, "o2")  # open-ended

    def test_states_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(tmp)
            store.upsert(_prof("alex", states=[
                StateEntry(kind="injury", text="wounded chest", from_beat="ch01_b003", to_beat="ch01_b020"),
            ]))
            self.assertTrue(store.states_at("alex", "ch01_b010"))
            self.assertFalse(store.states_at("alex", "ch01_b021"))


class MatchTests(unittest.TestCase):
    def test_exact_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(tmp)
            store.upsert(_prof("alex", aliases=["Alex (Human)"]))
            pid, score, decision = store.match("alex (human)")
            self.assertEqual(pid, "alex")
            self.assertEqual(decision, "exact")

    def test_auto_merge_and_suggest(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(tmp)
            store.upsert(_prof("alex"))
            pid, score, decision = store.match("mystery", score_fn=lambda p: 0.9)
            self.assertEqual(decision, "auto_merge")
            pid, score, decision = store.match("mystery", score_fn=lambda p: 0.75)
            self.assertEqual(decision, "suggest")
            pid, score, decision = store.match("mystery", score_fn=lambda p: 0.3)
            self.assertEqual(decision, "new")

    def test_never_merge_cooccurring_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(tmp)
            store.upsert(_prof("alex", cooccur=["ch01_b004"]))
            # candidate shares a panel beat -> must not auto-merge despite high score
            pid, score, decision = store.match("other", candidate_beats=["ch01_b004"], score_fn=lambda p: 0.99)
            self.assertEqual(decision, "new")


class LockedTests(unittest.TestCase):
    def test_locked_profile_only_accepts_added_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = ProfileStore(tmp)
            store.upsert(_prof("alex", status="locked", outfits=[
                Outfit(outfit_id="o1", valid_from_beat="ch01_b000"),
            ]))
            # attempt to overwrite outfits + add a crop
            overwrite = _prof("alex", status="locked", outfits=[
                Outfit(outfit_id="HACKED", valid_from_beat="ch01_b000"),
            ])
            overwrite.source_crops = [SourceCrop(path="/new/crop.png", beat_id="ch01_b002")]
            store.upsert(overwrite)
            prof = store.get("alex")
            self.assertEqual([o.outfit_id for o in prof.outfits], ["o1"])  # outfits preserved
            self.assertIn("/new/crop.png", [c.path for c in prof.source_crops])  # crop added


if __name__ == "__main__":
    unittest.main()
