import tempfile
import unittest
from types import SimpleNamespace

from utils.story_memory import StoryMemory, format_known_cast_block
from agents.story_memory_extractor import extract_chapter_memory
from interfaces.profile import CharacterProfile, StateEntry
from utils.profile_store import ProfileStore


def _beat(bid):
    return SimpleNamespace(beat_id=bid)


class StoryMemoryStoreTests(unittest.TestCase):
    def test_add_fact_requires_beats_and_kind(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = StoryMemory(tmp)
            self.assertIsNone(mem.add_fact({"kind": "event", "text": "x", "beat_ids": []}))
            self.assertIsNone(mem.add_fact({"kind": "bogus", "text": "x", "beat_ids": ["ch01_b001"]}))
            ok = mem.add_fact({"kind": "event", "text": "x", "beat_ids": ["ch01_b001"], "character_ids": ["a"]})
            self.assertIsNotNone(ok)
            self.assertEqual(len(mem.all_facts()), 1)

    def test_story_so_far_word_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = StoryMemory(tmp)
            mem.add_chapter(1, " ".join(["word"] * 500), [], [])
            out = mem.story_so_far(max_words=50)
            self.assertLessEqual(len(out.split()), 51)  # 50 + ellipsis token

    def test_known_cast_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = StoryMemory(tmp)
            mem.add_chapter(1, "s", [], [{"character_id": "alex", "aliases_used": ["Alex", "A"]}])
            block = format_known_cast_block(mem)
            self.assertIn("alex", block)
            self.assertIn("Alex", block)


class ExtractorTests(unittest.TestCase):
    def test_drops_fact_without_real_beat_and_applies_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = StoryMemory(tmp)
            store = ProfileStore(tmp)
            store.upsert(CharacterProfile(character_id="alex", display_name="alex",
                                          states=[StateEntry(kind="injury", text="old wound", from_beat="ch01_b000")]))

            def extract_fn(beats, prev_summary):
                return {
                    "facts": [
                        {"kind": "event", "text": "real", "beat_ids": ["ch01_b001"], "character_ids": ["alex"]},
                        {"kind": "event", "text": "fake", "beat_ids": ["ch99_b999"], "character_ids": ["alex"]},
                    ],
                    "state_changes": [
                        {"character_id": "alex", "kind": "injury", "text": "new bandage",
                         "from_beat": "ch01_b002", "evidence_beat_ids": ["ch01_b002"]},
                    ],
                    "summary": "a chapter", "open_threads": [], "cast": [],
                }

            res = extract_chapter_memory(1, [_beat("ch01_b001"), _beat("ch01_b002")], mem,
                                         extract_fn=extract_fn, profile_store=store)
            self.assertEqual(res["facts_added"], 1)
            self.assertEqual(res["facts_dropped"], 1)
            self.assertEqual(res["state_changes"], 1)
            prof = store.get("alex")
            injuries = [s for s in prof.states if s.kind == "injury"]
            # old injury closed, new injury open
            old = [s for s in injuries if s.text == "old wound"][0]
            self.assertEqual(old.to_beat, "ch01_b002")
            self.assertTrue(any(s.to_beat is None and s.text == "new bandage" for s in injuries))

    def test_conflict_detection_records_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = StoryMemory(tmp)
            mem.add_fact({"kind": "ability", "text": "cannot fly", "beat_ids": ["ch01_b000"], "character_ids": ["alex"]})

            def extract_fn(beats, prev_summary):
                return {"facts": [{"kind": "ability", "text": "can fly", "beat_ids": ["ch02_b001"], "character_ids": ["alex"]}],
                        "summary": "s", "open_threads": [], "cast": []}

            res = extract_chapter_memory(2, [_beat("ch02_b001")], mem,
                                         extract_fn=extract_fn, conflict_fn=lambda new_fact, prior_facts: True)
            self.assertEqual(res["conflicts"], 1)
            self.assertEqual(len(mem.conflicts()), 1)
            self.assertEqual(res["facts_added"], 1)  # pipeline keeps both facts


if __name__ == "__main__":
    unittest.main()
