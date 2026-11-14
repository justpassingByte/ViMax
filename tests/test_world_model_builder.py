import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from agents.world_model_builder import WorldModelBuilder, load_world_model
from interfaces import CharacterInScene, MangaBeat, MangaCharacterObservation, MangaChunkExtraction


_WORLD_ANALYSIS = {
    "environments": [
        {
            "key": "classroom",
            "name": "Rainy classroom",
            "env_type": "classroom",
            "indoor": True,
            "lighting": "dim fluorescent",
            "time_of_day": "afternoon",
            "weather": "rainy",
            "key_props": ["rain-streaked windows", "school desks"],
            "foreground_desc": "rows of school desks",
            "midground_desc": "",
            "background_desc": "rain-streaked windows",
            "beat_ids": ["ch01_b000"],
        },
        {
            "key": "classroom",
            "name": "Rainy classroom",
            "env_type": "classroom",
            "indoor": True,
            "beat_ids": ["ch01_b001"],
        },
    ],
    "beat_scene_graphs": [
        {
            "beat_id": "ch01_b000",
            "environment_key": "classroom",
            "entities": [
                {"entity_id": "Jiwoo", "entity_type": "character", "label": "Jiwoo"},
                {"entity_id": "window", "entity_type": "object", "label": "window"},
            ],
            "relations": [
                {"subject": "Jiwoo", "relation": "facing", "obj": "window", "confidence": 0.95},
                # Relation referencing an entity that does not exist -> must be dropped
                {"subject": "ghost", "relation": "facing", "obj": "Jiwoo", "confidence": 0.9},
            ],
            "camera_position": "behind Jiwoo, low angle",
        },
        # Graph for a beat that does not exist in the chapter -> must be dropped
        {
            "beat_id": "ch99_b999",
            "environment_key": "classroom",
            "entities": [],
            "relations": [],
            "camera_position": "",
        },
    ],
    "events": [
        {
            "description": "Jiwoo watches the rain start",
            "event_type": "action",
            "beat_ids": ["ch01_b000", "ch99_b999"],
            "character_keys": ["Jiwoo"],
            "environment_key": "classroom",
            "emotional_tone": "calm",
        },
        # Empty description -> must be dropped
        {"description": "", "beat_ids": ["ch01_b000"]},
    ],
}


class CountingFakeModel(Runnable):
    """Fake chat model returning the canned world analysis, counting LLM calls."""

    def __init__(self) -> None:
        self.call_count = 0

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError

    async def ainvoke(self, messages, config=None, **kwargs):
        self.call_count += 1
        return AIMessage(content=json.dumps(_WORLD_ANALYSIS))


def _make_chunk() -> MangaChunkExtraction:
    return MangaChunkExtraction(
        chapter=1,
        start_page_idx=0,
        end_page_idx=0,
        characters_observed=[
            MangaCharacterObservation(
                name="Jiwoo",
                species_or_form="human",
                appearance_summary="Teen boy with chestnut hair",
                outfit_summary="School uniform",
                page_idx=0,
            ),
            MangaCharacterObservation(
                name="Jiwoo",
                species_or_form="human",
                appearance_summary="Teen boy with chestnut hair",
                outfit_summary="Wet uniform sleeves",
                page_idx=0,
            ),
        ],
        beats=[
            MangaBeat(
                beat_id="ch01_b000",
                chapter=1,
                page_idx=0,
                setting="Classroom",
                action_desc="Jiwoo looks out the window.",
                characters_present=["Jiwoo"],
                anchor_image_path="",
            ),
            MangaBeat(
                beat_id="ch01_b001",
                chapter=1,
                page_idx=0,
                setting="Classroom",
                action_desc="Jiwoo sighs.",
                characters_present=["Jiwoo"],
                anchor_image_path="",
            ),
        ],
        chunk_tail_summary="",
    )


def _make_roster() -> list:
    return [
        CharacterInScene(
            idx=0,
            identifier_in_scene="Jiwoo",
            is_visible=True,
            static_features="Teen boy with chestnut hair",
            dynamic_features="School uniform",
        )
    ]


class _UnusedChatModel(Runnable):
    """Placeholder chat model — never invoked by the deterministic bible builder."""

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError


class WorldModelBuilderTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_world_model_merges_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_root = Path(tmp)
            model = await WorldModelBuilder(CountingFakeModel()).build_world_model(
                chunks=[_make_chunk()],
                characters=_make_roster(),
                working_dir=work_root,
            )
            # Two same-key environment candidates merge into ONE location.
            self.assertEqual(len(model.environments), 1)
            env = model.environments[0]
            self.assertEqual(env.environment_id, "env_001")
            self.assertEqual(sorted(env.source_beat_ids), ["ch01_b000", "ch01_b001"])
            self.assertIn("rain-streaked windows", env.background_desc)

            # Graph for unknown beat dropped; dangling relation dropped.
            self.assertEqual(len(model.beat_scene_graphs), 1)
            graph = model.beat_scene_graphs[0]
            self.assertEqual(graph.beat_id, "ch01_b000")
            self.assertEqual(graph.environment_id, "env_001")
            self.assertEqual(len(graph.relations), 1)
            self.assertEqual(graph.relations[0].relation, "facing")

            # Empty-description event dropped; unknown beat ids filtered out.
            self.assertEqual(len(model.events), 1)
            self.assertEqual(model.events[0].beat_ids, ["ch01_b000"])
            self.assertEqual(model.events[0].environment_id, "env_001")

            # Character bible is deterministic and keyed by canonical roster names.
            keys = [c.character_key for c in model.characters]
            self.assertIn("Jiwoo", keys)
            jiwoo = model.character_by_key("Jiwoo")
            assert jiwoo is not None
            self.assertIn("chestnut hair", jiwoo.permanent_features)
            # Beat evidence backfilled from characters_present.
            self.assertEqual(sorted(jiwoo.source_beat_ids), ["ch01_b000", "ch01_b001"])

            # World model persisted to disk.
            self.assertTrue((work_root / "vision_world" / "world_model.json").is_file())
            self.assertTrue((work_root / "vision_world" / "analysis_batch_000.json").is_file())

    async def test_second_build_reuses_cache_without_llm_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_root = Path(tmp)
            fake = CountingFakeModel()
            builder = WorldModelBuilder(fake)
            await builder.build_world_model(chunks=[_make_chunk()], characters=_make_roster(), working_dir=work_root)
            first_count = fake.call_count
            self.assertGreaterEqual(first_count, 1)

            await WorldModelBuilder(fake).build_world_model(
                chunks=[_make_chunk()], characters=_make_roster(), working_dir=work_root
            )
            self.assertEqual(fake.call_count, first_count, "Cached analysis batch must not trigger new LLM calls")

    def test_build_character_bible_deterministic_merge(self):
        builder = WorldModelBuilder(_UnusedChatModel())
        bible = builder.build_character_bible(chunks=[_make_chunk()], characters=_make_roster())
        by_key = {e.character_key: e for e in bible}
        self.assertEqual(set(by_key), {"Jiwoo"})
        jiwoo = by_key["Jiwoo"]
        # Two observations -> permanent features joined, confidence boosted.
        self.assertIn("chestnut hair", jiwoo.permanent_features)
        self.assertIn("School uniform", jiwoo.temporary_features)
        self.assertIn("Wet uniform sleeves", jiwoo.temporary_features)
        self.assertGreater(jiwoo.confidence, 0.35)

    def test_load_world_model_missing_or_corrupt_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_world_model(tmp))
            broken_dir = Path(tmp) / "vision_world"
            broken_dir.mkdir()
            (broken_dir / "world_model.json").write_text("{not valid json", encoding="utf-8")
            self.assertIsNone(load_world_model(tmp))


if __name__ == "__main__":
    unittest.main()
