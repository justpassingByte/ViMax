import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from agents.cinematic_director import (
    CinematicDirector,
    extract_shot_beat_refs,
    format_world_context,
)
from interfaces import (
    BeatExpansionDirective,
    CharacterInScene,
    ExpansionShotDirective,
    MangaBeat,
    MangaChunkExtraction,
    SceneExpansionPlan,
    ShotBriefDescription,
    VisionWorldModel,
)


class CountingExpansionModel(Runnable):
    """Fake chat model for the cinematic expansion call; counts invocations."""

    def __init__(self) -> None:
        self.call_count = 0

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError

    async def ainvoke(self, messages, config=None, **kwargs):
        self.call_count += 1
        payload = {
            "scene_idx": 0,
            "beat_directives": [
                {
                    "beat_id": "ch01_b000",
                    "preserve_panel_composition": True,
                    "additional_shots": [
                        {
                            "shot_type": "establishing_wide",
                            "category": "creative",
                            "purpose": "Establish the classroom before the action",
                            "visual_hint": "Rain-streaked windows, dim fluorescent light, rows of desks",
                            "camera_angle": "high_angle",
                            "camera_movement": "slow_dolly_in",
                            "framing": "wide",
                            "target_duration_sec": 2.0,
                        },
                        {"shot_type": "reaction_close_up", "category": "inferred", "purpose": "Jiwoo's quiet reaction"},
                        {"shot_type": "insert_detail", "category": "creative", "purpose": "Raindrops on glass"},
                        # Exceeds the per-beat cap -> trimmed to 3
                        {"shot_type": "return_to_panel", "category": "inferred", "purpose": "Back to panel"},
                    ],
                },
                # Unknown beat -> must be dropped by the known-beat filter
                {
                    "beat_id": "ch99_b999",
                    "preserve_panel_composition": True,
                    "additional_shots": [{"shot_type": "insert_detail", "purpose": "bogus"}],
                },
            ],
            "notes": "ok",
        }
        return AIMessage(content=json.dumps(payload))


def _make_beat(beat_id: str = "ch01_b000") -> MangaBeat:
    return MangaBeat(
        beat_id=beat_id,
        chapter=1,
        page_idx=0,
        setting="Classroom",
        action_desc="Jiwoo looks out the window.",
        characters_present=["Jiwoo"],
    )


class _UnusedChatModel(Runnable):
    """Placeholder chat model — never invoked by the deterministic bible builder."""

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError


def _make_world_model() -> VisionWorldModel:
    from agents.world_model_builder import WorldModelBuilder

    beat = _make_beat()
    chunk = MangaChunkExtraction(
        chapter=1,
        start_page_idx=0,
        end_page_idx=0,
        characters_observed=[],
        beats=[beat],
        chunk_tail_summary="",
    )
    roster = [
        CharacterInScene(
            idx=0,
            identifier_in_scene="Jiwoo",
            is_visible=True,
            static_features="Teen boy with chestnut hair",
        )
    ]
    builder = WorldModelBuilder(_UnusedChatModel())  # chat model unused for the deterministic bible
    return VisionWorldModel(
        chapters=[1],
        characters=builder.build_character_bible(chunks=[chunk], characters=roster),
        environments=[],
        beat_scene_graphs=[],
        events=[],
    )


class FormatWorldContextTests(unittest.TestCase):
    def test_contains_bible_and_is_deterministic(self):
        wm = _make_world_model()
        beats = [_make_beat()]
        ctx1 = format_world_context(wm, beats)
        ctx2 = format_world_context(wm, beats)
        self.assertEqual(ctx1, ctx2, "World context must be deterministic")
        self.assertIn("Jiwoo", ctx1)
        self.assertIn("CHARACTER BIBLE", ctx1)

    def test_respects_max_chars_truncation(self):
        wm = _make_world_model()
        beats = [_make_beat("ch01_b000"), _make_beat("ch01_b001"), _make_beat("ch01_b002")]
        ctx = format_world_context(wm, beats, max_chars=60)
        self.assertLessEqual(len(ctx), 60)
        self.assertTrue(ctx.endswith("..."))


class PlanSceneExpansionTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_filters_unknown_beats_and_caps_shots(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = CountingExpansionModel()
            director = CinematicDirector(chat_model=fake)
            plan = await director.plan_scene_expansion(
                scene_idx=0,
                scene_beats=[_make_beat()],
                world_model=_make_world_model(),
                working_dir=tmp,
            )
            self.assertEqual(len(plan.beat_directives), 1)
            directive = plan.beat_directives[0]
            self.assertEqual(directive.beat_id, "ch01_b000")
            self.assertEqual(len(directive.additional_shots), 3, "More than 3 per-beat shots must be trimmed")
            self.assertTrue(plan.world_context_block)
            self.assertTrue((Path(tmp) / "expansion_plan.json").is_file())

    async def test_cached_plan_reused_without_llm_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = CountingExpansionModel()
            director = CinematicDirector(chat_model=fake)
            await director.plan_scene_expansion(
                scene_idx=0, scene_beats=[_make_beat()], world_model=_make_world_model(), working_dir=tmp
            )
            calls_after_first = fake.call_count
            again = await director.plan_scene_expansion(
                scene_idx=0, scene_beats=[_make_beat()], world_model=_make_world_model(), working_dir=tmp
            )
            self.assertEqual(fake.call_count, calls_after_first)
            self.assertEqual(len(again.beat_directives), 1)


class ApplyExpansionToScriptTests(unittest.TestCase):
    def test_appends_blocks_and_keeps_canon_script_untouched(self):
        canon = "Shot 1 [RefBeat: ch01_b000] (dramatic):\n<Jiwoo> looks out the window.\n"
        plan = SceneExpansionPlan(
            scene_idx=0,
            beat_directives=[
                BeatExpansionDirective(
                    beat_id="ch01_b000",
                    preserve_panel_composition=True,
                    additional_shots=[
                        ExpansionShotDirective(
                            shot_type="establishing_wide",
                            category="creative",
                            purpose="Establish the classroom",
                            visual_hint="Rain-streaked windows",
                            camera_angle="high_angle",
                            camera_movement="slow_dolly_in",
                        )
                    ],
                )
            ],
            world_context_block="CHARACTER BIBLE (canonical names; use verbatim):\n- Jiwoo: chestnut hair",
        )
        augmented = CinematicDirector.apply_expansion_to_script(canon, plan)
        # Canon prefix untouched (bit-for-bit except trailing whitespace normalization)
        self.assertTrue(augmented.startswith(canon.rstrip()))
        self.assertIn("WORLD CONTEXT", augmented)
        self.assertIn("CINEMATIC EXPANSION DIRECTIVES", augmented)
        self.assertIn("[RefBeat: ch01_b000] ADD establishing_wide (creative)", augmented)
        self.assertIn("Rain-streaked windows", augmented)
        # The RefBeat tag format survives so the existing binder still understands it
        self.assertIn("[RefBeat: ch01_b000]", augmented)

    def test_preserve_composition_directive_emitted_without_shots(self):
        canon = "base"
        plan = SceneExpansionPlan(
            scene_idx=0,
            beat_directives=[BeatExpansionDirective(beat_id="ch01_b000", preserve_panel_composition=True)],
            world_context_block="",
        )
        out = CinematicDirector.apply_expansion_to_script(canon, plan)
        self.assertIn("PRESERVE original panel composition", out)

    def test_empty_plan_returns_script_unchanged(self):
        canon = "Shot 1: canon only.\n"
        plan = SceneExpansionPlan(scene_idx=0)
        self.assertEqual(CinematicDirector.apply_expansion_to_script(canon, plan), canon)


class BuildShotProvenanceTests(unittest.TestCase):
    def _shot(self, idx: int, visual: str, audio: str = "") -> ShotBriefDescription:
        return ShotBriefDescription(
            idx=idx,
            is_last=(idx == 2),
            cam_idx=0,
            visual_desc=visual,
            audio_desc=audio,
        )

    def test_categories_canon_inferred_creative(self):
        storyboard = [
            self._shot(0, "<Jiwoo> looks out the classroom window [RefBeat: ch01_b000]."),
            self._shot(1, "<Jiwoo> close-up reaction [RefBeat: ch01_b001]."),
            self._shot(2, "Wide establishing shot of rain over the city."),
        ]
        plan = SceneExpansionPlan(
            scene_idx=0,
            beat_directives=[
                BeatExpansionDirective(
                    beat_id="ch01_b001",
                    additional_shots=[ExpansionShotDirective(shot_type="reaction_close_up", purpose="Reaction")],
                )
            ],
        )
        provenance = CinematicDirector.build_shot_provenance(
            storyboard=storyboard,
            scene_idx=0,
            known_beat_ids=["ch01_b000", "ch01_b001", "ch01_b002"],
            expansion_plan=plan,
        )
        by_idx = {p.shot_idx: p for p in provenance}
        # Shot 0: explicit RefBeat tag, no expansion directive for that beat -> canon
        self.assertEqual(by_idx[0].category, "canon")
        self.assertEqual(by_idx[0].source_beat_ids, ["ch01_b000"])
        self.assertAlmostEqual(by_idx[0].confidence, 0.9)
        # Shot 1: tagged AND the beat has expansion coverage -> inferred
        self.assertEqual(by_idx[1].category, "inferred")
        # Shot 2: no beat binding at all -> creative
        self.assertEqual(by_idx[2].category, "creative")
        self.assertEqual(by_idx[2].source_beat_ids, [])
        self.assertAlmostEqual(by_idx[2].confidence, 0.4)

    def test_unknown_beat_tags_are_ignored(self):
        storyboard = [self._shot(0, "Mystery shot [RefBeat: ch99_b999].")]
        provenance = CinematicDirector.build_shot_provenance(
            storyboard=storyboard,
            scene_idx=0,
            known_beat_ids=["ch01_b000"],
            expansion_plan=None,
        )
        self.assertEqual(provenance[0].category, "creative")
        self.assertEqual(provenance[0].source_beat_ids, [])


class ExtractShotBeatRefsTests(unittest.TestCase):
    def test_parses_single_and_multiple_refs(self):
        self.assertEqual(extract_shot_beat_refs("a [RefBeat: ch01_b000] b"), ["ch01_b000"])
        self.assertEqual(
            extract_shot_beat_refs("a [RefBeat: ch01_b000, ch01_b001] b [BEAT: ch01_b002]"),
            ["ch01_b000", "ch01_b001", "ch01_b002"],
        )
        self.assertEqual(extract_shot_beat_refs("no tags here"), [])
        self.assertEqual(extract_shot_beat_refs(""), [])


if __name__ == "__main__":
    unittest.main()
