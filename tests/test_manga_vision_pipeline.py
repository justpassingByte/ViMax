"""Integration tests for the Manga Vision & Cinematic Director Layer wiring.

Runs the real ``Manga2VideoPipeline.plan_episode`` with a fully fake LLM and
verifies that, with the feature flag enabled:

- a chapter-level ``vision_world/world_model.json`` is built and cached,
- each planned scene gets ``expansion_plan.json`` (cinematic expansion) and
  ``shot_provenance.json`` (CANON/INFERRED/CREATIVE traceability),
- the canon ``script.txt`` on disk is NEVER modified,
- with the flag disabled, none of those artifacts appear (backward compat).
"""

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from PIL import Image, ImageDraw

from pipelines.manga2video_pipeline import Manga2VideoPipeline
from interfaces import MangaEpisodePlan


def _save_comic_panel(path: Path, *, width: int = 700, height: int = 900) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (235, 240, 250))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, width - 40, height - 40], outline=(20, 20, 30), width=4)
    img.save(path, format="JPEG", quality=90)
    img.close()


def _make_chapter_cbz(root: Path) -> Path:
    pages_dir = root / "raw"
    _save_comic_panel(pages_dir / "p01.jpg")
    _save_comic_panel(pages_dir / "p02.jpg")
    cbz = root / "ch01.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        zf.write(pages_dir / "p01.jpg", "p01.jpg")
        zf.write(pages_dir / "p02.jpg", "p02.jpg")
    return cbz


class VisionPipelineFakeLLM(Runnable):
    """Dispatches on the system-prompt text of every pipeline stage."""

    def __init__(self) -> None:
        self.calls: list = []
        self.storyboard_prompts: list = []

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError

    async def ainvoke(self, messages, config=None, **kwargs):
        if hasattr(messages, "to_messages"):
            messages = messages.to_messages()
        first = messages[0]
        sys_text = first.content if hasattr(first, "content") else str(first)
        self.calls.append(sys_text[:60])

        if "beat_directives" in sys_text:
            # Cinematic expansion plan for scene 0
            return AIMessage(
                content=json.dumps(
                    {
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
                                        "visual_hint": "Rain-streaked windows, dim fluorescent light",
                                        "camera_angle": "high_angle",
                                        "camera_movement": "slow_dolly_in",
                                        "framing": "wide",
                                        "target_duration_sec": 2.0,
                                    }
                                ],
                            }
                        ],
                        "notes": "",
                    }
                )
            )
        if "beat_scene_graphs" in sys_text:
            # World model analysis pass
            return AIMessage(
                content=json.dumps(
                    {
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
                                "foreground_desc": "rows of desks",
                                "midground_desc": "",
                                "background_desc": "rain-streaked windows",
                                "beat_ids": ["ch01_b000", "ch01_b001"],
                            }
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
                                    {"subject": "Jiwoo", "relation": "facing", "obj": "window", "confidence": 0.95}
                                ],
                                "camera_position": "behind Jiwoo",
                            },
                            {
                                "beat_id": "ch01_b001",
                                "environment_key": "classroom",
                                "entities": [{"entity_id": "Jiwoo", "entity_type": "character", "label": "Jiwoo"}],
                                "relations": [],
                                "camera_position": "",
                            },
                        ],
                        "events": [
                            {
                                "description": "Jiwoo watches the rain start",
                                "event_type": "action",
                                "beat_ids": ["ch01_b000"],
                                "character_keys": ["Jiwoo"],
                                "environment_key": "classroom",
                                "emotional_tone": "calm",
                            }
                        ],
                    }
                )
            )
        if "characters_observed" in sys_text:
            # MangaReader page-reading extraction (fake ignores attached images)
            return AIMessage(
                content=json.dumps(
                    {
                        "characters_observed": [
                            {
                                "name": "Jiwoo",
                                "species_or_form": "human",
                                "appearance_summary": "Teen boy with chestnut hair",
                                "outfit_summary": "School uniform",
                                "page_idx": 0,
                                "crop_box_norm": [100, 100, 800, 800],
                            }
                        ],
                        "beats": [
                            {
                                "page_idx": 0,
                                "is_scene_break": True,
                                "setting": "Classroom",
                                "action_desc": "Jiwoo looks out the window.",
                                "visual_details": "Raindrops on glass.",
                                "characters_present": ["Jiwoo"],
                                "dialogues": [
                                    {
                                        "speaker": "Jiwoo",
                                        "line": "Ame ga futte kita na.",
                                        "line_type": "spoken",
                                        "emotion": "Soft",
                                        "vietsub": "Trời lại bắt đầu mưa rồi.",
                                    }
                                ],
                                "sfx": [],
                                "crop_box_norm": [50, 50, 900, 900],
                            },
                            {
                                "page_idx": 1,
                                "is_scene_break": False,
                                "setting": "Classroom",
                                "action_desc": "Jiwoo sighs and smiles.",
                                "visual_details": "Soft window light.",
                                "characters_present": ["Jiwoo"],
                                "dialogues": [],
                                "sfx": [],
                                "crop_box_norm": [50, 50, 900, 900],
                            },
                        ],
                        "chunk_tail_summary": "",
                    }
                )
            )
        if "storyboard" in sys_text and "ff_desc" not in sys_text and "camera_parent_items" not in sys_text:
            self.storyboard_prompts.append(
                messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
            )
            return AIMessage(
                content=json.dumps(
                    {
                        "storyboard": [
                            {
                                "idx": 0,
                                "is_last": True,
                                "cam_idx": 0,
                                "visual_desc": "<Jiwoo> looks out classroom window [RefBeat: ch01_b000].",
                                "audio_desc": '[Speaker] Jiwoo (Soft): "Ame ga futte kita na." '
                                "(Vietsub: Trời lại bắt đầu mưa rồi.)",
                            }
                        ]
                    }
                )
            )
        if "ff_desc" in sys_text:
            return AIMessage(
                content=json.dumps(
                    {
                        "ff_desc": "Jiwoo at window",
                        "ff_vis_char_idxs": [0],
                        "lf_desc": "Jiwoo turns head slightly",
                        "lf_vis_char_idxs": [0],
                        "motion_desc": "Rain slides down pane",
                        "variation_type": "small",
                        "variation_reason": "subtle expression",
                    }
                )
            )
        # construct_camera_tree
        return AIMessage(content=json.dumps({"camera_parent_items": [None]}))


class MangaVisionPipelineIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_plan_episode_with_vision_flag_builds_world_model_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cbz = _make_chapter_cbz(root)
            fake = VisionPipelineFakeLLM()
            pipe = Manga2VideoPipeline(
                chat_model=fake,
                image_generator=SimpleNamespace(),
                video_generator=SimpleNamespace(),
                working_dir=root / "work",
            )
            with patch.dict(
                os.environ,
                {
                    "VIMAX_MANGA_VISION_ENABLED": "1",
                    "VIMAX_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER": "mock",
                },
            ):
                plan = await pipe.plan_episode(sources=[str(cbz)], quiet=True)

            self.assertIsInstance(plan, MangaEpisodePlan)
            self.assertGreaterEqual(len(plan.scenes), 1)
            work = root / "work"

            # 1. Chapter-level world model built and cached on disk.
            world_model_path = work / "vision_world" / "world_model.json"
            self.assertTrue(world_model_path.is_file())
            wm = json.loads(world_model_path.read_text(encoding="utf-8"))
            self.assertEqual(len(wm["environments"]), 1)
            self.assertEqual(len(wm["beat_scene_graphs"]), 2)

            # 1b. Report-only duplicate-identity analysis attached + cached
            # (fake extraction may produce an empty roster -> status "skipped").
            self.assertIn("identity_dedup", wm)
            self.assertIn(wm["identity_dedup"]["status"], {"ok", "skipped"})
            if wm["identity_dedup"]["status"] == "ok":
                self.assertTrue((work / "vision_world" / "identity_dedup.json").is_file())

            # 2. Every planned scene has cinematic expansion + shot provenance.
            scene_dirs = sorted(work.glob("scene_*"))
            self.assertEqual(len(scene_dirs), len(plan.scenes))
            for scene_dir in scene_dirs:
                expansion = json.loads((scene_dir / "expansion_plan.json").read_text(encoding="utf-8"))
                self.assertTrue(expansion["world_context_block"], "World context must be injected")
                provenance = json.loads((scene_dir / "shot_provenance.json").read_text(encoding="utf-8"))
                self.assertGreaterEqual(len(provenance), 1)
                for entry in provenance:
                    self.assertIn(entry["category"], {"canon", "inferred", "creative"})
                    self.assertIn("source_beat_ids", entry)

                # 3. Canon script.txt on disk must NEVER contain augmentation blocks.
                script_text = (scene_dir / "script.txt").read_text(encoding="utf-8")
                self.assertNotIn("WORLD CONTEXT", script_text)
                self.assertNotIn("CINEMATIC EXPANSION DIRECTIVES", script_text)

            # 4. The storyboard stage received the AUGMENTED script (in-memory only).
            self.assertTrue(fake.storyboard_prompts, "Storyboard design must have been invoked")
            augmented_seen = any(
                "WORLD CONTEXT" in str(p) and "CINEMATIC EXPANSION DIRECTIVES" in str(p)
                for p in fake.storyboard_prompts
            )
            self.assertTrue(augmented_seen, "Storyboard prompt must include the vision-layer augmentation")

            # 5. Analysis batch cached for reproducibility.
            self.assertTrue((work / "vision_world" / "analysis_batch_000.json").is_file())

    async def test_plan_episode_with_flag_off_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cbz = _make_chapter_cbz(root)
            fake = VisionPipelineFakeLLM()
            pipe = Manga2VideoPipeline(
                chat_model=fake,
                image_generator=SimpleNamespace(),
                video_generator=SimpleNamespace(),
                working_dir=root / "work",
            )
            with patch.dict(os.environ, {"VIMAX_MANGA_VISION_ENABLED": "0"}):
                plan = await pipe.plan_episode(sources=[str(cbz)], quiet=True)

            self.assertGreaterEqual(len(plan.scenes), 1)
            work = root / "work"
            self.assertFalse((work / "vision_world").exists(), "No vision artifacts when flag is off")
            for scene_dir in work.glob("scene_*"):
                self.assertFalse((scene_dir / "expansion_plan.json").is_file())
                self.assertFalse((scene_dir / "shot_provenance.json").is_file())
            self.assertFalse(
                any("WORLD CONTEXT" in str(p) for p in fake.storyboard_prompts),
                "Storyboard prompt must NOT include vision augmentation when flag is off",
            )


if __name__ == "__main__":
    unittest.main()
