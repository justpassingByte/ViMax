import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from PIL import Image

from agents.visual_qa import VisualQAReviewer
from agents.world_model_builder import WorldModelBuilder
from interfaces import (
    CharacterInScene,
    MangaBeat,
    MangaChunkExtraction,
    ShotBriefDescription,
    VisionWorldModel,
)


class FakeReviewModel(Runnable):
    """Fake chat model for the visual QA review call."""

    def __init__(self, *, payload=None, raise_error: bool = False) -> None:
        self.call_count = 0
        self.payload = payload or {
            "issues": [
                {
                    "shot_idx": 0,
                    "aspect": "character_identity",
                    "severity": "critical",
                    "description": "Hair color drifted from the bible",
                    "suggestion": "Re-generate with the character bible attached",
                },
                {
                    "shot_idx": 0,
                    "aspect": "totally_unknown_aspect",
                    "severity": "banana",
                    "description": "Should be normalized",
                    "suggestion": "",
                },
            ]
        }
        self.raise_error = raise_error

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError

    async def ainvoke(self, messages, config=None, **kwargs):
        self.call_count += 1
        if self.raise_error:
            raise RuntimeError("VLM unavailable")
        return AIMessage(content=json.dumps(self.payload))


def _make_scene_dir(tmp: Path, *, with_frame_for_shot0: bool = True, with_anchor: bool = True) -> Path:
    scene_dir = tmp / "scene_0"
    (scene_dir / "shots" / "0").mkdir(parents=True, exist_ok=True)
    (scene_dir / "shots" / "1").mkdir(parents=True, exist_ok=True)

    if with_frame_for_shot0:
        img = Image.new("RGB", (64, 64), (120, 140, 200))
        img.save(scene_dir / "shots" / "0" / "first_frame.png")
        img.close()

    anchor_path = ""
    if with_anchor:
        anchor = tmp / "panel_anchors" / "ch01_b000.jpg"
        anchor.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (64, 64), (235, 240, 250))
        img.save(anchor)
        img.close()
        anchor_path = str(anchor)

    beats = [
        MangaBeat(
            beat_id="ch01_b000",
            chapter=1,
            page_idx=0,
            setting="Classroom",
            action_desc="Jiwoo looks out the window.",
            characters_present=["Jiwoo"],
            anchor_image_path=anchor_path,
        )
    ]
    (scene_dir / "beat_map.json").write_text(json.dumps([b.model_dump() for b in beats]), encoding="utf-8")
    return scene_dir


def _make_storyboard() -> list:
    return [
        ShotBriefDescription(
            idx=0,
            is_last=False,
            cam_idx=0,
            visual_desc="<Jiwoo> looks out the classroom window [RefBeat: ch01_b000].",
            audio_desc='[Speaker] Jiwoo: "Ame da."',
        ),
        # Shot 1 has no first_frame.png -> deterministic missing_frame issue
        ShotBriefDescription(
            idx=1,
            is_last=True,
            cam_idx=0,
            visual_desc="<Jiwoo> reaction close-up [RefBeat: ch01_b000].",
            audio_desc="",
        ),
    ]


class _UnusedChatModel(Runnable):
    """Placeholder chat model — never invoked by the deterministic bible builder."""

    def invoke(self, input, config=None, **kwargs):
        raise NotImplementedError


def _make_world_model() -> VisionWorldModel:
    builder = WorldModelBuilder(_UnusedChatModel())
    chunk = MangaChunkExtraction(
        chapter=1,
        start_page_idx=0,
        end_page_idx=0,
        characters_observed=[],
        beats=[MangaBeat(beat_id="ch01_b000", chapter=1, setting="Classroom", characters_present=["Jiwoo"])],
    )
    roster = [
        CharacterInScene(
            idx=0,
            identifier_in_scene="Jiwoo",
            is_visible=True,
            static_features="Teen boy with chestnut hair",
        )
    ]
    return VisionWorldModel(chapters=[1], characters=builder.build_character_bible(chunks=[chunk], characters=roster))


class VisualQAReviewerTests(unittest.IsolatedAsyncioTestCase):
    async def test_review_scene_flags_and_persists_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scene_dir = _make_scene_dir(tmp_path)
            fake = FakeReviewModel()
            report = await VisualQAReviewer(fake).review_scene(
                scene_idx=0,
                scene_dir=scene_dir,
                shot_descriptions=_make_storyboard(),
                beats=[MangaBeat.model_validate(b) for b in json.loads((scene_dir / "beat_map.json").read_text(encoding="utf-8"))],
                world_model=_make_world_model(),
            )
            # Shot 0 reviewed via one VLM call; shot 1 flagged missing_frame.
            self.assertEqual(fake.call_count, 1)
            self.assertEqual(report.reviewed_frames, 1)
            missing = [i for i in report.issues if i.shot_idx == 1 and i.aspect == "missing_frame"]
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0].severity, "critical")
            # Unknown aspect/severity values normalized to safe defaults.
            aspects = {(i.aspect, i.severity) for i in report.issues if i.description == "Should be normalized"}
            self.assertEqual(aspects, {("artifact", "warning")})
            self.assertIn(0, report.failed_shots)
            self.assertIn(1, report.failed_shots)
            self.assertTrue((scene_dir / "visual_qa_report.json").is_file())
            saved = json.loads((scene_dir / "visual_qa_report.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["scene_idx"], 0)

    async def test_vlm_failure_never_breaks_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = _make_scene_dir(Path(tmp))
            fake = FakeReviewModel(raise_error=True)
            report = await VisualQAReviewer(fake).review_scene(
                scene_idx=0,
                scene_dir=scene_dir,
                shot_descriptions=_make_storyboard(),
                beats=[MangaBeat.model_validate(b) for b in json.loads((scene_dir / "beat_map.json").read_text(encoding="utf-8"))],
                world_model=_make_world_model(),
            )
            # Deterministic pre-checks still present; no exception escaped.
            self.assertTrue(any(i.aspect == "missing_frame" for i in report.issues))
            self.assertTrue(report.failed_shots)

    async def test_missing_anchor_records_info_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = _make_scene_dir(Path(tmp), with_anchor=False)
            fake = FakeReviewModel()
            report = await VisualQAReviewer(fake).review_scene(
                scene_idx=0,
                scene_dir=scene_dir,
                shot_descriptions=_make_storyboard(),
                beats=[MangaBeat.model_validate(b) for b in json.loads((scene_dir / "beat_map.json").read_text(encoding="utf-8"))],
                world_model=_make_world_model(),
            )
            info = [i for i in report.issues if i.shot_idx == 0 and i.aspect == "missing_frame" and i.severity == "info"]
            self.assertEqual(len(info), 1)
            self.assertIn("panel anchor", info[0].description.lower())

    async def test_report_only_when_no_shot_descriptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = _make_scene_dir(Path(tmp))
            fake = FakeReviewModel()
            report = await VisualQAReviewer(fake).review_scene(
                scene_idx=0,
                scene_dir=scene_dir,
                shot_descriptions=[],
                beats=[],
                world_model=None,
            )
            self.assertEqual(report.reviewed_frames, 0)
            self.assertEqual(report.failed_shots, [])
            self.assertEqual(fake.call_count, 0)


if __name__ == "__main__":
    unittest.main()
