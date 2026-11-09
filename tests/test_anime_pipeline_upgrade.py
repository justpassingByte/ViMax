import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
from moviepy import VideoFileClip
from PIL import Image

from agent_runtime.session_index import SessionIndex
from agent_runtime.vimax_adapters import (
    ViMaxAdapters,
    _GatedChatModel,
    invalidate_render_artifacts,
)
from agents.camera_image_generator import CameraImageGenerator
from interfaces import Camera, CharacterInScene, ShotBriefDescription, ShotDescription
from pipelines.script2video_pipeline import Script2VideoPipeline, shot_requires_last_frame
from utils.async_gate import AsyncConcurrencyGate
from utils.audio import build_subtitle_cues, extract_all_dialogues_from_shot, extract_vietsubs_from_shots
from utils.video import compose_timeline_clips, concatenate_video_files


def _make_synthetic_clip(path: Path, *, color: str, duration: float = 2.0, freq: int = 440, size: str = "160x90") -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s={size}:d={duration}:r=24",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency={freq}:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed creating synthetic clip {path}: {proc.stderr}")


class ShotSchemaAndStoryboardTests(unittest.IsolatedAsyncioTestCase):
    def test_shot_models_preserve_backward_compatible_defaults(self):
        brief = ShotBriefDescription.model_validate(
            {"idx": 0, "is_last": True, "cam_idx": 0, "visual_desc": "rain alley", "audio_desc": "[Sound Effect] Thunder."}
        )
        self.assertEqual(brief.transition_in, "cut")
        self.assertFalse(brief.continuity_from_prev)
        self.assertIsNone(brief.target_duration_sec)

        desc = ShotDescription.model_validate(
            {
                "idx": 0,
                "is_last": True,
                "cam_idx": 0,
                "visual_desc": "rain alley",
                "variation_type": "medium",
                "variation_reason": "fast dodge",
                "ff_desc": "ff",
                "ff_vis_char_idxs": [],
                "lf_desc": "lf",
                "lf_vis_char_idxs": [],
                "motion_desc": "dodge",
                "audio_desc": "wind",
                "transition_in": "fade_black",
                "continuity_from_prev": True,
                "target_duration_sec": 4.0,
            }
        )
        self.assertEqual(desc.transition_in, "fade_black")
        self.assertTrue(desc.continuity_from_prev)
        self.assertEqual(desc.target_duration_sec, 4.0)

    def test_shot_requires_last_frame_covers_action_and_continuity(self):
        s0 = ShotDescription(
            idx=0, is_last=False, cam_idx=0, visual_desc="clash 0", variation_type="small", variation_reason="r",
            ff_desc="ff0", ff_vis_char_idxs=[], lf_desc="lf0", lf_vis_char_idxs=[], motion_desc="m0", audio_desc="none",
        )
        s1 = ShotDescription(
            idx=1, is_last=False, cam_idx=1, visual_desc="clash 1", variation_type="small", variation_reason="r",
            ff_desc="ff1", ff_vis_char_idxs=[], lf_desc="lf1", lf_vis_char_idxs=[], motion_desc="m1", audio_desc="none",
            continuity_from_prev=True,
        )
        s2 = ShotDescription(
            idx=2, is_last=False, cam_idx=1, visual_desc="fast slash", variation_type="small", variation_reason="r",
            ff_desc="ff2", ff_vis_char_idxs=[], lf_desc="lf2", lf_vis_char_idxs=[], motion_desc="m2", audio_desc="none",
            target_duration_sec=4.0,
        )
        s3 = ShotDescription(
            idx=3, is_last=True, cam_idx=1, visual_desc="quiet close", variation_type="small", variation_reason="r",
            ff_desc="ff3", ff_vis_char_idxs=[], lf_desc="lf3", lf_vis_char_idxs=[], motion_desc="m3", audio_desc="none",
        )
        shots = [s0, s1, s2, s3]
        self.assertTrue(shot_requires_last_frame(shots, s0))  # s1 continuity_from_prev locks s0's last_frame
        self.assertTrue(shot_requires_last_frame(shots, s1))  # continuity_from_prev=True
        self.assertTrue(shot_requires_last_frame(shots, s2))  # fast action target_duration_sec <= 5.5
        self.assertFalse(shot_requires_last_frame(shots, s3))

    async def test_decompose_propagates_editorial_fields_and_serializes_gated_chat(self):
        from agents.storyboard_artist import StoryboardArtist, VisDescDecompositionResponse

        active = 0
        max_active = 0

        class FakeStructured:
            async def ainvoke(self, *args, **kwargs):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.02)
                active -= 1
                return VisDescDecompositionResponse(
                    ff_desc="start", ff_vis_char_idxs=[0], lf_desc="end", lf_vis_char_idxs=[0],
                    motion_desc="dash", variation_type="medium", variation_reason="action leap",
                )

        with tempfile.TemporaryDirectory() as tmp:
            gate = AsyncConcurrencyGate(1)
            gated_chat = _GatedChatModel(SimpleNamespace(), gate)
            pipeline = Script2VideoPipeline(chat_model=gated_chat, image_generator=object(), video_generator=object(), working_dir=tmp)
            pipeline.storyboard_artist = StoryboardArtist(chat_model=gated_chat)

            async def fake_decomp(shot_brief_desc, characters, retry_timeout=180):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.02)
                active -= 1
                return ShotDescription(
                    idx=shot_brief_desc.idx,
                    is_last=shot_brief_desc.is_last,
                    cam_idx=shot_brief_desc.cam_idx,
                    visual_desc=shot_brief_desc.visual_desc,
                    variation_type="medium",
                    variation_reason="fast movement",
                    ff_desc="ff",
                    ff_vis_char_idxs=[0],
                    lf_desc="lf",
                    lf_vis_char_idxs=[0],
                    motion_desc="m",
                    audio_desc=shot_brief_desc.audio_desc,
                    transition_in=shot_brief_desc.transition_in,
                    continuity_from_prev=shot_brief_desc.continuity_from_prev,
                    target_duration_sec=shot_brief_desc.target_duration_sec,
                )

            pipeline.storyboard_artist.decompose_visual_description = fake_decomp
            chars = [CharacterInScene(idx=0, identifier_in_scene="Kayden", is_visible=True, static_features="man")]
            briefs = [
                ShotBriefDescription(idx=i, is_last=(i == 3), cam_idx=0, visual_desc=f"shot {i}", audio_desc="sfx", transition_in="crossfade" if i == 1 else "cut", continuity_from_prev=(i == 1), target_duration_sec=4.0 if i == 2 else None)
                for i in range(4)
            ]
            descs = await pipeline.decompose_visual_descriptions(briefs, chars, quiet=True)
            self.assertEqual(max_active, 1)
            self.assertEqual(descs[1].transition_in, "crossfade")
            self.assertTrue(descs[1].continuity_from_prev)
            self.assertEqual(descs[2].target_duration_sec, 4.0)


class TimelineCompositionAndSubtitleSyncTests(unittest.TestCase):
    def test_compose_timeline_speeds_silent_action_never_spoken_and_syncs_subtitles(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            p0 = tmp_dir / "s0.mp4"
            p1 = tmp_dir / "s1.mp4"
            p2 = tmp_dir / "s2.mp4"
            _make_synthetic_clip(p0, color="0xDC143C", duration=2.0, freq=330, size="160x90")
            _make_synthetic_clip(p1, color="0x1E90FF", duration=2.0, freq=440, size="160x90")
            _make_synthetic_clip(p2, color="0x2E8B57", duration=2.0, freq=550, size="160x90")

            sd0 = ShotDescription(
                idx=0, is_last=False, cam_idx=0, visual_desc="silent blitz", variation_type="large", variation_reason="speed",
                ff_desc="ff0", ff_vis_char_idxs=[], lf_desc="lf0", lf_vis_char_idxs=[], motion_desc="dash", audio_desc="[Sound Effect] swoosh",
                transition_in="cut", target_duration_sec=1.0,
            )
            sd1 = ShotDescription(
                idx=1, is_last=False, cam_idx=1, visual_desc="jiwoo speaks", variation_type="small", variation_reason="dialogue",
                ff_desc="ff1", ff_vis_char_idxs=[], lf_desc="lf1", lf_vis_char_idxs=[], motion_desc="talk", audio_desc='[Speaker] Jiwoo: "Ikuzo!"',
                transition_in="crossfade", target_duration_sec=1.0,  # MUST be ignored because spoken=True
            )
            sd2 = ShotDescription(
                idx=2, is_last=True, cam_idx=2, visual_desc="aftermath", variation_type="medium", variation_reason="fade out",
                ff_desc="ff2", ff_vis_char_idxs=[], lf_desc="lf2", lf_vis_char_idxs=[], motion_desc="rain", audio_desc='[Speaker] Kayden: "Hmph."',
                transition_in="fade_black",
            )

            clips = [VideoFileClip(str(p0)), VideoFileClip(str(p1)), VideoFileClip(str(p2))]
            extra = []
            final = None
            out_mp4 = tmp_dir / "composed.mp4"
            try:
                final, offsets, durations, extra = compose_timeline_clips(
                    clips,
                    [sd0, sd1, sd2],
                    spoken_shot_mask=[False, True, True],
                    fade_black_sec=0.3,
                    crossfade_sec=0.4,
                )
                self.assertAlmostEqual(durations[0], 1.0, delta=0.05)
                self.assertAlmostEqual(durations[1], 2.0, delta=0.05)  # spoken shot NOT accelerated!
                self.assertAlmostEqual(durations[2], 2.0, delta=0.05)
                # xfade clamped by 0.25 * min(1.0, 2.0) = 0.25s -> offset[1] = 1.0 - 0.25 = 0.75
                self.assertAlmostEqual(offsets[0], 0.0, delta=0.05)
                self.assertAlmostEqual(offsets[1], 0.75, delta=0.05)
                self.assertAlmostEqual(offsets[2], 2.75, delta=0.05)

                final.write_videofile(str(out_mp4), codec="libx264", audio_codec="aac", preset="ultrafast")
            finally:
                if final is not None:
                    final.close()
                for c in extra:
                    c.close()
                for c in clips:
                    c.close()

            self.assertTrue(out_mp4.exists())
            cues = build_subtitle_cues(
                [sd0, sd1, sd2],
                shot_durations=durations,
                script_vietsubs={1: "Đi thôi!", 2: "Hừ."},
                shot_start_offsets=offsets,
            )
            self.assertEqual(len(cues), 2)
            self.assertAlmostEqual(cues[0][0], offsets[1] + 0.2, delta=0.02)
            self.assertLessEqual(cues[0][1], cues[1][0])

            rendered = VideoFileClip(str(out_mp4))
            try:
                self.assertIsNotNone(rendered.audio)
                self.assertAlmostEqual(rendered.duration, 4.75, delta=0.15)
                # At t=0.87s (midpoint of crossfade between red shot0 and blue shot1), pixels blend both channels
                mid_frame = rendered.get_frame(0.87)
                mean_r = float(mid_frame[:, :, 0].mean())
                mean_b = float(mid_frame[:, :, 2].mean())
                self.assertGreater(mean_r, 30.0)
                self.assertGreater(mean_b, 30.0)
                # At t=2.76s (right after fade_black boundary), brightness is near black
                black_frame = rendered.get_frame(2.76)
                self.assertLess(float(black_frame.mean()), 45.0)
            finally:
                rendered.close()

    def test_concatenate_video_files_with_scene_fade_and_atomic_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            a = tmp_dir / "scene_0.mp4"
            b = tmp_dir / "scene_1.mp4"
            out = tmp_dir / "episode.mp4"
            _make_synthetic_clip(a, color="0xFF4500", duration=2.0, freq=350, size="160x90")
            _make_synthetic_clip(b, color="0x00CED1", duration=2.0, freq=500, size="160x90")
            concatenate_video_files([str(a), str(b)], str(out), preset="ultrafast", scene_fade_sec=0.35)
            self.assertTrue(out.exists())
            clip = VideoFileClip(str(out))
            try:
                self.assertIsNotNone(clip.audio)
                self.assertAlmostEqual(clip.duration, 4.0, delta=0.15)
                # Scene boundary fades dark around 2.0s
                self.assertLess(float(clip.get_frame(1.98).mean()), 35.0)
            finally:
                clip.close()


class ContinuityGraphAndPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_continuity_from_prev_skips_transition_video_and_attaches_last_frame_without_deadlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "scene_0"
            for idx in (0, 1):
                (work_dir / "shots" / str(idx)).mkdir(parents=True, exist_ok=True)
            portrait_dir = work_dir / "portraits"
            portrait_dir.mkdir(parents=True, exist_ok=True)
            front_p = portrait_dir / "front.png"
            Image.new("RGB", (64, 64), "white").save(front_p)

            registry = {
                "Kayden": {
                    "front": {"path": str(front_p), "description": "front portrait of Kayden"},
                }
            }
            chars = [CharacterInScene(idx=0, identifier_in_scene="Kayden", is_visible=True, static_features="man")]
            s0 = ShotDescription(
                idx=0, is_last=False, cam_idx=0, visual_desc="Kayden charges", variation_type="medium", variation_reason="action",
                ff_desc="ff0", ff_vis_char_idxs=[0], lf_desc="lf0", lf_vis_char_idxs=[0], motion_desc="charge", audio_desc="crack",
            )
            s1 = ShotDescription(
                idx=1, is_last=True, cam_idx=1, visual_desc="Kayden strikes", variation_type="large", variation_reason="impact",
                ff_desc="ff1", ff_vis_char_idxs=[0], lf_desc="lf1", lf_vis_char_idxs=[0], motion_desc="strike",
                audio_desc='[Speaker] Kayden: "Kiero!" (Vietsub: Biến đi!)', continuity_from_prev=True,
            )
            shots = [s0, s1]

            seen_ref_inputs = []

            class FakeImageGen:
                async def generate_single_image(self, prompt, reference_image_paths=None, size="1600x900", **kwargs):
                    from interfaces import ImageOutput
                    return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (64, 36), "navy"))

            class FakeVideoGen:
                def __init__(self):
                    self.calls = []

                async def generate_single_video(self, prompt="", reference_image_paths=None, **kwargs):
                    self.calls.append((prompt, list(reference_image_paths or [])))
                    from interfaces import VideoOutput
                    return VideoOutput(fmt="bytes", ext="mp4", data=b"fake-mp4")

            vid_gen = FakeVideoGen()
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=FakeImageGen(), video_generator=vid_gen, working_dir=str(work_dir), num_image_candidates=1)

            async def fake_select(available_image_path_and_text_pairs, frame_description):
                seen_ref_inputs.append(([p for p, _ in available_image_path_and_text_pairs], [t for _, t in available_image_path_and_text_pairs]))
                chosen = available_image_path_and_text_pairs[-1:]
                return {"reference_image_path_and_text_pairs": chosen, "text_prompt": "Render matching frame referencing Image 0."}

            pipeline.reference_image_selector.select_reference_images_and_generate_prompt = fake_select
            pipeline.camera_image_generator.generate_transition_video = AsyncMock(side_effect=AssertionError("Should NOT call transition_video when continuity_from_prev=True"))

            pipeline._active_shot_descriptions = shots
            pipeline.frame_events = {
                0: {"first_frame": asyncio.Event(), "last_frame": asyncio.Event()},
                1: {"first_frame": asyncio.Event(), "last_frame": asyncio.Event()},
            }
            cam0 = Camera(idx=0, active_shot_idxs=[0])
            cam1 = Camera(idx=1, active_shot_idxs=[1], parent_cam_idx=0, parent_shot_idx=0)

            await asyncio.wait_for(
                asyncio.gather(
                    pipeline.generate_frames_for_single_camera(cam0, shots, chars, registry, priority_shot_idxs=[0]),
                    pipeline.generate_frames_for_single_camera(cam1, shots, chars, registry, priority_shot_idxs=[0]),
                    pipeline.generate_video_for_single_shot(s0),
                    pipeline.generate_video_for_single_shot(s1),
                ),
                timeout=5.0,
            )

            pipeline.camera_image_generator.generate_transition_video.assert_not_awaited()
            # Shot 1 first_frame selector received Shot 0's last_frame.png with continuity anchor text
            shot1_ff_paths, shot1_ff_texts = seen_ref_inputs[-2]
            self.assertTrue(any("0\\last_frame.png" in p or "0/last_frame.png" in p for p in shot1_ff_paths))
            self.assertTrue(any("CONTINUOUS ACTION ANCHOR" in t for t in shot1_ff_texts))

            # Both videos received [first_frame.png, last_frame.png] with Image 0 / Image 1 anchoring
            self.assertEqual(len(vid_gen.calls), 2)
            for prompt, frame_paths in vid_gen.calls:
                self.assertEqual(len(frame_paths), 2)
                self.assertIn("Image 0", prompt)
                self.assertIn("Image 1", prompt)
            s1_prompt = [p for p, paths in vid_gen.calls if "Kiero!" in p][0]
            self.assertIn("Spoken Delivery Pacing", s1_prompt)
            self.assertNotIn("Vietsub:", s1_prompt)

    def test_inline_vietsub_is_extracted_from_shots_and_stripped_from_spoken_dialogue(self):
        shots = [
            SimpleNamespace(
                idx=0,
                audio_desc='[Speaker] Jiwoo: "Daijoubu desu ka?" (Vietsub: Cậu có sao không?) [Sound Effect] Rain',
                visual_desc="rain",
            ),
            SimpleNamespace(
                idx=1,
                audio_desc='[Speaker] Kayden: "Hmph."\nVietsub: Hừ.',
                visual_desc="cat",
            ),
        ]
        vietsubs = extract_vietsubs_from_shots(shots)
        self.assertEqual(vietsubs, {0: "Cậu có sao không?", 1: "Hừ."})
        self.assertEqual(extract_all_dialogues_from_shot(shots[0].audio_desc), [("Jiwoo", "Daijoubu desu ka?")])
        self.assertEqual(extract_all_dialogues_from_shot(shots[1].audio_desc), [("Kayden", "Hmph.")])


class SelectiveCacheInvalidationAndStaleTests(unittest.IsolatedAsyncioTestCase):
    def test_session_index_mark_and_clear_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            rec = index.create(idea="test")
            sid = rec["session_id"]
            index.mark_stale(sid, ["storyboard", "frames", "final_video"])
            self.assertTrue(index.get(sid)["stale"]["frames"])
            self.assertTrue(index.get(sid)["stale"]["storyboard"])

            index.clear_stale(sid, ["frames"])
            self.assertFalse(index.get(sid)["stale"]["frames"])
            self.assertTrue(index.get(sid)["stale"]["storyboard"])

            index.clear_stale(sid)
            self.assertFalse(index.get(sid)["stale"]["storyboard"])
            self.assertFalse(index.get(sid)["stale"]["final_video"])

    async def test_invalidate_render_artifacts_supports_scene_idx_and_shot_idxs(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = SessionIndex(tmp)
            rec = index.create(idea="eleceed")
            sid = rec["session_id"]
            root = Path(tmp) / rec["working_dir"]
            idea_dir = root / "idea2video"
            for scene in (0, 1):
                s_dir = idea_dir / f"scene_{scene}"
                s_dir.mkdir(parents=True, exist_ok=True)
                (s_dir / "final_video.mp4").write_bytes(b"scene")
                for shot in (0, 1):
                    shot_dir = s_dir / "shots" / str(shot)
                    shot_dir.mkdir(parents=True, exist_ok=True)
                    (shot_dir / "shot_description.json").write_text("{}", encoding="utf-8")
                    (shot_dir / "first_frame.png").write_bytes(b"ff")
                    (shot_dir / "video.mp4").write_bytes(b"mp4")
            (idea_dir / "final_video.mp4").write_bytes(b"all")

            removed = invalidate_render_artifacts(root, scene_idx=0, shot_idxs=[1])
            self.assertTrue(removed)
            # Only scene_0/shots/1 media + scene_0 final + idea2video final removed
            self.assertFalse((idea_dir / "scene_0" / "shots" / "1" / "video.mp4").exists())
            self.assertTrue((idea_dir / "scene_0" / "shots" / "1" / "shot_description.json").exists())
            self.assertTrue((idea_dir / "scene_0" / "shots" / "0" / "video.mp4").exists())
            self.assertTrue((idea_dir / "scene_1" / "shots" / "1" / "video.mp4").exists())
            self.assertTrue((idea_dir / "scene_1" / "final_video.mp4").exists())
            self.assertFalse((idea_dir / "final_video.mp4").exists())

            adapter = ViMaxAdapters(Path(tmp), index)
            (idea_dir / "story.txt").write_text("s", encoding="utf-8")
            (idea_dir / "characters.json").write_text("[]", encoding="utf-8")
            (idea_dir / "script.json").write_text("[]", encoding="utf-8")
            (idea_dir / "scene_0" / "storyboard.json").write_text("[]", encoding="utf-8")
            (idea_dir / "scene_0" / "camera_tree.json").write_text("[]", encoding="utf-8")
            (idea_dir / "scene_1" / "storyboard.json").write_text("[]", encoding="utf-8")
            (idea_dir / "scene_1" / "camera_tree.json").write_text("[]", encoding="utf-8")

            bad_scene = await adapter.vimax_render_video({"session_id": sid, "scene_idx": -1})
            self.assertFalse(bad_scene.ok)
            self.assertEqual(bad_scene.metadata["error_type"], "invalid_argument")

            index.mark_stale(sid, ["final_video", "clips"])
            class OkIdeaPipeline:
                def __init__(self, *a, working_dir, **kw):
                    self.w = Path(working_dir)
                async def __call__(self, *a, **kw):
                    out = self.w / "final_video.mp4"
                    out.write_bytes(b"ok")
                    return str(out)

            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters.Idea2VideoPipeline", OkIdeaPipeline):
                ok_res = await adapter.vimax_render_video({"session_id": sid, "scene_idx": 0, "shot_idxs": [0]})
            self.assertTrue(ok_res.ok)
            self.assertFalse(index.get(sid)["stale"]["final_video"])
            self.assertFalse(index.get(sid)["stale"]["clips"])


class DegenerateNarrativeAndCacheGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_screenwriter_develop_story_rejects_verification_stub_and_retries(self):
        from agents.screenwriter import Screenwriter, is_valid_developed_story

        self.assertFalse(is_valid_developed_story("Verified before delivery — short stub."))
        valid_story = (
            "Title: Storm Over Seoul\n\n"
            + ("Lightning crackled above the rain-slicked rooftop as Kayden braced his stance against three ambushing Awakeners. " * 4)
        )
        self.assertTrue(is_valid_developed_story(valid_story))

        responses = iter([
            SimpleNamespace(content="Verified before delivery without any narrative prose."),
            SimpleNamespace(content=valid_story),
        ])

        class FakeChat:
            async def ainvoke(self, messages):
                return next(responses)

        writer = Screenwriter(FakeChat())
        with patch.object(Screenwriter.develop_story.retry, "sleep", AsyncMock()):
            story = await writer.develop_story("idea", "req")
        self.assertEqual(story, valid_story.strip())

    async def test_character_extractor_rejects_empty_list_on_nontrivial_script_and_retries(self):
        from agents.character_extractor import CharacterExtractor

        long_script = "INT. SEOUL ROOFTOP - RAINY NIGHT. " * 8
        char_json = json.dumps({
            "characters": [
                {
                    "idx": 0,
                    "identifier_in_scene": "Kayden",
                    "is_visible": True,
                    "static_features": "Tall silver-haired man with electric blue eyes.",
                    "dynamic_features": "Black trench coat drenched in storm rain.",
                }
            ]
        })

        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        class FakeResponseChat:
            def __init__(self):
                self.calls = 0

            async def ainvoke(self, input_val, config=None, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return AIMessage(content='{"characters": []}')
                return AIMessage(content=char_json)

        chat_impl = FakeResponseChat()
        chat_runnable = RunnableLambda(chat_impl.ainvoke)
        extractor = CharacterExtractor(chat_runnable)
        chars = await extractor.extract_characters(long_script)
        self.assertEqual(chat_impl.calls, 2)
        self.assertEqual(len(chars), 1)
        self.assertEqual(chars[0].identifier_in_scene, "Kayden")

    async def test_idea_pipeline_regenerates_corrupt_cached_story_characters_and_script(self):
        from pipelines.idea2video_pipeline import Idea2VideoPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_story = (
                "Title: Rainy Alley\n\n"
                + ("Jiwoo sprinted across the neon alley cradling the chubby orange tabby cat while purple static buzzed behind them. " * 4)
            )
            extracted_char = CharacterInScene(
                idx=0,
                identifier_in_scene="Seo Jiwoo",
                is_visible=True,
                static_features="17-year-old boy with neat chestnut hair.",
                dynamic_features="Navy hoodie.",
            )

            # Seed degenerate stubs like an aborted upstream chat turn
            (root / "story.txt").write_text("Verified before delivery.", encoding="utf-8")
            (root / "characters.json").write_text("[]", encoding="utf-8")
            (root / "script.json").write_text("[]", encoding="utf-8")

            pipeline = Idea2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            pipeline.screenwriter = SimpleNamespace(
                develop_story=AsyncMock(return_value=valid_story),
                write_script_based_on_story=AsyncMock(return_value=["SCENE 1 - ROOFTOP\nKayden lands in rain."]),
            )
            pipeline.character_extractor = SimpleNamespace(
                extract_characters=AsyncMock(return_value=[extracted_char]),
            )

            story = await pipeline.develop_story("idea", "req", quiet=True)
            chars = await pipeline.extract_characters(story, quiet=True)
            script = await pipeline.write_script_based_on_story(story, "req", quiet=True)

            self.assertEqual(story, valid_story)
            pipeline.screenwriter.develop_story.assert_awaited_once()
            self.assertEqual(len(chars), 1)
            self.assertEqual(chars[0].identifier_in_scene, "Seo Jiwoo")
            pipeline.character_extractor.extract_characters.assert_awaited_once()
            self.assertEqual(script, ["SCENE 1 - ROOFTOP\nKayden lands in rain."])
            pipeline.screenwriter.write_script_based_on_story.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
