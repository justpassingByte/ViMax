import os
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from interfaces import Camera, CharacterInScene, ShotBriefDescription, ShotDescription
from pipelines.script2video_pipeline import Script2VideoPipeline, _group_shots_into_cameras
from agents.character_portraits_generator import CharacterPortraitsGenerator, _is_quadruped_character


class FlakyCameraImageGenerator:
    def __init__(self):
        self.calls = 0

    async def construct_camera_tree(self, cameras, shot_descs):
        self.calls += 1
        if self.calls == 1:
            return ["not-a-camera"]
        return cameras


class Script2VideoPipelineGuardTests(unittest.IsolatedAsyncioTestCase):
    def test_group_shots_into_cameras_does_not_use_camera_idx_as_list_index(self):
        shots = [
            ShotDescription(idx=0, is_last=False, cam_idx=2, visual_desc="a", variation_type="small", variation_reason="same", ff_desc="a", ff_vis_char_idxs=[], lf_desc="a", lf_vis_char_idxs=[], motion_desc="a", audio_desc="none"),
            ShotDescription(idx=1, is_last=True, cam_idx=5, visual_desc="b", variation_type="small", variation_reason="same", ff_desc="b", ff_vis_char_idxs=[], lf_desc="b", lf_vis_char_idxs=[], motion_desc="b", audio_desc="none"),
            ShotDescription(idx=2, is_last=True, cam_idx=2, visual_desc="c", variation_type="small", variation_reason="same", ff_desc="c", ff_vis_char_idxs=[], lf_desc="c", lf_vis_char_idxs=[], motion_desc="c", audio_desc="none"),
        ]
        cameras = _group_shots_into_cameras(shots)
        self.assertEqual([camera.idx for camera in cameras], [2, 5])
        self.assertEqual(cameras[0].active_shot_idxs, [0, 2])
        self.assertEqual(cameras[1].active_shot_idxs, [1])

    async def test_plan_text_artifacts_retries_bad_camera_tree_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            pipeline.camera_image_generator = FlakyCameraImageGenerator()

            async def design_storyboard(script, characters, user_requirement, quiet=False):
                return [{"idx": 0, "is_last": True, "cam_idx": 3, "visual_desc": "wide shot", "audio_desc": "waves"}]

            async def decompose_visual_descriptions(shot_brief_descriptions, characters, quiet=False):
                return [{"idx": 0, "is_last": True, "cam_idx": 3, "visual_desc": "wide shot", "variation_type": "small", "variation_reason": "simple", "ff_desc": "start", "ff_vis_char_idxs": [], "lf_desc": "end", "lf_vis_char_idxs": [], "motion_desc": "walk", "audio_desc": "waves"}]

            pipeline.design_storyboard = design_storyboard
            pipeline.decompose_visual_descriptions = decompose_visual_descriptions
            events = []
            result = await pipeline.plan_text_artifacts(
                "script",
                "req",
                "style",
                characters=[{"idx": 0, "identifier_in_scene": "Man", "is_visible": True, "static_features": "adult", "dynamic_features": "coat"}],
                progress=lambda stage, message, metadata=None: events.append(stage),
                quiet=True,
            )

            self.assertEqual(pipeline.camera_image_generator.calls, 2)
            self.assertIn("construct_camera_tree_retry", events)
            self.assertEqual(result["camera_tree"][0].idx, 3)
            self.assertTrue((Path(tmp) / "camera_tree.json").exists())

    async def test_generate_character_portraits_skips_invisible_characters(self):
        from interfaces import CharacterInScene

        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            called = []

            async def fake_single(character, style, progress=None):
                called.append(character.identifier_in_scene)
                return {character.identifier_in_scene: {}}

            pipeline.generate_portraits_for_single_character = fake_single
            chars = [
                CharacterInScene(idx=0, identifier_in_scene="Kayden", is_visible=True, static_features="cat", dynamic_features="fat"),
                CharacterInScene(idx=1, identifier_in_scene="Narrator", is_visible=False, static_features="", dynamic_features=None),
            ]
            registry = await pipeline.generate_character_portraits(chars, None, "anime")
            self.assertEqual(called, ["Kayden"])
            self.assertIn("Kayden", registry)
            self.assertNotIn("Narrator", registry)

    async def test_quadruped_portrait_prompt_uses_four_paws_and_isolates_human_form(self):
        captured_prompts = []

        class RecordingImageGen:
            async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
                captured_prompts.append(prompt)
                return object()

        gen = CharacterPortraitsGenerator(RecordingImageGen())
        human_kayden = CharacterInScene(
            idx=0,
            identifier_in_scene="Kayden (Dạng Người)",
            is_visible=True,
            static_features="Người đàn ông trẻ tóc trắng bạc, mắt xanh lam.",
            dynamic_features="Áo choàng dài màu xám đen.",
        )
        cat_kayden = CharacterInScene(
            idx=1,
            identifier_in_scene="Kayden (Mèo Béo)",
            is_visible=True,
            static_features="Chú mèo mướp vàng cam béo tròn đi bằng bốn chân.",
            dynamic_features="Lông vàng cam sọc vằn, không mặc quần áo.",
        )
        self.assertFalse(_is_quadruped_character(human_kayden))
        self.assertTrue(_is_quadruped_character(cat_kayden))

        await gen.generate_front_portrait(human_kayden, "anime")
        await gen.generate_front_portrait(cat_kayden, "anime")
        self.assertIn("standing naturally on two feet", captured_prompts[0])
        self.assertIn("all four paws", captured_prompts[1])
        self.assertIn("NEVER bipedal", captured_prompts[1])

    async def test_native_audio_skips_tts_and_burns_synced_vietsubs(self):
        import imageio_ffmpeg
        import yaml
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "audio": {"model_provider": "native", "bgm_file": ""},
                "subtitles": {"enabled": True, "language": "vi", "burn_in": True, "font": "Arial", "font_size": 24},
            }), encoding="utf-8")

            work_dir = root / ".working_dir" / "ep1" / "script2video"
            work_dir.mkdir(parents=True)
            for s_idx in (0, 1):
                s_dir = work_dir / "shots" / str(s_idx)
                s_dir.mkdir(parents=True)
                clip_path = s_dir / "video.mp4"
                proc = subprocess.run(
                    [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=0x101828:s=320x180:d=2",
                     "-f", "lavfi", "-i", f"sine=frequency={440 + s_idx * 110}:duration=2",
                     "-c:v", "libx264", "-c:a", "aac", "-shortest", str(clip_path)],
                    capture_output=True, text=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)

            script = (
                "Shot 1 (20.0s):\n"
                '[Speaker] Kayden (Dạng Người): "Kuzu domo ga!"\n'
                "(Vietsub: Lũ chuột nhắt các ngươi!)\n\n"
                "Shot 2 (20.0s):\n"
                "Audio: [Sound Effect] Sấm sét im ắng.\n"
            )
            (work_dir / "script.txt").write_text(script, encoding="utf-8")

            pipeline = Script2VideoPipeline(
                chat_model=object(),
                image_generator=object(),
                video_generator=object(),
                working_dir=str(work_dir),
            )
            chars = [
                CharacterInScene(idx=0, identifier_in_scene="Kayden (Dạng Người)", is_visible=False, static_features="man"),
            ]
            storyboard = [
                ShotBriefDescription(idx=0, is_last=False, cam_idx=0, visual_desc="s0", audio_desc='[Speaker] Kayden (Dạng Người): "Kuzu domo ga!"'),
                ShotBriefDescription(idx=1, is_last=True, cam_idx=1, visual_desc="s1", audio_desc="[Sound Effect] Sấm sét."),
            ]
            shot_descs = [
                ShotDescription(idx=0, is_last=False, cam_idx=0, visual_desc="s0", variation_type="small", variation_reason="r", ff_desc="ff", ff_vis_char_idxs=[], lf_desc="lf", lf_vis_char_idxs=[], motion_desc="m", audio_desc='[Speaker] Kayden (Dạng Người): "Kuzu domo ga!"'),
                ShotDescription(idx=1, is_last=True, cam_idx=1, visual_desc="s1", variation_type="small", variation_reason="r", ff_desc="ff", ff_vis_char_idxs=[], lf_desc="lf", lf_vis_char_idxs=[], motion_desc="m", audio_desc="[Sound Effect] Sấm sét."),
            ]
            cameras = [Camera(idx=0, active_shot_idxs=[0]), Camera(idx=1, active_shot_idxs=[1])]

            pipeline.design_storyboard = AsyncMock(return_value=storyboard)
            pipeline.decompose_visual_descriptions = AsyncMock(return_value=shot_descs)
            pipeline.construct_camera_tree = AsyncMock(return_value=cameras)
            pipeline.generate_character_portraits = AsyncMock(return_value={})
            pipeline.generate_frames_for_single_camera = AsyncMock(return_value=None)
            pipeline.generate_video_for_single_shot = AsyncMock(return_value=None)

            events = []
            with patch("pipelines.script2video_pipeline.synthesize_speech") as mock_tts, \
                 patch.dict(os.environ, {}, clear=True):
                final_path = await pipeline(
                    script=script,
                    user_requirement="",
                    style="anime",
                    characters=chars,
                    character_portraits_registry={},
                    quiet=True,
                    progress=lambda stage, msg, meta=None: events.append(stage),
                )

            mock_tts.assert_not_called()
            self.assertIn("audio_native", events)
            self.assertIn("subtitles_done", events)
            srt_file = work_dir / "final_video_vietsub.srt"
            ass_file = work_dir / "final_video_vietsub.ass"
            self.assertTrue(srt_file.exists())
            self.assertTrue(ass_file.exists())
            srt_text = srt_file.read_text(encoding="utf-8")
            self.assertIn("Lũ chuột nhắt các ngươi!", srt_text)
            self.assertIn("00:00:00,200 --> 00:00:01,900", srt_text)

            from moviepy import VideoFileClip
            clip = VideoFileClip(final_path)
            try:
                self.assertIsNotNone(clip.audio)
            finally:
                clip.close()

    async def test_single_shot_video_prompt_separates_narrator_audio_without_pink_elephants(self):
        import asyncio
        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "audio": {"model_provider": "native", "bgm_file": ""},
                "subtitles": {"enabled": True, "language": "vi", "burn_in": True},
            }), encoding="utf-8")

            work_dir = root / ".working_dir" / "ep1" / "script2video"
            shot_dir = work_dir / "shots" / "5"
            shot_dir.mkdir(parents=True)
            (shot_dir / "first_frame.png").write_bytes(b"png")

            captured = {}

            class RecordingVideoGen:
                async def generate_single_video(self, prompt="", reference_image_paths=None, suppress_on_screen_subtitles=False, **kwargs):
                    captured["prompt"] = prompt
                    captured["suppress"] = suppress_on_screen_subtitles

                    class DummyOut:
                        def save(self, path):
                            Path(path).write_bytes(b"mp4")

                    return DummyOut()

            pipeline = Script2VideoPipeline(
                chat_model=object(),
                image_generator=object(),
                video_generator=RecordingVideoGen(),
                working_dir=str(work_dir),
            )
            ev = asyncio.Event()
            ev.set()
            pipeline.frame_events[5] = {"first_frame": ev}

            shot = ShotDescription(
                idx=5,
                is_last=False,
                cam_idx=5,
                visual_desc="sleeping cat",
                variation_type="small",
                variation_reason="still",
                ff_desc="ff",
                ff_vis_char_idxs=[1],
                lf_desc="lf",
                lf_vis_char_idxs=[1],
                motion_desc="Con mèo nằm thở nhẹ.",
                audio_desc='[Sound Effect] Nước nhỏ tí tách. [Speaker] Người Dẫn Chuyện: "Kesshi no kakugo de..."\n(Vietsub: Quyết tâm...)',
            )
            await pipeline.generate_video_for_single_shot(shot)

        self.assertTrue(captured["suppress"])
        self.assertIn("Active Speaker: Narrator", captured["prompt"])
        self.assertNotIn("Vietsub:", captured["prompt"])
        self.assertNotIn("karaoke", captured["prompt"].lower())
        self.assertNotIn("speech bubbles", captured["prompt"].lower())

    async def test_generate_and_select_best_image_count_one_forwards_progress_when_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            seen_with_progress = []

            class ProgressAwareImageGen:
                async def generate_single_image(self, prompt, reference_image_paths=None, size="1600x900", progress=None):
                    seen_with_progress.append(progress)
                    return "ok-progress"

            class MinimalImageGen:
                async def generate_single_image(self, prompt, reference_image_paths=None, size="1600x900"):
                    return "ok-minimal"

            cb = lambda stage, msg, meta=None: None
            p1 = Script2VideoPipeline(chat_model=object(), image_generator=ProgressAwareImageGen(), video_generator=object(), working_dir=tmp, num_image_candidates=1)
            res1 = await p1.generate_and_select_best_image("p", [], [], "target", progress=cb)
            self.assertEqual(res1, "ok-progress")
            self.assertEqual(seen_with_progress, [cb])

            p2 = Script2VideoPipeline(chat_model=object(), image_generator=MinimalImageGen(), video_generator=object(), working_dir=tmp, num_image_candidates=1)
            res2 = await p2.generate_and_select_best_image("p", [], [], "target", progress=cb)
            self.assertEqual(res2, "ok-minimal")

    async def test_resume_existing_final_video_burns_vietsubs_once_and_records_marker(self):
        import imageio_ffmpeg
        import yaml
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "agent.local.yaml").write_text(yaml.safe_dump({
                "audio": {"model_provider": "native", "bgm_file": ""},
                "subtitles": {"enabled": True, "language": "vi", "burn_in": True, "font": "Arial", "font_size": 24},
            }), encoding="utf-8")

            work_dir = root / ".working_dir" / "ep1" / "script2video"
            shot_dir = work_dir / "shots" / "0"
            shot_dir.mkdir(parents=True)
            shot_mp4 = shot_dir / "video.mp4"
            proc = subprocess.run(
                [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=0x101828:s=320x180:d=2",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                 "-c:v", "libx264", "-c:a", "aac", "-shortest", str(shot_mp4)],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            final_mp4 = work_dir / "final_video.mp4"
            final_mp4.write_bytes(shot_mp4.read_bytes())

            script = (
                "Shot 1 (8.0s):\n"
                '[Speaker] Kayden: "Kuzu domo ga!"\n'
                "(Vietsub: Lũ chuột nhắt các ngươi!)\n"
            )
            (work_dir / "script.txt").write_text(script, encoding="utf-8")

            pipeline = Script2VideoPipeline(
                chat_model=object(),
                image_generator=object(),
                video_generator=object(),
                working_dir=str(work_dir),
            )
            chars = [CharacterInScene(idx=0, identifier_in_scene="Kayden", is_visible=False, static_features="man")]
            storyboard = [ShotBriefDescription(idx=0, is_last=True, cam_idx=0, visual_desc="s0", audio_desc='[Speaker] Kayden: "Kuzu domo ga!"')]
            shot_descs = [ShotDescription(idx=0, is_last=True, cam_idx=0, visual_desc="s0", variation_type="small", variation_reason="r", ff_desc="ff", ff_vis_char_idxs=[], lf_desc="lf", lf_vis_char_idxs=[], motion_desc="m", audio_desc='[Speaker] Kayden: "Kuzu domo ga!"')]
            cameras = [Camera(idx=0, active_shot_idxs=[0])]

            for target_attr, val in (
                ("design_storyboard", storyboard),
                ("decompose_visual_descriptions", shot_descs),
                ("construct_camera_tree", cameras),
                ("generate_character_portraits", {}),
                ("generate_frames_for_single_camera", None),
                ("generate_video_for_single_shot", None),
            ):
                setattr(pipeline, target_attr, AsyncMock(return_value=val))

            subtitled_fallback = work_dir / "final_video_subtitled.mp4"
            burn_calls = []

            def fake_burn(video_path, ass_path, output_path):
                burn_calls.append((video_path, ass_path, output_path))
                subtitled_fallback.write_bytes(b"burned-fallback")
                return str(subtitled_fallback)

            with patch("pipelines.script2video_pipeline.burn_subtitles_into_video", side_effect=fake_burn), \
                 patch.dict(os.environ, {}, clear=True):
                first_result = await pipeline(
                    script=script,
                    user_requirement="",
                    style="anime",
                    characters=chars,
                    character_portraits_registry={},
                    quiet=True,
                )

            self.assertEqual(len(burn_calls), 1)
            self.assertEqual(Path(first_result).resolve(), subtitled_fallback.resolve())
            marker = work_dir / ".subtitles_burned_vietsub.json"
            self.assertTrue(marker.exists())
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
            self.assertEqual(marker_payload["output_path"], "final_video_subtitled.mp4")

            with patch("pipelines.script2video_pipeline.burn_subtitles_into_video", side_effect=fake_burn), \
                 patch.dict(os.environ, {}, clear=True):
                second_result = await pipeline(
                    script=script,
                    user_requirement="",
                    style="anime",
                    characters=chars,
                    character_portraits_registry={},
                    quiet=True,
                )

            self.assertEqual(len(burn_calls), 1)
            self.assertEqual(Path(second_result).resolve(), subtitled_fallback.resolve())

    async def test_frame_events_and_continuity_no_deadlock_on_speedstreak_shot(self):
        import asyncio
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = Script2VideoPipeline(chat_model=object(), image_generator=object(), video_generator=object(), working_dir=tmp)
            shots = [
                ShotDescription(
                    idx=0,
                    is_last=False,
                    cam_idx=0,
                    visual_desc="Speed streak across sky [AnimeFX: SpeedStreak] [AnimeFX: Vanish]",
                    variation_type="medium",
                    variation_reason="moving streak",
                    ff_desc="streak in sky",
                    ff_vis_char_idxs=[],
                    lf_desc="streak lower in sky",
                    lf_vis_char_idxs=[],
                    motion_desc="streak falls",
                    audio_desc="whoosh",
                    continuity_from_prev=False,
                ),
                ShotDescription(
                    idx=1,
                    is_last=True,
                    cam_idx=1,
                    visual_desc="Ground shot continuing impact",
                    variation_type="large",
                    variation_reason="impact",
                    ff_desc="ground crater",
                    ff_vis_char_idxs=[],
                    lf_desc="crater smoke",
                    lf_vis_char_idxs=[],
                    motion_desc="impact burst",
                    audio_desc="boom",
                    continuity_from_prev=True,
                ),
            ]
            pipeline.frame_events = {}
            for sd in shots:
                from pipelines.script2video_pipeline import shot_requires_last_frame
                ev_map = {"first_frame": asyncio.Event()}
                if shot_requires_last_frame(shots, sd):
                    ev_map["last_frame"] = asyncio.Event()
                pipeline.frame_events[sd.idx] = ev_map

            # Shot 0 is speedstreak -> must NOT require last_frame
            self.assertNotIn("last_frame", pipeline.frame_events[0])

            # Simulate Shot 0 first_frame completed
            shot0_ff = Path(tmp) / "shots" / "0" / "first_frame.png"
            shot0_ff.parent.mkdir(parents=True, exist_ok=True)
            shot0_ff.write_bytes(b"fake-png-data")
            pipeline.frame_events[0]["first_frame"].set()

            # Shot 1 resolves continuity from Shot 0 -> should resolve instantly without waiting for non-existent last_frame
            cont_pair = await asyncio.wait_for(pipeline._resolve_continuity_reference(shots, 1), timeout=1.0)
            self.assertIsNotNone(cont_pair)
            self.assertEqual(cont_pair[0], str(shot0_ff))

    async def test_stop_after_frames_never_calls_transition_video_and_skips_alley_anchor_in_pet_shop(self):
        import asyncio
        from types import SimpleNamespace
        from interfaces import Camera

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "ep1" / "script2video"
            for i in (0, 3, 4):
                s_dir = work_dir / "shots" / str(i)
                s_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "shots" / "0" / "first_frame.png").write_bytes(b"png0")
            (work_dir / "shots" / "3" / "first_frame.png").write_bytes(b"png3")

            captured_prompts = []
            captured_final_pairs = []
            pipeline = Script2VideoPipeline(
                chat_model=object(),
                image_generator=object(),
                video_generator=object(),
                working_dir=str(work_dir),
            )
            pipeline.generate_transition_video = AsyncMock(side_effect=AssertionError("must not invoke muse-video during frames stage"))

            async def fake_select(available_image_path_and_text_pairs, frame_description, **kwargs):
                return {"reference_image_path_and_text_pairs": [], "text_prompt": "clean anime prompt"}

            class DummyImg:
                def save(self, path):
                    Path(path).write_bytes(b"fake-png")

            async def fake_gen_best(prompt, reference_image_paths=None, reference_image_path_and_text_pairs=None, **kwargs):
                captured_prompts.append(prompt)
                captured_final_pairs.append(list(reference_image_path_and_text_pairs or []))
                return DummyImg()

            pipeline.reference_image_selector.select_reference_images_and_generate_prompt = AsyncMock(side_effect=fake_select)
            pipeline.generate_and_select_best_image = AsyncMock(side_effect=fake_gen_best)
            pipeline.frame_events = {
                0: {"first_frame": asyncio.Event()},
                3: {"first_frame": asyncio.Event()},
                4: {"first_frame": asyncio.Event()},
            }
            pipeline.frame_events[0]["first_frame"].set()
            pipeline.frame_events[3]["first_frame"].set()
            pipeline.camera_events = {0: asyncio.Event(), 1: asyncio.Event()}
            pipeline.camera_events[0].set()

            shots = [
                ShotDescription(
                    idx=0, is_last=False, cam_idx=0, visual_desc="Wide pet shop aisle",
                    variation_type="small", variation_reason="static", ff_desc="wide pet shop aisle",
                    ff_vis_char_idxs=[0], lf_desc="wide pet shop aisle", lf_vis_char_idxs=[0],
                    motion_desc="static", audio_desc="ambient", continuity_from_prev=False,
                ),
                ShotDescription(idx=1, is_last=False, cam_idx=0, visual_desc="s1", variation_type="small", variation_reason="s", ff_desc="s1", ff_vis_char_idxs=[], lf_desc="s1", lf_vis_char_idxs=[], motion_desc="m", audio_desc="a"),
                ShotDescription(idx=2, is_last=False, cam_idx=0, visual_desc="s2", variation_type="small", variation_reason="s", ff_desc="s2", ff_vis_char_idxs=[], lf_desc="s2", lf_vis_char_idxs=[], motion_desc="m", audio_desc="a"),
                ShotDescription(idx=3, is_last=False, cam_idx=0, visual_desc="s3", variation_type="small", variation_reason="s", ff_desc="s3", ff_vis_char_idxs=[], lf_desc="s3", lf_vis_char_idxs=[], motion_desc="m", audio_desc="a"),
                ShotDescription(
                    idx=4, is_last=True, cam_idx=1,
                    visual_desc="Side profile close-up inside pet shop under warm store lighting",
                    variation_type="small", variation_reason="static",
                    ff_desc="Side profile close-up inside bright pet shop under warm store lighting",
                    ff_vis_char_idxs=[0],
                    lf_desc="Side profile close-up inside bright pet shop under warm store lighting",
                    lf_vis_char_idxs=[0],
                    motion_desc="static", audio_desc="thought",
                    transition_in="cut", continuity_from_prev=False,
                ),
            ]
            chars = [CharacterInScene(idx=0, identifier_in_scene="Seo Jiwoo", is_visible=True, static_features="boy")]
            cam1 = Camera(idx=1, active_shot_idxs=[4], parent_shot_idx=0, variation_type="medium", variation_reason="angle change", missing_info=None)

            await pipeline.generate_frames_for_single_camera(
                camera=cam1,
                shot_descriptions=shots,
                characters=chars,
                character_portraits_registry={},
                priority_shot_idxs=[],
                shot_panel_refs=None,
                allow_transition_video=False,
            )

            pipeline.generate_transition_video.assert_not_called()
            joined_descs = " ".join(desc for _, desc in captured_final_pairs[0]).lower()
            self.assertIn("shot 3", joined_descs)
            self.assertNotIn("rainy navy", joined_descs)
            self.assertNotIn("stone wall", joined_descs)
            self.assertNotIn("asphalt", joined_descs)
            self.assertNotIn("rainy navy", captured_prompts[0].lower())


if __name__ == "__main__":
    unittest.main()
