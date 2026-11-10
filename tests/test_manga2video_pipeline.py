import asyncio
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from PIL import Image, ImageDraw

from agent_runtime.session_index import SessionIndex
from agent_runtime.vimax_adapters import ViMaxAdapters, build_vimax_adapter_specs, invalidate_render_artifacts
from agents.character_portraits_generator import CharacterPortraitsGenerator
from agents.manga_reader import MangaReader, split_dialogue_to_word_budget
from interfaces import (
    Camera,
    CharacterInScene,
    ImageOutput,
    MangaBeat,
    MangaCharacterObservation,
    MangaChunkExtraction,
    MangaDialogueLine,
    ShotBriefDescription,
    ShotDescription,
    VideoOutput,
)
from pipelines.manga2video_pipeline import Manga2VideoPipeline, bind_storyboard_to_panel_anchors
from pipelines.script2video_pipeline import Script2VideoPipeline
from tools.manga_source import (
    download_webtoon_chapter,
    extract_cbz_archive,
    extract_pdf_pages,
    ingest_manga_sources,
    parse_chapter_range,
    parse_webtoon_image_urls,
    split_and_normalize_images,
)
from utils.audio import extract_all_dialogues_from_shot, extract_spoken_dialogues_for_subtitles


def _make_synthetic_mp4(path: Path, *, color: str = "0x1E90FF", duration: float = 1.5, freq: int = 440) -> None:
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c={color}:s=320x180:d={duration}:r=24",
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
        raise RuntimeError(f"ffmpeg synthetic clip failed: {proc.stderr}")


def _save_comic_panel(path: Path, *, width: int = 700, height: int = 900, color=(235, 240, 250)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, width - 40, height - 40], outline=(20, 20, 30), width=4)
    img.save(path, format="JPEG", quality=90)
    img.close()


class MangaSourceIngestionTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_chapter_range_and_reject_invalid(self):
        self.assertEqual(parse_chapter_range("1-3"), {1, 2, 3})
        self.assertEqual(parse_chapter_range("ch01, ch03-04"), {1, 3, 4})
        self.assertEqual(parse_chapter_range([2, 5]), {2, 5})
        self.assertIsNone(parse_chapter_range("all"))
        with self.assertRaises(ValueError):
            parse_chapter_range("5-2")
        with self.assertRaises(ValueError):
            parse_chapter_range(-1)

    def test_split_tall_strip_cuts_cleanly_at_white_gutter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            # Create a 600x3200 vertical strip with busy art at top/bottom and a pure white gutter at y=1600..1760
            arr = np.full((3200, 600, 3), 255, dtype=np.uint8)
            rng = np.random.default_rng(42)
            arr[80:1520, 40:560] = rng.integers(20, 220, size=(1440, 520, 3), dtype=np.uint8)
            arr[1820:3120, 40:560] = rng.integers(30, 210, size=(1300, 520, 3), dtype=np.uint8)
            strip_path = tmp_dir / "long_strip.png"
            Image.fromarray(arr).save(strip_path)

            pages = split_and_normalize_images(
                [strip_path],
                tmp_dir / "out",
                chapter=1,
                split_height_threshold=2400,
                target_slice_height=1680,
            )
            self.assertEqual(len(pages), 2)
            self.assertGreater(pages[0].height, 1550)
            self.assertLess(pages[0].height, 1790)
            # Split boundary row inside original strip must lie inside the uniform white gutter (zero busy pixels!)
            cut_y = pages[0].height
            self.assertTrue(1600 <= cut_y <= 1760, f"Cut landed at {cut_y}, outside white gutter 1600..1760")

    def test_cbz_extraction_blocks_zip_slip_and_extracts_safe_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            evil_cbz = tmp_dir / "evil.cbz"
            with zipfile.ZipFile(evil_cbz, "w") as zf:
                zf.writestr("../../outside.png", b"evil")
            with self.assertRaises(ValueError):
                extract_cbz_archive(evil_cbz, tmp_dir / "dest")

            good_cbz = tmp_dir / "ch02.cbz"
            img1 = tmp_dir / "a.jpg"
            img2 = tmp_dir / "b.jpg"
            _save_comic_panel(img1)
            _save_comic_panel(img2)
            with zipfile.ZipFile(good_cbz, "w") as zf:
                zf.write(img2, "p10.jpg")
                zf.write(img1, "p02.jpg")
                zf.writestr("notes.txt", "ignore me")
            extracted = extract_cbz_archive(good_cbz, tmp_dir / "safe_dest")
            self.assertEqual([p.name for p in extracted], ["p02.jpg", "p10.jpg"])

    def test_pdf_ingestion_raises_helpful_runtime_error_when_libs_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            pdf_file = Path(tmp) / "ch01.pdf"
            pdf_file.write_bytes(b"%PDF-1.4 fake")
            with patch.dict("sys.modules", {"fitz": None, "pdf2image": None}):
                with self.assertRaises(RuntimeError) as ctx:
                    extract_pdf_pages(pdf_file, Path(tmp) / "out")
            self.assertIn("pymupdf", str(ctx.exception).lower())

    async def test_webtoon_html_parser_ssrf_guard_and_downloader(self):
        sample_html = """
        <html><head><title>Eleceed Ep 1</title></head>
        <body>
          <h1>Episode 1</h1>
          <div id="_imageList">
            <img class="_images" data-url="https://webtoon-phinf.pstatic.net/img_01.jpg" />
            <img class="_images" src="/img/blank.gif" />
            <img class="_images" data-url="https://webtoon-phinf.pstatic.net/img_02.png" />
          </div>
          <img class="avatar" src="https://example.com/avatar.png" />
        </body></html>
        """
        title, urls = parse_webtoon_image_urls(sample_html, base_url="https://www.webtoons.com/en/action/eleceed/ep1/viewer")
        self.assertEqual(title, "Episode 1")
        self.assertEqual(
            urls,
            [
                "https://webtoon-phinf.pstatic.net/img_01.jpg",
                "https://webtoon-phinf.pstatic.net/img_02.png",
            ],
        )

        # Private/localhost URL must be blocked by default
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                await download_webtoon_chapter("http://127.0.0.1:8080/ch1", tmp)

            img_buf_1 = Path(tmp) / "i1.jpg"
            img_buf_2 = Path(tmp) / "i2.png"
            _save_comic_panel(img_buf_1, width=700, height=500)
            _save_comic_panel(img_buf_2, width=700, height=500)

            referers = []

            async def fake_fetcher(url: str, *, referer: str = ""):
                referers.append((url, referer))
                if url.endswith("viewer"):
                    return sample_html.encode("utf-8")
                if url.endswith("img_01.jpg"):
                    return img_buf_1.read_bytes()
                return img_buf_2.read_bytes()

            dl_title, paths = await download_webtoon_chapter(
                "https://www.webtoons.com/en/action/eleceed/ep1/viewer",
                Path(tmp) / "dl",
                fetcher=fake_fetcher,
            )
            self.assertEqual(dl_title, "Episode 1")
            self.assertEqual(len(paths), 1)  # Two 500px strips stitched cleanly into one sheet
            self.assertTrue(all(ref for _, ref in referers))

    async def test_ingest_multi_chapter_folder_with_chapter_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "series"
            for ch in (1, 2, 3):
                for p in range(1, 3):
                    _save_comic_panel(src / f"ch{ch:02d}" / f"p{p:02d}.jpg")
            manifest = await ingest_manga_sources(src, root / "work", chapter_range="1-2")
            self.assertEqual([c.chapter for c in manifest.chapters], [1, 2])
            self.assertEqual(manifest.total_pages, 4)
            self.assertTrue((root / "work" / "comic_manifest.json").is_file())


class MangaReaderTests(unittest.IsolatedAsyncioTestCase):
    async def test_vision_batching_tail_continuity_form_separation_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src_ch = root / "ch01"
            for i in range(4):
                _save_comic_panel(src_ch / f"p{i + 1:02d}.jpg")
            manifest = await ingest_manga_sources(src_ch, root / "work")

            # User uploads explicitly tagged ref for Kayden (Human) only, not Cat Form
            uploads_dir = root / "uploads"
            uploads_dir.mkdir()
            human_upload = uploads_dir / "kayden_human.png"
            ignored_page = uploads_dir / "p01.jpg"
            _save_comic_panel(human_upload, width=400, height=400)
            _save_comic_panel(ignored_page, width=400, height=400)

            seen_batches = []
            seen_tails = []

            class FakeVisionReaderModel(Runnable):
                def invoke(self, input, config=None, **kwargs):
                    raise NotImplementedError

                async def ainvoke(self, messages, config=None, **kwargs):
                    human_items = messages[1].content
                    imgs = [item["image_url"]["url"] for item in human_items if item.get("type") == "image_url"]
                    header_txt = human_items[0]["text"]
                    seen_batches.append(len(imgs))
                    seen_tails.append(header_txt)

                    if len(seen_batches) == 1:
                        # Batch 1: pages [0, 1, 2]
                        payload = {
                            "characters_observed": [
                                {
                                    "name": "Kayden",
                                    "species_or_form": "human",
                                    "appearance_summary": "Wounded man with black messy hair",
                                    "outfit_summary": "Dark torn hoodie",
                                    "page_idx": 0,
                                    "crop_box_norm": [100, 100, 700, 700],
                                },
                                {
                                    "name": "Kayden",
                                    "species_or_form": "cat (four-legged quadruped)",
                                    "appearance_summary": "Fat orange tabby cat on four paws",
                                    "outfit_summary": "Orange tabby fur",
                                    "page_idx": 2,
                                    "crop_box_norm": [200, 200, 800, 800],
                                },
                            ],
                            "beats": [
                                {
                                    "page_idx": 0,
                                    "is_scene_break": True,
                                    "setting": "Dark rainy alley",
                                    "action_desc": "Wounded Kayden staggers through the rain.",
                                    "visual_details": "Lightning flickers.",
                                    "characters_present": ["Kayden (Human)"],
                                    "dialogues": [
                                        {
                                            "speaker": "Kayden (Human)",
                                            "line": "Kuso... kono mama de wa mazuina, hayaku kakureru basho wo sagasanakereba naranai.",
                                            "line_type": "thought",
                                            "emotion": "Pained",
                                            "vietsub": "Chết tiệt... Cứ thế này không ổn rồi, phải mau tìm chỗ trốn thôi.",
                                        }
                                    ],
                                    "sfx": ["ZAA"],
                                    "crop_box_norm": [50, 50, 850, 950],
                                }
                            ],
                            "chunk_tail_summary": "Kayden transforms near the cardboard box.",
                        }
                    else:
                        payload = {
                            "characters_observed": [],
                            "beats": [
                                {
                                    "page_idx": 3,
                                    "is_scene_break": False,
                                    "setting": "Rainy alley box",
                                    "action_desc": "Fat orange cat sits soaking wet.",
                                    "visual_details": "Grumpy expression.",
                                    "characters_present": ["Kayden (Cat)"],
                                    "dialogues": [
                                        {
                                            "speaker": "Kayden",
                                            "line": "Nyaa?",
                                            "line_type": "spoken",
                                            "emotion": "Grumpy",
                                            "vietsub": "Meo?",
                                        }
                                    ],
                                    "sfx": [],
                                    "crop_box_norm": [100, 100, 900, 900],
                                }
                            ],
                            "chunk_tail_summary": "",
                        }
                    return AIMessage(content=json.dumps(payload, ensure_ascii=False))

            reader = MangaReader(FakeVisionReaderModel())
            chunks = await reader.read_chapter_pages(
                manifest.chapters[0],
                root / "work" / "chunks",
                batch_size=3,
            )
            self.assertEqual(seen_batches, [3, 1])
            self.assertIn("cardboard box", seen_tails[1])
            self.assertEqual(len(chunks), 2)

            # Second call MUST hit JSON cache with 0 additional Vision calls
            chunks_cached = await reader.read_chapter_pages(
                manifest.chapters[0],
                root / "work" / "chunks",
                batch_size=3,
            )
            self.assertEqual(len(seen_batches), 2)
            self.assertEqual(len(chunks_cached), 2)

            chars, seed_refs, alias_map = reader.build_character_roster_and_seeds(
                manifest=manifest,
                chunks=chunks,
                working_dir=root / "work",
                upload_image_paths=[human_upload, ignored_page],
            )
            char_names = [c.identifier_in_scene for c in chars]
            self.assertEqual(len(chars), 2)  # Distinct Human vs Cat forms NEVER merged!
            human_name = next(n for n in char_names if "Human" in n)
            cat_name = next(n for n in char_names if "Human" not in n)
            # Human matched user upload, Cat cropped from comic panel (NOT polluted by kayden_human.png or p01.jpg!)
            self.assertEqual(seed_refs[human_name], [str(human_upload)])
            self.assertTrue(seed_refs[cat_name][0].endswith(".jpg"))
            self.assertIn("character_seeds", seed_refs[cat_name][0])

            plan, beats_by_scene = reader.build_episode_scenes(
                manifest=manifest,
                chunks=chunks,
                characters=chars,
                character_seed_refs=seed_refs,
                alias_map=alias_map,
                working_dir=root / "work",
                max_words_per_shot=14,
            )
            self.assertGreaterEqual(len(plan.scenes), 1)
            # Long dialogue from beat 0 should still stay valid & preserved in scene script with [Thought]
            script_txt = (root / "work" / "scene_0" / "script.txt").read_text(encoding="utf-8")
            self.assertIn("[Thought]", script_txt)
            self.assertIn("Vietsub:", script_txt)
            self.assertIn("[RefBeat:", script_txt)

    def test_split_dialogue_to_word_budget(self):
        long_line = "Kuso kono mama de wa mazuina hayaku kakureru basho wo sagasanakereba naranai zo hontou ni yabai."
        chunks = split_dialogue_to_word_budget(
            long_line,
            "Chết tiệt cứ thế này không ổn rồi, phải mau tìm chỗ trốn ngay thôi thật sự nguy to.",
            max_words=8,
        )
        self.assertGreaterEqual(len(chunks), 2)
        for line_part, viet_part in chunks:
            self.assertLessEqual(len(line_part.split()), 8)
            self.assertTrue(line_part)


class HybridRenderingGuardsTests(unittest.IsolatedAsyncioTestCase):
    async def test_character_portraits_generator_hybrid_refs_and_backwards_compat(self):
        calls = []

        class RecordingImageGen:
            async def generate_single_image(self, prompt, **kwargs):
                calls.append((prompt, kwargs))
                return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (64, 36), "white"))

        gen = CharacterPortraitsGenerator(RecordingImageGen())
        char = CharacterInScene(idx=0, identifier_in_scene="Jiwoo", is_visible=True, static_features="teen boy", dynamic_features="jacket")

        await gen.generate_front_portrait(char, "anime")
        self.assertNotIn("reference_image_paths", calls[0][1])

        await gen.generate_front_portrait(char, "anime", reference_image_paths=["seed1.jpg"])
        self.assertEqual(calls[1][1]["reference_image_paths"], ["seed1.jpg"])
        self.assertIn("HYBRID COMIC-TO-ANIME REDRAW", calls[1][0])

        await gen.generate_back_portrait(char, "front.png", "anime")
        self.assertIn("NEVER draw a necktie hanging down the back", calls[2][0])

    async def test_vimax_manga2video_frames_stage_wires_video_generator_for_camera_transitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = SessionIndex(root)
            rec = index.create(project_name="frames-video-wiring")
            sid = rec["session_id"]
            session_dir = index.working_dir(sid)
            manga_dir = session_dir / "manga2video"
            manga_dir.mkdir(parents=True, exist_ok=True)
            (manga_dir / "comic_manifest.json").write_text(
                json.dumps({"chapters": [], "total_pages": 0}), encoding="utf-8"
            )
            (manga_dir / "characters.json").write_text("[]", encoding="utf-8")
            (manga_dir / "episode_plan.json").write_text(
                json.dumps({"title": "ep", "chapters": [1], "total_beats": 0, "scenes": [], "character_seed_refs": {}}),
                encoding="utf-8",
            )
            adapter = ViMaxAdapters(root, index)
            seen_video_gens = []

            class DummyMangaPipe:
                def __init__(self, chat_model, image_generator, video_generator, working_dir, num_image_candidates=1):
                    self.video_generator = video_generator

                async def plan_episode(self, **kwargs):
                    from interfaces import MangaEpisodePlan
                    return MangaEpisodePlan(title="ep", chapters=[1], total_beats=0, scenes=[], character_seed_refs={})

                async def render_episode(self, **kwargs):
                    seen_video_gens.append(self.video_generator)
                    return []

            sentinel_vid = object()
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=sentinel_vid), \
                 patch("agent_runtime.vimax_adapters.Manga2VideoPipeline", DummyMangaPipe):
                res = await adapter.vimax_manga2video({"session_id": sid, "auto_render": True, "stop_after": "frames"})
            self.assertTrue(res.ok, res.content)
            self.assertEqual(seen_video_gens, [sentinel_vid])

    def test_audio_and_subtitles_support_thought_and_inner_monologue_tags(self):
        desc = (
            '[Sound Effect] Heavy rain falls. '
            '[Thought] Jiwoo (Shocked): "Ehh? Neko ga shabetta?!"\n'
            '(Vietsub: Hả?! Con mèo vừa nói chuyện à?!)'
        )
        self.assertEqual(extract_all_dialogues_from_shot(desc), [("Jiwoo", "Ehh? Neko ga shabetta?!")])
        self.assertEqual(extract_spoken_dialogues_for_subtitles(desc), [("Jiwoo", "Ehh? Neko ga shabetta?!")])

    async def test_script2video_panel_anchors_eight_ref_cap_and_thought_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "configs"
            cfg.mkdir()
            (cfg / "agent.local.yaml").write_text("subtitles:\n  enabled: true\n  burn_in: true\n  language: vi\n", encoding="utf-8")

            scene_dir = root / ".working_dir" / "ep1" / "manga2video" / "scene_0"
            scene_dir.mkdir(parents=True)
            portrait_img = root / "portrait.png"
            panel_img = root / "panel0.jpg"
            _save_comic_panel(portrait_img, width=100, height=100)
            _save_comic_panel(panel_img, width=200, height=300)

            (scene_dir / "shot_panel_refs.json").write_text(
                json.dumps(
                    {"0": [{"path": str(panel_img), "description": "PANEL ANCHOR 0"}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            chars = [CharacterInScene(idx=0, identifier_in_scene="Kayden", is_visible=True, static_features="man")]
            registry = {
                "Kayden": {
                    "front": {"path": str(portrait_img), "description": "front portrait of Kayden"},
                    "side": {"path": str(portrait_img), "description": "side portrait of Kayden"},
                }
            }
            brief = ShotBriefDescription(
                idx=0,
                is_last=True,
                cam_idx=0,
                visual_desc="<Kayden> stands in rain",
                audio_desc='[Thought] Kayden (Calm): "Mada owaranai."',
            )
            shot = ShotDescription(
                idx=0,
                is_last=True,
                cam_idx=0,
                visual_desc="<Kayden> stands in rain",
                variation_type="small",
                variation_reason="thought",
                ff_desc="Kayden standing in rain",
                ff_vis_char_idxs=[0],
                lf_desc="Kayden closing eyes",
                lf_vis_char_idxs=[0],
                motion_desc="Rain falls quietly",
                audio_desc='[Thought] Kayden (Calm): "Mada owaranai."',
            )
            cameras = [Camera(idx=0, active_shot_idxs=[0])]

            seen_selector_pairs = []
            seen_img_refs = []
            seen_vid_prompts = []

            class FakeImageGen:
                async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
                    seen_img_refs.append(list(reference_image_paths or []))
                    return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (64, 36), "blue"))

            class FakeVideoGen:
                async def generate_single_video(self, prompt="", reference_image_paths=None, **kwargs):
                    seen_vid_prompts.append(prompt)
                    return VideoOutput(fmt="bytes", ext="mp4", data=b"mp4")

            pipeline = Script2VideoPipeline(
                chat_model=object(),
                image_generator=FakeImageGen(),
                video_generator=FakeVideoGen(),
                working_dir=str(scene_dir),
                num_image_candidates=1,
            )
            pipeline.design_storyboard = AsyncMock(return_value=[brief])
            pipeline.decompose_visual_descriptions = AsyncMock(return_value=[shot])
            pipeline.construct_camera_tree = AsyncMock(return_value=cameras)

            async def fake_select(available_image_path_and_text_pairs, frame_description):
                seen_selector_pairs.append(available_image_path_and_text_pairs)
                # Intentionally return 10 unique items + duplicates to verify dedup AND hard [:8] cap!
                extra_unique = [(str(root / f"extra_{i}.png"), f"extra {i}") for i in range(10)]
                over_selected = available_image_path_and_text_pairs + [available_image_path_and_text_pairs[0]] + extra_unique
                return {"reference_image_path_and_text_pairs": over_selected, "text_prompt": "Render anime frame."}

            pipeline.reference_image_selector.select_reference_images_and_generate_prompt = fake_select

            frames = await pipeline(
                script="Shot 1:\n[Thought] Kayden (Calm): \"Mada owaranai.\"",
                user_requirement="",
                style="anime",
                characters=chars,
                character_portraits_registry=registry,
                quiet=True,
                stop_after="frames",
            )
            self.assertEqual(len(frames), 1)
            self.assertTrue(any("PANEL ANCHOR 0" in desc for _, desc in seen_selector_pairs[0]))
            self.assertEqual(len(seen_img_refs[0]), 8)
            self.assertEqual(len(set(seen_img_refs[0])), 8)
            self.assertEqual(seen_vid_prompts, [])  # stop_after="frames" did not trigger video generation

            # Verify Inner Monologue formatting in generate_video_for_single_shot
            pipeline.frame_events = {0: {"first_frame": asyncio.Event()}}
            pipeline.frame_events[0]["first_frame"].set()
            await pipeline.generate_video_for_single_shot(shot)
            self.assertEqual(len(seen_vid_prompts), 1)
            self.assertIn("Delivery Mode: Off-screen inner monologue voiceover", seen_vid_prompts[0])


class Manga2VideoEndToEndAndAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_bind_storyboard_strips_refbeat_tags_and_builds_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene_0"
            p0 = Path(tmp) / "a0.jpg"
            p1 = Path(tmp) / "a1.jpg"
            _save_comic_panel(p0)
            _save_comic_panel(p1)
            beats = [
                MangaBeat(beat_id="ch01_b000", chapter=1, page_idx=0, action_desc="Kayden leaps onto rooftop.", anchor_image_path=str(p0)),
                MangaBeat(beat_id="ch01_b001", chapter=1, page_idx=1, action_desc="Jiwoo pets orange cat.", anchor_image_path=str(p1)),
            ]
            raw_storyboard = [
                ShotBriefDescription(idx=0, is_last=False, cam_idx=0, visual_desc="<Kayden> leaps onto rooftop [RefBeat: ch01_b000].", audio_desc="[Sound Effect] Thunder"),
                ShotBriefDescription(idx=1, is_last=True, cam_idx=1, visual_desc="<Jiwoo> smiles holding orange cat.", audio_desc='[Speaker] Jiwoo: "Kawaii!"'),
            ]
            clean, refs = bind_storyboard_to_panel_anchors(raw_storyboard, beats, scene_dir)
            self.assertNotIn("RefBeat", clean[0].visual_desc)
            self.assertEqual(refs[0][0][0], str(p0))
            self.assertEqual(refs[1][0][0], str(p1))
            saved_sb = json.loads((scene_dir / "storyboard.json").read_text(encoding="utf-8"))
            self.assertNotIn("RefBeat", json.dumps(saved_sb))

    async def test_vimax_manga2video_adapter_uploads_planning_and_selective_invalidation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index = SessionIndex(root)
            rec = index.create(project_name="eleceed-ep1")
            sid = rec["session_id"]
            session_dir = index.working_dir(sid)
            uploads_dir = session_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)

            # Create a real ch01.cbz inside session uploads/ + user character upload
            tmp_src = root / "_raw"
            _save_comic_panel(tmp_src / "p01.jpg")
            _save_comic_panel(tmp_src / "p02.jpg")
            with zipfile.ZipFile(uploads_dir / "eleceed_ch01.cbz", "w") as zf:
                zf.write(tmp_src / "p01.jpg", "p01.jpg")
                zf.write(tmp_src / "p02.jpg", "p02.jpg")
            _save_comic_panel(uploads_dir / "jiwoo.png", width=300, height=300)

            specs = build_vimax_adapter_specs(root, index)
            self.assertIn("vimax_manga2video", {s.name for s in specs})
            adapter = ViMaxAdapters(root, index)

            class FakeLLM(Runnable):
                def invoke(self, input, config=None, **kwargs):
                    raise NotImplementedError

                async def ainvoke(self, messages, config=None, **kwargs):
                    if hasattr(messages, "to_messages"):
                        messages = messages.to_messages()
                    first = messages[0]
                    if isinstance(first, tuple):
                        sys_text = str(first[1])
                    else:
                        sys_text = first.content if hasattr(first, "content") else str(first)
                    if "characters_observed" in sys_text:
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
                                        }
                                    ],
                                    "chunk_tail_summary": "",
                                }
                            )
                        )
                    if "storyboard" in sys_text and "ff_desc" not in sys_text and "camera_parent_items" not in sys_text:
                        return AIMessage(
                            content=json.dumps(
                                {
                                    "storyboard": [
                                        {
                                            "idx": 0,
                                            "is_last": True,
                                            "cam_idx": 0,
                                            "visual_desc": "<Jiwoo> looks out classroom window [RefBeat: ch01_b000].",
                                            "audio_desc": '[Speaker] Jiwoo (Soft): "Ame ga futte kita na." (Vietsub: Trời lại bắt đầu mưa rồi.)',
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

            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeLLM()):
                plan_res = await adapter.vimax_manga2video(
                    {
                        "session_id": sid,
                        "auto_render": False,
                    }
                )
            self.assertTrue(plan_res.ok, plan_res.content)
            payload = json.loads(plan_res.content)
            self.assertTrue(payload["ready_for_render"])
            self.assertEqual(payload["chapters"], [1])

            checklist = index.artifact_checklist(sid)
            self.assertTrue(checklist["manga2video/comic_manifest.json"])
            self.assertTrue(checklist["manga2video/episode_plan.json"])
            self.assertTrue(checklist["manga2video/scene_*/storyboard.json"])
            self.assertTrue(checklist["manga2video/scene_*/camera_tree.json"])
            self.assertTrue(checklist["manga2video/scene_*/shots/*/shot_description.json"])
            self.assertFalse(checklist["manga2video/final_video.mp4"])

            # Verify upload character match wired into character_seed_refs.json
            manga_dir = session_dir / "manga2video"
            seed_refs = json.loads((manga_dir / "character_seed_refs.json").read_text(encoding="utf-8"))
            self.assertEqual(seed_refs["Jiwoo"], [str(uploads_dir / "jiwoo.png")])

            # Seed fake shot media & final videos, verify selective invalidation works for manga2video
            shot0 = manga_dir / "scene_0" / "shots" / "0"
            (shot0 / "first_frame.png").write_bytes(b"ff")
            (shot0 / "video.mp4").write_bytes(b"mp4")
            (manga_dir / "scene_0" / "final_video.mp4").write_bytes(b"s0")
            (manga_dir / "final_video.mp4").write_bytes(b"ep")
            invalidate_render_artifacts(session_dir, scene_idx=0, shot_idxs=[0])
            self.assertFalse((shot0 / "video.mp4").exists())
            self.assertTrue((shot0 / "shot_description.json").exists())
            self.assertFalse((manga_dir / "scene_0" / "final_video.mp4").exists())
            self.assertFalse((manga_dir / "final_video.mp4").exists())

            # Test render_episode A->Z concatenation & merged episode SRT sidecar using real synthetic mp4 clips
            _make_synthetic_mp4(shot0 / "video.mp4", color="0x3366CC", duration=1.5)
            scene0_srt = manga_dir / "scene_0" / "final_video_vietsub.srt"
            scene0_srt.write_text("1\n00:00:00,200 --> 00:00:01,200\nTrời lại bắt đầu mưa rồi.\n", encoding="utf-8")
            # Also save portraits registry so render_episode goes straight to scene compose test
            portrait_file = manga_dir / "portrait_jiwoo.png"
            _save_comic_panel(portrait_file, width=100, height=100)
            (manga_dir / "character_portraits_registry.json").write_text(
                json.dumps({"Jiwoo": {"front": {"path": str(portrait_file), "description": "front"}}}),
                encoding="utf-8",
            )
            (shot0 / "first_frame.png").write_bytes(b"ff")
            (root / "configs").mkdir(exist_ok=True)
            (root / "configs" / "agent.local.yaml").write_text(
                "audio:\n  model_provider: native\nsubtitles:\n  enabled: true\n  language: vi\n  burn_in: false\n",
                encoding="utf-8",
            )

            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeLLM()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=object()), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=object()):
                render_res = await adapter.vimax_render_video({"session_id": sid})
            self.assertTrue(render_res.ok, render_res.content)
            self.assertTrue((manga_dir / "final_video.mp4").is_file())
            self.assertTrue((manga_dir / "final_video_vietsub.srt").is_file())
            self.assertIn("Trời lại bắt đầu mưa rồi.", (manga_dir / "final_video_vietsub.srt").read_text(encoding="utf-8"))

            # Verify re-planning with force_replan=False preserves exact shot_panel_refs.json on resume
            (manga_dir / "scene_0" / "shot_panel_refs.json").write_text(
                json.dumps({"0": [{"path": str(uploads_dir / "jiwoo.png"), "description": "LOCKED ANCHOR", "beat_id": "ch01_b000"}]}),
                encoding="utf-8",
            )
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeLLM()):
                plan_res_2 = await adapter.vimax_manga2video(
                    {"session_id": sid, "auto_render": False}
                )
            self.assertTrue(plan_res_2.ok, plan_res_2.content)
            preserved_refs = json.loads((manga_dir / "scene_0" / "shot_panel_refs.json").read_text(encoding="utf-8"))
            self.assertEqual(preserved_refs["0"][0]["description"], "LOCKED ANCHOR")

    def test_merge_scene_subtitles_aligns_cumulative_offsets(self):
        from pipelines.manga2video_pipeline import _merge_scene_subtitles

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s0 = root / "scene_0"
            s1 = root / "scene_1"
            s0.mkdir()
            s1.mkdir()
            (s0 / "final_video_vietsub.srt").write_text("1\n00:00:00,500 --> 00:00:02,000\nCâu một.\n", encoding="utf-8")
            (s1 / "final_video_vietsub.srt").write_text("1\n00:00:00,250 --> 00:00:01,500\nCâu hai.\n", encoding="utf-8")
            srt_out, ass_out = _merge_scene_subtitles(
                [s0, s1],
                [4.0, 3.0],
                root,
                suffix="vietsub",
                font="Arial",
                font_size=28,
            )
            self.assertIsNotNone(srt_out)
            self.assertIsNotNone(ass_out)
            merged_txt = Path(srt_out).read_text(encoding="utf-8")
            self.assertIn("00:00:00,500 --> 00:00:02,000", merged_txt)
            self.assertIn("00:00:04,250 --> 00:00:05,500", merged_txt)
            self.assertIn("Câu hai.", merged_txt)

    async def test_invisible_character_skipped_in_frame_generation(self):
        from pipelines.script2video_pipeline import Script2VideoPipeline
        from interfaces import Camera, CharacterInScene, ShotDescription, ImageOutput
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_path = root / "jiwoo_front.png"
            _save_comic_panel(p_path, width=100, height=100)

            chars = [
                CharacterInScene(idx=0, identifier_in_scene="Jiwoo", is_visible=True, static_features="teen boy"),
                CharacterInScene(idx=1, identifier_in_scene="Narrator", is_visible=False, static_features="Voice only"),
            ]
            shots = [
                ShotDescription(
                    idx=0,
                    is_last=False,
                    cam_idx=0,
                    visual_desc="Jiwoo in rain",
                    variation_type="small",
                    variation_reason="minor",
                    ff_desc="Jiwoo in rain",
                    ff_vis_char_idxs=[0, 1, 99],
                    lf_desc="Jiwoo in rain",
                    lf_vis_char_idxs=[0, 1],
                    motion_desc="Static camera.",
                    audio_desc="Rain.",
                ),
                ShotDescription(
                    idx=1,
                    is_last=True,
                    cam_idx=0,
                    visual_desc="Jiwoo looks up",
                    variation_type="small",
                    variation_reason="minor",
                    ff_desc="Jiwoo looks up",
                    ff_vis_char_idxs=[0, 1],
                    lf_desc="Jiwoo looks up",
                    lf_vis_char_idxs=[0, 1],
                    motion_desc="Tilt up.",
                    audio_desc="Narrator speaks.",
                ),
            ]
            cameras = [Camera(idx=0, active_shot_idxs=[0, 1])]
            seen_prompts = []

            class DummySelector:
                async def select_reference_images_and_generate_prompt(self, available_image_path_and_text_pairs, frame_description):
                    seen_prompts.append(len(available_image_path_and_text_pairs))
                    return {"reference_image_path_and_text_pairs": available_image_path_and_text_pairs[:2], "text_prompt": frame_description}

            class DummyImgGen:
                async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
                    return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (320, 180), (40, 60, 80)))

            pipe = Script2VideoPipeline(chat_model=object(), image_generator=DummyImgGen(), video_generator=object(), working_dir=str(root), num_image_candidates=1)
            pipe.reference_image_selector = DummySelector()
            pipe.frame_events = {0: {"first_frame": asyncio.Event()}, 1: {"first_frame": asyncio.Event()}}
            await pipe.generate_frames_for_single_camera(
                camera=cameras[0],
                shot_descriptions=shots,
                characters=chars,
                character_portraits_registry={"Jiwoo": {"front": {"path": str(p_path), "description": "Jiwoo front"}}},
                priority_shot_idxs=[],
            )
            self.assertTrue((root / "shots" / "0" / "first_frame.png").is_file())
            self.assertTrue((root / "shots" / "1" / "first_frame.png").is_file())
            self.assertEqual(seen_prompts[0], 1)

    async def test_reference_image_selector_caps_over_eight_indices(self):
        from agents.reference_image_selector import ReferenceImageSelector
        from langchain_core.runnables import Runnable

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = []
            for i in range(11):
                p = root / f"img_{i}.png"
                _save_comic_panel(p, width=120, height=120)
                pairs.append((str(p), f"Ref {i}"))

            call_img_counts = []

            class OverSelectLLM(Runnable):
                def invoke(self, input, config=None, **kwargs):
                    raise NotImplementedError

                async def ainvoke(self, input, config=None, **kwargs):
                    msgs = input.to_messages() if hasattr(input, "to_messages") else list(input)
                    human = msgs[-1].content
                    imgs = [part for part in human if isinstance(part, dict) and part.get("type") == "image_url"]
                    call_img_counts.append(len(imgs))
                    if len(imgs) == 0:
                        return json.dumps({"ref_image_indices": list(range(11)), "text_prompt": "step1"})
                    return json.dumps({"ref_image_indices": list(range(len(imgs))), "text_prompt": "step2"})

            selector = ReferenceImageSelector(chat_model=OverSelectLLM())
            out = await selector.select_reference_images_and_generate_prompt(pairs, "Target anime frame")
            # Step 2 multimodal call must receive at most 8 images (<= muse2api's 10-image limit)
            self.assertEqual(call_img_counts, [0, 8])
            self.assertEqual(len(out["reference_image_path_and_text_pairs"]), 8)

    def test_resolve_script_vietsubs_preserves_shot_indices(self):
        from pipelines.script2video_pipeline import _resolve_script_vietsubs
        from interfaces import ShotDescription

        raw_script = (
            "Shot 1:\n[Speaker] A: \"One.\"\n(Vietsub: Beat một.)\n\n"
            "Shot 2:\n[Speaker] B: \"Two.\"\n(Vietsub: Beat hai.)\n"
        )
        # Storyboard expanded 2 beats into 3 shots: shot 0 is silent establishing, shots 1 & 2 carry speech
        shots = [
            ShotDescription(
                idx=0, is_last=False, cam_idx=0, visual_desc="Wide alley",
                variation_type="small", variation_reason="minor",
                ff_desc="Wide", lf_desc="Wide", motion_desc="Static",
                audio_desc="[Sound Effect] Rain.",
            ),
            ShotDescription(
                idx=1, is_last=False, cam_idx=0, visual_desc="A speaks",
                variation_type="small", variation_reason="minor",
                ff_desc="A", lf_desc="A", motion_desc="Static",
                audio_desc='[Speaker] A: "One." (Vietsub: Câu một chuẩn.)',
            ),
            ShotDescription(
                idx=2, is_last=True, cam_idx=0, visual_desc="B speaks",
                variation_type="small", variation_reason="minor",
                ff_desc="B", lf_desc="B", motion_desc="Static",
                audio_desc='[Speaker] B: "Two." (Vietsub: Câu hai chuẩn.)',
            ),
        ]
        resolved = _resolve_script_vietsubs(shots, raw_script)
        self.assertEqual(resolved, {1: 'A: "Câu một chuẩn."', 2: 'B: "Câu hai chuẩn."'})

    def test_bind_storyboard_backfills_missing_vietsub_only_for_speech_shots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor0 = root / "b0.jpg"
            anchor1 = root / "b1.jpg"
            anchor2 = root / "b2.jpg"
            _save_comic_panel(anchor0, width=200, height=200)
            _save_comic_panel(anchor1, width=200, height=200)
            _save_comic_panel(anchor2, width=200, height=200)

            beats = [
                MangaBeat(
                    beat_id="ch01_b000",
                    chapter=1,
                    page_idx=0,
                    setting="Alley at night",
                    action_desc="Lightning streak falls from dark clouds",
                    anchor_image_path=str(anchor0),
                    dialogues=[],
                ),
                MangaBeat(
                    beat_id="ch01_b001",
                    chapter=1,
                    page_idx=1,
                    setting="Alley at night",
                    action_desc="Kayden kneels exhausted in rain",
                    anchor_image_path=str(anchor1),
                    dialogues=[
                        MangaDialogueLine(
                            speaker="Kayden (Human)",
                            tag="Speaker",
                            line="Kuh... look at the state I'm in.",
                            vietsub="Ực... nhìn bộ dạng thảm hại này của mình xem.",
                        )
                    ],
                ),
                MangaBeat(
                    beat_id="ch01_b002",
                    chapter=1,
                    page_idx=2,
                    setting="Alley at night",
                    action_desc="Kayden presses hand on wall",
                    anchor_image_path=str(anchor2),
                    dialogues=[
                        MangaDialogueLine(
                            speaker="Kayden (Human)",
                            tag="Thought",
                            line="I need to hide.",
                            vietsub="Mình phải trốn đi mới được.",
                        )
                    ],
                ),
            ]
            storyboard = [
                ShotBriefDescription(
                    idx=0,
                    is_last=False,
                    cam_idx=0,
                    visual_desc="[RefBeat: ch01_b000] Wide shot of lightning streak.",
                    audio_desc="[Sound Effect] CRACKLE, rain pattering.",
                ),
                ShotBriefDescription(
                    idx=1,
                    is_last=False,
                    cam_idx=1,
                    visual_desc="[RefBeat: ch01_b001] <Kayden (Human)> kneels and says: 'Kuh... look at the state I'm in.'",
                    audio_desc="[Speaker] Kayden (Human) (Exhausted): 'Kuh... look at the state I'm in.'",
                ),
                ShotBriefDescription(
                    idx=2,
                    is_last=True,
                    cam_idx=1,
                    visual_desc="[RefBeat: ch01_b002] <Kayden (Human)> thinks urgently.",
                    audio_desc="[Thought] Kayden (Human): 'I need to hide.' (Vietsub: Mình phải trốn đi mới được.)",
                ),
            ]

            clean_shots, refs = bind_storyboard_to_panel_anchors(storyboard, beats, root / "scene_0")
            # Shot 0 is silent SFX only -> must NOT inject Vietsub even if nearby beats have speech
            self.assertNotIn("Vietsub", clean_shots[0].audio_desc)
            # Shot 1 had [Speaker] but dropped Vietsub -> backfilled from beat dialogue
            self.assertIn("(Vietsub: Ực... nhìn bộ dạng thảm hại này của mình xem.)", clean_shots[1].audio_desc)
            # Shot 2 already had Vietsub -> preserved once, not duplicated
            self.assertEqual(clean_shots[2].audio_desc.count("Vietsub:"), 1)
            self.assertNotIn("[RefBeat", clean_shots[1].visual_desc)
            self.assertIn(1, refs)

    async def test_child_camera_with_panel_refs_skips_transition_video_and_attaches_parent_lighting_ref(self):
        import asyncio
        from pipelines.script2video_pipeline import Script2VideoPipeline
        from interfaces import Camera, CharacterInScene, ShotDescription, ImageOutput, VideoOutput
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_path = root / "jiwoo_front.png"
            panel0_path = root / "panel_0.jpg"
            panel1_path = root / "panel_1.jpg"
            _save_comic_panel(p_path, width=100, height=100)
            _save_comic_panel(panel0_path, width=200, height=150)
            _save_comic_panel(panel1_path, width=200, height=150)

            chars = [
                CharacterInScene(idx=0, identifier_in_scene="Jiwoo", is_visible=True, static_features="teen boy"),
            ]
            shots = [
                ShotDescription(
                    idx=0, is_last=False, cam_idx=0,
                    visual_desc="Jiwoo stands in alley",
                    variation_type="small", variation_reason="minor",
                    ff_desc="Jiwoo stands in alley", ff_vis_char_idxs=[0],
                    lf_desc="Jiwoo stands in alley", lf_vis_char_idxs=[0],
                    motion_desc="Static", audio_desc="Rain.",
                ),
                ShotDescription(
                    idx=1, is_last=True, cam_idx=1,
                    visual_desc="Jiwoo turns toward cat",
                    variation_type="small", variation_reason="minor",
                    ff_desc="Jiwoo turns toward cat", ff_vis_char_idxs=[0],
                    lf_desc="Jiwoo turns toward cat", lf_vis_char_idxs=[0],
                    motion_desc="Pan", audio_desc="Meow.",
                ),
            ]
            parent_ff = root / "shots" / "0" / "first_frame.png"
            parent_ff.parent.mkdir(parents=True, exist_ok=True)
            _save_comic_panel(parent_ff, width=320, height=180)

            captured_candidates = []
            transition_calls = []

            class DummySelector:
                async def select_reference_images_and_generate_prompt(self, available_image_path_and_text_pairs, frame_description):
                    captured_candidates.append(list(available_image_path_and_text_pairs))
                    return {"reference_image_path_and_text_pairs": available_image_path_and_text_pairs[:3], "text_prompt": frame_description}

            class DummyImgGen:
                async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
                    return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (320, 180), (40, 60, 80)))

            class SpyVideoGen:
                async def generate_single_video(self, *args, **kwargs):
                    transition_calls.append((args, kwargs))
                    return VideoOutput(fmt="bytes", ext="mp4", data=b"fake-mp4")

            # (a) Child camera WITH panel refs skips transition_video and includes parent first_frame as style/lighting continuity ref
            pipe_with_panel = Script2VideoPipeline(
                chat_model=object(),
                image_generator=DummyImgGen(),
                video_generator=SpyVideoGen(),
                working_dir=str(root),
                num_image_candidates=1,
            )
            pipe_with_panel.reference_image_selector = DummySelector()
            pipe_with_panel.frame_events = {
                0: {"first_frame": asyncio.Event()},
                1: {"first_frame": asyncio.Event()},
            }
            pipe_with_panel.frame_events[0]["first_frame"].set()

            child_cam = Camera(idx=1, active_shot_idxs=[1], parent_cam_idx=0, parent_shot_idx=0, missing_info=None)
            await pipe_with_panel.generate_frames_for_single_camera(
                camera=child_cam,
                shot_descriptions=shots,
                characters=chars,
                character_portraits_registry={"Jiwoo": {"front": {"path": str(p_path), "description": "Jiwoo front"}}},
                priority_shot_idxs=[],
                shot_panel_refs={
                    0: [(str(panel0_path), "Panel 0")],
                    1: [(str(panel1_path), "ORIGINAL COMIC PANEL COMPOSITION & POSE REFERENCE")],
                },
            )
            self.assertEqual(len(transition_calls), 0)
            self.assertTrue((root / "shots" / "1" / "first_frame.png").is_file())
            self.assertTrue(
                any(
                    path == str(parent_ff) and "SCENE LIGHTING & ANIME STYLE CONTINUITY REFERENCE" in desc
                    for path, desc in captured_candidates[0]
                )
            )

            # (b) Child camera WITHOUT panel refs still calls transition_video as before
            root_no_panel = Path(tmp) / "no_panel"
            (root_no_panel / "shots" / "0").mkdir(parents=True, exist_ok=True)
            parent_ff_2 = root_no_panel / "shots" / "0" / "first_frame.png"
            _save_comic_panel(parent_ff_2, width=320, height=180)
            extracted_cam = Image.new("RGB", (320, 180), (90, 90, 90))

            pipe_no_panel = Script2VideoPipeline(
                chat_model=object(),
                image_generator=DummyImgGen(),
                video_generator=SpyVideoGen(),
                working_dir=str(root_no_panel),
                num_image_candidates=1,
            )
            pipe_no_panel.reference_image_selector = DummySelector()
            pipe_no_panel.frame_events = {
                0: {"first_frame": asyncio.Event()},
                1: {"first_frame": asyncio.Event()},
            }
            pipe_no_panel.frame_events[0]["first_frame"].set()
            pipe_no_panel.camera_image_generator.get_new_camera_image = lambda path: extracted_cam

            await pipe_no_panel.generate_frames_for_single_camera(
                camera=child_cam,
                shot_descriptions=shots,
                characters=chars,
                character_portraits_registry={"Jiwoo": {"front": {"path": str(p_path), "description": "Jiwoo front"}}},
                priority_shot_idxs=[],
                shot_panel_refs=None,
            )
            self.assertEqual(len(transition_calls), 1)
            self.assertTrue((root_no_panel / "shots" / "1" / "first_frame.png").is_file())


class FullSeriesAdaptationTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_sibling_dedup_nested_volumes_and_corrupt_image_resilience(self):
        from tools.manga_source import group_chapters_for_episodes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "eleceed_series"
            # Chapter 1 normalized + sibling raw (raw MUST be deduplicated) + corrupt/empty files inside ch01_pages
            _save_comic_panel(src / "ch01_pages" / "p01.jpg", width=650, height=900)
            _save_comic_panel(src / "ch01_pages" / "p02.jpg", width=650, height=900)
            (src / "ch01_pages" / "empty.jpg").write_bytes(b"")
            (src / "ch01_pages" / "broken.png").write_bytes(b"not-a-valid-image-payload")
            _save_comic_panel(src / "ch01_raw" / "raw01.jpg", width=650, height=900)
            # Standalone ch02_raw without non-raw sibling MUST be kept
            _save_comic_panel(src / "ch02_raw" / "p01.jpg", width=650, height=900)
            # Nested Volume/Chapter layout MUST be discovered at depth 2
            _save_comic_panel(src / "Vol 01" / "ch03" / "p01.jpg", width=650, height=900)

            manifest = await ingest_manga_sources(src, root / "work")
            self.assertEqual([c.chapter for c in manifest.chapters], [1, 2, 3])
            self.assertEqual(len(manifest.chapters[0].pages), 2)

            episodes = group_chapters_for_episodes(manifest, chapters_per_episode=2)
            self.assertEqual(len(episodes), 2)
            self.assertEqual([c.chapter for c in episodes[0].chapters], [1, 2])
            self.assertEqual([c.chapter for c in episodes[1].chapters], [3])

            # Directory containing ONLY unreadable images raises an explicit error
            bad_src = root / "all_bad" / "ch01"
            bad_src.mkdir(parents=True)
            (bad_src / "empty.jpg").write_bytes(b"")
            with self.assertRaises(ValueError):
                await ingest_manga_sources(bad_src.parent, root / "bad_work")

    async def test_webtoon_series_list_expansion_and_default_fetch_retry(self):
        from tools.manga_source import _default_fetch_url, parse_webtoon_episode_list_urls

        list_html = """
        <html><head><title>Eleceed</title></head>
        <body>
          <h1>Eleceed Official</h1>
          <ul id="_listUl">
            <li data-episode-no="2"><a href="/en/action/eleceed/episode-2/viewer?title_no=1571&episode_no=2">Ep 2</a></li>
            <li data-episode-no="1"><a href="/en/action/eleceed/episode-1/viewer?title_no=1571&episode_no=1">Ep 1</a></li>
            <li data-episode-no="3"><a href="/en/action/eleceed/episode-3/viewer?title_no=1571&episode_no=3">Ep 3</a></li>
          </ul>
        </body></html>
        """
        title, items = parse_webtoon_episode_list_urls(
            list_html,
            base_url="https://www.webtoons.com/en/action/eleceed/list?title_no=1571",
        )
        self.assertEqual(title, "Eleceed Official")
        self.assertEqual([ep for ep, _ in items], [1, 2, 3])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p_buf = root / "strip.jpg"
            _save_comic_panel(p_buf, width=700, height=600)
            img_bytes = p_buf.read_bytes()
            fetched_urls = []

            async def fake_fetcher(url: str, *, referer: str = ""):
                fetched_urls.append(url)
                if "/list" in url:
                    return list_html.encode("utf-8")
                if "/viewer" in url:
                    return (
                        b'<html><body><div id="_imageList">'
                        b'<img class="_images" data-url="https://webtoon-phinf.pstatic.net/panel_01.jpg" />'
                        b"</div></body></html>"
                    )
                return img_bytes

            manifest = await ingest_manga_sources(
                "https://www.webtoons.com/en/action/eleceed/list?title_no=1571",
                root / "wt_work",
                chapter_range="1-2",
                webtoon_fetcher=fake_fetcher,
            )
            self.assertEqual([c.chapter for c in manifest.chapters], [1, 2])
            self.assertTrue(any("episode_no=1" in u for u in fetched_urls))
            self.assertTrue(any("episode_no=2" in u for u in fetched_urls))
            self.assertFalse(any("episode_no=3" in u for u in fetched_urls))

        # Test _default_fetch_url transient HTTP 503 retry vs non-retryable HTTP 404
        class FakeResp:
            def __init__(self, status: int, body: bytes):
                self.status = status
                self._body = body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def read(self) -> bytes:
                return self._body

        statuses = [503, 200]
        calls = []

        class FakeSession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            def get(self, url: str):
                st = statuses.pop(0)
                calls.append(st)
                return FakeResp(st, b"ok-payload" if st == 200 else b"retry")

        with patch("tools.manga_source.aiohttp.ClientSession", FakeSession):
            data = await _default_fetch_url("https://example.com/p.jpg", retry_delay_sec=0.001)
        self.assertEqual(data, b"ok-payload")
        self.assertEqual(calls, [503, 200])

        statuses = [404, 200]
        calls.clear()
        with patch("tools.manga_source.aiohttp.ClientSession", FakeSession):
            with self.assertRaises(RuntimeError):
                await _default_fetch_url("https://example.com/missing.jpg", retry_delay_sec=0.001)
        self.assertEqual(calls, [404])

    async def test_reader_cross_episode_second_form_never_collides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p1 = root / "ch01" / "p01.jpg"
            _save_comic_panel(p1)
            manifest = await ingest_manga_sources(root / "ch01", root / "work")
            reader = MangaReader(object())
            series_chars = [
                CharacterInScene(
                    idx=0,
                    identifier_in_scene="Kayden",
                    is_visible=True,
                    static_features="Wounded black-haired awakener man",
                )
            ]
            cat_chunk = MangaChunkExtraction(
                chapter=2,
                start_page_idx=0,
                end_page_idx=0,
                characters_observed=[
                    MangaCharacterObservation(
                        name="Kayden",
                        species_or_form="cat (four-legged quadruped)",
                        appearance_summary="Fat orange tabby cat walking on all four paws",
                        outfit_summary="Orange tabby fur",
                        page_idx=0,
                    )
                ],
                beats=[
                    MangaBeat(
                        beat_id="ch02_b000",
                        chapter=2,
                        page_idx=0,
                        action_desc="Orange tabby cat glares angrily.",
                        characters_present=["Kayden"],
                    )
                ],
            )
            ep2_chars, _, _ = reader.build_character_roster_and_seeds(
                manifest=manifest,
                chunks=[cat_chunk],
                working_dir=root / "work_ep2",
                series_characters=series_chars,
            )
            self.assertEqual(len(ep2_chars), 1)
            self.assertNotEqual(ep2_chars[0].identifier_in_scene, "Kayden")
            self.assertIn("cat", ep2_chars[0].identifier_in_scene.lower())

    async def test_full_series_continuity_shared_portraits_resume_adapter_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "eleceed"
            _save_comic_panel(src / "ch01_pages" / "p01.jpg")
            _save_comic_panel(src / "ch02_pages" / "p01.jpg")
            _save_comic_panel(src / "ch01_raw" / "dup.jpg")  # Verify series skips raw sibling too!

            synth_clip = root / "synth.mp4"
            _make_synthetic_mp4(synth_clip, color="0x224488", duration=1.5)
            synth_bytes = synth_clip.read_bytes()

            class FakeSeriesLLM(Runnable):
                def invoke(self, input, config=None, **kwargs):
                    raise NotImplementedError

                async def ainvoke(self, messages, config=None, **kwargs):
                    if hasattr(messages, "to_messages"):
                        messages = messages.to_messages()
                    first = messages[0]
                    sys_text = str(first[1]) if isinstance(first, tuple) else str(getattr(first, "content", first))
                    human_content = messages[-1].content if hasattr(messages[-1], "content") else ""
                    if isinstance(human_content, list):
                        human_text = " ".join(str(x.get("text", "")) for x in human_content if isinstance(x, dict))
                    else:
                        human_text = str(human_content)

                    if "characters_observed" in sys_text:
                        if "Chapter: 1" in human_text:
                            payload = {
                                "characters_observed": [
                                    {
                                        "name": "Seo Jiwoo",
                                        "species_or_form": "human",
                                        "appearance_summary": "Teen boy with messy brown hair",
                                        "outfit_summary": "Dark school jacket",
                                        "page_idx": 0,
                                        "crop_box_norm": [100, 100, 700, 700],
                                    },
                                    {
                                        "name": "Kayden",
                                        "species_or_form": "cat (four-legged quadruped)",
                                        "appearance_summary": "Fat orange tabby cat on all four paws",
                                        "outfit_summary": "Orange tabby fur",
                                        "page_idx": 0,
                                        "crop_box_norm": [200, 200, 800, 800],
                                    },
                                ],
                                "beats": [
                                    {
                                        "page_idx": 0,
                                        "is_scene_break": True,
                                        "setting": "Alley",
                                        "action_desc": "Seo Jiwoo picks up orange cat Kayden.",
                                        "characters_present": ["Seo Jiwoo", "Kayden (Fat Orange Tabby Cat)"],
                                        "dialogues": [
                                            {
                                                "speaker": "Seo Jiwoo",
                                                "line": "Daijoubu ka, neko-chan?",
                                                "line_type": "spoken",
                                                "emotion": "Gentle",
                                                "vietsub": "Mày có sao không, mèo con?",
                                            }
                                        ],
                                        "sfx": [],
                                        "crop_box_norm": [50, 50, 900, 900],
                                    }
                                ],
                                "chunk_tail_summary": "",
                            }
                        else:
                            # Ch 2 uses shortened name "Jiwoo" and introduces new character "Curtin"
                            payload = {
                                "characters_observed": [
                                    {
                                        "name": "Jiwoo",
                                        "species_or_form": "human",
                                        "appearance_summary": "Teen boy smiling",
                                        "outfit_summary": "Casual shirt",
                                        "page_idx": 0,
                                        "crop_box_norm": [100, 100, 700, 700],
                                    },
                                    {
                                        "name": "Curtin",
                                        "species_or_form": "human",
                                        "appearance_summary": "Tall pale blonde healer man",
                                        "outfit_summary": "White coat",
                                        "page_idx": 0,
                                        "crop_box_norm": [250, 250, 850, 850],
                                    },
                                ],
                                "beats": [
                                    {
                                        "page_idx": 0,
                                        "is_scene_break": True,
                                        "setting": "Living room",
                                        "action_desc": "Jiwoo greets Curtin.",
                                        "characters_present": ["Jiwoo", "Curtin"],
                                        "dialogues": [
                                            {
                                                "speaker": "Curtin",
                                                "line": "Kore ga shin no chikara da.",
                                                "line_type": "spoken",
                                                "emotion": "Calm",
                                                "vietsub": "Đây mới là sức mạnh thật sự.",
                                            }
                                        ],
                                        "sfx": [],
                                        "crop_box_norm": [50, 50, 900, 900],
                                    }
                                ],
                                "chunk_tail_summary": "",
                            }
                        return AIMessage(content=json.dumps(payload, ensure_ascii=False))

                    if "storyboard" in sys_text and "ff_desc" not in sys_text and "camera_parent_items" not in sys_text:
                        beat_ref = "ch01_b000" if "ch01_b000" in human_text else "ch02_b000"
                        spk = "Seo Jiwoo" if beat_ref == "ch01_b000" else "Curtin"
                        line = "Daijoubu ka?" if beat_ref == "ch01_b000" else "Kore ga chikara da."
                        sub = "Có sao không?" if beat_ref == "ch01_b000" else "Đây là sức mạnh."
                        return AIMessage(
                            content=json.dumps(
                                {
                                    "storyboard": [
                                        {
                                            "idx": 0,
                                            "is_last": True,
                                            "cam_idx": 0,
                                            "visual_desc": f"<{spk}> anime shot [RefBeat: {beat_ref}].",
                                            "audio_desc": f'[Speaker] {spk}: "{line}" (Vietsub: {sub})',
                                        }
                                    ]
                                },
                                ensure_ascii=False,
                            )
                        )
                    if "ff_desc" in sys_text:
                        return AIMessage(
                            content=json.dumps(
                                {
                                    "ff_desc": "Character stands in frame",
                                    "ff_vis_char_idxs": [0],
                                    "lf_desc": "Character reacts subtly",
                                    "lf_vis_char_idxs": [0],
                                    "motion_desc": "Subtle breathing motion",
                                    "variation_type": "small",
                                    "variation_reason": "expression",
                                }
                            )
                        )
                    if "ref_image_indices" in sys_text:
                        return AIMessage(content=json.dumps({"ref_image_indices": [0], "text_prompt": "Render anime shot."}))
                    if "best_image_index" in sys_text or "BestImageResponse" in sys_text:
                        return AIMessage(content=json.dumps({"best_image_index": 0, "reason": "Matches anime style and panel pose."}))
                    return AIMessage(content=json.dumps({"camera_parent_items": [None]}))

            generated_prompts = []

            class SpyImgGen:
                async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
                    generated_prompts.append(prompt)
                    return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (320, 180), (45, 70, 110)))

            video_calls = []

            class SpyVidGen:
                async def generate_single_video(self, *args, **kwargs):
                    video_calls.append(kwargs.get("prompt", ""))
                    return VideoOutput(fmt="bytes", ext="mp4", data=synth_bytes)

            (root / "configs").mkdir(exist_ok=True)
            (root / "configs" / "agent.local.yaml").write_text(
                "audio:\n  model_provider: native\nsubtitles:\n  enabled: true\n  language: vi\n  burn_in: false\n",
                encoding="utf-8",
            )

            index = SessionIndex(root)
            adapter = ViMaxAdapters(root, index)

            # 1. Plan & verify series via ViMaxAdapters (without render first)
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeSeriesLLM()):
                plan_res = await adapter.vimax_manga2video(
                    {
                        "sources": str(src),
                        "chapters_per_episode": 1,
                        "auto_render": False,
                    }
                )
            self.assertTrue(plan_res.ok, plan_res.content)
            plan_payload = json.loads(plan_res.content)
            sid = plan_payload["session_id"]
            self.assertEqual(plan_payload["render_mode"], "manga2video_series")
            self.assertEqual(plan_payload["episode_count"], 2)
            self.assertTrue(plan_payload["ready_for_render"])

            manga_dir = index.working_dir(sid) / "manga2video"
            checklist = index.artifact_checklist(sid)
            self.assertTrue(checklist["manga2video/series_manifest.json"])
            self.assertTrue(checklist["manga2video/episodes/ep_*/episode_plan.json"])
            self.assertTrue(checklist["manga2video/episodes/ep_*/scene_*/storyboard.json"])
            self.assertFalse(checklist["manga2video/episodes/ep_*/final_video.mp4"])

            # Verify Ep 2 resolved shortened "Jiwoo" back to canonical Ep 1 "Seo Jiwoo"
            series_chars = json.loads((manga_dir / "series_characters.json").read_text(encoding="utf-8"))
            series_names = [c["identifier_in_scene"] for c in series_chars]
            self.assertIn("Seo Jiwoo", series_names)
            self.assertIn("Curtin", series_names)
            self.assertEqual(series_names.count("Seo Jiwoo"), 1)
            self.assertNotIn("Jiwoo", series_names)

            ep2_chars = json.loads((manga_dir / "episodes" / "ep_002" / "characters.json").read_text(encoding="utf-8"))
            self.assertIn("Seo Jiwoo", [c["identifier_in_scene"] for c in ep2_chars])

            # 2. Render Ep 1 ONLY via vimax_render_video(episode_idx=1) -> generates Ep 1 portraits only
            portrait_img_gen = SpyImgGen()
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeSeriesLLM()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=portrait_img_gen), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=SpyVidGen()):
                r1 = await adapter.vimax_render_video({"session_id": sid, "episode_idx": 1})
            self.assertTrue(r1.ok, r1.content)
            ep1_vid = manga_dir / "episodes" / "ep_001" / "final_video.mp4"
            ep2_vid = manga_dir / "episodes" / "ep_002" / "final_video.mp4"
            self.assertTrue(ep1_vid.is_file())
            self.assertFalse(ep2_vid.is_file())
            self.assertTrue((manga_dir / "episodes" / "ep_001" / "final_video_vietsub.srt").is_file())
            registry = json.loads((manga_dir / "character_portraits_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(set(registry.keys()), {"Seo Jiwoo", "Kayden"})

            # 3. Render whole series via vimax_render_video() -> MUST skip completed Ep 1 and ONLY generate portrait for new "Curtin"
            ep1_mtime_before = ep1_vid.stat().st_mtime_ns
            video_calls.clear()
            generated_prompts.clear()
            with patch("agent_runtime.vimax_adapters._build_chat_model", return_value=FakeSeriesLLM()), \
                 patch("agent_runtime.vimax_adapters._build_image_generator", return_value=portrait_img_gen), \
                 patch("agent_runtime.vimax_adapters._build_video_generator", return_value=SpyVidGen()):
                r_all = await adapter.vimax_render_video({"session_id": sid})
            self.assertTrue(r_all.ok, r_all.content)
            r_all_payload = json.loads(r_all.content)
            self.assertEqual(r_all_payload["completed_episodes"], 2)
            self.assertTrue(ep2_vid.is_file())
            self.assertEqual(ep1_vid.stat().st_mtime_ns, ep1_mtime_before)
            self.assertEqual(len(video_calls), 1)  # Only Ep 2's 1 shot rendered!
            registry_after = json.loads((manga_dir / "character_portraits_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(set(registry_after.keys()), {"Seo Jiwoo", "Kayden", "Curtin"})
            # Neither episode duplicated character portraits inside ep_XXX/
            self.assertFalse((manga_dir / "episodes" / "ep_001" / "character_portraits").exists())
            self.assertFalse((manga_dir / "episodes" / "ep_002" / "character_portraits").exists())

            # 4. Selective invalidation by episode_idx leaves Ep 1 intact while clearing Ep 2 final_video
            removed = invalidate_render_artifacts(index.working_dir(sid), episode_idx=2, scene_idx=0, shot_idxs=[0])
            self.assertTrue(any("ep_002" in r for r in removed))
            self.assertTrue(ep1_vid.is_file())
            self.assertFalse(ep2_vid.is_file())

            # 5. CLI runner test (--plan-only against series)
            from main_manga2video import run_manga2video_cli

            cli_dir = root / "cli_out"
            with patch("main_manga2video._build_chat_model", return_value=FakeSeriesLLM()):
                cli_summary = await run_manga2video_cli(
                    [
                        "--sources",
                        str(src),
                        "--working-dir",
                        str(cli_dir),
                        "--chapters-per-episode",
                        "1",
                        "--plan-only",
                    ]
                )
            self.assertEqual(cli_summary["mode"], "series")
            self.assertEqual(cli_summary["episode_count"], 2)
            self.assertTrue((cli_dir / "series_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
