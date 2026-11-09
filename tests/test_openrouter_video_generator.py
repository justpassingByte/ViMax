import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.video_generator_openrouter_api import VideoGeneratorOpenRouterAPI
from interfaces.video_output import VideoOutput


class OpenRouterVideoGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_duration_is_eight_seconds(self):
        captured = {}

        async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
            captured["payload"] = payload
            return 200, {"id": "job-1", "polling_url": "/videos/job-1", "status": "queued"}

        async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
            return 200, {"status": "completed", "unsigned_urls": ["https://cdn.example/out.mp4"]}

        async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
            return 200, b"video"

        async def fake_sleep(seconds):
            return None

        generator = VideoGeneratorOpenRouterAPI(api_key="test-key", model="google/veo-3.1-lite")
        with patch.dict(os.environ, {}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
             patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
             patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            output = await generator.generate_single_video(prompt="hello")

        self.assertIsInstance(output, VideoOutput)
        self.assertEqual(captured["payload"]["duration"], 8)
        self.assertEqual(captured["payload"]["model"], "google/veo-3.1-lite")

    async def test_seedance_fast_uses_supported_openrouter_payload(self):
        captured = {}

        async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
            captured["payload"] = payload
            return 200, {"id": "job-2", "polling_url": "/videos/job-2", "status": "queued"}

        async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
            return 200, {"status": "completed", "unsigned_urls": ["https://cdn.example/seedance.mp4"]}

        async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
            return 200, b"seedance-video"

        async def fake_sleep(seconds):
            return None

        with tempfile.TemporaryDirectory() as tmp:
            first_frame = Path(tmp) / "first.png"
            last_frame = Path(tmp) / "last.png"
            Image.new("RGB", (16, 9), "blue").save(first_frame)
            Image.new("RGB", (16, 9), "red").save(last_frame)
            generator = VideoGeneratorOpenRouterAPI(api_key="test-key", model="bytedance/seedance-2.0-fast")
            with patch.dict(os.environ, {}, clear=True), \
                 patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
                 patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
                 patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
                 patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
                output = await generator.generate_single_video(
                    prompt="a cinematic walk",
                    reference_image_paths=[str(first_frame), str(last_frame)],
                )

        payload = captured["payload"]
        self.assertIsInstance(output, VideoOutput)
        self.assertEqual(payload["model"], "bytedance/seedance-2.0-fast")
        self.assertEqual(payload["duration"], 8)
        self.assertEqual(payload["resolution"], "720p")
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertTrue(payload["generate_audio"])
        self.assertEqual(
            [frame["frame_type"] for frame in payload["frame_images"]],
            ["first_frame", "last_frame"],
        )


    async def test_muse2api_gateway_receives_images_and_resolves_relative_media_url(self):
        captured = {}

        async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
            captured["post_url"] = url
            captured["payload"] = payload
            return 200, {"id": "vid-1", "status": "queued"}

        async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
            captured["poll_url"] = url
            return 200, {"status": "completed", "url": "/v1/media/vid-1.mp4"}

        async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
            captured["download_url"] = url
            return 200, b"muse-mp4"

        async def fake_sleep(seconds):
            return None

        with tempfile.TemporaryDirectory() as tmp:
            first_frame = Path(tmp) / "first.png"
            Image.new("RGB", (16, 9), "blue").save(first_frame)
            generator = VideoGeneratorOpenRouterAPI(api_key="m2a_test", model="muse-video", base_url="http://127.0.0.1:18610/v1")
            with patch.dict(os.environ, {}, clear=True), \
                 patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
                 patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
                 patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
                 patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
                output = await generator.generate_single_video(prompt="cat walks", reference_image_paths=[str(first_frame)])

        self.assertEqual(output.data, b"muse-mp4")
        self.assertEqual(captured["post_url"], "http://127.0.0.1:18610/v1/videos")
        self.assertEqual(captured["poll_url"], "http://127.0.0.1:18610/v1/videos/vid-1")
        self.assertEqual(captured["download_url"], "http://127.0.0.1:18610/v1/media/vid-1.mp4")
        self.assertTrue(captured["payload"]["images"][0].startswith("data:image/png;base64,"))


    async def test_transient_failed_job_retries_once_and_releases_gate(self):
        import asyncio
        from utils.async_gate import AsyncConcurrencyGate

        create_calls = {"n": 0}
        gate = AsyncConcurrencyGate(1)
        events = []

        async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
            create_calls["n"] += 1
            return 200, {"id": f"vid-{create_calls['n']}", "status": "queued"}

        async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
            if url.endswith("vid-1"):
                return 200, {"status": "failed", "error": "Still sending timeout"}
            return 200, {"status": "completed", "url": "https://cdn.example/ok.mp4"}

        async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
            return 200, b"recovered-mp4"

        async def fake_sleep(seconds):
            return None

        generator = VideoGeneratorOpenRouterAPI(
            api_key="m2a_test",
            model="muse-video",
            base_url="http://127.0.0.1:18610/v1",
            concurrency_gate=gate,
        )
        with patch.dict(os.environ, {"VIMAX_VIDEO_MAX_ATTEMPTS": "2", "VIMAX_VIDEO_RETRY_DELAY_SECONDS": "0"}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
             patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
             patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            output = await generator.generate_single_video(
                prompt="cat jumps",
                progress=lambda stage, message, meta: events.append(stage),
            )

        self.assertEqual(create_calls["n"], 2)
        self.assertEqual(output.data, b"recovered-mp4")
        self.assertIn("video_retry", events)
        async with asyncio.timeout(0.2):
            async with gate:
                pass

    async def test_auth_error_and_refusal_fail_fast_without_retry(self):
        calls = {"n": 0}

        async def fake_post_401(url, *, headers, payload, timeout, hard_timeout_seconds):
            calls["n"] += 1
            return 401, {"error": "invalid api key"}

        async def fake_sleep(seconds):
            return None

        generator = VideoGeneratorOpenRouterAPI(api_key="bad", model="muse-video", base_url="http://127.0.0.1:18610/v1")
        with patch.dict(os.environ, {"VIMAX_VIDEO_MAX_ATTEMPTS": "3"}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_401), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            with self.assertRaises(RuntimeError):
                await generator.generate_single_video(prompt="cat")
        self.assertEqual(calls["n"], 1)

        calls["n"] = 0

        async def fake_post_ok(url, *, headers, payload, timeout, hard_timeout_seconds):
            calls["n"] += 1
            return 200, {"id": "vid-refuse", "status": "queued"}

        async def fake_poll_refusal(url, *, headers, timeout, hard_timeout_seconds):
            return 200, {"status": "failed", "error": "模型未生成媒体，仅返回文本: cannot generate"}

        with patch.dict(os.environ, {"VIMAX_VIDEO_MAX_ATTEMPTS": "3"}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_ok), \
             patch("tools.video_generator_openrouter_api._get_json", fake_poll_refusal), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            with self.assertRaises(RuntimeError):
                await generator.generate_single_video(prompt="cat")
        self.assertEqual(calls["n"], 1)

    async def test_suppress_on_screen_subtitles_appends_visual_constraint(self):
        captured = {}

        async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
            captured["payload"] = payload
            return 200, {"id": "job-sub", "polling_url": "/videos/job-sub", "status": "queued"}

        async def fake_get_json(url, *, headers, timeout, hard_timeout_seconds):
            return 200, {"status": "completed", "unsigned_urls": ["https://cdn.example/out.mp4"]}

        async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
            return 200, b"video"

        async def fake_sleep(seconds):
            return None

        generator = VideoGeneratorOpenRouterAPI(api_key="test-key", model="muse-video")
        with patch.dict(os.environ, {}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
             patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
             patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            await generator.generate_single_video(prompt="hello", suppress_on_screen_subtitles=True)

        self.assertIn("clean and textless", captured["payload"]["prompt"])
        self.assertNotIn("suppress_on_screen_subtitles", captured["payload"])

        muse_gen = VideoGeneratorOpenRouterAPI(api_key="test-key", model="muse-video", base_url="http://127.0.0.1:18610/v1")
        with patch.dict(os.environ, {}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
             patch("tools.video_generator_openrouter_api._get_json", fake_get_json), \
             patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            await muse_gen.generate_single_video(prompt="hello", suppress_on_screen_subtitles=True)

        self.assertTrue(captured["payload"]["suppress_on_screen_subtitles"])
        self.assertIn("textless", captured["payload"]["prompt"])


    async def test_transient_koda_tool_path_error_retries_while_region_refusal_fails_fast(self):
        create_calls = {"n": 0}

        async def fake_post_json(url, *, headers, payload, timeout, hard_timeout_seconds):
            create_calls["n"] += 1
            return 200, {"id": f"vid-koda-{create_calls['n']}", "status": "queued"}

        async def fake_get_json_tool_err(url, *, headers, timeout, hard_timeout_seconds):
            if url.endswith("vid-koda-1"):
                return 200, {
                    "status": "failed",
                    "error": "模型未生成媒体，仅返回文本: 生成工具报错找不到我传入的文件路径",
                }
            return 200, {"status": "completed", "url": "https://cdn.example/fixed.mp4"}

        async def fake_get_bytes(url, *, headers, timeout, hard_timeout_seconds):
            return 200, b"fixed-mp4"

        async def fake_sleep(seconds):
            return None

        generator = VideoGeneratorOpenRouterAPI(
            api_key="m2a_test",
            model="muse-video",
            base_url="http://127.0.0.1:18610/v1",
        )
        with patch.dict(os.environ, {"VIMAX_VIDEO_RETRY_DELAY_SECONDS": "0"}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
             patch("tools.video_generator_openrouter_api._get_json", fake_get_json_tool_err), \
             patch("tools.video_generator_openrouter_api._get_bytes", fake_get_bytes), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            output = await generator.generate_single_video(prompt="rain scene")

        self.assertEqual(create_calls["n"], 2)
        self.assertEqual(output.data, b"fixed-mp4")

        create_calls["n"] = 0

        async def fake_get_json_region(url, *, headers, timeout, hard_timeout_seconds):
            return 200, {
                "status": "failed",
                "error": "模型未生成媒体，仅返回文本: 该国家/地区当前不支持视频生成",
            }

        with patch.dict(os.environ, {"VIMAX_VIDEO_RETRY_DELAY_SECONDS": "0"}, clear=True), \
             patch("tools.video_generator_openrouter_api._post_json", fake_post_json), \
             patch("tools.video_generator_openrouter_api._get_json", fake_get_json_region), \
             patch("tools.video_generator_openrouter_api.asyncio.sleep", fake_sleep):
            with self.assertRaises(RuntimeError):
                await generator.generate_single_video(prompt="rain scene")

        self.assertEqual(create_calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
