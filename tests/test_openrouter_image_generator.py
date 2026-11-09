import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

from PIL import Image

from agent_runtime.vimax_adapters import _build_image_generator, _build_video_generator
from tools.image_generator_openrouter_api import (
    ImageGeneratorOpenRouterAPI,
    OpenRouterImageAPIError,
    _is_retryable_image_error,
)


def _encoded_png(size: tuple[int, int] = (16, 9)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size, "blue").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class OpenRouterImageGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_generates_image_with_dedicated_images_api(self):
        captured = {}

        async def fake_post(url, *, headers, payload, timeout):
            captured.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return 200, {"data": [{"b64_json": _encoded_png(), "media_type": "image/png"}]}

        progress = []
        generator = ImageGeneratorOpenRouterAPI(api_key="secret", model="openai/gpt-image-2")
        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            result = await generator.generate_single_image(
                "a cinematic beach",
                aspect_ratio="16:9",
                progress=lambda stage, message, metadata: progress.append((stage, message, metadata)),
            )

        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/images")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(captured["payload"]["model"], "openai/gpt-image-2")
        self.assertNotIn("aspect_ratio", captured["payload"])
        self.assertIn("landscape image", captured["payload"]["prompt"])
        self.assertEqual(result.data.size, (16, 9))
        self.assertEqual(result.ext, "png")
        self.assertEqual([item[0] for item in progress], ["image_generation", "image_completed"])

    async def test_reference_images_use_data_urls(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.png"
            Image.new("RGB", (16, 9), "red").save(reference_path)
            post = AsyncMock(return_value=(200, {"data": [{"b64_json": _encoded_png()}]}))
            generator = ImageGeneratorOpenRouterAPI(api_key="secret")
            with patch("tools.image_generator_openrouter_api._post_json", post):
                await generator.generate_single_image("edit this", [str(reference_path)])

        payload = post.await_args.kwargs["payload"]
        reference_url = payload["input_references"][0]["image_url"]["url"]
        self.assertTrue(reference_url.startswith("data:image/png;base64,"))

    async def test_non_retryable_client_error_is_not_repeated(self):
        post = AsyncMock(return_value=(400, {"error": {"message": "bad request"}}))
        generator = ImageGeneratorOpenRouterAPI(api_key="secret")
        with patch("tools.image_generator_openrouter_api._post_json", post):
            with self.assertRaises(OpenRouterImageAPIError):
                await generator.generate_single_image("bad request")
        self.assertEqual(post.await_count, 1)

    def test_retry_policy_is_bounded_to_transient_errors_and_portrait_outputs(self):
        self.assertTrue(_is_retryable_image_error(OpenRouterImageAPIError(429, {})))
        self.assertTrue(_is_retryable_image_error(OpenRouterImageAPIError(500, {})))
        self.assertTrue(_is_retryable_image_error(ValueError("Generated image is portrait-oriented (9x16); retrying for a landscape frame")))
        self.assertFalse(_is_retryable_image_error(OpenRouterImageAPIError(401, {})))
        self.assertFalse(_is_retryable_image_error(ValueError("invalid reference image")))

    def test_agent_factory_selects_openrouter_from_image_base_url(self):
        with patch("agent_runtime.vimax_adapters.image_api_key", return_value="secret"), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="openai/gpt-image-2"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="https://openrouter.ai/api/v1"):
            generator = _build_image_generator()
        self.assertIsInstance(generator, ImageGeneratorOpenRouterAPI)
        self.assertEqual(generator.model, "openai/gpt-image-2")


    def test_agent_factory_selects_openrouter_for_local_muse2api(self):
        with patch("agent_runtime.vimax_adapters.image_api_key", return_value="m2a_test"), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="muse-image"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="http://127.0.0.1:18610/v1"), \
             patch("agent_runtime.vimax_adapters.image_provider", return_value=""):
            generator = _build_image_generator()
        self.assertIsInstance(generator, ImageGeneratorOpenRouterAPI)
        self.assertEqual(generator.model, "muse-image")

    async def test_non_openrouter_gateway_receives_images_and_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.png"
            Image.new("RGB", (16, 9), "red").save(reference_path)
            post = AsyncMock(return_value=(200, {"data": [{"b64_json": _encoded_png()}]}))
            generator = ImageGeneratorOpenRouterAPI(api_key="m2a_test", model="muse-image", base_url="http://127.0.0.1:18610/v1")
            with patch("tools.image_generator_openrouter_api._post_json", post):
                await generator.generate_single_image("draw a cat", [str(reference_path)], aspect_ratio="16:9")

        payload = post.await_args.kwargs["payload"]
        self.assertEqual(payload["aspect_ratio"], "16:9")
        self.assertTrue(payload["images"][0].startswith("data:image/png;base64,"))

    async def test_muse_image_and_video_generators_share_serial_concurrency_gate(self):
        import asyncio

        with patch("agent_runtime.vimax_adapters.image_api_key", return_value="m2a_test"), \
             patch("agent_runtime.vimax_adapters.image_model", return_value="muse-image"), \
             patch("agent_runtime.vimax_adapters.image_base_url", return_value="http://127.0.0.1:18610/v1"), \
             patch("agent_runtime.vimax_adapters.image_provider", return_value=""), \
             patch("agent_runtime.vimax_adapters.video_api_key", return_value="m2a_test"), \
             patch("agent_runtime.vimax_adapters.video_model", return_value="muse-video"), \
             patch("agent_runtime.vimax_adapters.video_base_url", return_value="http://127.0.0.1:18610/v1"), \
             patch("agent_runtime.vimax_adapters.video_provider", return_value="muse2api"):
            img_gen = _build_image_generator()
            vid_gen = _build_video_generator()

        self.assertIsNotNone(img_gen.concurrency_gate)
        self.assertIs(img_gen.concurrency_gate, vid_gen.concurrency_gate)

        active = 0
        max_active = 0

        async def fake_post(url, *, headers, payload, timeout):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.03)
            active -= 1
            return 200, {"data": [{"b64_json": _encoded_png(), "media_type": "image/png"}]}

        with patch("tools.image_generator_openrouter_api._post_json", fake_post):
            await asyncio.gather(
                img_gen.generate_single_image("one"),
                img_gen.generate_single_image("two"),
            )

        self.assertEqual(max_active, 1)

    async def test_url_image_download_sends_auth_only_same_origin_and_raises_on_http_error(self):
        captured_headers = []

        async def fake_post_same_origin(url, *, headers, payload, timeout):
            return 200, {"data": [{"url": "/v1/media/generated.png"}]}

        async def fake_download_ok(url, *, headers, timeout):
            captured_headers.append((url, dict(headers)))
            buffer = BytesIO()
            Image.new("RGB", (32, 18), "green").save(buffer, format="PNG")
            return 200, buffer.getvalue()

        generator = ImageGeneratorOpenRouterAPI(
            api_key="secret-token",
            model="muse-image",
            base_url="http://127.0.0.1:18610/v1",
        )
        with patch("tools.image_generator_openrouter_api._post_json", fake_post_same_origin), \
             patch("tools.image_generator_openrouter_api._download_image_bytes", fake_download_ok):
            out = await generator.generate_single_image("cat")

        self.assertEqual(out.data.size, (32, 18))
        self.assertEqual(captured_headers[0][0], "http://127.0.0.1:18610/v1/media/generated.png")
        self.assertEqual(captured_headers[0][1].get("Authorization"), "Bearer secret-token")

        async def fake_post_cross_origin(url, *, headers, payload, timeout):
            return 200, {"data": [{"url": "https://cdn.other.example/img.png"}]}

        with patch("tools.image_generator_openrouter_api._post_json", fake_post_cross_origin), \
             patch("tools.image_generator_openrouter_api._download_image_bytes", fake_download_ok):
            await generator.generate_single_image("cat")

        self.assertEqual(captured_headers[1][0], "https://cdn.other.example/img.png")
        self.assertNotIn("Authorization", captured_headers[1][1])

        async def fake_download_404(url, *, headers, timeout):
            return 404, b"not found"

        with patch("tools.image_generator_openrouter_api._post_json", fake_post_cross_origin), \
             patch("tools.image_generator_openrouter_api._download_image_bytes", fake_download_404):
            with self.assertRaises(OpenRouterImageAPIError):
                await generator.generate_single_image("cat")


if __name__ == "__main__":
    unittest.main()
