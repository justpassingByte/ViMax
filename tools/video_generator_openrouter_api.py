import asyncio
import json
import logging
import os
from typing import Any, List
from urllib.parse import urljoin

import aiohttp

from interfaces.video_output import VideoOutput
from utils.image import image_path_to_b64


class OpenRouterVideoAPIError(RuntimeError):
    def __init__(self, status_code: int | None, message: str, *, retryable: bool | None = None) -> None:
        self.status_code = status_code
        self.explicit_retryable = retryable
        super().__init__(message)


_NON_RETRYABLE_VIDEO_MARKERS = (
    "key limit",
    "quota",
    "额度不足",
    "insufficient",
    "invalid api key",
    "unauthorized",
    "forbidden",
    "payment required",
    "模型未生成媒体",
    "仅返回文本",
    "已被 muse.ai 登出",
    "refusal",
    "refused",
    "content filter",
    "policy",
    "prohibited",
    "nsfw",
)


def _is_retryable_video_error(exc: BaseException) -> bool:
    if isinstance(exc, (aiohttp.ClientError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, OpenRouterVideoAPIError):
        if exc.explicit_retryable is not None:
            return exc.explicit_retryable
        if exc.status_code is not None:
            return exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500
    if isinstance(exc, ValueError):
        return False
    text = str(exc).lower()
    if any(marker in text for marker in _NON_RETRYABLE_VIDEO_MARKERS):
        return False
    return isinstance(exc, OpenRouterVideoAPIError)


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.environ.get(name, str(default))))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _emit_progress(progress, stage: str, message: str, metadata: dict | None = None) -> None:
    if progress is not None:
        progress(stage, message, metadata or {})


class VideoGeneratorOpenRouterAPI:
    def __init__(
        self,
        api_key: str,
        model: str = "google/veo-3.1-lite",
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str = "",
        app_title: str = "ViMax",
        concurrency_gate: Any = None,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.http_referer = http_referer
        self.app_title = app_title
        self.concurrency_gate = concurrency_gate

    async def generate_single_video(
        self,
        prompt: str = "",
        reference_image_paths: List[str] = [],
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> VideoOutput:
        progress = kwargs.get("progress")
        request_timeout_seconds = _env_float("VIMAX_VIDEO_REQUEST_TIMEOUT_SECONDS", 60.0)
        query_timeout_seconds = _env_float("VIMAX_VIDEO_QUERY_TIMEOUT_SECONDS", 600.0)
        poll_interval_seconds = _env_float("VIMAX_VIDEO_POLL_INTERVAL_SECONDS", 10.0)
        duration = _env_int("VIMAX_OPENROUTER_VIDEO_DURATION", 8)
        resolution = os.environ.get("VIMAX_OPENROUTER_VIDEO_RESOLUTION", "720p")
        generate_audio = _env_bool("VIMAX_OPENROUTER_GENERATE_AUDIO", True)
        suppress_subtitles = kwargs.get("suppress_on_screen_subtitles")
        if suppress_subtitles is None:
            suppress_subtitles = _env_bool("VIMAX_VIDEO_SUPPRESS_SUBTITLES", False)
        is_openrouter = "openrouter.ai" in self.base_url.lower()
        effective_prompt = prompt or ""
        if suppress_subtitles:
            constraint = (
                "Keep the entire video frame clean and textless (pure borderless animation visuals only, zero on-screen text overlays; render any spoken dialogue strictly as audio track only)."
                if is_openrouter
                else "Keep the visual video frames pure, clean, and textless (borderless anime visuals only; all spoken lines and narration belong strictly to the Japanese audio soundtrack)."
            )
            if "textless" not in effective_prompt.lower() and "zero on-screen text" not in effective_prompt.lower():
                effective_prompt = (
                    f"{effective_prompt.rstrip()}\nVisual direction: {constraint}"
                    if effective_prompt.strip()
                    else constraint
                )

        payload = {
            "model": self.model,
            "prompt": effective_prompt,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
            "resolution": resolution,
            "generate_audio": generate_audio,
        }
        if suppress_subtitles and not is_openrouter:
            payload["suppress_on_screen_subtitles"] = True
        frame_images = _frame_images(reference_image_paths)
        if frame_images:
            payload["frame_images"] = frame_images
            if "openrouter.ai" not in self.base_url.lower():
                payload["images"] = [image_path_to_b64(path, mime=True) for path in reference_image_paths]

        async def _run_job() -> VideoOutput:
            headers = self._headers()
            timeout = aiohttp.ClientTimeout(total=request_timeout_seconds)
            _emit_progress(progress, "video_create", f"Creating OpenRouter video generation task with {self.model}", {"model": self.model, "duration": duration, "resolution": resolution, "frame_count": len(frame_images)})

            create_status, create_payload = await _post_json(
                f"{self.base_url}/videos",
                headers=headers,
                payload=payload,
                timeout=timeout,
                hard_timeout_seconds=request_timeout_seconds,
            )
            if create_status >= 400:
                raise OpenRouterVideoAPIError(
                    create_status,
                    f"OpenRouter video create failed with HTTP {create_status}: {create_payload}",
                )
            job_id = create_payload.get("id")
            polling_url = create_payload.get("polling_url")
            if not job_id:
                raise OpenRouterVideoAPIError(
                    None,
                    f"Video create response missing id: {create_payload}",
                    retryable=False,
                )
            if not polling_url:
                polling_url = f"/videos/{job_id}"
            _emit_progress(progress, "video_task_created", "OpenRouter video generation task created", {"model": self.model, "job_id": job_id, "status": create_payload.get("status")})

            poll_url = _absolute_url(self.base_url, polling_url)
            deadline = asyncio.get_running_loop().time() + query_timeout_seconds if query_timeout_seconds > 0 else None
            last_status = create_payload.get("status")
            last_payload = create_payload
            while deadline is None or asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(poll_interval_seconds)
                poll_status, poll_payload = await _get_json(
                    poll_url,
                    headers=headers,
                    timeout=timeout,
                    hard_timeout_seconds=request_timeout_seconds,
                )
                if poll_status >= 400:
                    raise OpenRouterVideoAPIError(
                        poll_status,
                        f"OpenRouter video poll failed with HTTP {poll_status}: {poll_payload}",
                    )
                last_payload = poll_payload
                status = poll_payload.get("status")
                last_status = status
                _emit_progress(progress, "video_status", f"OpenRouter video generation status: {status}", {"model": self.model, "job_id": job_id, "status": status})

                if status in {"completed", "succeeded", "success"}:
                    urls = poll_payload.get("unsigned_urls") or []
                    if not urls:
                        job_obj = poll_payload.get("job") or {}
                        out_url = poll_payload.get("url") or job_obj.get("outputUrl") or job_obj.get("outputPath")
                        if out_url:
                            urls = [out_url]
                    if urls:
                        content_url = urls[0]
                        if content_url.startswith("/"):
                            if "zerakeyai.com" in self.base_url:
                                content_url = f"https://media.zerakeyai.com{content_url}"
                            elif content_url.startswith("/v1/") and self.base_url.rstrip("/").endswith("/v1"):
                                content_url = urljoin(self.base_url, content_url)
                            else:
                                content_url = _absolute_url(self.base_url, content_url)
                    else:
                        content_url = f"{self.base_url}/videos/{job_id}/content?index=0"
                    _emit_progress(progress, "video_download_start", "Downloading OpenRouter video output", {"model": self.model, "job_id": job_id})
                    download_status, data = await _get_bytes(
                        content_url,
                        headers=headers if _needs_authorization(content_url) else {},
                        timeout=timeout,
                        hard_timeout_seconds=request_timeout_seconds,
                    )
                    if download_status >= 400:
                        raise OpenRouterVideoAPIError(
                            download_status,
                            f"OpenRouter video content download failed with HTTP {download_status}: {data[:500]!r}",
                        )
                    _emit_progress(progress, "video_completed", "OpenRouter video generation completed and downloaded", {"model": self.model, "job_id": job_id})
                    return VideoOutput(fmt="bytes", ext="mp4", data=data)
                if status in {"failed", "cancelled", "expired"}:
                    err_detail = poll_payload.get("error") or poll_payload
                    err_lower = str(err_detail).lower()
                    retryable = (
                        status == "failed"
                        and not any(marker in err_lower for marker in _NON_RETRYABLE_VIDEO_MARKERS)
                    )
                    raise OpenRouterVideoAPIError(
                        None,
                        f"OpenRouter video generation {status} for job {job_id}: {err_detail}",
                        retryable=retryable,
                    )

            raise OpenRouterVideoAPIError(
                None,
                f"OpenRouter video generation timed out after {query_timeout_seconds:g}s for job {job_id}; last_status={last_status}; last_payload={last_payload}",
                retryable=False,
            )

        max_attempts = max(1, _env_int("VIMAX_VIDEO_MAX_ATTEMPTS", 2))
        retry_delay_seconds = _env_float("VIMAX_VIDEO_RETRY_DELAY_SECONDS", 2.0)

        for attempt in range(1, max_attempts + 1):
            try:
                if self.concurrency_gate is not None:
                    async with self.concurrency_gate:
                        return await _run_job()
                return await _run_job()
            except Exception as exc:
                if attempt >= max_attempts or not _is_retryable_video_error(exc):
                    raise
                logging.warning(
                    "Retrying video generation (attempt %s/%s) after transient failure: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )
                _emit_progress(
                    progress,
                    "video_retry",
                    f"Retrying video generation after transient error (attempt {attempt + 1}/{max_attempts})",
                    {"model": self.model, "attempt": attempt + 1, "max_attempts": max_attempts, "error": str(exc)},
                )
                if retry_delay_seconds > 0:
                    await asyncio.sleep(retry_delay_seconds)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-OpenRouter-Title"] = self.app_title
        return headers


def _frame_images(reference_image_paths: List[str]) -> list[dict]:
    if len(reference_image_paths) > 2:
        raise ValueError("OpenRouter video generation supports at most first and last frame images")
    frame_types = ["first_frame", "last_frame"]
    return [
        {
            "type": "image_url",
            "image_url": {"url": image_path_to_b64(path, mime=True)},
            "frame_type": frame_types[index],
        }
        for index, path in enumerate(reference_image_paths)
    ]


def _absolute_url(base_url: str, url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return urljoin(f"{base_url.rstrip('/')}/", url.lstrip("/"))


def _needs_authorization(url: str) -> bool:
    return url.startswith("https://openrouter.ai/api/")


def _decode_json_body(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"message": text}
    return parsed if isinstance(parsed, dict) else {"data": parsed}


async def _post_json(url: str, *, headers: dict[str, str], payload: dict, timeout: aiohttp.ClientTimeout, hard_timeout_seconds: float) -> tuple[int, dict]:
    async def request() -> tuple[int, dict]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as response:
                return response.status, _decode_json_body(await response.text())

    return await asyncio.wait_for(request(), timeout=hard_timeout_seconds + 5)


async def _get_json(url: str, *, headers: dict[str, str], timeout: aiohttp.ClientTimeout, hard_timeout_seconds: float) -> tuple[int, dict]:
    async def request() -> tuple[int, dict]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                return response.status, _decode_json_body(await response.text())

    return await asyncio.wait_for(request(), timeout=hard_timeout_seconds + 5)


async def _get_bytes(url: str, *, headers: dict[str, str], timeout: aiohttp.ClientTimeout, hard_timeout_seconds: float) -> tuple[int, bytes]:
    async def request() -> tuple[int, bytes]:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                return response.status, await response.read()

    return await asyncio.wait_for(request(), timeout=hard_timeout_seconds + 5)
