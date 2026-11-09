from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from utils.image_selection import image_candidate_count_from_config

DEFAULT_LLM_MODEL = "gpt-5.5"
DEFAULT_LLM_MODEL_PROVIDER = "openai"
DEFAULT_LLM_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_IMAGE_MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_IMAGE_BASE_URL = "https://yunwu.ai"
DEFAULT_VIDEO_MODEL = "veo3.1-fast"
DEFAULT_VIDEO_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_MODEL_PROVIDER = "openai"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@lru_cache(maxsize=8)
def _load_agent_config_by_stat(resolved_path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    if mtime_ns < 0:
        return {}
    path = Path(resolved_path)
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid configs/agent.local.yaml: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("configs/agent.local.yaml must be a YAML mapping")
    return payload


def load_agent_config(workspace_root: str | Path = ".") -> dict[str, Any]:
    path = Path(workspace_root).resolve() / "configs" / "agent.local.yaml"
    try:
        st = path.stat()
        mtime_ns, size = int(st.st_mtime_ns), int(st.st_size)
    except OSError:
        mtime_ns, size = -1, -1
    return _load_agent_config_by_stat(str(path), mtime_ns, size)


load_agent_config.cache_clear = _load_agent_config_by_stat.cache_clear  # type: ignore[attr-defined]


def config_value(section: str, key: str, env_names: list[str], default: str = "", workspace_root: str | Path = ".") -> str:
    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value
    section_payload = load_agent_config(workspace_root).get(section, {})
    if isinstance(section_payload, dict):
        value = section_payload.get(key)
        if isinstance(value, str) and value:
            return value
    return default


def llm_model(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "model", ["VIMAX_LLM_MODEL"], DEFAULT_LLM_MODEL, workspace_root)


def llm_model_provider(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "model_provider", ["VIMAX_LLM_MODEL_PROVIDER"], DEFAULT_LLM_MODEL_PROVIDER, workspace_root)


def llm_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "base_url", ["VIMAX_LLM_BASE_URL"], DEFAULT_LLM_BASE_URL, workspace_root)


def llm_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("llm", "api_key", ["VIMAX_LLM_API_KEY", "VIMAX_API_KEY"], "", workspace_root)


def image_model(workspace_root: str | Path = ".") -> str:
    return config_value("image", "model", ["VIMAX_IMAGE_MODEL"], DEFAULT_IMAGE_MODEL, workspace_root)


def image_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("image", "base_url", ["VIMAX_IMAGE_BASE_URL"], DEFAULT_IMAGE_BASE_URL, workspace_root)


def image_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("image", "api_key", ["VIMAX_IMAGE_API_KEY", "VIMAX_LLM_API_KEY", "VIMAX_API_KEY"], llm_api_key(workspace_root), workspace_root)


def image_num_candidates(workspace_root: str | Path = ".") -> int:
    return image_candidate_count_from_config(load_agent_config(workspace_root))



def embedding_model(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "model", ["VIMAX_EMBEDDING_MODEL"], DEFAULT_EMBEDDING_MODEL, workspace_root)


def embedding_model_provider(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "model_provider", ["VIMAX_EMBEDDING_MODEL_PROVIDER"], DEFAULT_EMBEDDING_MODEL_PROVIDER, workspace_root)


def embedding_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "base_url", ["VIMAX_EMBEDDING_BASE_URL"], "", workspace_root)


def embedding_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("embedding", "api_key", ["VIMAX_EMBEDDING_API_KEY"], "", workspace_root)


def reranker_model(workspace_root: str | Path = ".") -> str:
    return config_value("reranker", "model", ["VIMAX_RERANKER_MODEL"], DEFAULT_RERANKER_MODEL, workspace_root)


def reranker_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("reranker", "base_url", ["VIMAX_RERANKER_BASE_URL"], "", workspace_root)


def reranker_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("reranker", "api_key", ["VIMAX_RERANKER_API_KEY"], "", workspace_root)


def video_model(workspace_root: str | Path = ".") -> str:
    return config_value("video", "model", ["VIMAX_VIDEO_MODEL"], DEFAULT_VIDEO_MODEL, workspace_root)


def video_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("video", "base_url", ["VIMAX_VIDEO_BASE_URL"], DEFAULT_VIDEO_BASE_URL, workspace_root)


def video_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("video", "api_key", ["VIMAX_VIDEO_API_KEY", "VIMAX_LLM_API_KEY", "VIMAX_API_KEY"], llm_api_key(workspace_root), workspace_root)


def api_provider_from_base_url(base_url: str) -> str:
    normalized = base_url.strip().lower()
    if "openrouter.ai" in normalized:
        return "openrouter"
    if "yunwu.ai" in normalized:
        return "yunwu"
    return ""


def image_provider(workspace_root: str | Path = ".") -> str:
    """Infer the image API relay/provider from image.base_url or image.model_provider."""
    explicit = config_value("image", "model_provider", ["VIMAX_IMAGE_PROVIDER", "VIMAX_IMAGE_MODEL_PROVIDER"], "", workspace_root).strip().lower()
    if explicit:
        return explicit
    return api_provider_from_base_url(image_base_url(workspace_root))


def video_provider(workspace_root: str | Path = ".") -> str:
    """Infer the video API relay/provider from video.base_url or video.model_provider."""
    explicit = config_value("video", "model_provider", ["VIMAX_VIDEO_PROVIDER", "VIMAX_VIDEO_MODEL_PROVIDER"], "", workspace_root).strip().lower()
    if explicit:
        return explicit
    return api_provider_from_base_url(video_base_url(workspace_root))

DEFAULT_AUDIO_MODEL_PROVIDER = "elevenlabs"
DEFAULT_AUDIO_MODEL = "eleven_multilingual_v2"
DEFAULT_AUDIO_BASE_URL = "https://api.elevenlabs.io/v1"
DEFAULT_AUDIO_VOICE = "21m00Tcm4TlvDq8ikWAM"

def audio_model_provider(workspace_root: str | Path = ".") -> str:
    return config_value("audio", "model_provider", ["VIMAX_AUDIO_PROVIDER"], DEFAULT_AUDIO_MODEL_PROVIDER, workspace_root)

def audio_model(workspace_root: str | Path = ".") -> str:
    return config_value("audio", "model", ["VIMAX_AUDIO_MODEL"], DEFAULT_AUDIO_MODEL, workspace_root)

def audio_base_url(workspace_root: str | Path = ".") -> str:
    return config_value("audio", "base_url", ["VIMAX_AUDIO_BASE_URL"], DEFAULT_AUDIO_BASE_URL, workspace_root)

def audio_api_key(workspace_root: str | Path = ".") -> str:
    return config_value("audio", "api_key", ["VIMAX_AUDIO_API_KEY", "ELEVENLABS_API_KEY"], "", workspace_root)

def audio_voice(workspace_root: str | Path = ".") -> str:
    return config_value("audio", "voice", ["VIMAX_AUDIO_VOICE"], DEFAULT_AUDIO_VOICE, workspace_root)

def audio_bgm_file(workspace_root: str | Path = ".") -> str:
    return config_value("audio", "bgm_file", ["VIMAX_BGM_FILE"], "", workspace_root)

def audio_bgm_volume(workspace_root: str | Path = ".") -> float:
    val = config_value("audio", "bgm_volume", ["VIMAX_BGM_VOLUME"], "0.15", workspace_root)
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.15

def audio_character_voices(workspace_root: str | Path = ".") -> dict[str, str]:
    section = load_agent_config(workspace_root).get("audio", {})
    if isinstance(section, dict):
        cv = section.get("character_voices", {})
        if isinstance(cv, dict):
            return {str(k).strip(): str(v).strip() for k, v in cv.items()}
    return {}

DEFAULT_SUBTITLES_ENABLED = False
DEFAULT_SUBTITLES_LANGUAGE = "vi"
DEFAULT_SUBTITLES_BURN_IN = True
DEFAULT_SUBTITLES_FONT = "Arial"
DEFAULT_SUBTITLES_FONT_SIZE = 34


def _parse_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def subtitles_enabled(workspace_root: str | Path = ".") -> bool:
    raw = os.environ.get("VIMAX_SUBTITLES_ENABLED")
    if raw is not None and raw.strip() != "":
        return _parse_bool(raw, DEFAULT_SUBTITLES_ENABLED)
    section = load_agent_config(workspace_root).get("subtitles", {})
    if isinstance(section, dict) and "enabled" in section:
        return _parse_bool(section.get("enabled"), DEFAULT_SUBTITLES_ENABLED)
    return DEFAULT_SUBTITLES_ENABLED


def subtitles_language(workspace_root: str | Path = ".") -> str:
    return config_value("subtitles", "language", ["VIMAX_SUBTITLES_LANGUAGE"], DEFAULT_SUBTITLES_LANGUAGE, workspace_root)


def subtitles_burn_in(workspace_root: str | Path = ".") -> bool:
    raw = os.environ.get("VIMAX_SUBTITLES_BURN_IN")
    if raw is not None and raw.strip() != "":
        return _parse_bool(raw, DEFAULT_SUBTITLES_BURN_IN)
    section = load_agent_config(workspace_root).get("subtitles", {})
    if isinstance(section, dict) and "burn_in" in section:
        return _parse_bool(section.get("burn_in"), DEFAULT_SUBTITLES_BURN_IN)
    return DEFAULT_SUBTITLES_BURN_IN


def subtitles_font(workspace_root: str | Path = ".") -> str:
    return config_value("subtitles", "font", ["VIMAX_SUBTITLES_FONT"], DEFAULT_SUBTITLES_FONT, workspace_root)


def subtitles_font_size(workspace_root: str | Path = ".") -> int:
    raw = os.environ.get("VIMAX_SUBTITLES_FONT_SIZE")
    if raw is not None and raw.strip():
        try:
            return max(12, int(raw))
        except ValueError:
            return DEFAULT_SUBTITLES_FONT_SIZE
    section = load_agent_config(workspace_root).get("subtitles", {})
    if isinstance(section, dict) and section.get("font_size") is not None:
        try:
            return max(12, int(section.get("font_size")))
        except (TypeError, ValueError):
            return DEFAULT_SUBTITLES_FONT_SIZE
    return DEFAULT_SUBTITLES_FONT_SIZE
