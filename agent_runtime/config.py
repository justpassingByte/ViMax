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


DEFAULT_MANGA_VISION_ENABLED = False
DEFAULT_MANGA_VISION_MAX_CONTEXT_CHARS = 2400
DEFAULT_MANGA_VISION_QA_MAX_FRAME_PAIRS = 4
DEFAULT_MANGA_VISION_AUTO_RERENDER_CRITICAL = True
DEFAULT_MANGA_VISION_MAX_RERENDER_CYCLES = 2
DEFAULT_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER = "auto"
DEFAULT_MANGA_VISION_IDENTITY_DEDUP_THRESHOLD = 0.82


def manga_vision_qa_max_frame_pairs(workspace_root: str | Path = ".") -> int:
    raw = os.environ.get("VIMAX_MANGA_VISION_QA_MAX_FRAME_PAIRS")
    if raw is not None and raw.strip():
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    section = load_agent_config(workspace_root).get("manga_vision", {})
    if isinstance(section, dict) and section.get("qa_max_frame_pairs") is not None:
        try:
            return max(1, int(section.get("qa_max_frame_pairs")))
        except (TypeError, ValueError):
            pass
    return DEFAULT_MANGA_VISION_QA_MAX_FRAME_PAIRS
    """Embedder used by the report-only duplicate-identity detector.

    - ``auto`` (default): use the real OpenAI-compatible provider from the
      ``embedding`` section when configured/reachable, else fall back to the
      deterministic offline mock.
    - ``mock``: always use the offline mock (no network).
    - ``off``: skip identity dedup entirely.
    """
    raw = os.environ.get("VIMAX_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER")
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        return value if value in {"auto", "mock", "off"} else DEFAULT_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER
    section = load_agent_config(workspace_root).get("manga_vision", {})
    if isinstance(section, dict):
        value = str(section.get("identity_dedup_embedder", "")).strip().lower()
        if value in {"auto", "mock", "off"}:
            return value
    return DEFAULT_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER


def manga_vision_auto_rerender_critical(workspace_root: str | Path = ".") -> bool:
    raw = os.environ.get("VIMAX_MANGA_VISION_AUTO_RERENDER_CRITICAL")
    if raw is not None and raw.strip() != "":
        return _parse_bool(raw, DEFAULT_MANGA_VISION_AUTO_RERENDER_CRITICAL)
    section = load_agent_config(workspace_root).get("manga_vision", {})
    if isinstance(section, dict) and "auto_rerender_critical" in section:
        return _parse_bool(section.get("auto_rerender_critical"), DEFAULT_MANGA_VISION_AUTO_RERENDER_CRITICAL)
    return DEFAULT_MANGA_VISION_AUTO_RERENDER_CRITICAL


def manga_vision_max_rerender_cycles(workspace_root: str | Path = ".") -> int:
    raw = os.environ.get("VIMAX_MANGA_VISION_MAX_RERENDER_CYCLES")
    if raw is not None and raw.strip():
        try:
            return max(1, min(5, int(raw)))
        except ValueError:
            pass
    section = load_agent_config(workspace_root).get("manga_vision", {})
    if isinstance(section, dict) and section.get("max_rerender_cycles") is not None:
        try:
            return max(1, min(5, int(section.get("max_rerender_cycles"))))
        except (TypeError, ValueError):
            pass
    return DEFAULT_MANGA_VISION_MAX_RERENDER_CYCLES


def manga_vision_identity_dedup_embedder(workspace_root: str | Path = ".") -> str:
    """Embedder used by the report-only duplicate-identity detector."""
    raw = os.environ.get("VIMAX_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER")
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        return value if value in {"auto", "mock", "off"} else DEFAULT_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER
    section = load_agent_config(workspace_root).get("manga_vision", {})
    if isinstance(section, dict):
        value = str(section.get("identity_dedup_embedder", "")).strip().lower()
        if value in {"auto", "mock", "off"}:
            return value
    return DEFAULT_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER


def manga_vision_identity_dedup_threshold(workspace_root: str | Path = ".") -> float:
    raw = os.environ.get("VIMAX_MANGA_VISION_IDENTITY_DEDUP_THRESHOLD")
    if raw is not None and raw.strip():
        try:
            return min(1.0, max(0.0, float(raw)))
        except ValueError:
            pass
    section = load_agent_config(workspace_root).get("manga_vision", {})
    if isinstance(section, dict) and section.get("identity_dedup_threshold") is not None:
        try:
            return min(1.0, max(0.0, float(section.get("identity_dedup_threshold"))))
        except (TypeError, ValueError):
            pass
    return DEFAULT_MANGA_VISION_IDENTITY_DEDUP_THRESHOLD


def manga_vision_enabled(workspace_root: str | Path = ".") -> bool:
    """Feature flag for the Manga Vision & Cinematic Director layer.

    Disabled by default so existing pipeline behavior is fully preserved;
    enable via ``manga_vision.enabled: true`` in configs/agent.local.yaml or
    the ``VIMAX_MANGA_VISION_ENABLED`` environment variable.
    """
    raw = os.environ.get("VIMAX_MANGA_VISION_ENABLED")
    if raw is not None and raw.strip() != "":
        return _parse_bool(raw, DEFAULT_MANGA_VISION_ENABLED)
    section = load_agent_config(workspace_root).get("manga_vision", {})
    if isinstance(section, dict) and "enabled" in section:
        return _parse_bool(section.get("enabled"), DEFAULT_MANGA_VISION_ENABLED)
    return DEFAULT_MANGA_VISION_ENABLED


def manga_vision_max_context_chars(workspace_root: str | Path = ".") -> int:
    raw = os.environ.get("VIMAX_MANGA_VISION_MAX_CONTEXT_CHARS")
    if raw is not None and raw.strip():
        try:
            return max(400, int(raw))
        except ValueError:
            pass
    section = load_agent_config(workspace_root).get("manga_vision", {})
    if isinstance(section, dict) and section.get("max_context_chars") is not None:
        try:
            return max(400, int(section.get("max_context_chars")))
        except (TypeError, ValueError):
            pass
    return DEFAULT_MANGA_VISION_MAX_CONTEXT_CHARS


# ---------------------------------------------------------------------------
# Incremental-quality feature flags (``quality:`` section).
#
# Each flag defaults to the *upgraded* behaviour so a fresh run uses the new
# pipeline; set the flag to the legacy value in configs/agent.local.yaml (or
# via the matching VIMAX_QUALITY_* env var) to fall back to the old behaviour.
# Accessors follow the same env -> config-section -> default precedence as the
# manga_vision_* helpers above.
# ---------------------------------------------------------------------------

DEFAULT_QUALITY_CANON_GATE = True
DEFAULT_QUALITY_GENERIC_LINTER = True
DEFAULT_QUALITY_SERIES_SLUG = ""
DEFAULT_QUALITY_REF_POLICY = "profile"            # legacy | all_characters | profile
DEFAULT_QUALITY_MAX_REFERENCE_IMAGES = 4
DEFAULT_QUALITY_EMBEDDINGS = "dinov2"             # off | dinov2
DEFAULT_QUALITY_TEXT_EMBEDDINGS = "e5"            # off | e5
DEFAULT_QUALITY_SCENE_SPLIT_THRESHOLD = 0.80
DEFAULT_QUALITY_PROFILES = True
DEFAULT_QUALITY_PROFILE_MIN_APPEARANCES = 5
DEFAULT_QUALITY_IDENTITY_QA = True
DEFAULT_QUALITY_IDENTITY_MARGIN = 0.15
DEFAULT_QUALITY_MAX_KEYFRAME_RETAKES = 2
DEFAULT_QUALITY_VOICE_MODE = "native_fixed"       # native | native_fixed | native_with_reference
DEFAULT_QUALITY_SHOT_PLANNER = True
DEFAULT_QUALITY_STORY_MEMORY = True
DEFAULT_QUALITY_EXCEPTIONS_QUEUE = True
DEFAULT_QUALITY_MAX_CLIP_RETAKES = 1
DEFAULT_QUALITY_CHAPTER_BUDGET_IMAGES = 0         # 0 = unlimited
DEFAULT_QUALITY_CHAPTER_BUDGET_CLIPS = 0          # 0 = unlimited

_QUALITY_REF_POLICIES = {"legacy", "all_characters", "profile"}
_QUALITY_EMBEDDINGS = {"off", "dinov2"}
_QUALITY_TEXT_EMBEDDINGS = {"off", "e5"}
_QUALITY_VOICE_MODES = {"native", "native_fixed", "native_with_reference"}


def _quality_section(workspace_root: str | Path = ".") -> dict[str, Any]:
    section = load_agent_config(workspace_root).get("quality", {})
    return section if isinstance(section, dict) else {}


def _quality_bool(key: str, env: str, default: bool, workspace_root: str | Path = ".") -> bool:
    raw = os.environ.get(env)
    if raw is not None and raw.strip() != "":
        return _parse_bool(raw, default)
    section = _quality_section(workspace_root)
    if key in section:
        return _parse_bool(section.get(key), default)
    return default


def _quality_int(key: str, env: str, default: int, minimum: int, workspace_root: str | Path = ".") -> int:
    raw = os.environ.get(env)
    if raw is not None and raw.strip():
        try:
            return max(minimum, int(raw))
        except ValueError:
            pass
    section = _quality_section(workspace_root)
    if section.get(key) is not None:
        try:
            return max(minimum, int(section.get(key)))
        except (TypeError, ValueError):
            pass
    return default


def _quality_float(key: str, env: str, default: float, workspace_root: str | Path = ".") -> float:
    raw = os.environ.get(env)
    if raw is not None and raw.strip():
        try:
            return min(1.0, max(0.0, float(raw)))
        except ValueError:
            pass
    section = _quality_section(workspace_root)
    if section.get(key) is not None:
        try:
            return min(1.0, max(0.0, float(section.get(key))))
        except (TypeError, ValueError):
            pass
    return default


def _quality_enum(key: str, env: str, default: str, allowed: set[str], workspace_root: str | Path = ".") -> str:
    raw = os.environ.get(env)
    if raw is not None and raw.strip() != "":
        value = raw.strip().lower()
        return value if value in allowed else default
    section = _quality_section(workspace_root)
    if key in section and section.get(key) is not None:
        value = str(section.get(key)).strip().lower()
        if value in allowed:
            return value
    return default


def quality_canon_gate(workspace_root: str | Path = ".") -> bool:
    return _quality_bool("canon_gate", "VIMAX_QUALITY_CANON_GATE", DEFAULT_QUALITY_CANON_GATE, workspace_root)


def quality_generic_linter(workspace_root: str | Path = ".") -> bool:
    return _quality_bool("generic_linter", "VIMAX_QUALITY_GENERIC_LINTER", DEFAULT_QUALITY_GENERIC_LINTER, workspace_root)


def quality_series_slug(workspace_root: str | Path = ".") -> str:
    raw = os.environ.get("VIMAX_QUALITY_SERIES_SLUG")
    if raw is not None and raw.strip() != "":
        return raw.strip()
    section = _quality_section(workspace_root)
    value = section.get("series_slug")
    if value is not None and str(value).strip():
        return str(value).strip()
    return DEFAULT_QUALITY_SERIES_SLUG


def quality_ref_policy(workspace_root: str | Path = ".") -> str:
    return _quality_enum("ref_policy", "VIMAX_QUALITY_REF_POLICY", DEFAULT_QUALITY_REF_POLICY, _QUALITY_REF_POLICIES, workspace_root)


def quality_max_reference_images(workspace_root: str | Path = ".") -> int:
    return _quality_int("max_reference_images", "VIMAX_QUALITY_MAX_REFERENCE_IMAGES", DEFAULT_QUALITY_MAX_REFERENCE_IMAGES, 1, workspace_root)


def quality_embeddings(workspace_root: str | Path = ".") -> str:
    return _quality_enum("embeddings", "VIMAX_QUALITY_EMBEDDINGS", DEFAULT_QUALITY_EMBEDDINGS, _QUALITY_EMBEDDINGS, workspace_root)


def quality_text_embeddings(workspace_root: str | Path = ".") -> str:
    return _quality_enum("text_embeddings", "VIMAX_QUALITY_TEXT_EMBEDDINGS", DEFAULT_QUALITY_TEXT_EMBEDDINGS, _QUALITY_TEXT_EMBEDDINGS, workspace_root)


def quality_scene_split_threshold(workspace_root: str | Path = ".") -> float:
    return _quality_float("scene_split_threshold", "VIMAX_QUALITY_SCENE_SPLIT_THRESHOLD", DEFAULT_QUALITY_SCENE_SPLIT_THRESHOLD, workspace_root)


def quality_profiles(workspace_root: str | Path = ".") -> bool:
    return _quality_bool("profiles", "VIMAX_QUALITY_PROFILES", DEFAULT_QUALITY_PROFILES, workspace_root)


def quality_profile_min_appearances(workspace_root: str | Path = ".") -> int:
    return _quality_int("profile_min_appearances", "VIMAX_QUALITY_PROFILE_MIN_APPEARANCES", DEFAULT_QUALITY_PROFILE_MIN_APPEARANCES, 1, workspace_root)


def quality_identity_qa(workspace_root: str | Path = ".") -> bool:
    return _quality_bool("identity_qa", "VIMAX_QUALITY_IDENTITY_QA", DEFAULT_QUALITY_IDENTITY_QA, workspace_root)


def quality_identity_margin(workspace_root: str | Path = ".") -> float:
    return _quality_float("identity_margin", "VIMAX_QUALITY_IDENTITY_MARGIN", DEFAULT_QUALITY_IDENTITY_MARGIN, workspace_root)


def quality_max_keyframe_retakes(workspace_root: str | Path = ".") -> int:
    return _quality_int("max_keyframe_retakes", "VIMAX_QUALITY_MAX_KEYFRAME_RETAKES", DEFAULT_QUALITY_MAX_KEYFRAME_RETAKES, 0, workspace_root)


def quality_voice_mode(workspace_root: str | Path = ".") -> str:
    return _quality_enum("voice_mode", "VIMAX_QUALITY_VOICE_MODE", DEFAULT_QUALITY_VOICE_MODE, _QUALITY_VOICE_MODES, workspace_root)


def quality_shot_planner(workspace_root: str | Path = ".") -> bool:
    return _quality_bool("shot_planner", "VIMAX_QUALITY_SHOT_PLANNER", DEFAULT_QUALITY_SHOT_PLANNER, workspace_root)


def quality_story_memory(workspace_root: str | Path = ".") -> bool:
    return _quality_bool("story_memory", "VIMAX_QUALITY_STORY_MEMORY", DEFAULT_QUALITY_STORY_MEMORY, workspace_root)


def quality_exceptions_queue(workspace_root: str | Path = ".") -> bool:
    return _quality_bool("exceptions_queue", "VIMAX_QUALITY_EXCEPTIONS_QUEUE", DEFAULT_QUALITY_EXCEPTIONS_QUEUE, workspace_root)


def quality_max_clip_retakes(workspace_root: str | Path = ".") -> int:
    return _quality_int("max_clip_retakes", "VIMAX_QUALITY_MAX_CLIP_RETAKES", DEFAULT_QUALITY_MAX_CLIP_RETAKES, 0, workspace_root)


def quality_chapter_budget_images(workspace_root: str | Path = ".") -> int:
    return _quality_int("chapter_budget_images", "VIMAX_QUALITY_CHAPTER_BUDGET_IMAGES", DEFAULT_QUALITY_CHAPTER_BUDGET_IMAGES, 0, workspace_root)


def quality_chapter_budget_clips(workspace_root: str | Path = ".") -> int:
    return _quality_int("chapter_budget_clips", "VIMAX_QUALITY_CHAPTER_BUDGET_CLIPS", DEFAULT_QUALITY_CHAPTER_BUDGET_CLIPS, 0, workspace_root)
