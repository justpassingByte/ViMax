"""Detect whether the video provider (Muse) accepts a reference-audio input.

``native_with_reference`` voice mode needs the provider to accept a short audio
sample so a character keeps one voice across clips. We probe once, cache the
result in ``<working_dir>/provider_capabilities.json``, and fall back to
``native_fixed`` (never an error) when it isn't supported.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_CACHE_FILE = "provider_capabilities.json"
_KEY = "audio_reference"


def _load_caps(working_dir: str | Path) -> dict:
    p = Path(working_dir) / _CACHE_FILE
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}
    return {}


def _save_caps(working_dir: str | Path, caps: dict) -> None:
    try:
        (Path(working_dir) / _CACHE_FILE).write_text(json.dumps(caps, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def probe_audio_reference_support(
    video_generator: Any,
    working_dir: str | Path,
    *,
    probe_fn: Optional[Any] = None,
    force: bool = False,
) -> bool:
    """Return True if the provider accepts reference audio. Cached after first probe.

    ``probe_fn(video_generator) -> bool`` performs the real one-off request; when
    omitted we look for a ``supports_audio_reference`` attribute/callable on the
    generator, else assume unsupported. Any failure => False (fall back).
    """
    caps = _load_caps(working_dir)
    if not force and _KEY in caps:
        return bool(caps[_KEY])

    supported = False
    try:
        if probe_fn is not None:
            supported = bool(probe_fn(video_generator))
        else:
            attr = getattr(video_generator, "supports_audio_reference", None)
            if callable(attr):
                supported = bool(attr())
            elif isinstance(attr, bool):
                supported = attr
            else:
                supported = False
    except Exception as exc:  # noqa: BLE001 - never fatal
        logger.warning("audio-reference probe failed (%s); assuming unsupported.", exc)
        supported = False

    caps[_KEY] = supported
    _save_caps(working_dir, caps)
    return supported


def effective_voice_mode(configured_mode: str, video_generator: Any, working_dir: str | Path, **kwargs) -> str:
    """Resolve the voice mode actually usable: downgrade native_with_reference to
    native_fixed when the provider can't accept reference audio."""
    if configured_mode != "native_with_reference":
        return configured_mode
    if probe_audio_reference_support(video_generator, working_dir, **kwargs):
        return "native_with_reference"
    logger.info("Provider does not accept reference audio; using native_fixed.")
    return "native_fixed"
