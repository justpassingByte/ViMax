"""Series bible loader: per-series canon data under ``series/<slug>/``.

Keeps all series-specific prompt content (style palette, per-character notes,
wardrobe rules, voice descriptions) out of the generic pipeline code. When no
series is configured or the files are missing, every accessor returns an empty
value so the pipeline runs unchanged.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a hard dep elsewhere
    yaml = None


def _series_root(workspace_root: str | Path = ".") -> Path:
    return Path(workspace_root) / "series"


@functools.lru_cache(maxsize=16)
def _load_yaml(path_str: str, mtime_ns: int) -> Dict[str, Any]:
    path = Path(path_str)
    if yaml is None or not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_yaml_cached(path: Path) -> Dict[str, Any]:
    try:
        stat = path.stat()
    except OSError:
        return {}
    return _load_yaml(str(path), stat.st_mtime_ns)


class SeriesBible:
    """In-memory view of one series' canon data. Empty when the series is missing."""

    def __init__(self, slug: str, workspace_root: str | Path = "."):
        self.slug = (slug or "").strip()
        self._root = _series_root(workspace_root) / self.slug if self.slug else None
        self._series: Dict[str, Any] = {}
        self._characters: List[Dict[str, Any]] = []
        if self._root and self._root.is_dir():
            self._series = _load_yaml_cached(self._root / "series.yaml")
            chars = _load_yaml_cached(self._root / "characters.yaml")
            raw = chars.get("characters") if isinstance(chars, dict) else None
            if isinstance(raw, list):
                self._characters = [c for c in raw if isinstance(c, dict)]

    @property
    def available(self) -> bool:
        return bool(self._series or self._characters)

    def title(self) -> str:
        return str(self._series.get("title", "")).strip()

    def style_prompt(self) -> str:
        return str(self._series.get("style_prompt", "") or "").strip()

    def negative_prompt(self) -> str:
        return str(self._series.get("negative_prompt", "") or "").strip()

    def all_names(self) -> List[str]:
        """Every title/name/alias (for the no-series-names-in-code test)."""
        names: List[str] = []
        if self.title():
            names.append(self.title())
        for ch in self._characters:
            for alias in ch.get("aliases", []) or []:
                if isinstance(alias, str) and alias.strip():
                    names.append(alias.strip())
        return names

    def _match_character(self, name_or_alias: str) -> Optional[Dict[str, Any]]:
        if not name_or_alias:
            return None
        needle = name_or_alias.strip().lower()
        for ch in self._characters:
            for alias in ch.get("aliases", []) or []:
                a = str(alias).strip().lower()
                if a and (a == needle or a in needle or needle in a):
                    return ch
        return None

    def character_notes(self, name_or_alias: str, scene_idx: Optional[int] = None) -> str:
        """Prompt notes + the wardrobe rule covering ``scene_idx`` for one character."""
        ch = self._match_character(name_or_alias)
        if ch is None:
            return ""
        parts: List[str] = []
        notes = str(ch.get("prompt_notes", "") or "").strip()
        if notes:
            parts.append(notes)
        for rule in ch.get("wardrobe_rules", []) or []:
            if not isinstance(rule, dict):
                continue
            lo = rule.get("from_scene")
            hi = rule.get("to_scene")
            text = str(rule.get("outfit_text", "") or "").strip()
            if not text:
                continue
            if scene_idx is None:
                parts.append(text)
            else:
                try:
                    if int(lo) <= int(scene_idx) <= int(hi):
                        parts.append(text)
                except (TypeError, ValueError):
                    parts.append(text)
        return " ".join(parts).strip()

    def voice_description(self, name_or_alias: str) -> str:
        ch = self._match_character(name_or_alias)
        if ch is None:
            return ""
        return str(ch.get("voice_description", "") or "").strip()


@functools.lru_cache(maxsize=8)
def _cached_bible(slug: str, workspace_root: str) -> SeriesBible:
    return SeriesBible(slug, workspace_root)


def load_series(slug: str, workspace_root: str | Path = ".") -> SeriesBible:
    """Load (and cache) a series bible by slug. Empty bible when slug is blank/missing."""
    return _cached_bible(slug or "", str(workspace_root))


def load_configured_series(workspace_root: str | Path = ".") -> SeriesBible:
    """Load the series named by ``quality.series_slug`` (empty bible if unset)."""
    try:
        from agent_runtime.config import quality_series_slug
        slug = quality_series_slug(workspace_root)
    except Exception:
        slug = ""
    return load_series(slug, workspace_root)
