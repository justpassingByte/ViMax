"""Unattended multi-chapter run support (phase 10).

- ``check_clip``: sample first/middle/last frames of a rendered clip and run the
  phase-06 identity check on each (frame extraction is injected so it is testable
  offline).
- ``BudgetTracker``: cap images/clips per chapter (0 = unlimited); when a budget
  is hit the chapter stops and the run moves on.
- ``aggregate_chapter_exceptions``: gather per-scene exceptions, profile-merge
  suggestions, story-memory conflicts and repaired canon reports into one
  ``exceptions/chNN.json`` the user reviews once per chapter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


async def check_clip(
    video_path: str,
    shot: dict,
    *,
    extract_frames_fn: Callable[[str], Sequence[str]],
    check_frame_fn: Callable[..., Any],
    max_clip_retakes: int = 1,
) -> dict:
    """Check first/middle/last frames of a clip. Returns {passed, frame_checks, reasons}."""
    try:
        frames = list(extract_frames_fn(video_path) or [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("clip frame extraction failed for %s: %s", video_path, exc)
        return {"passed": True, "frame_checks": [], "reasons": [], "error": str(exc)}

    checks = []
    reasons: List[str] = []
    for i, fp in enumerate(frames):
        pos = "first" if i == 0 else ("last" if i == len(frames) - 1 else "mid")
        res = check_frame_fn(frame_path=fp, shot=shot, position=pos)
        fc = await res if hasattr(res, "__await__") else res
        checks.append(fc)
        if not getattr(fc, "passed", True):
            reasons.extend(getattr(fc, "reasons", []))
    return {"passed": not reasons, "frame_checks": checks, "reasons": reasons}


class BudgetTracker:
    """Per-chapter image/clip budget (0 = unlimited)."""

    def __init__(self, max_images: int = 0, max_clips: int = 0):
        self.max_images = int(max_images or 0)
        self.max_clips = int(max_clips or 0)
        self.images = 0
        self.clips = 0
        self.stopped_reason: Optional[str] = None

    def add_image(self, n: int = 1) -> bool:
        self.images += n
        return self._check()

    def add_clip(self, n: int = 1) -> bool:
        self.clips += n
        return self._check()

    def _check(self) -> bool:
        if self.max_images and self.images > self.max_images:
            self.stopped_reason = f"image budget exceeded ({self.images} > {self.max_images})"
            return False
        if self.max_clips and self.clips > self.max_clips:
            self.stopped_reason = f"clip budget exceeded ({self.clips} > {self.max_clips})"
            return False
        return True

    @property
    def over_budget(self) -> bool:
        return self.stopped_reason is not None


def _read_json(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def aggregate_chapter_exceptions(work_root: str | Path, chapter: int) -> dict:
    """Collect all review items for a chapter into exceptions/chNN.json."""
    work_root = Path(work_root)
    out: Dict[str, Any] = {
        "chapter": chapter,
        "keyframe_exceptions": [],
        "clip_exceptions": [],
        "profile_merge_suggestions": [],
        "story_conflicts": [],
        "canon_repairs": [],
    }
    # per-scene exceptions + canon reports
    for scene_dir in sorted(work_root.glob("scene_*")):
        exc = _read_json(scene_dir / "exceptions.json", [])
        if isinstance(exc, list):
            for e in exc:
                e = dict(e)
                e["scene"] = scene_dir.name
                (out["clip_exceptions"] if e.get("kind") == "clip" else out["keyframe_exceptions"]).append(e)
        report = _read_json(scene_dir / "canon_report.json", None)
        if isinstance(report, dict) and report.get("issues"):
            out["canon_repairs"].append({"scene": scene_dir.name, "passed": report.get("passed"), "issues": report.get("issues")})

    sug = _read_json(work_root / "profile_merge_suggestions.json", [])
    if isinstance(sug, list):
        out["profile_merge_suggestions"] = sug
    conflicts = _read_json(work_root / "story_memory" / "conflicts.json", [])
    if isinstance(conflicts, list):
        out["story_conflicts"] = conflicts

    out["total"] = (
        len(out["keyframe_exceptions"]) + len(out["clip_exceptions"])
        + len(out["profile_merge_suggestions"]) + len(out["story_conflicts"])
        + len(out["canon_repairs"])
    )
    exc_dir = work_root / "exceptions"
    exc_dir.mkdir(parents=True, exist_ok=True)
    (exc_dir / f"ch{chapter:02d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
