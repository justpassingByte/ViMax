"""Identity QA for keyframes (phase 06).

Checks a rendered keyframe *before* the costly video step: the expected
characters are present, each matches its own reference identity (self-calibrated
threshold, no labels needed), wears the right outfit, and the frame carries no
comic lettering/bubbles. A small retry orchestrator re-renders failing keyframes
up to a budget, then records an exception so the run continues unattended.

Everything is dependency-injected (embedder + VLM callables) so it is testable
offline with fakes and never forces a model download.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FrameCheck(BaseModel):
    shot_idx: int
    frame: Literal["first", "last"] = "first"
    expected: List[str] = Field(default_factory=list)
    detected: List[dict] = Field(default_factory=list)
    identity_scores: Dict[str, float] = Field(default_factory=dict)
    baselines: Dict[str, float] = Field(default_factory=dict)
    outfit_ok: Dict[str, bool] = Field(default_factory=dict)
    has_lettering: bool = False
    passed: bool = True
    reasons: List[str] = Field(default_factory=list)


def _mean_pairwise_cosine(vectors) -> float:
    from tools.embeddings import cosine
    n = len(vectors)
    if n < 2:
        return 1.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += cosine(vectors[i], vectors[j])
            count += 1
    return total / count if count else 1.0


def _centroid(vectors):
    import numpy as np
    return np.asarray(vectors, dtype="float32").mean(axis=0)


def _reference_paths(profile) -> List[str]:
    paths: List[str] = []
    for o in getattr(profile, "outfits", []) or []:
        sheet = getattr(o, "sheet", None)
        if sheet is not None:
            for v in (sheet.front, sheet.side, sheet.back):
                if v and os.path.exists(v):
                    paths.append(v)
    for c in getattr(profile, "source_crops", []) or []:
        if getattr(c, "usable", True) and c.path and os.path.exists(c.path):
            paths.append(c.path)
    # de-dupe, keep order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _crop_to_temp(frame_path: str, box) -> Optional[str]:
    try:
        from PIL import Image
        with Image.open(frame_path) as im:
            w, h = im.size
            x0, y0, x1, y1 = box
            # accept normalized 0..1000 boxes or raw pixel boxes
            if max(box) <= 1000 and max(w, h) > 1000:
                x0, y0, x1, y1 = x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000
            crop = im.convert("RGB").crop((int(x0), int(y0), int(max(x0 + 1, x1)), int(max(y0 + 1, y1))))
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        crop.save(tmp)
        return tmp
    except Exception:
        return None


async def check_frame(
    frame_path: str,
    shot_idx: int,
    expected_character_ids: Sequence[str],
    beat_id: str,
    profiles: Any,                      # ProfileStore-like: .get(id), .outfit_at(id, beat)
    image_embedder: Optional[Any],
    vlm: Optional[Callable[..., Any]],
    *,
    frame: str = "first",
    identity_margin: float = 0.15,
) -> FrameCheck:
    """Run a single VLM pass + embedding identity check on one keyframe."""
    expected = list(expected_character_ids or [])
    check = FrameCheck(shot_idx=shot_idx, frame=frame, expected=expected)

    outfit_desc = {}
    for cid in expected:
        prof = profiles.get(cid) if profiles else None
        if prof is not None:
            o = profiles.outfit_at(prof.character_id, beat_id)
            outfit_desc[cid] = (o.description if o else "") or getattr(prof, "prompt_notes", "")

    vlm_result: Dict[str, Any] = {}
    if vlm is not None:
        try:
            res = vlm(frame_path=frame_path, expected=expected, outfit_desc=outfit_desc)
            vlm_result = await res if hasattr(res, "__await__") else res
        except Exception as exc:  # noqa: BLE001 - never fatal
            logger.warning("identity_qa VLM failed: %s", exc)
            vlm_result = {}

    detected = vlm_result.get("detected", []) or []
    check.detected = detected
    check.has_lettering = bool(vlm_result.get("has_lettering", False))
    check.outfit_ok = {cid: bool(vlm_result.get("outfit_ok", {}).get(cid, True)) for cid in expected}

    # Identity scoring via embeddings (self-calibrated baseline).
    if image_embedder is not None and profiles is not None:
        box_by_name = {}
        for d in detected:
            nm = d.get("character_id") or d.get("name")
            if nm and d.get("box"):
                box_by_name[nm] = d["box"]
        for cid in expected:
            prof = profiles.get(cid)
            if prof is None:
                continue
            ref_paths = _reference_paths(prof)
            if not ref_paths:
                continue
            try:
                ref_vecs = image_embedder.embed_images(ref_paths)
                baseline = _mean_pairwise_cosine(ref_vecs)
                centroid = _centroid(ref_vecs)
                probe_path = frame_path
                tmp_crop = None
                if cid in box_by_name:
                    tmp_crop = _crop_to_temp(frame_path, box_by_name[cid])
                    if tmp_crop:
                        probe_path = tmp_crop
                frame_vec = image_embedder.embed_images([probe_path])[0]
                if tmp_crop:
                    try:
                        os.remove(tmp_crop)
                    except OSError:
                        pass
                from tools.embeddings import cosine
                score = cosine(frame_vec, centroid)
                check.baselines[cid] = round(baseline, 4)
                check.identity_scores[cid] = round(score, 4)
                if score < baseline - identity_margin:
                    check.reasons.append(f"{cid}: identity {score:.3f} < baseline {baseline:.3f} - margin {identity_margin}")
            except Exception as exc:  # noqa: BLE001
                logger.warning("identity_qa embed failed for %s: %s", cid, exc)

    # Count check.
    detected_names = {d.get("character_id") or d.get("name") for d in detected if (d.get("character_id") or d.get("name"))}
    if detected and expected and len(detected_names) != len(set(expected)):
        check.reasons.append(f"character count mismatch: expected {len(set(expected))}, detected {len(detected_names)}")

    for cid, ok in check.outfit_ok.items():
        if not ok:
            check.reasons.append(f"{cid}: wrong outfit")
    if check.has_lettering:
        check.reasons.append("comic lettering / speech bubble detected")

    check.passed = not check.reasons
    return check


async def qa_and_retake_keyframes(
    shots: Sequence[dict],
    *,
    render_frame: Callable[[int, str, int], Any],   # (shot_idx, frame, attempt) -> frame_path
    check: Callable[..., Any],                       # (frame_path, shot) -> awaitable FrameCheck
    max_retakes: int = 2,
) -> Dict[str, Any]:
    """Render → check → re-render loop. Returns {checks, exceptions, n_renders}.

    ``shots`` items: {"shot_idx": int, "frame": "first"|"last"}.
    """
    checks: List[FrameCheck] = []
    exceptions: List[dict] = []
    n_renders = 0
    for shot in shots:
        shot_idx = shot["shot_idx"]
        frame = shot.get("frame", "first")
        best: Optional[FrameCheck] = None
        best_score = -1.0
        passed = False
        for attempt in range(max_retakes + 1):
            frame_path = render_frame(shot_idx, frame, attempt)
            n_renders += 1
            res = check(frame_path=frame_path, shot=shot)
            fc: FrameCheck = await res if hasattr(res, "__await__") else res
            checks.append(fc)
            min_score = min(fc.identity_scores.values()) if fc.identity_scores else 1.0
            if min_score > best_score:
                best_score = min_score
                best = fc
            if fc.passed:
                passed = True
                break
        if not passed and best is not None:
            exceptions.append({
                "shot_idx": shot_idx,
                "frame": frame,
                "reasons": best.reasons,
                "best_score": round(best_score, 4),
            })
    return {"checks": checks, "exceptions": exceptions, "n_renders": n_renders}
