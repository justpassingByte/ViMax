"""Deterministic story-fidelity gate for scene storyboards.

Pure-code checks (no LLM) that a draft storyboard stays grounded in the
scene's canon beats before keyframes are rendered:

- every shot cites a ``[RefBeat: <beat_id>]`` of a beat in this scene,
- the cited beat order never regresses (except ``[Flashback]`` shots),
- dialogue is drawn only from the beats (no invented lines, no missing
  lines, no duplicates),
- only characters present in a cited beat appear in the shot.

``check_storyboard`` reports issues; ``repair_storyboard`` applies a fixed
(non-LLM) repair so a run can continue unattended. Both are gated by the
``quality.canon_gate`` flag at the call site.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from agents.cinematic_director import extract_shot_beat_refs
from interfaces.manga import MangaBeat
from interfaces.shot_description import ShotBriefDescription
from utils.audio import extract_all_dialogues_from_shot

CanonIssueKind = Literal[
    "uncited",
    "unknown_beat",
    "invented_dialogue",
    "missing_dialogue",
    "duplicate_dialogue",
    "foreign_character",
    "order_violation",
    "coverage_ratio",
]

_FLASHBACK_RE = re.compile(r"\[\s*(?:AnimeFX:\s*)?Flashback\s*\]", re.IGNORECASE)
_CHAR_TAG_RE = re.compile(r"<([^>]+)>")
_DIALOGUE_MATCH_RATIO = 0.9


class CanonIssue(BaseModel):
    shot_idx: int
    kind: CanonIssueKind
    detail: str


class CanonReport(BaseModel):
    scene_idx: int
    passed: bool
    issues: List[CanonIssue] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace for fuzzy compare."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _normalize_name(name: str) -> str:
    return _normalize_text(name)


def _build_alias_map(characters: Optional[Sequence[Any]]) -> Dict[str, str]:
    """Map a normalized alias/name -> canonical identifier for roster characters."""
    alias_map: Dict[str, str] = {}
    for ch in characters or []:
        canonical = str(
            getattr(ch, "identifier_in_scene", None)
            or getattr(ch, "name", None)
            or ch
        )
        keys = {canonical}
        for attr in ("aliases", "alias", "names"):
            val = getattr(ch, attr, None)
            if isinstance(val, (list, tuple, set)):
                keys.update(str(v) for v in val)
            elif isinstance(val, str) and val:
                keys.add(val)
        # also register a bare first token (e.g. "Alex" from "Alex (Human)")
        bare = re.sub(r"\(.*?\)", "", canonical).strip()
        if bare:
            keys.add(bare)
        for key in keys:
            norm = _normalize_name(key)
            if norm:
                alias_map[norm] = canonical
    return alias_map


def _resolve_character(tag: str, alias_map: Dict[str, str]) -> str:
    norm = _normalize_name(tag)
    if norm in alias_map:
        return alias_map[norm]
    # fuzzy containment: "<Young Boy Alex>" -> "Alex"
    for alias_norm, canonical in alias_map.items():
        if alias_norm and (alias_norm in norm or norm in alias_norm):
            return canonical
    return tag.strip()


def _beat_character_set(beat: MangaBeat, alias_map: Dict[str, str]) -> set[str]:
    present: set[str] = set()
    for name in beat.characters_present or []:
        present.add(_resolve_character(str(name), alias_map))
    return present


def _beat_dialogue_variants(beat: MangaBeat) -> List[str]:
    """Normalized dialogue strings for a beat (both original line and vietsub)."""
    variants: List[str] = []
    for d in beat.dialogues or []:
        for field in ("line", "vietsub", "source_text"):
            val = _normalize_text(getattr(d, field, "") or "")
            if val:
                variants.append(val)
    return variants


def _shot_character_tags(shot: ShotBriefDescription, alias_map: Dict[str, str]) -> List[str]:
    tags: List[str] = []
    for raw in _CHAR_TAG_RE.findall(shot.visual_desc or ""):
        resolved = _resolve_character(raw, alias_map)
        if resolved:
            tags.append(resolved)
    return tags


def _shot_dialogue_texts(shot: ShotBriefDescription) -> List[str]:
    """Normalized spoken/thought texts present in a shot."""
    out: List[str] = []
    for _speaker, text in extract_all_dialogues_from_shot(shot.audio_desc or "", shot.visual_desc or ""):
        norm = _normalize_text(text)
        if norm:
            out.append(norm)
    # also capture inline (Vietsub: ...) text as a valid grounded variant
    for m in re.finditer(r"\(\s*Vietsub\s*:\s*([^)]+?)\s*\)", shot.audio_desc or "", re.IGNORECASE):
        norm = _normalize_text(m.group(1))
        if norm:
            out.append(norm)
    return out


# ---------------------------------------------------------------------------
# Checker
# ---------------------------------------------------------------------------

def check_storyboard(
    storyboard: Sequence[ShotBriefDescription],
    beats: Sequence[MangaBeat],
    characters: Optional[Sequence[Any]] = None,
    *,
    scene_idx: int = 0,
    min_coverage_ratio: float = 0.6,
) -> CanonReport:
    """Validate a storyboard against its scene beats. Pure code, no LLM."""
    beats = list(beats)
    beat_by_id = {b.beat_id: b for b in beats}
    beat_order = {b.beat_id: i for i, b in enumerate(beats)}
    alias_map = _build_alias_map(characters)

    # All canon dialogue variants across the scene, grouped per beat.
    all_dialogue_variants = [v for b in beats for v in _beat_dialogue_variants(b)]

    issues: List[CanonIssue] = []
    cited_beats: set[str] = set()
    seen_dialogues: List[str] = []
    last_order = -1

    for shot in storyboard:
        combined = f"{shot.visual_desc or ''} {shot.audio_desc or ''}"
        beat_ids = extract_shot_beat_refs(combined)
        is_flashback = bool(_FLASHBACK_RE.search(combined))

        if not beat_ids:
            issues.append(CanonIssue(shot_idx=shot.idx, kind="uncited", detail="Shot has no [RefBeat] tag."))
        else:
            valid_ids = [bid for bid in beat_ids if bid in beat_by_id]
            for bid in beat_ids:
                if bid not in beat_by_id:
                    issues.append(CanonIssue(shot_idx=shot.idx, kind="unknown_beat", detail=f"Cites unknown beat '{bid}'."))
            if valid_ids:
                cited_beats.update(valid_ids)
                order = min(beat_order[bid] for bid in valid_ids)
                if not is_flashback and order < last_order:
                    issues.append(CanonIssue(
                        shot_idx=shot.idx,
                        kind="order_violation",
                        detail=f"Beat order regresses to {valid_ids} after position {last_order}.",
                    ))
                last_order = max(last_order, order)

                # Foreign-character check against the union of cited beats' rosters.
                allowed: set[str] = set()
                for bid in valid_ids:
                    allowed |= _beat_character_set(beat_by_id[bid], alias_map)
                for tag in _shot_character_tags(shot, alias_map):
                    if allowed and tag not in allowed:
                        issues.append(CanonIssue(
                            shot_idx=shot.idx,
                            kind="foreign_character",
                            detail=f"<{tag}> not present in cited beat(s) {valid_ids}.",
                        ))

        # Dialogue grounding: each spoken line must match some canon variant.
        for text in _shot_dialogue_texts(shot):
            matched = any(_fuzzy_ratio(text, variant) >= _DIALOGUE_MATCH_RATIO for variant in all_dialogue_variants)
            if not matched:
                issues.append(CanonIssue(
                    shot_idx=shot.idx,
                    kind="invented_dialogue",
                    detail=f"Dialogue not found in canon beats: '{text[:60]}'.",
                ))
                continue
            if any(_fuzzy_ratio(text, prev) >= _DIALOGUE_MATCH_RATIO for prev in seen_dialogues):
                issues.append(CanonIssue(
                    shot_idx=shot.idx,
                    kind="duplicate_dialogue",
                    detail=f"Dialogue repeated across shots: '{text[:60]}'.",
                ))
            seen_dialogues.append(text)

    # Missing dialogue: every canon line should appear once somewhere.
    for beat in beats:
        for d in beat.dialogues or []:
            variants = [_normalize_text(getattr(d, f, "") or "") for f in ("line", "vietsub", "source_text")]
            variants = [v for v in variants if v]
            if not variants:
                continue
            found = any(
                any(_fuzzy_ratio(seen, v) >= _DIALOGUE_MATCH_RATIO for v in variants)
                for seen in seen_dialogues
            )
            if not found:
                issues.append(CanonIssue(
                    shot_idx=-1,
                    kind="missing_dialogue",
                    detail=f"Canon line never shown ({beat.beat_id}): '{variants[0][:60]}'.",
                ))

    # Beat coverage ratio.
    coverage = (len(cited_beats) / len(beats)) if beats else 1.0
    if beats and coverage < min_coverage_ratio:
        issues.append(CanonIssue(
            shot_idx=-1,
            kind="coverage_ratio",
            detail=f"Only {coverage:.0%} of beats are illustrated (min {min_coverage_ratio:.0%}).",
        ))

    total_canon = len(all_dialogue_variants)
    dialogue_fidelity = 1.0
    if beats:
        canon_lines = [d for b in beats for d in (b.dialogues or [])]
        if canon_lines:
            shown = sum(
                1
                for d in canon_lines
                if any(
                    _fuzzy_ratio(seen, _normalize_text(getattr(d, "vietsub", "") or getattr(d, "line", "") or "")) >= _DIALOGUE_MATCH_RATIO
                    for seen in seen_dialogues
                )
            )
            dialogue_fidelity = shown / len(canon_lines)

    stats = {
        "beat_coverage": round(coverage, 4),
        "dialogue_fidelity": round(dialogue_fidelity, 4),
        "coverage_shot_ratio": round((len(storyboard) / len(beats)) if beats else 0.0, 4),
        "n_shots": len(storyboard),
        "n_beats": len(beats),
    }
    report = CanonReport(scene_idx=scene_idx, passed=not issues, issues=issues, stats=stats)
    return report


# ---------------------------------------------------------------------------
# Deterministic repair (no LLM)
# ---------------------------------------------------------------------------

def _speaker_for_beat_line(beat: MangaBeat, text_norm: str) -> Tuple[str, str, str]:
    """Return (speaker, line, vietsub) of the beat dialogue matching text_norm."""
    for d in beat.dialogues or []:
        for field in ("vietsub", "line", "source_text"):
            if _fuzzy_ratio(text_norm, _normalize_text(getattr(d, field, "") or "")) >= _DIALOGUE_MATCH_RATIO:
                return (d.speaker or "", d.line or "", d.vietsub or "")
    return ("", "", "")


def repair_storyboard(
    storyboard: Sequence[ShotBriefDescription],
    beats: Sequence[MangaBeat],
    report: CanonReport,
    characters: Optional[Sequence[Any]] = None,
) -> List[ShotBriefDescription]:
    """Fixed, deterministic repair driven by a CanonReport.

    - drop shots flagged uncited / unknown_beat / invented_dialogue / foreign_character,
    - add a simple medium shot for each missing canon dialogue line,
    - re-sort by cited beat order (stable) and re-index.
    """
    beats = list(beats)
    beat_order = {b.beat_id: i for i, b in enumerate(beats)}
    beat_by_id = {b.beat_id: b for b in beats}

    drop_kinds = {"uncited", "unknown_beat", "invented_dialogue", "foreign_character", "order_violation"}
    drop_idxs = {iss.shot_idx for iss in report.issues if iss.kind in drop_kinds and iss.shot_idx >= 0}

    kept: List[ShotBriefDescription] = [s for s in storyboard if s.idx not in drop_idxs]

    # Build new shots for missing dialogue lines.
    added: List[ShotBriefDescription] = []
    for iss in report.issues:
        if iss.kind != "missing_dialogue":
            continue
        # detail format: "Canon line never shown (beat_id): '...'."
        m = re.search(r"\(([^)]+)\)", iss.detail)
        beat_id = m.group(1) if m else ""
        beat = beat_by_id.get(beat_id)
        if beat is None:
            continue
        for d in beat.dialogues or []:
            speaker = d.speaker or "Narrator"
            line = d.line or d.vietsub or ""
            if not line:
                continue
            tag = "Thought" if d.line_type == "thought" else "Speaker"
            audio = f"[{tag}] {speaker} ({d.emotion or 'neutral'}): '{line}'"
            if d.vietsub:
                audio += f" (Vietsub: {d.vietsub})"
            vis = f"[RefBeat: {beat_id}] Medium shot of <{speaker}> delivering the line, facing the camera, scene-consistent lighting."
            added.append(ShotBriefDescription(
                idx=10_000 + len(added),
                is_last=False,
                cam_idx=0,
                visual_desc=vis,
                audio_desc=audio,
            ))

    combined = kept + added

    def _sort_key(shot: ShotBriefDescription) -> Tuple[int, int]:
        refs = extract_shot_beat_refs(f"{shot.visual_desc or ''} {shot.audio_desc or ''}")
        orders = [beat_order[b] for b in refs if b in beat_order]
        return (min(orders) if orders else len(beats) + 1, shot.idx)

    combined.sort(key=_sort_key)

    repaired: List[ShotBriefDescription] = []
    for new_idx, shot in enumerate(combined):
        is_last = new_idx == len(combined) - 1
        repaired.append(shot.model_copy(update={"idx": new_idx, "is_last": is_last}))
    return repaired
