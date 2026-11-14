"""Rule-based shot planner (phase 08, quality.shot_planner).

Picks shot size / camera angle / movement / fx per beat type from
``configs/shot_rules.yaml`` with weighted, seeded randomness (reproducible per
scene) and a small backtracking pass to satisfy variety rules. No LLM calls and
no character/series names in code — all cues live in the rules file.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

_BEAT_TYPES = [
    "establishing", "dialogue", "exchange", "reaction",
    "action", "impact", "reveal", "monologue", "transition",
]


class ShotSlot(BaseModel):
    beat_ids: List[str] = Field(default_factory=list)
    beat_type: str = "dialogue"
    shot_size: str = "medium"
    camera_angle: str = "eye"
    camera_movement: str = "static"
    fx: List[str] = Field(default_factory=list)
    is_coverage: bool = False


def load_rules(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "shot_rules.yaml"
    path = Path(path)
    if yaml is None or not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _get(beat, attr, default=""):
    if isinstance(beat, dict):
        return beat.get(attr, default)
    return getattr(beat, attr, default)


def _beat_dialogues(beat):
    return _get(beat, "dialogues", []) or []


def infer_beat_type(beat, prev_beat=None, rules: Optional[dict] = None) -> str:
    """Infer a beat type for caches without an explicit one (pure code, keyword-driven)."""
    explicit = _get(beat, "beat_type", None)
    if explicit in _BEAT_TYPES:
        return explicit

    rules = rules or {}
    cues = rules.get("inference", {}) if isinstance(rules, dict) else {}
    action_kw = [k.lower() for k in cues.get("action_keywords", [])]
    impact_sfx = [k.lower() for k in cues.get("impact_sfx", [])]
    reveal_kw = [k.lower() for k in cues.get("reveal_keywords", [])]

    dialogues = _beat_dialogues(beat)
    action_desc = str(_get(beat, "action_desc", "")).lower()
    sfx = " ".join(str(s) for s in (_get(beat, "sfx", []) or [])).lower()
    is_break = bool(_get(beat, "is_scene_break", False))

    # impact: strong sfx
    if impact_sfx and any(k in sfx for k in impact_sfx):
        return "impact"
    # action: action keywords + some sfx
    if action_kw and any(k in action_desc for k in action_kw):
        return "action"
    # reveal
    if reveal_kw and any(k in action_desc for k in reveal_kw):
        return "reveal"

    if dialogues:
        kinds = {str(_get(d, "line_type", "spoken")) for d in dialogues}
        if kinds <= {"thought", "narration"}:
            return "monologue"
        # exchange if speaker differs from previous beat (same scene) else dialogue
        if prev_beat is not None and _beat_dialogues(prev_beat):
            cur_spk = str(_get(dialogues[0], "speaker", "")).lower()
            prev_spk = str(_get(_beat_dialogues(prev_beat)[0], "speaker", "")).lower()
            if cur_spk and prev_spk and cur_spk != prev_spk and not is_break:
                return "exchange"
        return "dialogue"

    # no dialogue
    if is_break or prev_beat is None:
        return "establishing"
    present = _get(beat, "characters_present", []) or []
    if present:
        return "reaction"
    return "transition"


def _weighted_choice(rng: random.Random, options: List[dict]) -> str:
    if not options:
        return ""
    values = [o["v"] for o in options]
    weights = [max(0.0, float(o.get("w", 1))) for o in options]
    if sum(weights) <= 0:
        return rng.choice(values)
    return rng.choices(values, weights=weights, k=1)[0]


def _rule_for(rules: dict, beat_type: str) -> dict:
    return (rules.get("beat_types", {}) or {}).get(beat_type, {}) if rules else {}


def plan_scene(beats: Sequence[Any], rules: Optional[dict] = None, seed: int = 0) -> List[ShotSlot]:
    """One ShotSlot per beat, picked by weighted seeded choice, then variety-repaired."""
    rules = rules or load_rules()
    variety = rules.get("variety", {}) if rules else {}
    max_run = int(variety.get("max_same_size_angle_run", 2))
    rng = random.Random(seed)

    beats = list(beats)
    slots: List[ShotSlot] = []
    prev_beat = None
    for b in beats:
        bt = infer_beat_type(b, prev_beat, rules)
        rule = _rule_for(rules, bt)
        beat_id = _get(b, "beat_id", "")
        size = _weighted_choice(rng, rule.get("sizes", [])) or "medium"
        angle = _weighted_choice(rng, rule.get("angles", [])) or "eye"
        move = _weighted_choice(rng, rule.get("movements", [])) or "static"
        fx_opt = rule.get("fx", [])
        fx = []
        fx_pick = _weighted_choice(rng, fx_opt) if fx_opt else ""
        if fx_pick:
            fx = [fx_pick]
        slots.append(ShotSlot(
            beat_ids=[beat_id] if beat_id else [],
            beat_type=bt, shot_size=size, camera_angle=angle, camera_movement=move, fx=fx,
        ))
        prev_beat = b

    # Variety repair: break runs of identical (size, angle) longer than max_run.
    for i in range(len(slots)):
        run = 1
        j = i - 1
        while j >= 0 and (slots[j].shot_size, slots[j].camera_angle) == (slots[i].shot_size, slots[i].camera_angle):
            run += 1
            j -= 1
        if run > max_run:
            rule = _rule_for(rules, slots[i].beat_type)
            # try a few re-rolls for a different size/angle pair
            for _ in range(6):
                new_size = _weighted_choice(rng, rule.get("sizes", [])) or slots[i].shot_size
                new_angle = _weighted_choice(rng, rule.get("angles", [])) or slots[i].camera_angle
                if (new_size, new_angle) != (slots[i].shot_size, slots[i].camera_angle):
                    slots[i].shot_size = new_size
                    slots[i].camera_angle = new_angle
                    break
    return slots


def check_variety(slots: Sequence[ShotSlot], rules: Optional[dict] = None) -> List[str]:
    """Return a list of variety-rule violations (empty = clean)."""
    rules = rules or load_rules()
    variety = rules.get("variety", {}) if rules else {}
    violations: List[str] = []

    max_run = int(variety.get("max_same_size_angle_run", 2))
    run = 0
    prev_key = None
    for idx, s in enumerate(slots):
        key = (s.shot_size, s.camera_angle)
        if key == prev_key:
            run += 1
        else:
            run = 1
            prev_key = key
        if run > max_run:
            violations.append(f"shot {idx}: same size+angle {key} repeated more than {max_run} times")

    thresh = variety.get("min_distinct_sizes_if_shots_ge", {})
    if thresh and len(slots) >= int(thresh.get("shots", 10**9)):
        distinct = {s.shot_size for s in slots}
        if len(distinct) < int(thresh.get("sizes", 0)):
            violations.append(f"only {len(distinct)} distinct shot sizes across {len(slots)} shots")

    share = float(variety.get("max_low_angle_share_non_action", 1.0))
    non_action = [s for s in slots if s.beat_type not in ("action", "impact")]
    if non_action:
        low = sum(1 for s in non_action if s.camera_angle == "low")
        if low / len(non_action) > share + 1e-9:
            violations.append(f"low-angle share {low/len(non_action):.0%} exceeds {share:.0%} on non-action shots")

    allowed_fx_types = set(variety.get("fx_allowed_beat_types", []))
    for idx, s in enumerate(slots):
        if s.fx and s.beat_type not in allowed_fx_types:
            violations.append(f"shot {idx}: fx {s.fx} not allowed for beat type '{s.beat_type}'")
    return violations


_ANIMEFX_RE = re.compile(r"\[AnimeFX:\s*[^\]]+\]", re.IGNORECASE)


def fx_tokens_to_tags(fx: Sequence[str], rules: Optional[dict] = None):
    """Map fx tokens to ('[AnimeFX: ...]' tags, transition_in | None)."""
    rules = rules or load_rules()
    fx_map = rules.get("fx_map", {}) if rules else {}
    tags: List[str] = []
    transition = None
    for token in fx:
        mapped = fx_map.get(token)
        if not mapped:
            continue
        if mapped.startswith("[AnimeFX"):
            tags.append(mapped)
        else:
            transition = mapped  # e.g. white_flash
    return tags, transition


def strip_forbidden_fx_tags(visual_desc: str, beat_type: str, rules: Optional[dict] = None) -> str:
    """Remove any [AnimeFX: ...] tags the LLM added when this beat type disallows fx."""
    rules = rules or load_rules()
    allowed_types = set((rules.get("variety", {}) or {}).get("fx_allowed_beat_types", []))
    if beat_type in allowed_types:
        return visual_desc
    return _ANIMEFX_RE.sub("", visual_desc or "").replace("  ", " ").strip()
