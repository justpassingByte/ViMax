"""Cast lock: keep off-frame characters out of the video prompt (phase 11).

A single-character keyframe whose motion/audio prompt still names a *second*
character ("the cat faces left toward the off-frame boy") pushes the video model
to pan the camera and invent that second character from scratch — and, lacking a
reference image, it draws the wrong person. This module rewrites the prompt so
only the characters actually present in the keyframe are describable:

- ``visible_char_idxs`` / ``offscreen_characters``: who is in the frame vs merely
  referenced.
- ``sanitize_offscreen_motion``: turn "toward the off-frame boy" into a plain gaze
  direction and delete camera moves that would reveal an absent character.
- ``cast_lock_block``: an explicit "only these N characters exist in this clip"
  instruction.
- ``speaker_is_offscreen``: whether the active speaker is a character not shown in
  the frame (so their line becomes an off-screen voice).

Everything here is pure text/data work with no character- or series-specific
names hardcoded; the names come from the shot's own ``characters.json``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# A short, generic noun that often tags an off-frame person in a description.
_PERSON_NOUN = (
    r"(?:boy|girl|man|woman|child|kid|figure|person|people|guy|lady|"
    r"cat|dog|animal|creature|character|stranger|friend|shadow|silhouette|"
    r"soldier|warrior|hunter|monster|beast|elder|teacher|student|owner|clerk)"
)


def _char_id(c: Any) -> str:
    if isinstance(c, dict):
        return str(c.get("identifier_in_scene", "") or "")
    return str(getattr(c, "identifier_in_scene", "") or "")


def _char_idx(c: Any) -> Optional[int]:
    val = c.get("idx") if isinstance(c, dict) else getattr(c, "idx", None)
    if val is None:
        val = c.get("char_idx") if isinstance(c, dict) else getattr(c, "char_idx", None)
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _base_name(identifier: str) -> str:
    """Strip a trailing "(...)" qualifier: "Alex (Cat)" -> "Alex"."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", identifier or "").strip()


def visible_char_idxs(shot: Any) -> Set[int]:
    """Union of the first- and last-frame visible character indices."""
    out: Set[int] = set()
    for attr in ("ff_vis_char_idxs", "lf_vis_char_idxs"):
        vals = shot.get(attr) if isinstance(shot, dict) else getattr(shot, attr, None)
        for v in vals or []:
            try:
                out.add(int(v))
            except (TypeError, ValueError):
                continue
    return out


def _name_variants(identifier: str) -> List[str]:
    """Searchable name forms for one character identifier (longest first).

    Deliberately conservative: the full identifier and the whole base name, plus a
    single bare token only when the base name is itself one word (a real proper
    name like "Alex"). Multi-word identifiers such as "Tall Masked Figure"
    are only matched as a whole, so a descriptive pseudo-name never hits a loose
    word ("tabby", "ball") in the prompt and scrubs a legitimate reference.
    """
    variants: Set[str] = set()
    ident = (identifier or "").strip()
    if ident:
        variants.add(ident)
    base = _base_name(ident)
    if base:
        variants.add(base)
        tokens = base.split()
        if len(tokens) == 1 and len(base) >= 3 and not re.fullmatch(_PERSON_NOUN, base, re.IGNORECASE):
            variants.add(base)
    return sorted((v for v in variants if v), key=len, reverse=True)


def _visible_variant_set(shot: Any, characters: Sequence[Any]) -> Set[str]:
    """Lower-cased name variants of every character visible in the keyframes."""
    visible = visible_char_idxs(shot)
    out: Set[str] = set()
    for c in characters or []:
        if _char_idx(c) in visible:
            for v in _name_variants(_char_id(c)):
                out.add(v.lower())
    return out


def offscreen_characters(
    shot: Any,
    characters: Sequence[Any],
    *,
    extra_text: str = "",
) -> List[Any]:
    """Characters referenced by the shot text but NOT visible in its keyframes.

    A candidate that shares any name token with a visible character (e.g. the same
    person shown in a different form) is treated as in-frame, not off-frame.
    """
    visible = visible_char_idxs(shot)
    visible_variants = _visible_variant_set(shot, characters)
    text_parts = [
        str(shot.get(k, "") if isinstance(shot, dict) else getattr(shot, k, "") or "")
        for k in ("visual_desc", "motion_desc", "ff_desc", "lf_desc", "audio_desc")
    ]
    text_parts.append(extra_text or "")
    text = " ".join(text_parts)
    out: List[Any] = []
    for c in characters or []:
        idx = _char_idx(c)
        if idx is not None and idx in visible:
            continue
        variants = _name_variants(_char_id(c))
        if any(v.lower() in visible_variants for v in variants):
            continue  # shares a name with someone on screen -> same identity
        for variant in variants:
            if re.search(rf"\b{re.escape(variant)}\b", text, re.IGNORECASE):
                out.append(c)
                break
    return out


def references_offscreen_person(text: str) -> bool:
    """True when the text mentions an explicitly off-frame/off-screen person."""
    if not text:
        return False
    return bool(re.search(
        rf"\boff[\s-]?(?:frame|screen)\s+(?:the\s+)?(?:young\s+|small\s+|other\s+)?{_PERSON_NOUN}",
        text, re.IGNORECASE,
    ))


def _offscreen_target_re(offscreen_names: Iterable[str]) -> str:
    """Regex alternation matching an off-frame person: explicit name OR off-frame noun."""
    name_alts = []
    for ident in offscreen_names:
        for variant in _name_variants(ident):
            name_alts.append(re.escape(variant))
    offframe = rf"off[\s-]?(?:frame|screen)\s+(?:the\s+)?(?:young\s+|small\s+|other\s+)?{_PERSON_NOUN}"
    parts = [offframe]
    if name_alts:
        parts.append("|".join(sorted(set(name_alts), key=len, reverse=True)))
    return "(?:" + "|".join(parts) + ")"


def sanitize_offscreen_motion(motion: str, offscreen_names: Sequence[str]) -> str:
    """Rewrite a motion/visual prompt so no off-frame character is describable.

    Keeps the on-screen subject and its gaze *direction*, but turns references to
    an absent character into neutral off-frame space and deletes camera moves that
    would pan/tilt/cut to reveal that character.
    """
    if not motion:
        return ""
    target = _offscreen_target_re(offscreen_names)
    text = motion

    # 1. "looks/gazes down at <target>" -> eyes lowered (the look-down bug).
    text = re.sub(
        rf"\b(?:looks?|looking|glanc\w+|gaz\w+|star\w+|peer\w+|gaz\w+)\s+down\s+"
        rf"(?:at|toward|towards|to|upon)\s+(?:the\s+)?{target}",
        "with eyes lowered toward the lower edge of the frame",
        text,
        flags=re.IGNORECASE,
    )
    # 2. "looks/facing at|toward <target>" -> gaze toward off-frame space.
    text = re.sub(
        rf"\b(?:looks?|looking|glanc\w+|gaz\w+|star\w+|peer\w+|facing|turned)\s+"
        rf"(?:at|toward|towards|to|upon)\s+(?:the\s+)?{target}",
        "with gaze held toward off-frame space",
        text,
        flags=re.IGNORECASE,
    )
    # 3. camera-reveal clause that targets the off-frame character -> hold static.
    text = re.sub(
        rf"\b(?:pan(?:s|ning)?|tilt(?:s|ing)?|dolly(?:ing)?|truck(?:s|ing)?|"
        rf"crane(?:s|ing)?|whip[\s-]?pan\w*|cut(?:s|ting)?|reverse(?:\s+(?:angle|shot))?|"
        rf"camera\s+(?:moves?|swings?|pans?|tilts?|cuts?|shifts?))"
        rf"[^.,;]*?\b(?:to|toward|towards|over to|onto|reveal(?:ing)?)\b[^.,;]*?{target}[^.,;]*",
        "camera holds static",
        text,
        flags=re.IGNORECASE,
    )
    # 4. bare directional "toward/at/to <target>" -> toward off-frame space.
    text = re.sub(
        rf"\b(?:toward|towards|at|to|upon)\s+(?:the\s+)?{target}",
        "toward off-frame space",
        text,
        flags=re.IGNORECASE,
    )
    # 5. any leftover "off-frame <noun>" -> off-frame space.
    text = re.sub(
        rf"\boff[\s-]?(?:frame|screen)\s+(?:the\s+)?(?:young\s+|small\s+|other\s+)?{_PERSON_NOUN}",
        "off-frame space",
        text,
        flags=re.IGNORECASE,
    )
    # tidy whitespace and stray punctuation left behind
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r",\s*,", ",", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+\.", ".", text)
    return text


def cast_lock_block(visible_characters: Sequence[str], offscreen_present: bool) -> str:
    """An explicit instruction fixing exactly who may appear in the clip."""
    names = [n for n in (visible_characters or []) if n]
    if names:
        who = "; ".join(names)
        count_line = (
            f"Exactly {len(names)} character(s) exist in this entire clip: {who}."
        )
    else:
        count_line = "No human or animal characters exist in this entire clip."
    lines = [
        "[CAST LOCK]",
        f"- {count_line}",
        "- The camera never pans, tilts, dollies, cuts, or zooms to reveal anyone else.",
        "- No additional person, face, hand, body, or shadow of another character enters the frame at any point.",
    ]
    if offscreen_present:
        lines.append(
            "- Any character being spoken to or looked at stays entirely off-frame for the whole clip; "
            "only their direction is implied, they are never shown."
        )
    return "\n".join(lines)


def speaker_is_offscreen(
    speaker_block: str,
    shot: Any,
    characters: Sequence[Any],
) -> bool:
    """True when the active speaker is a character not visible in the keyframes."""
    if not speaker_block:
        return False
    full = speaker_block.strip()
    clean = re.sub(r"\s*\([^)]*\)", "", full).strip()
    if not clean or re.fullmatch(r"(?:narrator|người dẫn chuyện)", clean, re.IGNORECASE):
        return False
    visible = visible_char_idxs(shot)

    # Collect every character the speaker block could refer to, exact identifier
    # match first so "Alex (Cat)" never gets confused with "Alex (Human)".
    exact = [c for c in (characters or []) if _char_id(c).strip().lower() == full.lower()]
    candidates = exact
    if not candidates:
        for c in characters or []:
            for variant in _name_variants(_char_id(c)):
                if re.search(rf"\b{re.escape(variant)}\b", clean, re.IGNORECASE):
                    candidates.append(c)
                    break
    if not candidates:
        return False
    # If any matching character is on screen, the speaker is in frame.
    if any(_char_idx(c) in visible for c in candidates):
        return False
    return True


def visible_identifiers(shot: Any, characters: Sequence[Any]) -> List[str]:
    """Readable identifiers of the characters visible in the shot keyframes."""
    visible = visible_char_idxs(shot)
    out: List[str] = []
    for c in characters or []:
        idx = _char_idx(c)
        if idx is not None and idx in visible:
            ident = _char_id(c)
            if ident:
                out.append(ident)
    return out
