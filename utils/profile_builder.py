"""Build draft character profiles from the reader roster (phase 05).

Lightweight and fail-soft: creates/updates one draft profile per roster
character using names/aliases, permanent features, seed crops, series-bible
notes and wardrobe rules, and records which beats each character co-occurs in
(so matching never merges two characters seen in the same panel). It does not
generate new portrait sheets; those stay the existing registry portraits.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from interfaces.profile import CharacterProfile, Outfit, SourceCrop


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    return s or "character"


def _first_beat(beats_by_scene: Dict[int, List[Any]]) -> str:
    for _idx in sorted(beats_by_scene or {}):
        for b in beats_by_scene[_idx]:
            bid = getattr(b, "beat_id", "")
            if bid:
                return bid
    return ""


def _cooccurrence(beats_by_scene: Dict[int, List[Any]]) -> Dict[str, List[str]]:
    """character_name -> list of beat_ids where >=2 named characters appear together."""
    out: Dict[str, set] = {}
    for beats in (beats_by_scene or {}).values():
        for b in beats:
            present = [str(c) for c in getattr(b, "characters_present", []) or []]
            if len(present) >= 2:
                bid = getattr(b, "beat_id", "")
                for name in present:
                    out.setdefault(name, set()).add(bid)
    return {k: sorted(v) for k, v in out.items()}


def build_draft_profiles(
    characters: Sequence[Any],
    character_seed_refs: Dict[str, List[str]],
    beats_by_scene: Dict[int, List[Any]],
    store,
    series_bible=None,
) -> List[str]:
    """Create/update draft profiles. Returns the list of profile ids touched."""
    touched: List[str] = []
    cooccur = _cooccurrence(beats_by_scene)
    default_from = _first_beat(beats_by_scene) or "ch01_b000"

    for ch in characters or []:
        ident = getattr(ch, "identifier_in_scene", "") or getattr(ch, "name", "")
        if not ident:
            continue
        cid = _slug(ident)
        existing = store.get(ident)
        if existing is not None and existing.status == "locked":
            touched.append(existing.character_id)
            continue

        aliases = [ident]
        notes = ""
        if series_bible is not None and getattr(series_bible, "available", False):
            notes = series_bible.character_notes(ident) or ""

        crops = [
            SourceCrop(path=p, beat_id="", form_id="", outfit_id="o_default")
            for p in (character_seed_refs.get(ident) or [])
        ]

        # One default outfit; series wardrobe text (if any) becomes its description.
        outfit = Outfit(
            outfit_id="o_default",
            label="default",
            description=notes,
            valid_from_beat=default_from,
            valid_to_beat=None,
        )

        prof = CharacterProfile(
            character_id=cid,
            display_name=ident,
            aliases=aliases,
            permanent_features=str(getattr(ch, "static_features", "") or ""),
            outfits=[outfit],
            source_crops=crops,
            prompt_notes=notes,
            cooccur_beat_ids=cooccur.get(ident, []),
            status="draft",
        )
        if existing is not None:
            # preserve prior status/outfits structure where present; just refresh crops/notes/cooccur
            prof.status = existing.status
            if existing.outfits:
                prof.outfits = existing.outfits
        store.upsert(prof)
        touched.append(cid)
    return touched
