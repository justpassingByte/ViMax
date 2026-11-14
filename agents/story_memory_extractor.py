"""Per-chapter story-memory extraction (phase 09).

Runs once per chapter after the beats are read: an injected LLM returns facts,
state_changes, a summary and open threads; pure-code validation drops any fact
that doesn't cite a real beat of this chapter, applies validated state changes to
the profile store (closing the previous same-kind state), and records suspected
contradictions without blocking the pipeline.

The LLM is dependency-injected (``extract_fn`` / ``conflict_fn``) so this is
testable offline with fakes and never forces a network call.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from interfaces.profile import StateEntry

logger = logging.getLogger(__name__)


def _beat_id(b) -> str:
    return b.get("beat_id", "") if isinstance(b, dict) else getattr(b, "beat_id", "")


def extract_chapter_memory(
    chapter: int,
    beats: Sequence[Any],
    memory,
    *,
    extract_fn: Callable[..., Dict[str, Any]],
    prev_summary: str = "",
    profile_store=None,
    conflict_fn: Optional[Callable[..., bool]] = None,
    conflict_topk: int = 5,
) -> Dict[str, Any]:
    """Extract and persist one chapter's memory. Returns a summary dict.

    ``extract_fn(beats, prev_summary) -> {"facts": [...], "state_changes": [...],
    "summary": str, "open_threads": [...], "cast": [...]}``.
    """
    valid_beats = {_beat_id(b) for b in beats if _beat_id(b)}
    try:
        result = extract_fn(beats=beats, prev_summary=prev_summary) or {}
    except Exception as exc:  # noqa: BLE001 - never fatal
        logger.warning("story memory extraction failed for ch%s: %s", chapter, exc)
        return {"facts_added": 0, "facts_dropped": 0, "state_changes": 0, "conflicts": 0}

    facts_added = 0
    facts_dropped = 0
    stored_facts: List[dict] = []
    for fact in result.get("facts", []) or []:
        fact = dict(fact)
        fact.setdefault("chapter", chapter)
        cited = [b for b in (fact.get("beat_ids") or []) if b in valid_beats]
        if not cited:
            facts_dropped += 1
            continue
        fact["beat_ids"] = cited
        stored = memory.add_fact(fact)
        if stored is None:
            facts_dropped += 1
        else:
            facts_added += 1
            stored_facts.append(stored)

    # State changes -> profile store (with evidence beat), closing old same-kind state.
    state_changes = 0
    if profile_store is not None:
        for sc in result.get("state_changes", []) or []:
            evidence = [b for b in (sc.get("evidence_beat_ids") or []) if b in valid_beats]
            if not evidence:
                continue
            cid = sc.get("character_id")
            prof = profile_store.get(cid) if cid else None
            if prof is None:
                continue
            kind = sc.get("kind", "other")
            from_beat = sc.get("from_beat") or evidence[0]
            # close previous open state of same kind
            for s in prof.states:
                if s.kind == kind and s.to_beat is None:
                    s.to_beat = from_beat
            prof.states.append(StateEntry(
                kind=kind, text=sc.get("text", ""), from_beat=from_beat,
                to_beat=sc.get("to_beat"), evidence_beat_ids=evidence,
            ))
            profile_store.upsert(prof)
            state_changes += 1

    # Conflict detection against prior facts for the same characters.
    conflicts = 0
    if conflict_fn is not None:
        for fact in stored_facts:
            prior = [
                f for f in memory.facts_for(fact.get("character_ids", []), limit=conflict_topk)
                if f.get("fact_id") != fact.get("fact_id")
            ]
            if not prior:
                continue
            try:
                if conflict_fn(new_fact=fact, prior_facts=prior):
                    memory.add_conflict({"new": fact, "prior": prior})
                    conflicts += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("conflict check failed: %s", exc)

    memory.add_chapter(
        chapter,
        result.get("summary", ""),
        result.get("open_threads", []),
        result.get("cast", []),
    )
    return {
        "facts_added": facts_added,
        "facts_dropped": facts_dropped,
        "state_changes": state_changes,
        "conflicts": conflicts,
    }
