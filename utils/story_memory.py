"""Cross-chapter story memory (phase 09).

Stores three things under ``<base>/story_memory/``:
- ``facts.jsonl`` — canon events/meanings/relationships/abilities/world-rules/deaths,
  each citing real beat ids (transient scene details are NEVER stored here).
- ``chapters/chNN.json`` — per-chapter summary, open threads, cast.
- ``conflicts.json`` — suspected contradictions awaiting review.

Long-lived character state (outfits, injuries) lives in the profile store
(phase 05); this module references it rather than duplicating it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_FACT_KINDS = {"event", "dialogue_meaning", "relationship", "ability", "world_rule", "death"}


class StoryMemory:
    def __init__(self, base_dir: str | Path):
        self.base = Path(base_dir)
        self.root = self.base / "story_memory"
        self.facts_path = self.root / "facts.jsonl"
        self.chapters_dir = self.root / "chapters"
        self.conflicts_path = self.root / "conflicts.json"

    # ------------------------------------------------------------------ io
    def _ensure(self) -> None:
        self.chapters_dir.mkdir(parents=True, exist_ok=True)

    def all_facts(self) -> List[dict]:
        if not self.facts_path.is_file():
            return []
        out = []
        for line in self.facts_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def add_fact(self, fact: dict) -> Optional[dict]:
        """Append a fact. Returns the stored fact, or None if invalid (bad kind / no beats)."""
        kind = str(fact.get("kind", "")).lower()
        if kind not in _FACT_KINDS:
            return None
        if not fact.get("beat_ids"):
            return None
        self._ensure()
        stored = {
            "fact_id": fact.get("fact_id") or f"f{len(self.all_facts()) + 1:04d}",
            "chapter": int(fact.get("chapter", 0) or 0),
            "kind": kind,
            "text": str(fact.get("text", "")).strip(),
            "character_ids": list(fact.get("character_ids", []) or []),
            "beat_ids": list(fact.get("beat_ids", []) or []),
        }
        with self.facts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(stored, ensure_ascii=False) + "\n")
        return stored

    def add_chapter(self, chapter: int, summary: str, open_threads: Sequence[str], cast: Sequence[dict]) -> None:
        self._ensure()
        data = {
            "chapter": chapter,
            "summary": str(summary or "").strip(),
            "open_threads": list(open_threads or []),
            "cast": list(cast or []),
        }
        (self.chapters_dir / f"ch{chapter:02d}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def chapters(self) -> List[dict]:
        if not self.chapters_dir.is_dir():
            return []
        out = []
        for p in sorted(self.chapters_dir.glob("ch*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except ValueError:
                continue
        return out

    def add_conflict(self, conflict: dict) -> None:
        self._ensure()
        existing = []
        if self.conflicts_path.is_file():
            try:
                existing = json.loads(self.conflicts_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except ValueError:
                existing = []
        existing.append(conflict)
        self.conflicts_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    def conflicts(self) -> List[dict]:
        if not self.conflicts_path.is_file():
            return []
        try:
            data = json.loads(self.conflicts_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except ValueError:
            return []

    # -------------------------------------------------------------- queries
    def facts_for(self, character_ids: Sequence[str], limit: int = 20) -> List[dict]:
        wanted = {str(c) for c in (character_ids or [])}
        facts = self.all_facts()
        if wanted:
            facts = [f for f in facts if wanted & set(f.get("character_ids", []))]
        # most recent first
        facts.sort(key=lambda f: f.get("chapter", 0), reverse=True)
        return facts[:limit]

    def search(self, text: str, k: int = 10, text_embedder=None) -> List[dict]:
        facts = self.all_facts()
        if not facts:
            return []
        if text_embedder is not None:
            try:
                from tools.embeddings import cosine
                fact_vecs = text_embedder.embed_texts([f.get("text", "") for f in facts])
                qv = text_embedder.embed_texts([text or ""])[0]
                scored = sorted(zip(facts, fact_vecs), key=lambda t: cosine(qv, t[1]), reverse=True)
                return [f for f, _v in scored[:k]]
            except Exception:
                pass
        # fallback: naive token overlap, recent first
        q = set((text or "").lower().split())
        facts.sort(key=lambda f: (len(q & set(str(f.get("text", "")).lower().split())), f.get("chapter", 0)), reverse=True)
        return facts[:k]

    def story_so_far(self, max_words: int = 300) -> str:
        """Concatenate chapter summaries (most recent last), truncated to max_words."""
        parts = [c.get("summary", "") for c in self.chapters() if c.get("summary")]
        text = " ".join(parts).strip()
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]).rstrip() + " …"

    def known_cast(self) -> List[dict]:
        seen: Dict[str, set] = {}
        for ch in self.chapters():
            for c in ch.get("cast", []):
                cid = c.get("character_id", "")
                if not cid:
                    continue
                seen.setdefault(cid, set()).update(c.get("aliases_used", []) or [])
        return [{"character_id": k, "aliases_used": sorted(v)} for k, v in sorted(seen.items())]


def format_known_cast_block(memory: StoryMemory) -> str:
    cast = memory.known_cast()
    if not cast:
        return ""
    lines = [f"- {c['character_id']}: {', '.join(c['aliases_used']) or '(no aliases yet)'}" for c in cast]
    return "Known cast (keep these names/aliases stable across chapters):\n" + "\n".join(lines)
