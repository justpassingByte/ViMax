"""Character profile store (phase 05).

Profiles live under ``<base>/profiles/<character_id>/profile.json`` (plus
``sheets/`` and ``crops/`` folders). The store loads/saves them, resolves the
outfit and states in effect at a given beat, and matches a new roster candidate
to an existing profile (exact alias, auto-merge, or a mid-zone suggestion),
never merging two names that have co-occurred in the same panel.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from interfaces.profile import CharacterProfile, Outfit, SourceCrop, StateEntry

_BEAT_RE = re.compile(r"ch0*([0-9]+)_b0*([0-9]+)", re.IGNORECASE)

# Match decision thresholds (cosine-ish scores in 0..1).
AUTO_MERGE_SCORE = 0.86
SUGGEST_SCORE = 0.70


def beat_key(beat_id: str) -> Tuple[int, int]:
    """Parse 'ch01_b012' -> (1, 12) for ordered comparison. Unknown -> (0, 0)."""
    m = _BEAT_RE.search(str(beat_id or ""))
    if not m:
        return (0, 0)
    return (int(m.group(1)), int(m.group(2)))


def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip().lower())


class ProfileStore:
    def __init__(self, base_dir: str | Path):
        self.base = Path(base_dir)
        self.root = self.base / "profiles"
        self._profiles: Dict[str, CharacterProfile] = {}
        self.load_all()

    # ------------------------------------------------------------------ io
    def load_all(self) -> Dict[str, CharacterProfile]:
        self._profiles = {}
        if self.root.is_dir():
            for pdir in sorted(self.root.iterdir()):
                pfile = pdir / "profile.json"
                if pfile.is_file():
                    try:
                        prof = CharacterProfile.model_validate_json(pfile.read_text(encoding="utf-8"))
                        self._profiles[prof.character_id] = prof
                    except Exception:
                        continue
        return self._profiles

    def all(self) -> List[CharacterProfile]:
        return list(self._profiles.values())

    def get(self, name_or_alias: str) -> Optional[CharacterProfile]:
        needle = _norm(name_or_alias)
        if not needle:
            return None
        for prof in self._profiles.values():
            if prof.character_id == name_or_alias:
                return prof
            names = {_norm(prof.display_name), _norm(prof.character_id)}
            names.update(_norm(a) for a in prof.aliases)
            if needle in names:
                return prof
        return None

    def upsert(self, profile: CharacterProfile) -> CharacterProfile:
        """Save a profile. A ``locked`` profile only accepts added source_crops/states."""
        existing = self._profiles.get(profile.character_id)
        if existing is not None and existing.status == "locked":
            merged = existing.model_copy(deep=True)
            seen = {c.path for c in merged.source_crops}
            for c in profile.source_crops:
                if c.path not in seen:
                    merged.source_crops.append(c)
                    seen.add(c.path)
            seen_states = {(s.kind, s.text, s.from_beat) for s in merged.states}
            for s in profile.states:
                key = (s.kind, s.text, s.from_beat)
                if key not in seen_states:
                    merged.states.append(s)
                    seen_states.add(key)
            profile = merged
        self._profiles[profile.character_id] = profile
        self._write(profile)
        return profile

    def _write(self, profile: CharacterProfile) -> None:
        pdir = self.root / profile.character_id
        (pdir / "sheets").mkdir(parents=True, exist_ok=True)
        (pdir / "crops").mkdir(parents=True, exist_ok=True)
        (pdir / "profile.json").write_text(profile.model_dump_json(indent=2), encoding="utf-8")

    # -------------------------------------------------------------- queries
    def outfit_at(self, character_id: str, beat_id: str) -> Optional[Outfit]:
        prof = self._profiles.get(character_id) or self.get(character_id)
        if prof is None or not prof.outfits:
            return None
        bk = beat_key(beat_id)
        best: Optional[Outfit] = None
        best_from = (-1, -1)
        for o in prof.outfits:
            fk = beat_key(o.valid_from_beat)
            tk = beat_key(o.valid_to_beat) if o.valid_to_beat else None
            if fk <= bk and (tk is None or bk <= tk):
                if fk >= best_from:
                    best = o
                    best_from = fk
        # Fall back to the most recent outfit that started on/before this beat.
        if best is None:
            candidates = [o for o in prof.outfits if beat_key(o.valid_from_beat) <= bk]
            if candidates:
                best = max(candidates, key=lambda o: beat_key(o.valid_from_beat))
        return best

    def states_at(self, character_id: str, beat_id: str) -> List[StateEntry]:
        prof = self._profiles.get(character_id) or self.get(character_id)
        if prof is None:
            return []
        bk = beat_key(beat_id)
        out = []
        for s in prof.states:
            fk = beat_key(s.from_beat)
            tk = beat_key(s.to_beat) if s.to_beat else None
            if fk <= bk and (tk is None or bk <= tk):
                out.append(s)
        return out

    def add_crops(self, character_id: str, crops: Sequence[SourceCrop]) -> None:
        prof = self._profiles.get(character_id)
        if prof is None:
            return
        seen = {c.path for c in prof.source_crops}
        for c in crops:
            if c.path not in seen:
                prof.source_crops.append(c)
                seen.add(c.path)
        self.upsert(prof)

    # --------------------------------------------------------------- match
    def match(
        self,
        candidate_name: str,
        candidate_aliases: Optional[Sequence[str]] = None,
        candidate_beats: Optional[Sequence[str]] = None,
        score_fn=None,
    ) -> Tuple[Optional[str], float, str]:
        """Match a roster candidate to an existing profile.

        Returns (profile_id|None, score, decision) where decision is
        'exact' | 'auto_merge' | 'suggest' | 'new'. Two names that have ever
        co-occurred in the same beat/panel are never merged.
        """
        cand_names = {_norm(candidate_name)} | {_norm(a) for a in (candidate_aliases or [])}
        cand_beats = {str(b) for b in (candidate_beats or [])}

        # 1. exact alias/name match
        for prof in self._profiles.values():
            names = {_norm(prof.display_name), _norm(prof.character_id)} | {_norm(a) for a in prof.aliases}
            if cand_names & names:
                return (prof.character_id, 1.0, "exact")

        # 2. scored match (optional embedding/text score function)
        best_id = None
        best_score = 0.0
        for prof in self._profiles.values():
            if cand_beats and set(prof.cooccur_beat_ids) & cand_beats:
                continue  # never merge characters seen together in one panel
            score = float(score_fn(prof)) if score_fn else 0.0
            if score > best_score:
                best_score = score
                best_id = prof.character_id
        if best_id is not None and best_score >= AUTO_MERGE_SCORE:
            return (best_id, best_score, "auto_merge")
        if best_id is not None and best_score >= SUGGEST_SCORE:
            return (best_id, best_score, "suggest")
        return (None, best_score, "new")

    # -------------------------------------------------------- status ops
    def set_status(self, character_id: str, status: str) -> bool:
        prof = self._profiles.get(character_id)
        if prof is None or status not in {"draft", "approved", "locked"}:
            return False
        prof.status = status  # type: ignore[assignment]
        self._profiles[character_id] = prof
        self._write(prof)
        return True

    def merge(self, keep_id: str, drop_id: str) -> bool:
        keep = self._profiles.get(keep_id)
        drop = self._profiles.get(drop_id)
        if keep is None or drop is None or keep.status == "locked":
            return False
        for a in [drop.display_name, drop.character_id, *drop.aliases]:
            if a and _norm(a) not in {_norm(x) for x in [keep.display_name, keep.character_id, *keep.aliases]}:
                keep.aliases.append(a)
        seen = {c.path for c in keep.source_crops}
        for c in drop.source_crops:
            if c.path not in seen:
                keep.source_crops.append(c)
        keep.cooccur_beat_ids = sorted(set(keep.cooccur_beat_ids) | set(drop.cooccur_beat_ids))
        self.upsert(keep)
        try:
            import shutil
            shutil.rmtree(self.root / drop_id, ignore_errors=True)
        except Exception:
            pass
        self._profiles.pop(drop_id, None)
        return True


def _cli(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m utils.profile_store <approve|lock|merge|list> [id] [id2] [--base DIR]")
        return 2
    base = "."
    args = list(argv)
    if "--base" in args:
        i = args.index("--base")
        base = args[i + 1]
        del args[i:i + 2]
    cmd = args[0]
    store = ProfileStore(base)
    if cmd == "list":
        for p in store.all():
            print(f"{p.character_id}\t{p.status}\taliases={p.aliases}")
        return 0
    if cmd in {"approve", "lock"} and len(args) >= 2:
        ok = store.set_status(args[1], "approved" if cmd == "approve" else "locked")
        print("ok" if ok else "not found")
        return 0 if ok else 1
    if cmd == "merge" and len(args) >= 3:
        ok = store.merge(args[1], args[2])
        print("merged" if ok else "failed")
        return 0 if ok else 1
    print("unknown command or missing args")
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
