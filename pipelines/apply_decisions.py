"""Apply a chapter's review decisions (phase 10).

Reads ``exceptions/chNN.decisions.json`` and applies each decision:
- ``retake``: delete the shot's keyframe(s) so the next render regenerates only
  that shot (the existing resume logic skips intact files).
- ``merge``: merge two character profiles via the profile store.
- ``accept`` / ``skip``: mark reviewed, no change.

Usage: ``python -m pipelines.apply_decisions <work_root> chNN``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def apply_decisions(work_root: str | Path, chapter_tag: str) -> Dict[str, Any]:
    work_root = Path(work_root)
    dec_path = work_root / "exceptions" / f"{chapter_tag}.decisions.json"
    decisions = _load(dec_path, [])
    result = {"retakes": 0, "merges": 0, "accepted": 0, "skipped": 0, "deleted_frames": []}
    if not isinstance(decisions, list):
        return result

    store = None
    for d in decisions:
        action = str(d.get("action", "")).lower()
        if action == "retake":
            scene = d.get("scene", "")
            shot_idx = d.get("shot_idx")
            frame = d.get("frame", "first")
            fname = "first_frame.png" if frame != "last" else "last_frame.png"
            fp = work_root / scene / "shots" / str(shot_idx) / fname
            if fp.is_file():
                try:
                    fp.unlink()
                    result["deleted_frames"].append(str(fp))
                except OSError:
                    pass
            result["retakes"] += 1
        elif action == "merge":
            if store is None:
                from utils.profile_store import ProfileStore
                store = ProfileStore(str(work_root))
            if store.merge(d.get("keep_id", ""), d.get("drop_id", "")):
                result["merges"] += 1
        elif action == "accept":
            result["accepted"] += 1
        else:
            result["skipped"] += 1

    # record that decisions were applied
    (work_root / "exceptions" / f"{chapter_tag}.applied.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main(argv: List[str]) -> int:
    if len(argv) < 2:
        print("usage: python -m pipelines.apply_decisions <work_root> chNN")
        return 2
    res = apply_decisions(argv[0], argv[1])
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
