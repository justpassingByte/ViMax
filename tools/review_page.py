"""One-page HTML review of a chapter's exceptions (phase 10).

Renders the aggregated ``exceptions/chNN.json`` so the user can review a whole
chapter in one place (keyframe/clip exceptions, profile-merge suggestions,
story conflicts, repaired canon reports). Pure stdlib + inline HTML, no server.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Optional


def _section(title: str, items, render_item) -> str:
    if not items:
        return f"<section><h2>{html.escape(title)} <small>(0)</small></h2><p class=empty>None.</p></section>"
    rows = "".join(render_item(it) for it in items)
    return f"<section><h2>{html.escape(title)} <small>({len(items)})</small></h2>{rows}</section>"


def render(work_root: str | Path, chapter_tag: str, out_path: Optional[str | Path] = None) -> str:
    work_root = Path(work_root)
    data_path = work_root / "exceptions" / f"{chapter_tag}.json"
    try:
        data: Any = json.loads(data_path.read_text(encoding="utf-8")) if data_path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    out = Path(out_path) if out_path else work_root / "exceptions" / f"{chapter_tag}.review.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    def kf(it):
        reasons = html.escape("; ".join(it.get("reasons", [])))
        return (f'<div class=item><b>{html.escape(str(it.get("scene","")))} '
                f'shot {html.escape(str(it.get("shot_idx","")))}</b> ({html.escape(str(it.get("frame","")))})<br>'
                f'<span class=why>{reasons}</span></div>')

    def merge(it):
        return f'<div class=item>suggest merge: {html.escape(json.dumps(it, ensure_ascii=False))}</div>'

    def conflict(it):
        return f'<div class=item>conflict: {html.escape(json.dumps(it.get("new", it), ensure_ascii=False))}</div>'

    def canon(it):
        n = len(it.get("issues", []))
        return f'<div class=item><b>{html.escape(str(it.get("scene","")))}</b>: {n} canon issue(s), passed={it.get("passed")}</div>'

    body = (
        _section("Keyframe exceptions", data.get("keyframe_exceptions", []), kf)
        + _section("Clip exceptions", data.get("clip_exceptions", []), kf)
        + _section("Profile merge suggestions", data.get("profile_merge_suggestions", []), merge)
        + _section("Story conflicts", data.get("story_conflicts", []), conflict)
        + _section("Canon repairs", data.get("canon_repairs", []), canon)
    )
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Review {html.escape(chapter_tag)}</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;background:#faf9f7;color:#222}}
 h1{{margin-bottom:4px}} section{{margin:18px 0}} h2{{font-size:16px;border-bottom:1px solid #ddd;padding-bottom:4px}}
 .item{{padding:8px;border:1px solid #e3e0da;border-radius:6px;margin:6px 0;background:#fff}}
 .why{{color:#a33}} .empty{{color:#999}} small{{color:#888;font-weight:normal}}
</style></head><body>
<h1>Chapter review — {html.escape(chapter_tag)}</h1>
<p>Total items: {data.get("total", 0)}</p>
{body}
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m tools.review_page <work_root> chNN")
        raise SystemExit(2)
    print(render(sys.argv[1], sys.argv[2]))
