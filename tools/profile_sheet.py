"""One-page HTML review of character profiles (phase 05).

Renders one row per profile (name, aliases, status, a few source crops, outfit
labels) so the user can review the main cast once. Pure stdlib + inline HTML;
no server. Images are referenced by relative path next to the HTML file.
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Optional

from utils.profile_store import ProfileStore


def _img_tag(path: str, base: Path) -> str:
    if not path or not os.path.exists(path):
        return '<div class="noimg">—</div>'
    try:
        rel = os.path.relpath(path, base)
    except ValueError:
        rel = path
    return f'<img src="{html.escape(rel)}" loading="lazy" />'


def render(base_dir: str | Path, out_path: Optional[str | Path] = None, max_crops: int = 6) -> str:
    base = Path(base_dir)
    store = ProfileStore(base)
    out = Path(out_path) if out_path else base / "profiles" / "profiles_review.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for prof in sorted(store.all(), key=lambda p: p.character_id):
        crops = "".join(_img_tag(c.path, out.parent) for c in prof.source_crops[:max_crops])
        outfits = ", ".join(html.escape(o.label or o.outfit_id) for o in prof.outfits) or "—"
        aliases = html.escape(", ".join(prof.aliases)) or "—"
        rows.append(
            f'<tr><td class="name">{html.escape(prof.display_name)}<br><span class="st st-{prof.status}">'
            f'{prof.status}</span></td><td>{aliases}</td><td class="crops">{crops}</td>'
            f'<td>{outfits}</td></tr>'
        )

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Character profiles</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px;background:#faf9f7;color:#222}}
 table{{border-collapse:collapse;width:100%}}
 th,td{{border:1px solid #ddd;padding:8px;vertical-align:top;text-align:left}}
 th{{background:#f0ede8}}
 .crops img{{height:84px;margin:2px;border:1px solid #ccc;border-radius:4px}}
 .noimg{{display:inline-block;width:60px;height:84px;background:#eee;text-align:center;line-height:84px;color:#999}}
 .name{{font-weight:600;white-space:nowrap}}
 .st{{font-size:11px;padding:1px 6px;border-radius:8px}}
 .st-draft{{background:#fde8c7}} .st-approved{{background:#cde8c7}} .st-locked{{background:#c7d4e8}}
</style></head><body>
<h1>Character profiles ({len(rows)})</h1>
<table><thead><tr><th>Character</th><th>Aliases</th><th>Source crops</th><th>Outfits</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan=4>No profiles yet.</td></tr>'}</tbody></table>
</body></html>"""
    out.write_text(doc, encoding="utf-8")
    return str(out)


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "."
    print(render(base))
