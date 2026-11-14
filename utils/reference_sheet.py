"""Compose several character portraits into one labelled reference sheet.

Used when a shot has more present characters than the image model's reference
slot budget: instead of dropping characters, their portraits are tiled left to
right into a single sheet image (neutral background, small name label under
each), so every character still reaches the model in one reference slot.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

_BG = (245, 245, 245)
_LABEL_BG = (30, 30, 30)
_LABEL_FG = (255, 255, 255)


def _load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def compose(
    entries: Sequence[Tuple[str, str]],
    *,
    cell: int = 384,
    pad: int = 12,
    label_h: int = 28,
) -> Image.Image:
    """Tile (path, label) portraits horizontally into one sheet image.

    Entries whose image can't be opened are skipped. Order is preserved.
    """
    loaded: List[Tuple[Image.Image, str]] = []
    for path, label in entries:
        try:
            with Image.open(path) as im:
                loaded.append((im.convert("RGB").copy(), str(label or "")))
        except Exception:
            continue
    if not loaded:
        # Return a small blank sheet rather than crashing.
        return Image.new("RGB", (cell, cell + label_h), _BG)

    n = len(loaded)
    cell_w = cell
    cell_h = cell
    sheet_w = n * cell_w + (n + 1) * pad
    sheet_h = cell_h + label_h + 2 * pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), _BG)
    draw = ImageDraw.Draw(sheet)
    font = _load_font(18)

    x = pad
    for im, label in loaded:
        im_fit = im.copy()
        im_fit.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        off_x = x + (cell_w - im_fit.width) // 2
        off_y = pad + (cell_h - im_fit.height) // 2
        sheet.paste(im_fit, (off_x, off_y))
        # label strip
        ly0 = pad + cell_h
        draw.rectangle([x, ly0, x + cell_w, ly0 + label_h], fill=_LABEL_BG)
        text = label if len(label) <= 28 else label[:27] + "…"
        try:
            tw = draw.textlength(text, font=font)
        except Exception:
            tw = len(text) * 9
        draw.text((x + max(4, (cell_w - tw) // 2), ly0 + 5), text, fill=_LABEL_FG, font=font)
        x += cell_w + pad

    return sheet


def compose_to_file(entries: Sequence[Tuple[str, str]], out_path: str | Path, **kwargs) -> Optional[str]:
    """Compose and save a sheet; returns the saved path, or None if nothing to compose."""
    valid = [(p, l) for p, l in entries if p and Path(p).exists()]
    if not valid:
        return None
    sheet = compose(valid, **kwargs)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(out))
    return str(out)


def order_label(entries: Sequence[Tuple[str, str]]) -> str:
    """Left-to-right description listing the characters on a composed sheet."""
    names = [str(l).strip() for _p, l in entries if str(l).strip()]
    if not names:
        return ""
    ordered = ", ".join(f"{i + 1}) {n}" for i, n in enumerate(names))
    return (
        "Character turnaround reference sheet (one neutral-background portrait per character, "
        f"left to right: {ordered}). Match each character's exact face, hair, and wardrobe; "
        "do NOT copy the sheet background or labels."
    )
