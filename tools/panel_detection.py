"""Deterministic comic panel detection & reading order (OpenCV, CPU-only).

This complements the VLM reading pass in ``agents/manga_reader.py``:

- The VLM remains the semantic source of truth for beat interpretation.
- This module provides a fast, offline, deterministic geometric prior that the
  Vision Layer uses to (a) sanity-check / refine VLM ``crop_box_norm`` boxes and
  (b) establish a page-level reading order for multi-panel manga pages.

No new third-party dependency is required: only ``cv2`` + ``PIL`` + ``numpy``,
which are already part of the project environment. The algorithm follows the
classic contour/gutter heuristic popularized by open-source panel extractors
(e.g. Kumiko-style tools), implemented here from scratch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ReadingMode = Literal["auto", "rtl", "ltr"]

_MIN_PANEL_AREA_RATIO = 0.035   # ignore blobs smaller than 3.5% of the page
_MAX_PANELS = 24                # hard cap to avoid pathological over-segmentation


class PanelBox(BaseModel):
    """Detected panel rectangle in normalized 0..1000 coordinates.

    Uses the same ``[ymin, xmin, ymax, xmax]`` convention as
    ``MangaBeat.crop_box_norm`` so boxes can be compared directly.
    """

    ymin: int = Field(default=0)
    xmin: int = Field(default=0)
    ymax: int = Field(default=1000)
    xmax: int = Field(default=1000)
    order: int = Field(default=0, description="0-based reading-order index.")

    def as_list(self) -> List[int]:
        return [self.ymin, self.xmin, self.ymax, self.xmax]

    def area_ratio(self) -> float:
        return max(0.0, (self.ymax - self.ymin) * (self.xmax - self.xmin)) / (1000.0 * 1000.0)


def _clamp01k(value: int) -> int:
    return max(0, min(1000, int(value)))


def _box_from_rect(x: int, y: int, w: int, h: int, page_w: int, page_h: int) -> PanelBox:
    return PanelBox(
        ymin=_clamp01k(round(1000.0 * y / page_h)),
        xmin=_clamp01k(round(1000.0 * x / page_w)),
        ymax=_clamp01k(round(1000.0 * (y + h) / page_h)),
        xmax=_clamp01k(round(1000.0 * (x + w) / page_w)),
        order=0,
    )


def _iou(a: PanelBox, b: PanelBox) -> float:
    inter_y0, inter_x0 = max(a.ymin, b.ymin), max(a.xmin, b.xmin)
    inter_y1, inter_x1 = min(a.ymax, b.ymax), min(a.xmax, b.xmax)
    if inter_y1 <= inter_y0 or inter_x1 <= inter_x0:
        return 0.0
    inter = (inter_y1 - inter_y0) * (inter_x1 - inter_x0)
    area_a = (a.ymax - a.ymin) * (a.xmax - a.xmin)
    area_b = (b.ymax - b.ymin) * (b.xmax - b.xmin)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedup_nested(boxes: List[PanelBox]) -> List[PanelBox]:
    """Drop boxes almost fully contained in another (nested contour duplicates)."""
    kept: List[PanelBox] = []
    ranked = sorted(boxes, key=lambda b: -b.area_ratio())
    for box in ranked:
        contained = False
        for keeper in kept:
            if _iou(box, keeper) > 0.82:
                contained = True
                break
            if keeper.ymin <= box.ymin and keeper.xmin <= box.xmin and keeper.ymax >= box.ymax and keeper.xmax >= box.xmax:
                contained = True
                break
        if not contained:
            kept.append(box)
    return kept


def _assign_reading_order(boxes: List[PanelBox], mode: ReadingMode) -> List[PanelBox]:
    """Sort panels into page reading order.

    Row-band clustering with generous vertical overlap tolerance, then
    left-to-right (LTR: manhwa/webtoon, western comics) or right-to-left
    (RTL: Japanese manga) within each band.
    """
    direction: Literal["ltr", "rtl"]
    if mode == "auto":
        direction = "ltr"
    else:
        direction = mode

    def _row_key(box: PanelBox) -> int:
        return box.ymin

    bands: List[List[PanelBox]] = []
    for box in sorted(boxes, key=_row_key):
        placed = False
        for band in bands:
            band_top = min(b.ymin for b in band)
            band_bottom = max(b.ymax for b in band)
            overlap = min(band_bottom, box.ymax) - max(band_top, box.ymin)
            band_height = max(1, band_bottom - band_top)
            box_height = max(1, box.ymax - box.ymin)
            if overlap > 0.45 * min(band_height, box_height):
                band.append(box)
                placed = True
                break
        if not placed:
            bands.append([box])

    ordered: List[PanelBox] = []
    for band in bands:
        band.sort(key=(lambda b: -b.xmax) if direction == "rtl" else (lambda b: b.xmin))
        ordered.extend(band)
    for idx, box in enumerate(ordered):
        box.order = idx
    return ordered


def detect_panels(
    image_path: str | Path,
    *,
    reading_mode: ReadingMode = "auto",
    min_area_ratio: float = _MIN_PANEL_AREA_RATIO,
    max_panels: int = _MAX_PANELS,
) -> List[PanelBox]:
    """Detect panel rectangles on a single comic page image.

    Returns panels in reading order. When detection is inconclusive (e.g. a
    borderless webtoon strip), a single full-page panel is returned.
    """
    with Image.open(image_path) as raw_im:
        rgb = ImageOps.exif_transpose(raw_im).convert("RGB")
        try:
            arr = np.asarray(rgb)
        finally:
            rgb.close()

    page_h, page_w = arr.shape[:2]
    if page_h == 0 or page_w == 0:
        return [PanelBox(order=0)]

    # Downscale for speed, keeping aspect ratio.
    scale = 900.0 / max(page_h, page_w)
    if scale < 1.0:
        resized = cv2.resize(arr, (max(1, int(page_w * scale)), max(1, int(page_h * scale))), interpolation=cv2.INTER_AREA)
    else:
        resized = arr

    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    # Panels are dark regions separated by white gutters.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(page_w * page_h)
    candidates: List[PanelBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w <= 0 or h <= 0:
            continue
        fill_ratio = cv2.contourArea(contour) / float(w * h)
        if fill_ratio < 0.35:
            continue  # not a solid panel-shaped region
        # Translate candidate rect back to full-res coordinates.
        full_x = int(round(x / (scale if scale < 1.0 else 1.0)))
        full_y = int(round(y / (scale if scale < 1.0 else 1.0)))
        full_w = int(round(w / (scale if scale < 1.0 else 1.0)))
        full_h = int(round(h / (scale if scale < 1.0 else 1.0)))
        box = _box_from_rect(full_x, full_y, full_w, full_h, page_w, page_h)
        if box.area_ratio() >= min_area_ratio:
            candidates.append(box)

    panels = _dedup_nested(candidates)
    if len(panels) > max_panels:
        panels = sorted(panels, key=lambda b: -b.area_ratio())[:max_panels]
    if len(panels) == 0:
        panels = [PanelBox(order=0)]
    return _assign_reading_order(panels, reading_mode)


def refine_vlm_crop_box(
    vlm_box: Sequence[int],
    page_panels: Sequence[PanelBox],
    *,
    snap_iou: float = 0.55,
) -> List[int]:
    """Snap a VLM-emitted ``crop_box_norm`` onto the nearest detected panel when it overlaps well.

    Falls back to the (clamped) VLM box otherwise. Never invents a box that was
    not already supported by either the VLM or a detected panel.
    """
    try:
        ymin, xmin, ymax, xmax = (int(round(float(v))) for v in vlm_box)
    except (TypeError, ValueError):
        ymin, xmin, ymax, xmax = 0, 0, 1000, 1000
    if ymax - ymin < 20 or xmax - xmin < 20 or ymax <= ymin or xmax <= xmin:
        ymin, xmin, ymax, xmax = 0, 0, 1000, 1000
    ymin, xmin = max(0, min(1000, ymin)), max(0, min(1000, xmin))
    ymax, xmax = max(0, min(1000, ymax)), max(0, min(1000, xmax))
    proposed = PanelBox(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)

    best: tuple[float, PanelBox | None] = (0.0, None)
    for panel in page_panels:
        iou = _iou(proposed, panel)
        if iou > best[0]:
            best = (iou, panel)
    if best[1] is not None and best[0] >= snap_iou:
        return best[1].as_list()
    return [ymin, xmin, ymax, xmax]
