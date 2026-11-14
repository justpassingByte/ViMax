import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.panel_detection import PanelBox, detect_panels, refine_vlm_crop_box


def _save_two_stacked_panels(path: Path, *, width: int = 800, height: int = 1200) -> None:
    """A classic manga page: two solid dark panels separated by a white gutter."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    px = img.load()
    for x0, y0, x1, y1 in [(40, 60, width - 40, 560), (40, 640, width - 40, 1140)]:
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = (25, 25, 35)
    img.save(path, format="PNG")
    img.close()


def _save_two_side_by_side_panels(path: Path, *, width: int = 1200, height: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    px = img.load()
    for x0, y0, x1, y1 in [(40, 40, 560, height - 40), (640, 40, width - 40, height - 40)]:
        for y in range(y0, y1):
            for x in range(x0, x1):
                px[x, y] = (30, 40, 55)
    img.save(path, format="PNG")
    img.close()


class PanelDetectionTests(unittest.TestCase):
    def test_detects_two_stacked_panels_in_reading_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "page.png"
            _save_two_stacked_panels(page)
            panels = detect_panels(page)
            self.assertEqual(len(panels), 2)
            self.assertEqual([p.order for p in panels], [0, 1])
            # Top panel comes first; both boxes stay within the normalized 0..1000 space.
            self.assertLess(panels[0].ymin, 500)
            self.assertGreater(panels[1].ymin, 450)
            for p in panels:
                self.assertTrue(0 <= p.ymin <= 1000 and 0 <= p.xmin <= 1000)
                self.assertTrue(0 <= p.ymax <= 1000 and 0 <= p.xmax <= 1000)
                self.assertGreaterEqual(p.area_ratio(), 0.035)

    def test_reading_order_ltr_vs_rtl(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "page.png"
            _save_two_side_by_side_panels(page)
            ltr = detect_panels(page, reading_mode="ltr")
            rtl = detect_panels(page, reading_mode="rtl")
            self.assertEqual(len(ltr), 2)
            self.assertEqual(len(rtl), 2)
            self.assertLess(ltr[0].xmin, ltr[1].xmin, "LTR: left panel read first")
            self.assertGreater(rtl[0].xmin, rtl[1].xmin, "RTL: right panel read first")

    def test_blank_page_returns_single_full_page_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "blank.png"
            img = Image.new("RGB", (600, 800), (255, 255, 255))
            img.save(page)
            img.close()
            panels = detect_panels(page)
            self.assertEqual(len(panels), 1)
            self.assertEqual(panels[0].as_list(), [0, 0, 1000, 1000])
            self.assertEqual(panels[0].order, 0)


class RefineVlmCropBoxTests(unittest.TestCase):
    def test_snaps_close_vlm_box_onto_detected_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            page = Path(tmp) / "page.png"
            _save_two_stacked_panels(page)
            panels = detect_panels(page)
            vlm_box = [60, 60, 460, 940]  # slightly sloppy estimate of the top panel
            snapped = refine_vlm_crop_box(vlm_box, panels)
            self.assertEqual(snapped, panels[0].as_list())

    def test_degenerate_box_falls_back_to_full_page(self):
        self.assertEqual(refine_vlm_crop_box([5, 5, 6, 7], []), [0, 0, 1000, 1000])
        self.assertEqual(refine_vlm_crop_box("not-a-box", []), [0, 0, 1000, 1000])

    def test_low_overlap_box_is_clamped_but_not_snapped(self):
        panels = [PanelBox(ymin=0, xmin=0, ymax=400, xmax=400)]
        # Overlaps the panel only slightly -> must NOT snap, but stays clamped.
        out = refine_vlm_crop_box([500, 500, 900, 900], panels)
        self.assertEqual(out, [500, 500, 900, 900])

    def test_out_of_range_coordinates_are_clamped(self):
        panels = [PanelBox(ymin=0, xmin=0, ymax=1000, xmax=1000)]
        out = refine_vlm_crop_box([-50, -50, 1500, 1500], panels)
        self.assertEqual(out, [0, 0, 1000, 1000])


if __name__ == "__main__":
    unittest.main()
