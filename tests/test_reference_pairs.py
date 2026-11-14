import json
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pipelines.script2video_pipeline import (
    _ensure_character_portraits_and_spatial_anchors,
    _write_refs_used,
)
from utils import reference_sheet


class _Char:
    def __init__(self, ident, visible=True):
        self.identifier_in_scene = ident
        self.is_visible = visible


def _png(path, color=(200, 100, 50), size=(64, 48)):
    Image.new("RGB", size, color).save(path)
    return path


def _registry(tmp, names):
    reg = {}
    for n in names:
        p = os.path.join(tmp, f"{n}.png")
        _png(p)
        reg[n] = {"front": {"path": p, "description": f"{n} front portrait"}}
    return reg


class ReferencePairsTests(unittest.TestCase):
    def test_three_chars_max4_gives_three_portraits_plus_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = _registry(tmp, ["A", "B", "C"])
            chars = [_Char("A"), _Char("B"), _Char("C")]
            anchor = (_png(os.path.join(tmp, "anchor.png")), "scene establishing anchor")
            pairs = _ensure_character_portraits_and_spatial_anchors(
                pairs=[], visible_characters=chars, character_portraits_registry=reg,
                scene_anchor_pair=anchor, limit=4, sheet_dir=tmp, sheet_tag="ff",
            )
            paths = [p for p, _t in pairs]
            # 3 portraits + 1 anchor, no sheet
            self.assertEqual(len(pairs), 4)
            self.assertFalse(any("reference_sheet" in p for p in paths))
            self.assertIn(anchor[0], paths)

    def test_three_chars_max2_composes_sheet_plus_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = _registry(tmp, ["A", "B", "C"])
            chars = [_Char("A"), _Char("B"), _Char("C")]
            anchor = (_png(os.path.join(tmp, "anchor.png")), "scene establishing anchor")
            pairs = _ensure_character_portraits_and_spatial_anchors(
                pairs=[], visible_characters=chars, character_portraits_registry=reg,
                scene_anchor_pair=anchor, limit=2, sheet_dir=tmp, sheet_tag="ff",
            )
            paths = [p for p, _t in pairs]
            self.assertTrue(any("reference_sheet" in p for p in paths), msg=str(paths))
            self.assertLessEqual(len(pairs), 2)
            self.assertIn(anchor[0], paths)

    def test_legacy_no_sheet_even_with_many_chars(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = _registry(tmp, ["A", "B", "C"])
            chars = [_Char("A"), _Char("B"), _Char("C")]
            pairs = _ensure_character_portraits_and_spatial_anchors(
                pairs=[], visible_characters=chars, character_portraits_registry=reg,
                scene_anchor_pair=None, limit=2, sheet_dir=None,
            )
            self.assertFalse(any("reference_sheet" in p for p, _t in pairs))

    def test_compose_sheet_dimensions_and_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries = [(_png(os.path.join(tmp, f"{n}.png")), n) for n in ("A", "B", "C")]
            sheet = reference_sheet.compose(entries, cell=100, pad=10, label_h=20)
            # 3 cells of 100 + 4 pads of 10
            self.assertEqual(sheet.width, 3 * 100 + 4 * 10)
            label = reference_sheet.order_label(entries)
            self.assertLess(label.index("1) A"), label.index("2) B"))
            self.assertLess(label.index("2) B"), label.index("3) C"))

    def test_write_refs_used_records_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = _registry(tmp, ["A"])
            chars = [_Char("A")]
            portrait_path = reg["A"]["front"]["path"]
            panel_path = os.path.join(tmp, "panel_anchors", "p.jpg")
            os.makedirs(os.path.dirname(panel_path), exist_ok=True)
            _png(panel_path)
            pairs = [
                (portrait_path, "Canonical character portrait reference for <A>"),
                (panel_path, "original comic panel"),
                (os.path.join(tmp, "anchor.png"), "scene establishing anchor from Shot 2"),
            ]
            _png(pairs[2][0])
            _write_refs_used(tmp, 7, "first_frame", pairs, chars, reg)
            data = json.loads(Path(tmp, "shots", "7", "first_frame_refs_used.json").read_text(encoding="utf-8"))
            roles = [r["role"] for r in data["refs"]]
            self.assertEqual(roles[0], "portrait")
            self.assertEqual(roles[1], "panel")
            self.assertEqual(roles[2], "scene_anchor")
            self.assertEqual(data["n_present"], 1)
            self.assertEqual(data["n_portrait_refs"], 1)


if __name__ == "__main__":
    unittest.main()
