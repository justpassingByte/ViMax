import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.embeddings import build_image_embedder, build_text_embedder, cosine
from agents.identity_dedup import audit_seed_crops, filter_seed_refs


class FakeImageEmbedder:
    """Deterministic fake: vector keyed by a tag embedded in the file name."""
    name = "fake"

    def __init__(self, mapping):
        self.mapping = mapping  # path -> vector list

    def embed_images(self, paths):
        return np.asarray([self.mapping[p] for p in paths], dtype="float32")


def _png(path, color=(10, 20, 30)):
    Image.new("RGB", (8, 8), color).save(path)
    return path


class EmbeddingBuilderTests(unittest.TestCase):
    def test_off_and_unknown_return_none(self):
        self.assertIsNone(build_image_embedder("off"))
        self.assertIsNone(build_text_embedder("off"))
        self.assertIsNone(build_image_embedder("unknown"))

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine([0, 0], [1, 1]), 0.0)


class AuditSeedCropsTests(unittest.TestCase):
    def test_detects_hash_duplicate_between_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            # identical bytes for A and B
            same = _png(os.path.join(tmp, "same_A.png"), (1, 2, 3))
            b_copy = os.path.join(tmp, "same_B.png")
            Path(b_copy).write_bytes(Path(same).read_bytes())
            seed_refs = {"A": [same], "B": [b_copy]}
            report = audit_seed_crops(seed_refs, panel_dir=None, image_embedder=None)
            self.assertTrue(report["hash_duplicate_between_characters"])
            self.assertIn(b_copy, report["flagged"])

    def test_detects_hash_duplicate_with_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel_dir = os.path.join(tmp, "panel_anchors")
            os.makedirs(panel_dir)
            panel = _png(os.path.join(panel_dir, "b000.jpg"), (9, 9, 9))
            seed = os.path.join(tmp, "seed_A.png")
            Path(seed).write_bytes(Path(panel).read_bytes())
            report = audit_seed_crops({"A": [seed]}, panel_dir=panel_dir, image_embedder=None)
            self.assertTrue(report["hash_duplicate_with_panel"])
            self.assertIn(seed, report["flagged"])

    def test_ambiguous_identity_with_fake_embedder(self):
        with tempfile.TemporaryDirectory() as tmp:
            a1 = _png(os.path.join(tmp, "a1.png"))
            a2 = _png(os.path.join(tmp, "a2.png"))
            b1 = _png(os.path.join(tmp, "b1.png"))
            # a2 sits between the A and B centroids -> ambiguous
            mapping = {
                a1: [1.0, 0.0],
                a2: [0.72, 0.70],
                b1: [0.0, 1.0],
            }
            report = audit_seed_crops(
                {"A": [a1, a2], "B": [b1]}, panel_dir=None,
                image_embedder=FakeImageEmbedder(mapping), ambiguous_margin=0.3,
            )
            flagged = report["flagged"]
            self.assertIn(a2, flagged)
            self.assertNotIn(a1, flagged)

    def test_filter_keeps_at_least_one_seed(self):
        out = filter_seed_refs({"A": ["x", "y"], "B": ["z"]}, ["x", "y", "z"])
        self.assertEqual(out["A"], ["x"])  # never strip the only seed
        self.assertEqual(out["B"], ["z"])


if __name__ == "__main__":
    unittest.main()
