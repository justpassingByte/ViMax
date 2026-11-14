"""Unit tests for the report-only duplicate-identity detector.

Uses a fully controlled dictionary-backed embedder so similarity thresholds
are deterministic, plus the offline MockHashingEmbedder for the fallback path.
No network is used anywhere in this file.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents.identity_dedup import (
    MockHashingEmbedder,
    build_identity_dedup_embedder,
    detect_duplicate_identities,
)
from interfaces import CharacterBibleEntry


class DictMockEmbedder:
    """Returns a fixed vector per exact input text; unknown texts -> zeros."""

    def __init__(self, vectors: dict, fail: bool = False) -> None:
        self.vectors = vectors
        self.fail = fail
        self.calls: list = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if self.fail:
            raise RuntimeError("embedding provider down")
        return [list(self.vectors.get(t, [0.0, 0.0, 0.0, 0.0])) for t in texts]

    def embed_query(self, text):
        return list(self.vectors.get(text, [0.0, 0.0, 0.0, 0.0]))


def _entry(key: str, features: str) -> CharacterBibleEntry:
    return CharacterBibleEntry(character_key=key, permanent_features=features)


# Orthogonal unit vectors -> cosine 0. Vector "same" -> cosine 1 with itself.
_UNIT = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "B": [0.0, 1.0, 0.0, 0.0],
    "C": [0.0, 0.0, 1.0, 0.0],
    "same": [0.0, 0.0, 0.0, 1.0],
}


class TestDetectDuplicateIdentities(unittest.TestCase):
    def test_flags_high_similarity_pair(self):
        chars = [
            _entry("Hojin", "small boy, dark hair"),
            _entry("Young Boy with Soccer Ball", "small boy, dark hair, chasing ball"),
            _entry("Kayden", "fat orange cat"),
        ]
        chars[0].source_beat_ids = ["ch01_b040"]
        chars[1].source_beat_ids = ["ch01_b025"]
        embedder = DictMockEmbedder(
            {
                "Hojin: small boy, dark hair": _UNIT["same"],
                "Young Boy with Soccer Ball: small boy, dark hair, chasing ball": _UNIT["same"],
                "Kayden: fat orange cat": _UNIT["A"],
            }
        )
        report = detect_duplicate_identities(chars, embedder=embedder, threshold=0.82)
        self.assertEqual(report.status, "ok")
        suspected = report.suspected_pairs()
        self.assertEqual(len(suspected), 1)
        self.assertEqual({suspected[0].a, suspected[0].b}, {"Hojin", "Young Boy with Soccer Ball"})
        self.assertEqual(suspected[0].similarity, 1.0)
        # Unrelated pair included as a near-miss? cosine 0 < floor -> excluded.
        keys = {(p.a, p.b) for p in report.pairs}
        self.assertNotIn(("Hojin", "Kayden"), keys)
        self.assertNotIn(("Kayden", "Young Boy with Soccer Ball"), keys)

    def test_below_threshold_not_suspected(self):
        chars = [_entry("Alpha", "f1"), _entry("Beta", "f2")]
        embedder = DictMockEmbedder(
            {"Alpha: f1": [1.0, 0.0], "Beta: f2": [0.75, 0.6614378277661477]}  # cosine 0.75
        )
        report = detect_duplicate_identities(chars, embedder=embedder, threshold=0.82)
        self.assertEqual(report.status, "ok")
        self.assertEqual(report.suspected_pairs(), [])
        # Near-miss kept for visibility.
        self.assertEqual(len(report.pairs), 1)
        self.assertFalse(report.pairs[0].suspected)
        self.assertAlmostEqual(report.pairs[0].similarity, 0.75, places=3)

    def test_no_embedder_is_skipped(self):
        report = detect_duplicate_identities([_entry("A", "x")], embedder=None)
        self.assertEqual(report.status, "skipped")
        self.assertEqual(report.pairs, [])

    def test_single_character_is_skipped(self):
        report = detect_duplicate_identities([_entry("A", "x")], embedder=DictMockEmbedder(_UNIT))
        self.assertEqual(report.status, "skipped")

    def test_blank_features_ignored(self):
        chars = [_entry("A", "described"), _entry("B", ""), _entry("C", "   ")]
        embedder = DictMockEmbedder({"A: described": [1.0, 0.0]})
        report = detect_duplicate_identities(chars, embedder=embedder)
        self.assertEqual(report.status, "skipped")  # <2 described entries

    def test_embedding_failure_is_reported_not_raised(self):
        chars = [_entry("A", "f1"), _entry("B", "f2")]
        report = detect_duplicate_identities(chars, embedder=DictMockEmbedder(_UNIT, fail=True))
        self.assertEqual(report.status, "skipped")
        self.assertIn("embedding call failed", report.reason)


class TestMockHashingEmbedder(unittest.TestCase):
    def test_deterministic_and_normalized(self):
        e = MockHashingEmbedder()
        v1 = e.embed_documents(["hello world", "hello world"])
        self.assertEqual(v1[0], v1[1])
        norm = sum(x * x for x in v1[0]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=6)
        self.assertEqual(e.embed_query("hello world"), v1[0])

    def test_disjoint_texts_low_similarity(self):
        e = MockHashingEmbedder(dim=64)
        va, vb = e.embed_documents(["quantum physics laboratory", "chocolate cake recipe"])[0:2]
        sim = sum(x * y for x, y in zip(va, vb))
        self.assertLess(sim, 0.2)


class TestBuildIdentityDedupEmbedder(unittest.TestCase):
    def test_off_mode(self):
        with patch.dict(os.environ, {"VIMAX_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER": "off"}):
            embedder, kind = build_identity_dedup_embedder(".")
        self.assertIsNone(embedder)
        self.assertEqual(kind, "off")

    def test_mock_mode(self):
        with patch.dict(os.environ, {"VIMAX_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER": "mock"}):
            embedder, kind = build_identity_dedup_embedder(".")
        self.assertIsInstance(embedder, MockHashingEmbedder)
        self.assertEqual(kind, "mock")

    def test_auto_falls_back_to_mock_when_provider_raises(self):
        def _boom():
            raise RuntimeError("no provider")

        with patch.dict(os.environ, {"VIMAX_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER": "auto"}):
            with patch("agent_runtime.vimax_adapters._build_embedding_model", side_effect=_boom):
                embedder, kind = build_identity_dedup_embedder(".")
        self.assertIsInstance(embedder, MockHashingEmbedder)
        self.assertEqual(kind, "mock-fallback")

    def test_auto_uses_real_provider_when_working(self):
        class FakeReal:
            def embed_query(self, text):
                return [1.0, 0.0]

        with patch.dict(os.environ, {"VIMAX_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER": "auto"}):
            with patch("agent_runtime.vimax_adapters._build_embedding_model", return_value=FakeReal()):
                embedder, kind = build_identity_dedup_embedder(".")
        self.assertIsInstance(embedder, FakeReal)
        self.assertEqual(kind, "openai-compatible")


class TestConfigHelpers(unittest.TestCase):
    def test_threshold_env_override(self):
        from agent_runtime.config import manga_vision_identity_dedup_threshold

        with patch.dict(os.environ, {"VIMAX_MANGA_VISION_IDENTITY_DEDUP_THRESHOLD": "0.9"}):
            self.assertAlmostEqual(manga_vision_identity_dedup_threshold("."), 0.9)

    def test_embedder_env_invalid_falls_back(self):
        from agent_runtime.config import (
            DEFAULT_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER,
            manga_vision_identity_dedup_embedder,
        )

        with patch.dict(os.environ, {"VIMAX_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER": "bogus"}):
            self.assertEqual(manga_vision_identity_dedup_embedder("."), DEFAULT_MANGA_VISION_IDENTITY_DEDUP_EMBEDDER)


class TestWorldModelIntegration(unittest.TestCase):
    def test_detect_writes_pairs_sorted_desc(self):
        chars = [
            _entry("A", "f1"),
            _entry("B", "f2"),
            _entry("C", "f3"),
        ]
        embedder = DictMockEmbedder(
            {
                "A: f1": _UNIT["same"],
                "B: f2": _UNIT["same"],
                "C: f3": _UNIT["A"],
            }
        )
        report = detect_duplicate_identities(chars, embedder=embedder, threshold=0.82)
        sims = [p.similarity for p in report.pairs]
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertEqual((report.pairs[0].a, report.pairs[0].b), ("A", "B"))
        dumped = json.loads(report.model_dump_json())
        self.assertEqual(dumped["status"], "ok")


if __name__ == "__main__":
    unittest.main()
