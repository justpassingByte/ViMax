"""Report-only duplicate-identity detection for the Manga Vision Layer.

Problem this solves: the reading pass may describe the SAME character under
two different names (e.g. "Alex" in one panel, "Young Boy with Soccer Ball"
in another). Token overlap on permanent features is too weak to catch this
(measured ~0.20 Jaccard for confirmed duplicates), so we compare the SEMANTIC
similarity of the bible descriptions with embeddings and flag suspected
duplicates above a threshold.

Design constraints:
- REPORT-ONLY: nothing is merged or renamed automatically. The report is
  written to ``vision_world/identity_dedup.json`` and attached to the
  ``VisionWorldModel`` so Visual QA / humans can act on it.
- OFFLINE-BY-DEFAULT: a deterministic local ``MockHashingEmbedder`` is used
  when no real embedding provider is configured, so the pipeline never
  depends on network availability and tests cost nothing.
- PLUGGABLE: configure a real provider any time via the existing
  ``embedding`` section in ``configs/agent.local.yaml`` (OpenAI-compatible
  ``base_url`` + ``api_key`` + ``model``). With
  ``manga_vision.identity_dedup_embedder: auto`` (default) the real provider
  is used as soon as it works; any failure falls back to the mock.
"""

from __future__ import annotations

import math
import re
from typing import Any, List, Optional, Sequence, Tuple

from interfaces import CharacterBibleEntry, IdentityDedupPair, IdentityDedupReport

DEFAULT_DEDUP_THRESHOLD = 0.82

_MOCK_DIM = 256
_PAIR_MARGIN = 0.15  # keep near-miss pairs (threshold - margin) for visibility
_MAX_PAIRS = 12


class MockHashingEmbedder:
    """Deterministic offline placeholder embedder (bag of hashed tokens).

    NOT semantically deep — it is a safe stand-in so the pipeline runs and
    tests stay hermetic. Configure a real provider (``embedding`` section in
    configs/agent.local.yaml) for genuine semantic duplicate detection.
    """

    def __init__(self, dim: int = _MOCK_DIM) -> None:
        self.dim = dim

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9]+", str(text).lower()):
            if len(tok) < 3:
                continue
            h = 0
            for ch in tok:
                h = (h * 31 + ord(ch)) % self.dim
            vec[h] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vector(text)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def build_identity_dedup_embedder(workspace_root: str | Any = ".") -> Tuple[Optional[Any], str]:
    """Return ``(embedder, kind)`` for the identity-dedup pass.

    Never raises. Kinds: ``"off"`` (dedup disabled -> embedder None),
    ``"mock"`` (explicit mock), ``"openai-compatible"`` (real provider),
    ``"mock-fallback"`` (auto but provider missing/unreachable).
    """
    from agent_runtime.config import manga_vision_identity_dedup_embedder

    mode = manga_vision_identity_dedup_embedder(workspace_root)
    if mode == "off":
        return None, "off"
    if mode == "mock":
        return MockHashingEmbedder(), "mock"

    # auto: try the real OpenAI-compatible provider, fall back to the mock.
    from agent_runtime.vimax_adapters import _build_embedding_model

    try:
        model = _build_embedding_model()
        model.embed_query("identity dedup smoke test")  # fail fast on bad config
        return model, "openai-compatible"
    except Exception:  # noqa: BLE001 - dedup must never break the pipeline
        return MockHashingEmbedder(), "mock-fallback"


def detect_duplicate_identities(
    characters: Sequence[CharacterBibleEntry],
    embedder: Optional[Any] = None,
    threshold: float = DEFAULT_DEDUP_THRESHOLD,
    max_pairs: int = _MAX_PAIRS,
) -> IdentityDedupReport:
    """Compare permanent features pairwise and flag suspected duplicates."""
    if embedder is None:
        return IdentityDedupReport(status="skipped", reason="no embedder configured")

    entries = [c for c in characters if str(c.permanent_features or "").strip()]
    if len(entries) < 2:
        return IdentityDedupReport(status="skipped", embedder=_embedder_kind(embedder), reason="fewer than 2 described characters")

    texts = [f"{c.character_key}: {c.permanent_features}" for c in entries]
    try:
        vectors = embedder.embed_documents(texts)
    except Exception as exc:  # noqa: BLE001 - report-only, never fatal
        return IdentityDedupReport(
            status="skipped",
            embedder=_embedder_kind(embedder),
            threshold=threshold,
            reason=f"embedding call failed: {exc}",
        )
    if len(vectors) != len(entries):
        return IdentityDedupReport(
            status="skipped",
            embedder=_embedder_kind(embedder),
            threshold=threshold,
            reason="embedder returned a mismatched number of vectors",
        )

    floor = max(0.0, threshold - _PAIR_MARGIN)
    pairs: List[IdentityDedupPair] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            sim = _cosine(vectors[i], vectors[j])
            if sim >= floor:
                beats_i = {str(b) for b in entries[i].source_beat_ids}
                beats_j = {str(b) for b in entries[j].source_beat_ids}
                pairs.append(
                    IdentityDedupPair(
                        a=entries[i].character_key,
                        b=entries[j].character_key,
                        similarity=round(min(1.0, max(0.0, sim)), 4),
                        suspected=sim >= threshold,
                        never_cooccur=not (beats_i & beats_j),
                    )
                )
    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return IdentityDedupReport(
        status="ok",
        embedder=_embedder_kind(embedder),
        threshold=threshold,
        pairs=pairs[:max_pairs],
    )


def _embedder_kind(embedder: Any) -> str:
    name = type(embedder).__name__
    return "mock" if "Mock" in name else name
