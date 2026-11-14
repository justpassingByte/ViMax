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
import os
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


def _file_sha1(path: str) -> Optional[str]:
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return None


def audit_seed_crops(
    seed_refs: dict,
    panel_dir: Optional[str] = None,
    image_embedder: Optional[Any] = None,
    *,
    ambiguous_margin: float = 0.05,
) -> dict:
    """Audit character seed crops for cross-character or panel duplication.

    ``seed_refs``: {character_key: [image_path, ...]}.  Returns a JSON-able report
    with ``flagged`` paths to drop. Never raises; degrades to hash-only checks when
    no image embedder is available.
    """
    report: dict = {
        "embedder": _embedder_kind(image_embedder) if image_embedder else "none",
        "hash_duplicate_between_characters": [],
        "hash_duplicate_with_panel": [],
        "ambiguous_identity": [],
        "flagged": [],
    }

    # Path -> owning character; and hash -> first (char, path) seen.
    hash_owner: dict = {}
    for char, paths in (seed_refs or {}).items():
        for p in (paths or []):
            h = _file_sha1(p)
            if not h:
                continue
            if h in hash_owner:
                owner_char, owner_path = hash_owner[h]
                if owner_char != char:
                    report["hash_duplicate_between_characters"].append({"a": owner_char, "b": char, "path": p})
                    report["flagged"].append(p)
            else:
                hash_owner[h] = (char, p)

    # Hash duplicate with a whole panel anchor.
    if panel_dir and os.path.isdir(panel_dir):
        panel_hashes = {}
        for fn in os.listdir(panel_dir):
            fp = os.path.join(panel_dir, fn)
            h = _file_sha1(fp)
            if h:
                panel_hashes[h] = fn
        for char, paths in (seed_refs or {}).items():
            for p in (paths or []):
                h = _file_sha1(p)
                if h and h in panel_hashes:
                    report["hash_duplicate_with_panel"].append({"character": char, "path": p, "panel": panel_hashes[h]})
                    if p not in report["flagged"]:
                        report["flagged"].append(p)

    # Embedding-based ambiguity: a crop close to >= 2 character centroids.
    if image_embedder is not None:
        try:
            import numpy as np
            from tools.embeddings import cosine
            char_vecs = {}
            for char, paths in (seed_refs or {}).items():
                valid = [p for p in (paths or []) if os.path.exists(p)]
                if not valid:
                    continue
                vecs = image_embedder.embed_images(valid)
                char_vecs[char] = (valid, np.asarray(vecs))
            centroids = {c: v[1].mean(axis=0) for c, v in char_vecs.items() if len(v[1])}
            for char, (valid, vecs) in char_vecs.items():
                for idx, p in enumerate(valid):
                    sims = sorted(
                        ((other, cosine(vecs[idx], cen)) for other, cen in centroids.items()),
                        key=lambda t: t[1], reverse=True,
                    )
                    if len(sims) >= 2 and sims[0][1] - sims[1][1] < ambiguous_margin and sims[1][0] != char:
                        report["ambiguous_identity"].append({"path": p, "owner": char, "top": sims[0][0], "second": sims[1][0]})
                        if p not in report["flagged"]:
                            report["flagged"].append(p)
        except Exception as exc:  # noqa: BLE001 - never fatal
            report["embedding_error"] = str(exc)

    return report


def filter_seed_refs(seed_refs: dict, flagged_paths: Sequence[str]) -> dict:
    """Return a copy of ``seed_refs`` with flagged paths removed (keeping >=1 per char when possible)."""
    flagged = set(flagged_paths or [])
    out: dict = {}
    for char, paths in (seed_refs or {}).items():
        kept = [p for p in (paths or []) if p not in flagged]
        if not kept and paths:
            kept = [paths[0]]  # never strip a character's only seed entirely
        out[char] = kept
    return out
