"""Optional image/text embedders for reference selection and grouping.

Lazy, fail-soft: ``torch``/``transformers`` and the model weights are imported
only when an embedder is actually built. If anything is missing the builder
returns ``None`` and the caller disables the feature (never a hard error).

Embeddings only help *select and check* reference images and *group* scenes /
locations — they never make the image model obey a reference.

Models & licenses (published on Hugging Face):
- ``facebook/dinov2-small``   — image embedder — Apache-2.0 (commercial use OK).
- ``intfloat/multilingual-e5-small`` — text embedder — MIT (commercial use OK).
Both are permissive; no non-commercial model is used.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

_IMAGE_MODELS = {"dinov2": "facebook/dinov2-small"}
_TEXT_MODELS = {"e5": "intfloat/multilingual-e5-small"}


class ImageEmbedder(Protocol):
    name: str
    def embed_images(self, paths: List[str]):  # -> np.ndarray (n, d), L2-normalized
        ...


class TextEmbedder(Protocol):
    name: str
    def embed_texts(self, texts: List[str]):  # -> np.ndarray (n, d), L2-normalized
        ...


def _import_deps():
    import numpy as np  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    return np, torch, transformers


def _resolve_device(device: str):
    import torch
    if device and device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


class _CachingMixin:
    """Content-addressed vector cache under ``<cache_dir>/<model>/``."""

    def __init__(self, cache_dir: Optional[str], model_key: str):
        self._cache_dir = None
        if cache_dir:
            self._cache_dir = Path(cache_dir) / "embeddings" / model_key.replace("/", "_")
            try:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                self._cache_dir = None

    def _cache_path(self, key: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}.npy"

    def _load(self, key: str):
        p = self._cache_path(key)
        if p is not None and p.is_file():
            try:
                import numpy as np
                return np.load(str(p))
            except Exception:
                return None
        return None

    def _store(self, key: str, vec) -> None:
        p = self._cache_path(key)
        if p is not None:
            try:
                import numpy as np
                np.save(str(p), vec)
            except Exception:
                pass


def _l2_normalize(mat):
    import numpy as np
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class _Dinov2Embedder(_CachingMixin):
    def __init__(self, model_id: str, device: str, cache_dir: Optional[str]):
        super().__init__(cache_dir, model_id)
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.name = "dinov2"
        self._device = device
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(device).eval()
        self._torch = torch

    def embed_images(self, paths: List[str]):
        import numpy as np
        from PIL import Image
        out = []
        for path in paths:
            key = None
            try:
                with open(path, "rb") as f:
                    key = _sha1(f.read())
            except OSError:
                key = None
            cached = self._load(key) if key else None
            if cached is not None:
                out.append(cached)
                continue
            try:
                with Image.open(path) as im:
                    img = im.convert("RGB")
                inputs = self._processor(images=img, return_tensors="pt").to(self._device)
                with self._torch.no_grad():
                    feats = self._model(**inputs).last_hidden_state[:, 0]  # CLS token
                vec = feats[0].cpu().numpy().astype("float32")
            except Exception as exc:
                logger.warning("dinov2 embed failed for %s: %s", path, exc)
                vec = np.zeros((self._model.config.hidden_size,), dtype="float32")
            out.append(vec)
            if key:
                self._store(key, vec)
        return _l2_normalize(np.vstack(out))


class _E5Embedder(_CachingMixin):
    def __init__(self, model_id: str, device: str, cache_dir: Optional[str]):
        super().__init__(cache_dir, model_id)
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.name = "e5"
        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(device).eval()
        self._torch = torch

    def embed_texts(self, texts: List[str]):
        import numpy as np
        out = []
        pending_idx = []
        pending_text = []
        results = [None] * len(texts)
        for i, t in enumerate(texts):
            key = _sha1(("e5:" + (t or "")).encode("utf-8"))
            cached = self._load(key)
            if cached is not None:
                results[i] = cached
            else:
                pending_idx.append((i, key))
                pending_text.append("query: " + (t or ""))
        if pending_text:
            try:
                inputs = self._tokenizer(pending_text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(self._device)
                with self._torch.no_grad():
                    outputs = self._model(**inputs)
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    summed = (outputs.last_hidden_state * mask).sum(1)
                    counts = mask.sum(1).clamp(min=1e-9)
                    mean = (summed / counts).cpu().numpy().astype("float32")
            except Exception as exc:
                logger.warning("e5 embed failed: %s", exc)
                mean = np.zeros((len(pending_text), self._model.config.hidden_size), dtype="float32")
            for (orig_i, key), vec in zip(pending_idx, mean):
                results[orig_i] = vec
                self._store(key, vec)
        return _l2_normalize(np.vstack(results))


def build_image_embedder(kind: str, device: str = "auto", cache_dir: Optional[str] = None) -> Optional[ImageEmbedder]:
    kind = (kind or "off").lower()
    if kind == "off" or kind not in _IMAGE_MODELS:
        return None
    try:
        _import_deps()
    except Exception as exc:
        logger.warning("Image embeddings disabled (missing deps): %s", exc)
        return None
    try:
        dev = _resolve_device(device)
        return _Dinov2Embedder(_IMAGE_MODELS[kind], dev, cache_dir)
    except Exception as exc:
        logger.warning("Image embedder '%s' unavailable (%s); feature disabled.", kind, exc)
        return None


def build_text_embedder(kind: str, device: str = "auto", cache_dir: Optional[str] = None) -> Optional[TextEmbedder]:
    kind = (kind or "off").lower()
    if kind == "off" or kind not in _TEXT_MODELS:
        return None
    try:
        _import_deps()
    except Exception as exc:
        logger.warning("Text embeddings disabled (missing deps): %s", exc)
        return None
    try:
        dev = _resolve_device(device)
        return _E5Embedder(_TEXT_MODELS[kind], dev, cache_dir)
    except Exception as exc:
        logger.warning("Text embedder '%s' unavailable (%s); feature disabled.", kind, exc)
        return None


def cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float32").ravel()
    b = np.asarray(b, dtype="float32").ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
