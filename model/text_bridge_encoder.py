"""Text bridge encoder for ChoreoRAG.

Safe design:
- Uses sentence-transformers when available.
- Falls back to deterministic hashed n-gram embeddings when unavailable.

Recommended experiment model:
  BAAI/bge-small-zh-v1.5
or:
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

import numpy as np


def _l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / max(float(np.linalg.norm(x)), eps)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


class TextBridgeEncoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: str = "cpu",
        fallback_dim: int = 384,
        prefer_sentence_transformers: bool = True,
    ):
        self.model_name = model_name or "hash"
        self.device = device
        self.fallback_dim = int(fallback_dim)
        self.model = None
        self.backend = "hash"

        if prefer_sentence_transformers and self.model_name.lower() not in {"", "none", "hash"}:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self.model = SentenceTransformer(self.model_name, device=device)
                self.backend = "sentence-transformers"
            except Exception as exc:
                print(
                    f"⚠️ sentence-transformers unavailable ({exc}); "
                    f"using deterministic hash encoder dim={self.fallback_dim}."
                )

    @staticmethod
    def _tokens(text: str) -> List[str]:
        text = str(text or "").strip().lower()
        units = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text)
        if not units:
            units = ["empty"]
        grams = list(units)
        joined = "".join(units)
        for n in (2, 3, 4):
            for i in range(max(0, len(joined) - n + 1)):
                grams.append(joined[i : i + n])
        return grams

    def _hash_one(self, text: str) -> np.ndarray:
        v = np.zeros((self.fallback_dim,), dtype=np.float32)
        for tok in self._tokens(text):
            d = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(d[:4], "little") % self.fallback_dim
            sign = 1.0 if d[4] % 2 == 0 else -1.0
            v[idx] += sign
        return _l2_normalize(v)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = [str(t or "") for t in texts]
        if self.backend == "sentence-transformers" and self.model is not None:
            emb = self.model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=64,
                show_progress_bar=False,
            )
            return np.asarray(emb, dtype=np.float32)
        emb = np.stack([self._hash_one(t) for t in texts], axis=0)
        return _l2_normalize(emb).astype(np.float32)


def encode_texts(
    texts: Iterable[str],
    model_name: str = "BAAI/bge-small-zh-v1.5",
    device: str = "cpu",
    fallback_dim: int = 384,
) -> np.ndarray:
    return TextBridgeEncoder(
        model_name=model_name,
        device=device,
        fallback_dim=fallback_dim,
    ).encode(texts)
