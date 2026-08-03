from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Iterable, List


BASE_DIR = Path(__file__).resolve().parent.parent
HF_CACHE_DIR = BASE_DIR / ".hf_cache"
HF_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))
TRANSFORMERS_CACHE = HF_CACHE_DIR / "transformers"
TRANSFORMERS_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE))


class SentenceTransformerEmbeddings:
    """Semantic embeddings backed by sentence-transformers with a local cache."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, cache_folder=str(HF_CACHE_DIR))

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=False).tolist()

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return self.model.encode(list(texts), convert_to_numpy=False).tolist()


class LocalHashingEmbeddings:
    """Deterministic offline embeddings based on token hashing.

    This is intentionally simple and dependency-light so the project can run
    in offline or restricted environments while still providing a stable
    vector representation for semantic-style retrieval.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int = 384):
        self.model_name = model_name
        self.dim = dim

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in re.findall(r"\w+", (text or "").lower()) if token]

    def _embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for token in self._tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            vector[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self._embed_text(text) for text in texts]


def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Prefer the real sentence-transformer model from a repo-local cache."""
    try:
        return SentenceTransformerEmbeddings(model_name=model_name)
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"[embeddings] Falling back to local hashing embeddings: {exc}")
        return LocalHashingEmbeddings(model_name=model_name)
