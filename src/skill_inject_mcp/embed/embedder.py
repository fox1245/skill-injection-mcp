from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

import httpx
import numpy as np


def format_query(query: str, *, task: str = "Given a skill requirement, retrieve matching agent skills") -> str:
    """Qwen instruction-aware query format."""
    return f"Instruct: {task}\nQuery: {query}"


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n == 0.0:
        return vec
    return vec / n


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        ...

    @abstractmethod
    def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        ...


class FakeEmbedder(Embedder):
    """Deterministic bag-of-tokens embedder for offline tests.

    Synonym groups share hashed dimensions so dense retrieval can match
    paraphrases without an API key.
    """

    SYNONYMS: dict[str, str] = {
        "install": "pkg_install",
        "installation": "pkg_install",
        "installer": "pkg_install",
        "packages": "pkg_install",
        "package": "pkg_install",
        "pip": "pkg_install",
        "dependency": "pkg_install",
        "dependencies": "pkg_install",
        "build": "build_sys",
        "cmake": "build_sys",
        "compile": "build_sys",
        "compilation": "build_sys",
        "builder": "build_sys",
        "bm25": "sparse_ret",
        "fts5": "sparse_ret",
        "sparse": "sparse_ret",
        "fulltext": "sparse_ret",
        "full-text": "sparse_ret",
        "embedding": "dense_ret",
        "vector": "dense_ret",
        "dense": "dense_ret",
        "sqlite-vector": "dense_ret",
        "sqlite_vector": "dense_ret",
        "semantic": "dense_ret",
        "hybrid": "hybrid_ret",
        "rrf": "hybrid_ret",
        "retrieval": "ret_generic",
        "retrieve": "ret_generic",
        "search": "ret_generic",
        "index": "ret_generic",
        "python": "lang_py",
        "adapter": "adapter_tok",
    }

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim

    def _tokenize(self, text: str) -> list[str]:
        raw = re.findall(r"[a-z0-9_\-]+", text.lower())
        out: list[str] = []
        for t in raw:
            canon = self.SYNONYMS.get(t, t)
            out.append(canon)
            out.append(t)
        return out

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        toks = self._tokenize(text)
        if not toks:
            vec[0] = 1.0
            return l2_normalize(vec)
        for tok in toks:
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "little") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            weight = 1.0 + (h[5] / 255.0)
            vec[idx] += sign * weight
            # secondary dim for stability
            idx2 = int.from_bytes(h[6:10], "little") % self.dim
            vec[idx2] += 0.35 * sign
        return l2_normalize(vec)

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        return [self._embed_one(t) for t in texts]

    def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        # Apply instruction format then embed (Fake ignores instruction text noise lightly)
        return [self._embed_one(format_query(t)) for t in texts]


class OpenRouterEmbedder(Embedder):
    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-embedding-8b",
        dim: int = 1024,
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")

    def _call(self, texts: Sequence[str]) -> list[np.ndarray]:
        payload = {
            "model": self.model,
            "input": list(texts),
            "dimensions": self.dim,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{self.base_url}/embeddings", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()["data"]
        out: list[np.ndarray] = []
        for item in sorted(data, key=lambda x: x["index"]):
            arr = np.asarray(item["embedding"], dtype=np.float64)
            if arr.shape[0] != self.dim:
                # MRL truncate or pad
                if arr.shape[0] > self.dim:
                    arr = arr[: self.dim]
                else:
                    pad = np.zeros(self.dim, dtype=np.float64)
                    pad[: arr.shape[0]] = arr
                    arr = pad
            out.append(l2_normalize(arr))
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        return self._call(texts)

    def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        formatted = [format_query(t) for t in texts]
        return self._call(formatted)


def build_embedder(
    *,
    api_key: str | None,
    use_fake: bool = False,
    model: str = "qwen/qwen3-embedding-8b",
    dim: int = 1024,
    base_url: str = "https://openrouter.ai/api/v1",
) -> tuple[Embedder, bool]:
    """Return (embedder, degraded). degraded=True when falling back to FakeEmbedder."""
    if use_fake or not api_key:
        return FakeEmbedder(dim=dim), bool(not use_fake and not api_key)
    return OpenRouterEmbedder(api_key=api_key, model=model, dim=dim, base_url=base_url), False
