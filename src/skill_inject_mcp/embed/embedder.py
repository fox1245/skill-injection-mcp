from __future__ import annotations

import hashlib
import math
import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import RLock
from typing import Sequence

import httpx
import numpy as np

from skill_inject_mcp.text import words
from skill_inject_mcp.timeouts import EMBEDDING_READ_TIMEOUT_S, http_timeout


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

    def __init__(self, dim: int = 1024, query_cache_size: int = 512) -> None:
        self.dim = dim
        self.query_cache_size = query_cache_size
        self._query_cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cache_lock = RLock()

    def _query_keys(self, texts):
        identity = (getattr(self, "model", None), self.dim, getattr(self, "base_url", None), format_query(""))
        return [(*identity, hashlib.sha256(text.encode("utf-8")).digest()) for text in texts]

    def _query_hits(self, texts):
        keys = self._query_keys(texts)
        hits = {}
        with self._cache_lock:
            for key in keys:
                if key in self._query_cache:
                    hits[key] = self._query_cache[key]
                    self._query_cache.move_to_end(key)
        misses = list(dict.fromkeys(key for key in keys if key not in hits))
        return keys, hits, misses

    def _publish_queries(self, keys, vectors):
        if len(keys) != len(vectors):
            raise ValueError("Invalid query embedding count")
        arrays = [np.asarray(v, dtype=np.float64).copy() for v in vectors]
        if any(v.shape != (self.dim,) or not np.isfinite(v).all() for v in arrays):
            raise ValueError("Invalid query embedding shape or non-finite values")
        values = dict(zip(keys, arrays))
        with self._cache_lock:
            for key, vector in values.items():
                vector.setflags(write=False)
                if self.query_cache_size > 0:
                    self._query_cache[key] = vector
                    self._query_cache.move_to_end(key)
            while len(self._query_cache) > max(0, self.query_cache_size):
                self._query_cache.popitem(last=False)
        return values

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[np.ndarray]:
        ...

    def embed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        keys, hits, missing = self._query_hits(texts)
        if missing:
            originals = dict(zip(keys, texts))
            values = self._embed_queries_uncached([originals[key] for key in missing])
            hits.update(self._publish_queries(missing, values))
        return [hits[key].copy() for key in keys]

    async def aembed_queries(self, texts: Sequence[str]) -> list[np.ndarray]:
        keys, hits, missing = self._query_hits(texts)
        if missing:
            originals = dict(zip(keys, texts))
            values = await self._aembed_queries_uncached([originals[key] for key in missing])
            hits.update(self._publish_queries(missing, values))
        return [hits[key].copy() for key in keys]

    def _embed_queries_uncached(self, texts: Sequence[str]) -> list[np.ndarray]:
        raise NotImplementedError

    async def _aembed_queries_uncached(self, texts: Sequence[str]) -> list[np.ndarray]:
        # Fake/local embedders do no I/O. Remote implementations must override this.
        return self._embed_queries_uncached(texts)

    def close(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


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

    def __init__(self, dim: int = 1024, query_cache_size: int = 512) -> None:
        super().__init__(dim, query_cache_size)

    def _tokenize(self, text: str) -> list[str]:
        raw = words(text)
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

    def _embed_queries_uncached(self, texts: Sequence[str]) -> list[np.ndarray]:
        # Apply instruction format then embed (Fake ignores instruction text noise lightly)
        return [self._embed_one(format_query(t)) for t in texts]


class OpenRouterEmbedder(Embedder):
    def __init__(
        self,
        api_key: str,
        model: str = "qwen/qwen3-embedding-8b",
        dim: int = 1024,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_s: float = EMBEDDING_READ_TIMEOUT_S,
        query_cache_size: int = 512,
    ) -> None:
        super().__init__(dim, query_cache_size)
        self.api_key = api_key
        self.model = model
        self.dim = dim
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client_lock = RLock()
        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._async_loop = None

    def _get_client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(timeout=http_timeout(self.timeout_s))
            return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._async_client is not None and self._async_loop is not loop:
            raise RuntimeError("Close the embedding async client before switching event loops")
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=http_timeout(self.timeout_s))
            self._async_loop = loop
        return self._async_client

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    async def aclose(self) -> None:
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None
            self._async_loop = None

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
        try:
            resp = self._get_client().post(f"{self.base_url}/embeddings", json=payload, headers=headers,
                                           timeout=http_timeout(self.timeout_s))
            resp.raise_for_status()
            data = resp.json()["data"]
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"Embedding request timed out ({type(exc).__name__}; read_timeout_s={self.timeout_s:g}). "
                "Check provider availability or SKILL_INJECT_EMBEDDING_TIMEOUT_S."
            ) from exc
        return self._decode(data, len(texts))

    def _decode(self, data, expected_count: int) -> list[np.ndarray]:
        if (not isinstance(data, list) or len(data) != expected_count
                or any(not isinstance(item, dict) or type(item.get("index")) is not int for item in data)
                or sorted(item["index"] for item in data) != list(range(expected_count))):
            raise ValueError("Invalid embedding response count or indices")
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

    def _embed_queries_uncached(self, texts: Sequence[str]) -> list[np.ndarray]:
        if not texts:
            return []
        formatted = [format_query(t) for t in texts]
        return self._call(formatted)

    async def _aembed_queries_uncached(self, texts: Sequence[str]) -> list[np.ndarray]:
        response = await self._get_async_client().post(
            f"{self.base_url}/embeddings",
            json={"model": self.model, "input": [format_query(t) for t in texts], "dimensions": self.dim},
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            timeout=http_timeout(self.timeout_s),
        )
        response.raise_for_status()
        return self._decode(response.json()["data"], len(texts))


def build_embedder(
    *,
    api_key: str | None,
    use_fake: bool = False,
    model: str = "qwen/qwen3-embedding-8b",
    dim: int = 1024,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout_s: float = EMBEDDING_READ_TIMEOUT_S,
    query_cache_size: int = 512,
) -> tuple[Embedder, bool]:
    """Return (embedder, degraded). degraded=True when falling back to FakeEmbedder."""
    if use_fake or not api_key:
        return FakeEmbedder(dim=dim, query_cache_size=query_cache_size), bool(not use_fake and not api_key)
    return OpenRouterEmbedder(api_key=api_key, model=model, dim=dim, base_url=base_url,
                              timeout_s=timeout_s, query_cache_size=query_cache_size), False
