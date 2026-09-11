from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from skill_inject_mcp.index.native import NativeVectorStore


class VectorIndex(ABC):
    """Common interface for dense skill vectors."""

    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def upsert(self, skill_id: str, vector: np.ndarray) -> None:
        ...

    def upsert_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        for skill_id, vector in items:
            self.upsert(skill_id, vector)

    def close(self) -> None:
        pass

    def search_readonly(self, query: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        return self.search(query, top_k)

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        """Return list of (skill_id, score) sorted by score desc."""
        ...

    @abstractmethod
    def count(self) -> int:
        ...


class NumpyVectorIndex(VectorIndex):
    """In-memory / npz-persisted cosine similarity index (fallback when sqlite-vector is unavailable)."""

    def __init__(self, dim: int = 1024, persist_path: Path | None = None) -> None:
        self.dim = dim
        self.persist_path = Path(persist_path) if persist_path else None
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None
        if self.persist_path and self.persist_path.exists():
            self._load()

    def clear(self) -> None:
        self._ids = []
        self._mat = None
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    def upsert(self, skill_id: str, vector: np.ndarray) -> None:
        v = np.asarray(vector, dtype=np.float64).reshape(-1)
        if v.shape[0] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {v.shape[0]}")
        if skill_id in self._ids:
            idx = self._ids.index(skill_id)
            assert self._mat is not None
            self._mat[idx] = v
        else:
            self._ids.append(skill_id)
            if self._mat is None:
                self._mat = v.reshape(1, -1)
            else:
                self._mat = np.vstack([self._mat, v.reshape(1, -1)])
        self._save()

    def search(self, query: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        if not self._ids or self._mat is None:
            return []
        q = np.asarray(query, dtype=np.float64).reshape(-1)
        # cosine: assume L2-normalized
        scores = self._mat @ q
        order = sorted(range(len(self._ids)), key=lambda i: (-scores[i], self._ids[i]))
        out: list[tuple[str, float]] = []
        for i in order[:top_k]:
            out.append((self._ids[int(i)], float(scores[int(i)])))
        return out

    def count(self) -> int:
        return len(self._ids)

    def upsert_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        values = {sid: self._mat[i] for i, sid in enumerate(self._ids)}
        for sid, vector in items:
            vector = np.asarray(vector, dtype=np.float64).reshape(-1)
            if vector.shape != (self.dim,):
                raise ValueError(f"expected dim {self.dim}, got {vector.shape}")
            values[sid] = vector
        self._ids = list(values)
        self._mat = np.stack(list(values.values())) if values else None
        self._save()

    def _save(self) -> None:
        if not self.persist_path or self._mat is None:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.persist_path,
            ids=np.array(self._ids, dtype=str),
            mat=self._mat,
        )

    def _load(self) -> None:
        assert self.persist_path is not None
        with np.load(self.persist_path, allow_pickle=False) as data:
            self._ids = [str(x) for x in data["ids"].tolist()]
            self._mat = data["mat"]


class SqliteVectorIndex(NativeVectorStore, VectorIndex):
    """sqliteai/sqlite-vector native exact cosine search over FLOAT32 blobs."""

    table = "skill_vectors"
    id_column = "skill_id"


def build_vector_index(
    index_dir: Path, dim: int = 1024, *, backend: str = "sqlite-vector",
    extension_path: Path | None = None,
) -> tuple[VectorIndex, str]:
    """Use the selected backend. Native loading failures never select NumPy."""
    index_dir = Path(index_dir)
    if backend not in ("sqlite-vector", "numpy"):
        raise ValueError(f"Unknown dense backend: {backend}")
    index_dir.mkdir(parents=True, exist_ok=True)
    if backend == "numpy":
        return NumpyVectorIndex(dim=dim, persist_path=index_dir / "dense_numpy.npz"), "numpy"
    return SqliteVectorIndex(index_dir / "dense.sqlite", dim=dim, extension_path=extension_path), "sqlite-vector"
