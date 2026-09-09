from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np


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


class SqliteVectorIndex(VectorIndex):
    """Optional sqlite-vector backed index when the extension is loadable."""

    def __init__(self, db_path: Path, dim: int = 1024) -> None:
        import sqlite3

        self.dim = dim
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.enable_load_extension(True)
        # Try common extension names; caller should catch failures
        loaded = False
        for name in ("vector", "sqlite_vector", "sqlitevector"):
            try:
                self._conn.load_extension(name)
                loaded = True
                break
            except Exception:
                continue
        if not loaded:
            # Attempt import sqlite_vector package helper if present
            try:
                import sqlite_vector  # type: ignore

                sqlite_vector.load(self._conn)
                loaded = True
            except Exception as e:
                self._conn.close()
                raise RuntimeError("sqlite-vector extension unavailable") from e
        self._conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS skill_vectors (
                skill_id TEXT PRIMARY KEY,
                embedding BLOB
            )
            """
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM skill_vectors")
        self._conn.commit()

    def upsert(self, skill_id: str, vector: np.ndarray) -> None:
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT OR REPLACE INTO skill_vectors(skill_id, embedding) VALUES (?, ?)",
            (skill_id, blob),
        )
        self._conn.commit()

    def search(self, query: np.ndarray, top_k: int = 20) -> list[tuple[str, float]]:
        # Exact cosine in Python over stored blobs (no native ANN implementation).
        cur = self._conn.execute("SELECT skill_id, embedding FROM skill_vectors")
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        scored: list[tuple[str, float]] = []
        for sid, blob in cur.fetchall():
            v = np.frombuffer(blob, dtype=np.float32)
            if v.shape[0] == 0:
                continue
            denom = float(np.linalg.norm(v) * np.linalg.norm(q)) or 1.0
            score = float(np.dot(v, q) / denom)
            scored.append((sid, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    def upsert_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO skill_vectors(skill_id, embedding) VALUES (?, ?)",
                [(sid, np.asarray(vector, dtype=np.float32).tobytes()) for sid, vector in items],
            )

    def close(self) -> None:
        self._conn.close()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM skill_vectors").fetchone()
        return int(row[0]) if row else 0


def build_vector_index(index_dir: Path, dim: int = 1024) -> tuple[VectorIndex, str]:
    """Prefer optional sqlite-vector native ext; fall back to numpy. Returns (index, backend_name)."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = index_dir / "dense.sqlite"
    try:
        idx = SqliteVectorIndex(sqlite_path, dim=dim)
        return idx, "sqlite-vector"
    except Exception:
        npz_path = index_dir / "dense_numpy.npz"
        return NumpyVectorIndex(dim=dim, persist_path=npz_path), "numpy"
