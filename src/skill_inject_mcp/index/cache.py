"""Optional content-addressed embedding cache with atomic SQLite writes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path

import numpy as np


class EmbeddingCache:
    def __init__(self, root: Path, identity: tuple, dim: int) -> None:
        self.version = hashlib.sha256(json.dumps(["v1", identity], sort_keys=True).encode()).hexdigest()
        # A single short filename also works with Windows' legacy path-length limit.
        self.path = Path(root) / "embedding-cache.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "model_hash TEXT, doc_hash TEXT, vector BLOB, PRIMARY KEY (model_hash, doc_hash))"
            )
            connection.commit()

    def _decode_vector(self, raw: object) -> np.ndarray | None:
        try:
            vector = np.frombuffer(raw, dtype=np.float64).copy()
            if vector.shape != (self.dim,) or not np.isfinite(vector).all():
                return None
            return vector
        except (ValueError, TypeError):
            return None

    def get(self, digest: str) -> np.ndarray | None:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT vector FROM embeddings WHERE model_hash=? AND doc_hash=?",
                (self.version, digest),
            ).fetchone()
        if row is None:
            return None
        return self._decode_vector(row[0])

    def get_many(self, digests: Sequence[str]) -> dict[str, np.ndarray]:
        unique = list(dict.fromkeys(digests))
        if not unique:
            return {}
        found: dict[str, np.ndarray] = {}
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            for offset in range(0, len(unique), 500):
                chunk = unique[offset:offset + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT doc_hash, vector FROM embeddings WHERE model_hash=? AND doc_hash IN ({placeholders})",
                    (self.version, *chunk),
                ).fetchall()
                for doc_hash, raw in rows:
                    vector = self._decode_vector(raw)
                    if vector is not None:
                        found[doc_hash] = vector
        return found

    def put(self, digest: str, vector: np.ndarray) -> None:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO embeddings(model_hash, doc_hash, vector) VALUES (?,?,?)",
                (self.version, digest, np.asarray(vector, dtype=np.float64).tobytes()),
            )
            connection.commit()

    def put_many(
        self,
        items: Sequence[tuple[str, np.ndarray]] | Mapping[str, np.ndarray],
    ) -> None:
        pairs = list(items.items() if isinstance(items, Mapping) else items)
        if not pairs:
            return
        rows = [
            (self.version, digest, np.asarray(vector, dtype=np.float64).tobytes())
            for digest, vector in pairs
        ]
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings(model_hash, doc_hash, vector) VALUES (?,?,?)",
                rows,
            )
            connection.commit()
