"""Optional content-addressed embedding cache with atomic SQLite writes."""
from __future__ import annotations

import hashlib
import json
import sqlite3
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

    def get(self, digest: str) -> np.ndarray | None:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            row = connection.execute(
                "SELECT vector FROM embeddings WHERE model_hash=? AND doc_hash=?",
                (self.version, digest),
            ).fetchone()
        if row is None:
            return None
        try:
            vector = np.frombuffer(row[0], dtype=np.float64).copy()
            if vector.shape != (self.dim,) or not np.isfinite(vector).all():
                return None
            return vector
        except (ValueError, TypeError):
            return None

    def put(self, digest: str, vector: np.ndarray) -> None:
        with closing(sqlite3.connect(self.path, timeout=30)) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO embeddings(model_hash, doc_hash, vector) VALUES (?,?,?)",
                (self.version, digest, np.asarray(vector, dtype=np.float64).tobytes()),
            )
            connection.commit()
