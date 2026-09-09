"""Build independent index generations and publish only successful snapshots."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skill_inject_mcp.config import Settings
from skill_inject_mcp.embed.embedder import Embedder, build_embedder
from skill_inject_mcp.index.sparse import SparseIndex
from skill_inject_mcp.index.cache import EmbeddingCache
from skill_inject_mcp.index.vector import VectorIndex, build_vector_index
from skill_inject_mcp.registry.scan import SkillRegistry


def embedding_identity(settings: Settings) -> tuple:
    return (
        "fake" if settings.use_fake_embedder or not settings.resolve_api_key() else "openrouter",
        settings.embedding_model, settings.embedding_dim, settings.embedding_base_url,
    )


def fingerprint(registry: SkillRegistry, root: Path, identity: tuple) -> str:
    data = {
        "root": str(root.resolve()), "embedding": identity,
        "skills": [s.model_dump(mode="json") for s in registry.all()],
        "errors": [e.model_dump(mode="json") for e in registry.validation_errors],
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _remove_generation(directory: Path, index_root: Path) -> None:
    # Never recursively delete caller-supplied paths. Only our generated child.
    target, parent = directory.resolve(), index_root.resolve()
    if target.parent != parent or not target.name.startswith("snapshot-"):
        raise ValueError("Refusing to remove a path outside the managed index generation")
    shutil.rmtree(target)


@dataclass
class IndexSnapshot:
    registry: SkillRegistry
    root: Path
    snapshot_id: str
    identity: tuple
    embedder: Embedder
    embedder_degraded: bool
    sparse: SparseIndex
    dense: VectorIndex
    backend: str
    vectors: dict[str, np.ndarray]
    directory: Path
    index_root: Path

    def info(self) -> dict:
        return {
            "skills_indexed": len(self.registry.skills),
            "dense_backend": self.backend,
            "embedder": type(self.embedder).__name__,
            "embedder_degraded": self.embedder_degraded,
            "validation_errors": [e.model_dump() for e in self.registry.validation_errors],
            "skills_dir": str(self.root),
            "registry_snapshot": self.snapshot_id,
        }

    def close(self) -> None:
        self.sparse.close()
        self.dense.close()
        if self.directory.exists():
            _remove_generation(self.directory, self.index_root)


def build_snapshot(
    settings: Settings, registry: SkillRegistry, root: Path, snapshot_id: str,
    previous: IndexSnapshot | None,
) -> IndexSnapshot:
    identity = embedding_identity(settings)
    if previous is not None and previous.identity == identity:
        embedder, degraded = previous.embedder, previous.embedder_degraded
        cached = dict(previous.vectors)
    else:
        embedder, degraded = build_embedder(
            api_key=settings.resolve_api_key(), use_fake=settings.use_fake_embedder,
            model=settings.embedding_model, dim=settings.embedding_dim,
            base_url=settings.embedding_base_url, timeout_s=settings.embedding_timeout_s,
        )
        cached = {}
    skills = registry.all()
    documents = {
        s.skill_id: f"{s.name}\n{s.description}\n{s.body}\n{' '.join(s.tags)}"
        for s in skills
    }
    hashes = {sid: hashlib.sha256(text.encode()).hexdigest() for sid, text in documents.items()}
    disk_cache = EmbeddingCache(Path(settings.index_dir), identity, settings.embedding_dim) if settings.persistent_embedding_cache else None
    pending = {}
    vectors = {}
    for sid, digest in hashes.items():
        if digest in cached:
            vectors[digest] = cached[digest]
        else:
            stored = disk_cache.get(digest) if disk_cache else None
            if stored is not None:
                vectors[digest] = stored
            else:
                pending[digest] = documents[sid]
    # Complete remote work before creating or touching any live index files.
    items = list(pending.items())
    for offset in range(0, len(items), settings.embedding_batch_size):
        batch = items[offset:offset + settings.embedding_batch_size]
        embedded = embedder.embed_documents([text for _, text in batch])
        if len(embedded) != len(batch):
            raise ValueError("Embedding response count does not match document count")
        for (digest, _), vector in zip(batch, embedded):
            array = np.asarray(vector, dtype=np.float64)
            if array.shape != (settings.embedding_dim,) or not np.isfinite(array).all():
                raise ValueError("Embedding response has invalid shape or non-finite values")
            vectors[digest] = array.copy()
            if disk_cache is not None:
                disk_cache.put(digest, vectors[digest])

    index_root = Path(settings.index_dir).resolve()
    index_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="snapshot-", dir=index_root))
    sparse = dense = None
    try:
        dense, backend = build_vector_index(directory, dim=settings.embedding_dim)
        sparse = SparseIndex(directory / "sparse.sqlite")
        sparse.upsert_many(skills)
        dense.upsert_many([(s.skill_id, vectors[hashes[s.skill_id]]) for s in skills])
        return IndexSnapshot(
            registry, root, snapshot_id, identity, embedder, degraded, sparse, dense,
            backend, vectors, directory, index_root,
        )
    except Exception:
        if sparse is not None:
            sparse.close()
        if dense is not None:
            dense.close()
        _remove_generation(directory, index_root)
        raise
