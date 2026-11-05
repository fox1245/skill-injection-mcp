"""Build independent index generations and publish only successful snapshots."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np

from skill_inject_mcp.config import Settings
from skill_inject_mcp.embed.admission import build_admission
from skill_inject_mcp.embed.embedder import Embedder, build_embedder
from skill_inject_mcp.index.locks import FileLock
from skill_inject_mcp.index.sparse import SparseIndex
from skill_inject_mcp.index.cache import EmbeddingCache
from skill_inject_mcp.index.vector import VectorIndex, build_vector_index
from skill_inject_mcp.registry.scan import SkillRegistry


def embedding_identity(settings: Settings) -> tuple:
    if settings.retrieval_mode == "lexical":
        return ("lexical",)
    return (
        "fake" if settings.use_fake_embedder else "openrouter",
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


def _create_generation(index_root: Path) -> tuple[Path, FileLock]:
    root = Path(index_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="snapshot-", dir=root))
    lease = FileLock(directory / "lease.lock")
    try:
        if not lease.try_acquire():
            raise RuntimeError(f"Could not acquire lease for index generation: {directory}")
        return directory, lease
    except BaseException:
        lease.release()
        if directory.exists():
            _remove_generation(directory, root)
        raise


def sweep_orphan_generations(index_root: Path, build_lock: FileLock) -> list[str]:
    if build_lock is None or not build_lock.held:
        raise RuntimeError("sweep_orphan_generations requires a held build_lock")
    root = Path(index_root).resolve()
    if not root.is_dir():
        return []
    removed: list[str] = []
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if not entry.name.startswith("snapshot-") or not entry.is_dir():
            continue
        lease_path = entry / "lease.lock"
        if lease_path.exists():
            probe = FileLock(lease_path)
            if not probe.try_acquire():
                continue
            probe.release()
        try:
            _remove_generation(entry, root)
            removed.append(entry.name)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Orphan index generation cleanup failed (%s)", type(exc).__name__,
            )
    return removed


@dataclass
class IndexSnapshot:
    registry: SkillRegistry
    root: Path
    snapshot_id: str
    identity: tuple
    embedder: Embedder | None
    embedder_degraded: bool
    sparse: SparseIndex
    dense: VectorIndex | None
    backend: str
    vectors: dict[str, np.ndarray]
    directory: Path
    index_root: Path
    source_signature: tuple | None = None
    lease: FileLock | None = None
    _lease_lock: Lock = field(default_factory=Lock, repr=False)
    _leases: int = field(default=0, repr=False)
    _retired: bool = field(default=False, repr=False)
    _disposed: bool = field(default=False, repr=False)

    def acquire(self) -> bool:
        with self._lease_lock:
            if self._retired:
                return False
            self._leases += 1
            return True

    def release(self, *, defer_disposal: bool = False):
        with self._lease_lock:
            if self._leases < 1:
                raise RuntimeError("Snapshot lease released without acquisition")
            self._leases -= 1
            dispose = self._retired and not self._leases and not self._disposed
            if dispose:
                self._disposed = True
        if dispose:
            if defer_disposal:
                return self._dispose
            self._dispose()
        return None

    def info(self) -> dict:
        return {
            "skills_indexed": len(self.registry.skills),
            "dense_backend": self.backend,
            "embedder": type(self.embedder).__name__ if self.embedder is not None else "none",
            "embedder_degraded": self.embedder_degraded,
            "validation_errors": [e.model_dump() for e in self.registry.validation_errors],
            "skills_dir": str(self.root),
            "registry_snapshot": self.snapshot_id,
        }

    def close(self) -> None:
        with self._lease_lock:
            self._retired = True
            dispose = not self._leases and not self._disposed
            if dispose:
                self._disposed = True
        if dispose:
            self._dispose()

    def _dispose(self) -> None:
        try:
            self.sparse.close()
        finally:
            try:
                if self.dense is not None:
                    self.dense.close()
            finally:
                if self.lease is not None:
                    self.lease.release()
                if self.directory.exists():
                    _remove_generation(self.directory, self.index_root)


def build_snapshot(
    settings: Settings, registry: SkillRegistry, root: Path, snapshot_id: str,
    previous: IndexSnapshot | None,
    *, embedder_state: tuple[Embedder, bool] | None = None,
) -> IndexSnapshot:
    identity = embedding_identity(settings)
    if settings.retrieval_mode == "lexical":
        index_root = Path(settings.index_dir).resolve()
        directory, lease = _create_generation(index_root)
        sparse = None
        try:
            sparse = SparseIndex(directory / "sparse.sqlite")
            sparse.upsert_many(registry.all())
            return IndexSnapshot(
                registry, root, snapshot_id, identity, None, False, sparse, None,
                "off", {}, directory, index_root, lease=lease,
            )
        except BaseException:
            if sparse is not None:
                sparse.close()
            lease.release()
            if directory.exists():
                _remove_generation(directory, index_root)
            raise
    if previous is not None and previous.identity == identity:
        embedder, degraded = embedder_state or (previous.embedder, previous.embedder_degraded)
        cached = dict(previous.vectors)
    else:
        embedder, degraded = embedder_state or build_embedder(
            api_key=settings.resolve_api_key(), use_fake=settings.use_fake_embedder,
            model=settings.embedding_model, dim=settings.embedding_dim,
            base_url=settings.embedding_base_url, timeout_s=settings.embedding_timeout_s,
            query_cache_size=settings.query_cache_size,
            admission=None if settings.use_fake_embedder else build_admission(settings),
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
    directory, lease = _create_generation(index_root)
    sparse = dense = None
    try:
        dense, backend = build_vector_index(directory, dim=settings.embedding_dim,
                                            backend=settings.dense_backend, extension_path=settings.sqlite_vector_path)
        sparse = SparseIndex(directory / "sparse.sqlite")
        sparse.upsert_many(skills)
        dense.upsert_many([(s.skill_id, vectors[hashes[s.skill_id]]) for s in skills])
        return IndexSnapshot(
            registry, root, snapshot_id, identity, embedder, degraded, sparse, dense,
            backend, vectors, directory, index_root, lease=lease,
        )
    except BaseException:
        if sparse is not None:
            sparse.close()
        if dense is not None:
            dense.close()
        lease.release()
        if directory.exists():
            _remove_generation(directory, index_root)
        raise
