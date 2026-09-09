from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skill_inject_mcp.config import Settings
from skill_inject_mcp.embed.embedder import FakeEmbedder
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest


def write_skill(root: Path, sid: str, text: str):
    path = root / sid / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\nname: {sid}\ndescription: {text}\n---\n{text}\n", encoding="utf-8")
    return path


@pytest.fixture
def harness(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    write_skill(root, "installer", "Install Python packages with pip")
    write_skill(root, "indexer", "Build BM25 indexes")
    calls = []
    original = FakeEmbedder.embed_documents

    def record(self, texts):
        calls.append(list(texts))
        return original(self, texts)

    monkeypatch.setattr(FakeEmbedder, "embed_documents", record)
    engine = SkillInjectEngine(Settings(
        _env_file=None, skills_dir=root, index_dir=tmp_path / "index",
        use_fake_embedder=True, multi_query=False,
    ))
    yield engine, root, calls
    engine.close()


def req():
    return SkillInjectRequest(requirements=[
        Requirement(id="r", description="Install Python packages with pip"),
    ])


def test_repeated_resolve_reuses_snapshot_and_embeddings(harness):
    engine, root, calls = harness
    first = engine.resolve(req())
    retriever = engine.retriever
    second = engine.resolve(req())
    assert first.match_status == second.match_status == "complete"
    assert engine.retriever is retriever
    assert first.registry_snapshot == second.registry_snapshot
    assert len(calls) == 1
    assert len(calls[0]) == 2


def test_change_reembeds_only_changed_document(harness):
    engine, root, calls = harness
    first = engine.resolve(req())
    old_directory = engine._snapshot.directory
    write_skill(root, "indexer", "Build SQLite BM25 indexes")
    second = engine.resolve(req())
    assert first.registry_snapshot != second.registry_snapshot
    assert [len(batch) for batch in calls] == [2, 1]
    assert not old_directory.exists()
    stale = engine.get_skill_body("installer", first.registry_snapshot)
    assert stale["reason"] == "snapshot_mismatch"
    assert engine.get_skill_body("installer", second.registry_snapshot)["found"]


def test_delete_does_not_reembed_unchanged_documents(harness):
    engine, root, calls = harness
    engine.resolve(req())
    (root / "indexer" / "SKILL.md").unlink()
    engine.resolve(req())
    assert [len(batch) for batch in calls] == [2]
    assert engine.registry.get("indexer") is None
    assert engine.sparse.count() == engine.dense.count() == 1


def test_embedding_failure_preserves_previous_snapshot(harness, monkeypatch):
    engine, root, calls = harness
    first = engine.resolve(req())
    previous = engine._snapshot

    def fail(*args, **kwargs):
        raise RuntimeError("simulated embedding outage")

    monkeypatch.setattr(FakeEmbedder, "embed_documents", fail)
    write_skill(root, "installer", "Install Python packages with pip and uv")
    with pytest.raises(RuntimeError, match="simulated"):
        engine.reindex()
    assert engine._snapshot is previous
    assert previous.directory.exists()
    assert engine.sparse.count() == engine.dense.count() == 2
    assert engine.retriever.retrieve("Install Python packages with pip")
    body = engine.get_skill_body("installer", first.registry_snapshot)
    assert body["found"]
    assert "uv" not in body["description"]


def test_database_failure_preserves_live_snapshot_and_cleans_candidate(harness, monkeypatch):
    from skill_inject_mcp.index.sparse import SparseIndex
    engine, root, calls = harness
    engine.resolve(req())
    previous = engine._snapshot

    def fail(*args, **kwargs):
        raise RuntimeError("simulated database write failure")

    monkeypatch.setattr(SparseIndex, "upsert_many", fail)
    write_skill(root, "indexer", "Build SQLite indexes")
    with pytest.raises(RuntimeError, match="database"):
        engine.reindex()
    assert engine._snapshot is previous
    assert list(previous.index_root.iterdir()) == [previous.directory]
    assert engine.sparse.count() == 2


def test_wrong_embedding_count_is_not_published(harness, monkeypatch):
    engine, root, calls = harness
    engine.resolve(req())
    previous = engine._snapshot
    write_skill(root, "indexer", "Build SQLite indexes")
    monkeypatch.setattr(FakeEmbedder, "embed_documents", lambda *args: [])
    with pytest.raises(ValueError, match="count"):
        engine.reindex()
    assert engine._snapshot is previous


def test_root_switch_and_model_change_invalidate_snapshot(harness, tmp_path):
    engine, root, calls = harness
    first = engine.resolve(req())
    other = tmp_path / "other"
    write_skill(other, "installer", "Install Ruby gems")
    info = engine.ensure_index(other)
    assert info["registry_snapshot"] != first.registry_snapshot
    assert engine.get_skill_body("installer")["description"] == "Install Ruby gems"
    engine.ensure_index(root)
    before = len(calls)
    engine.settings.embedding_dim = 256
    engine.ensure_index(root)
    assert len(calls) == before + 1
    assert engine.dense.dim == 256


def test_invalid_request_does_not_index_or_call_network(harness, monkeypatch):
    engine, root, calls = harness

    def unexpected(**kwargs):
        pytest.fail("Invalid references must be rejected before indexing")

    monkeypatch.setattr(engine, "ensure_index", unexpected)
    bad = req()
    bad.requirements[0].depends_on = ["missing"]
    result = engine.resolve(bad)
    assert result.match_status == "no_match"
    assert calls == []


def test_numpy_bulk_save_round_trip_without_pickle(tmp_path, monkeypatch):
    from skill_inject_mcp.index.vector import NumpyVectorIndex
    path = tmp_path / "vectors.npz"
    index = NumpyVectorIndex(dim=2, persist_path=path)
    calls = []
    original = np.savez_compressed

    def record(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "savez_compressed", record)
    index.upsert_many([("b", np.array([1., 0.])), ("a", np.array([1., 0.]))])
    assert calls == [1]
    loaded = NumpyVectorIndex(dim=2, persist_path=path)
    assert loaded.search(np.array([1., 0.])) == [("a", 1.), ("b", 1.)]


def test_embedding_batches_are_bounded(harness):
    engine, root, calls = harness
    engine.settings.embedding_batch_size = 1
    engine.reindex()
    assert [len(batch) for batch in calls] == [1, 1]


def test_persistent_cache_survives_engine_restart(harness, monkeypatch):
    engine, root, calls = harness
    engine.settings.persistent_embedding_cache = True
    settings = engine.settings.model_copy()
    engine.resolve(req())
    engine.close()
    restarted = SkillInjectEngine(settings)
    try:
        result = restarted.resolve(req())
        assert result.match_status == "complete"
        assert [len(batch) for batch in calls] == [2]
    finally:
        restarted.close()


def test_corrupt_persistent_vector_is_recomputed(harness):
    engine, root, calls = harness
    engine.settings.persistent_embedding_cache = True
    settings = engine.settings.model_copy()
    engine.resolve(req())
    engine.close()
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(Path(settings.index_dir) / "embedding-cache.sqlite")) as connection:
        connection.execute("UPDATE embeddings SET vector=? WHERE doc_hash=(SELECT doc_hash FROM embeddings LIMIT 1)",
                           (b"invalid cached vector",))
        connection.commit()
    restarted = SkillInjectEngine(settings)
    try:
        restarted.resolve(req())
        assert [len(batch) for batch in calls] == [2, 1]
    finally:
        restarted.close()
