from __future__ import annotations

from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.schemas import (
    MatchStatus,
    Requirement,
    SkillInjectRequest,
    Constraints,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "skills"


def _req(**kwargs) -> Requirement:
    return Requirement(**kwargs)


def test_install_intent_not_false_complete_with_build_only(tmp_path: Path):
    """Install intent must not falsely complete using a build-only skill."""
    only_build = tmp_path / "skills"
    src = FIXTURES / "cmake-builder"
    dest = only_build / "cmake-builder"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text((src / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8")

    eng = SkillInjectEngine(
        Settings(skills_dir=only_build, index_dir=tmp_path / "idx", use_fake_embedder=True)
    )
    eng.reindex()

    resp = eng.resolve(
        SkillInjectRequest(
            requirements=[
                _req(
                    id="install-deps",
                    description="Install Python packages and project dependencies with pip",
                    required=True,
                )
            ]
        )
    )
    assert resp.match_status != MatchStatus.complete

    eng2 = SkillInjectEngine(
        Settings(skills_dir=FIXTURES, index_dir=tmp_path / "idx2", use_fake_embedder=True)
    )
    eng2.reindex()
    resp2 = eng2.resolve(
        SkillInjectRequest(
            requirements=[
                _req(
                    id="install-deps",
                    description="Install Python packages and project dependencies with pip",
                    required=True,
                )
            ]
        )
    )
    assert resp2.match_status == MatchStatus.complete
    assert resp2.checks[0].skill_id == "package-installer"


def test_sparse_and_dense_need_with_sparse_only_is_partial(tmp_path: Path):
    """Two required needs (sparse + dense) with only sparse skill -> partial/no_match."""
    only_sparse = tmp_path / "skills"
    src = FIXTURES / "python-bm25"
    dest = only_sparse / "python-bm25"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text((src / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8")

    eng = SkillInjectEngine(
        Settings(skills_dir=only_sparse, index_dir=tmp_path / "idx", use_fake_embedder=True)
    )
    eng.reindex()

    resp = eng.resolve(
        SkillInjectRequest(
            requirements=[
                _req(
                    id="need-sparse",
                    description="BM25 FTS5 sparse full-text search in Python",
                    required=True,
                ),
                _req(
                    id="need-dense",
                    description="Dense vector embedding search with sqlite-vector adapter",
                    required=True,
                ),
            ]
        )
    )
    assert resp.match_status in (MatchStatus.partial, MatchStatus.no_match)
    assert resp.match_status != MatchStatus.complete
    dense_check = next(c for c in resp.checks if c.requirement_id == "need-dense")
    assert dense_check.matched is False
    assert any(g.requirement_id == "need-dense" for g in resp.gaps)


def test_missing_skill_no_match(engine: SkillInjectEngine):
    resp = engine.resolve(
        SkillInjectRequest(
            requirements=[
                _req(
                    id="need-kubernetes",
                    description="Deploy Helm charts to a Kubernetes cluster with kubectl",
                    required=True,
                    search_query="HelmKubernetesKubectlZZZNOMATCH",
                )
            ],
            constraints=Constraints(min_score=0.05, top_k=3),
        )
    )
    assert resp.match_status == MatchStatus.no_match
    assert resp.gaps


def test_synonym_via_dense_fake_embedder(engine: SkillInjectEngine):
    resp = engine.resolve(
        SkillInjectRequest(
            requirements=[
                _req(
                    id="syn-install",
                    description="Add project dependencies using the pip installer",
                    required=True,
                )
            ]
        )
    )
    assert resp.match_status == MatchStatus.complete
    assert resp.checks[0].skill_id == "package-installer"
    assert resp.checks[0].dense_rank is not None


def test_unique_token_via_bm25(engine: SkillInjectEngine):
    resp = engine.resolve(
        SkillInjectRequest(
            requirements=[
                _req(
                    id="uniq",
                    description="Locate the diagnostic token FTS5PorterTokenXyz in sparse index",
                    required=True,
                    search_query="FTS5PorterTokenXyz",
                )
            ]
        )
    )
    assert resp.match_status == MatchStatus.complete
    assert resp.checks[0].skill_id == "python-bm25"
    assert resp.checks[0].sparse_rank == 1


def test_dag_cycle_and_duplicate_ids_validation(tmp_path: Path):
    eng_dup = SkillInjectEngine(
        Settings(
            skills_dir=ROOT / "fixtures" / "bad_skills",
            index_dir=tmp_path / "idx_dup",
            use_fake_embedder=True,
        )
    )
    info = eng_dup.reindex()
    assert any(e["code"] == "duplicate_skill_id" for e in info["validation_errors"])

    resp = eng_dup.resolve(
        SkillInjectRequest(
            requirements=[_req(id="r1", description="anything about dup", required=True)]
        )
    )
    assert any(e.code == "duplicate_skill_id" for e in resp.validation_errors)
    assert resp.match_status != MatchStatus.complete

    eng_cyc = SkillInjectEngine(
        Settings(
            skills_dir=ROOT / "fixtures" / "bad_skills_cycle",
            index_dir=tmp_path / "idx_cyc",
            use_fake_embedder=True,
        )
    )
    info2 = eng_cyc.reindex()
    assert any(e["code"] == "dependency_cycle" for e in info2["validation_errors"])

    eng_ok = SkillInjectEngine(
        Settings(skills_dir=FIXTURES, index_dir=tmp_path / "idx_ok", use_fake_embedder=True)
    )
    eng_ok.reindex()
    resp_cyc = eng_ok.resolve(
        SkillInjectRequest(
            requirements=[
                _req(id="a", description="Install packages with pip", required=True, depends_on=["b"]),
                _req(id="b", description="BM25 FTS5 sparse search", required=True, depends_on=["a"]),
            ]
        )
    )
    assert any(e.code == "requirement_dependency_cycle" for e in resp_cyc.validation_errors)
    assert resp_cyc.match_status != MatchStatus.complete


def test_required_unresolved_not_complete(engine: SkillInjectEngine):
    resp = engine.resolve(
        SkillInjectRequest(
            requirements=[
                _req(
                    id="ok-install",
                    description="Install Python packages with pip",
                    required=True,
                ),
                _req(
                    id="missing-k8s",
                    description="Orchestrate multi-cluster Istio service mesh federation",
                    required=True,
                    search_query="IstioXYZUNIQUE999Federation",
                ),
            ],
            constraints=Constraints(min_score=0.05, top_k=3),
        )
    )
    missing = next(c for c in resp.checks if c.requirement_id == "missing-k8s")
    assert missing.matched is False
    assert resp.match_status != MatchStatus.complete
    assert any(g.requirement_id == "missing-k8s" for g in resp.gaps)


def test_rerank_stub_sets_degraded(engine: SkillInjectEngine):
    resp = engine.resolve(
        SkillInjectRequest(
            requirements=[
                _req(id="r", description="Install Python packages with pip", required=True)
            ],
            constraints=Constraints(rerank="qwen3-0.6b"),
        )
    )
    assert resp.retriever_degraded is True
