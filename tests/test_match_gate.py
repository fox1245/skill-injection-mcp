from __future__ import annotations

from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.retrieve.checks import evaluate_candidate, tokenize
from skill_inject_mcp.schemas import (
    MatchStatus,
    Requirement,
    SkillInjectRequest,
    SkillMeta,
)


def _hit(skill_id: str, *, sparse=None, dense=None, ranking=0.1):
    class _H:
        pass

    h = _H()
    h.skill_id = skill_id
    h.sparse_score = sparse
    h.dense_score = dense
    h.ranking_score = ranking
    h.dense_rank = 1
    h.sparse_rank = 1
    return h


def test_tokenize_drops_stopwords():
    toks = tokenize("The origami paper folding ritual guide and 이것은")
    assert "the" not in toks
    assert "and" not in toks
    assert "origami" in toks
    assert "paper" in toks


def test_origami_like_no_lexical_against_bm25():
    skill = SkillMeta(
        skill_id="python-bm25",
        name="Python BM25 Sparse Retriever",
        description="Implement BM25 and SQLite FTS5 sparse full-text search in Python.",
        body="Build SQLite FTS5 indexes with porter tokenization.",
        path="python-bm25/SKILL.md",
        tags=["bm25", "fts5", "sparse", "python", "retrieval"],
    )
    hit = _hit("python-bm25", sparse=0.02, dense=0.62)
    # High dense cosine (tiny-registry nearest neighbor) must not fulfill.
    assessment = evaluate_candidate(
        "origami paper folding ritual guide",
        skill,
        hit,
        runner_up_hit=None,
    )
    assert assessment.matched is False
    assert assessment.reason == "no_lexical_evidence"


def test_lexical_plus_sparse_accepts():
    skill = SkillMeta(
        skill_id="python-bm25",
        name="Python BM25 Sparse Retriever",
        description="Implement BM25 and SQLite FTS5 sparse full-text search in Python.",
        body="Unique diagnostic token: FTS5PorterTokenXyz",
        path="python-bm25/SKILL.md",
        tags=["bm25", "fts5"],
    )
    hit = _hit("python-bm25", sparse=0.4, dense=0.1)
    assessment = evaluate_candidate("FTS5PorterTokenXyz", skill, hit, None)
    assert assessment.matched is True
    assert assessment.reason == "lexical+sparse"


def test_unrelated_origami_intent_not_complete(tmp_path: Path):
    """Live-bug regression: origami-like intent must not complete on python-bm25."""
    only = tmp_path / "skills"
    src = Path(__file__).resolve().parents[1] / "fixtures" / "skills" / "python-bm25"
    dest = only / "python-bm25"
    dest.mkdir(parents=True)
    (dest / "SKILL.md").write_text((src / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8")

    eng = SkillInjectEngine(
        Settings(skills_dir=only, index_dir=tmp_path / "idx", use_fake_embedder=True)
    )
    eng.reindex()
    resp = eng.resolve(
        SkillInjectRequest(
            requirements=[
                Requirement(
                    id="origami",
                    description="origami paper folding ritual guide",
                    required=True,
                )
            ]
        )
    )
    assert resp.match_status != MatchStatus.complete
    assert resp.match_status in (MatchStatus.no_match, MatchStatus.partial)
    assert resp.checks[0].matched is False
    assert resp.checks[0].reason == "no_lexical_evidence"
    assert any(g.requirement_id == "origami" for g in resp.gaps)
