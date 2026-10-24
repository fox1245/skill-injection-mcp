from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.schemas import Constraints, Requirement, SkillInjectRequest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "skills"


def request(description: str, **kwargs) -> SkillInjectRequest:
    return SkillInjectRequest(requirements=[Requirement(id="r", description=description, **kwargs)])


def write_skill(root: Path, sid: str, description: str, *, body: str = "", deps=()) -> None:
    target = root / sid / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    dependency_line = "depends_on: [" + ", ".join(deps) + "]\n"
    target.write_text(
        f"---\nid: {sid}\nname: {sid}\ndescription: {description}\n"
        f"{dependency_line}---\n{body}\n", encoding="utf-8",
    )


def make_engine(tmp_path: Path, root: Path | None = None) -> SkillInjectEngine:
    return SkillInjectEngine(Settings(
        _env_file=None, skills_dir=root or FIXTURES, index_dir=tmp_path / "idx",
        use_fake_embedder=True, multi_query=False,
    ))


@pytest.mark.parametrize("description", [
    "python chess engine using alpha beta pruning",
    "Implement BM25 FTS5 sparse search and dense vector embeddings",
])
def test_partial_vocabulary_is_not_fulfillment(engine, description):
    result = engine.resolve(request(description))
    assert result.match_status != "complete"
    assert not result.checks[0].matched


def test_search_hint_cannot_replace_requirement(engine):
    result = engine.resolve(request(
        "Deploy a Kubernetes cluster with Helm and enforce Istio network policy",
        search_query="Install Python packages and project dependencies with pip",
    ))
    assert result.match_status != "complete"
    assert not result.checks[0].matched


def test_non_goals_are_not_capabilities(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "builder", "Compile CMake binaries", body=(
        "# Capabilities\nCompile CMake binaries.\n"
        "## Non-goals\nDoes **not** install Python packages with pip.\n"
        "### Details\nPython packages pip installation is unsupported.\n"
    ))
    for n in range(20):
        write_skill(root, f"paper-{n}", "Fold origami paper cranes")
    result = make_engine(tmp_path, root).resolve(request("Install Python packages with pip"))
    assert result.match_status != "complete"


def test_korean_exact_match(tmp_path):
    root = tmp_path / "skills"
    description = "파이썬 패키지 설치와 의존성 관리"
    write_skill(root, "korean", description, body=description + " 절차")
    result = make_engine(tmp_path, root).resolve(request(description))
    assert result.match_status == "complete"
    assert result.checks[0].skill_id == "korean"


def test_unknown_requirement_dependency_blocks_completion(engine):
    result = engine.resolve(request(
        "Install Python packages and project dependencies with pip", depends_on=["missing"],
    ))
    assert result.match_status != "complete"
    assert any(e.code == "unknown_requirement_dependency" for e in result.validation_errors)


def test_missing_skill_dependency_blocks_binding(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "installer", "Install Python packages with pip", deps=["missing"])
    result = make_engine(tmp_path, root).resolve(request("Install Python packages with pip"))
    assert result.match_status != "complete"
    assert not result.checks[0].matched


@pytest.mark.parametrize("steps", [
    [{"id": "s", "summary": "install", "requirement_ids": ["missing"]}],
    [{"id": "s", "summary": "one", "requirement_ids": ["r"]},
     {"id": "s", "summary": "two", "requirement_ids": ["r"]}],
])
def test_invalid_plan_references_are_rejected(engine, steps):
    req = request("Install Python packages and project dependencies with pip")
    from skill_inject_mcp.schemas import DraftPlan
    req.draft_plan = DraftPlan.model_validate({"steps": steps})
    result = engine.resolve(req)
    assert result.match_status != "complete"
    assert result.validation_errors
    assert not result.plan_bindings


def test_unmatched_optional_dependency_blocks_required_dependent(engine):
    req = SkillInjectRequest(requirements=[
        Requirement(id="dependency", description="Configure Kubernetes Helm Istio", required=False),
        Requirement(id="dependent", description="Install Python packages with pip", depends_on=["dependency"]),
    ])
    result = engine.resolve(req)
    assert result.match_status != "complete"
    assert not next(c for c in result.checks if c.requirement_id == "dependent").matched


def test_skill_dependency_closure_is_bound_in_order(tmp_path):
    root = tmp_path / "skills"
    write_skill(root, "base", "Prepare environment")
    write_skill(root, "installer", "Install Python packages with pip", deps=["base"])
    result = make_engine(tmp_path, root).resolve(request("Install Python packages with pip"))
    assert result.match_status == "complete"
    assert result.plan_bindings[0].skill_ids == ["base", "installer"]


@pytest.mark.parametrize("payload", [
    {"schema_version": "99.0"},
    {"goal": "silently ignored"},
    {"requirements": [{"id": "r", "description": "   "}]},
    {"requirements": [{"id": "r", "description": "Install packages", "task_kind": "verify"}]},
    {"constraints": {"must": ["No network"]}},
])
def test_invalid_contract_does_not_silently_drop_fields(payload):
    with pytest.raises(ValidationError):
        SkillInjectRequest.model_validate(payload)


def test_mcp_exposes_typed_schemas():
    from skill_inject_mcp.server import mcp
    tool = next(t for t in asyncio.run(mcp.list_tools()) if t.name == "resolve_skills")
    assert "anyOf" not in tool.inputSchema["properties"]["request"]
    assert "match_status" in tool.outputSchema["properties"]
    assert "checks" in tool.outputSchema["properties"]


def test_top_k_does_not_hide_dense_runner_up(engine, monkeypatch):
    from skill_inject_mcp.retrieve.hybrid import RRFResult
    hits = [
        RRFResult("python-bm25", .03, 1, None, .60, None),
        RRFResult("sqlite-vector-adapter", .02, 2, None, .59, None),
    ]
    monkeypatch.setattr(engine, "ensure_index", lambda **kw: {})
    monkeypatch.setattr(engine.retriever, "retrieve",
                        lambda query, top_k=None, queries=None: hits[:top_k])
    for width in (1, 2):
        req = request("BM25")
        req.constraints = Constraints(top_k=width)
        result = engine.resolve(req)
        assert not result.checks[0].matched


def test_negative_requirement_is_not_silently_rewritten(engine):
    result = engine.resolve(request("Do not install Python packages with pip"))
    assert result.match_status != "complete"


def test_mcp_call_returns_validated_structured_output(engine, monkeypatch):
    from skill_inject_mcp import server
    from skill_inject_mcp.schemas import SkillInjectResponse
    monkeypatch.setattr(server, "_engine", engine)
    result = asyncio.run(server.mcp.call_tool(
        "resolve_skills",
        {"request": request("Install Python packages with pip").model_dump()},
    ))
    content, structured = result
    assert content
    response = SkillInjectResponse.model_validate(structured)
    assert response.match_status == "complete"
    assert response.checks[0].assessment == "supported"
