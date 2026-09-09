from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "1.0"


class MatchStatus(str, Enum):
    complete = "complete"
    partial = "partial"
    no_match = "no_match"


class Requirement(BaseModel):
    id: str
    description: str
    required: bool = True
    search_query: str | None = None
    depends_on: list[str] = Field(default_factory=list)


class Constraints(BaseModel):
    skills_dir: str | None = None
    rerank: Literal["off", "qwen3-0.6b"] = "off"
    top_k: int = Field(
        default=5,
        ge=1,
        description=(
            "How many hybrid candidates to walk / keep as evidence per requirement. "
            "Optional; default 5. Internal sparse/dense channel remains Settings.retrieve_top_k (20)."
        ),
    )
    min_score: float | None = None


class DraftPlanStep(BaseModel):
    id: str
    summary: str
    requirement_ids: list[str] = Field(default_factory=list)


class DraftPlan(BaseModel):
    steps: list[DraftPlanStep] = Field(default_factory=list)


class SkillInjectRequest(BaseModel):
    schema_version: str = SCHEMA_VERSION
    requirements: list[Requirement] = Field(default_factory=list)
    draft_plan: DraftPlan | None = None
    constraints: Constraints | None = None


class CheckResult(BaseModel):
    requirement_id: str
    matched: bool
    skill_id: str | None = None
    ranking_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    reason: str | None = None


class EvidenceItem(BaseModel):
    skill_id: str
    requirement_id: str
    ranking_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    snippet: str | None = None
    name: str | None = None
    description: str | None = None


class GapItem(BaseModel):
    requirement_id: str
    description: str
    reason: str


class ValidationErrorItem(BaseModel):
    code: str
    message: str
    path: str | None = None


class PlanBinding(BaseModel):
    step_id: str
    requirement_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)


class SkillInjectResponse(BaseModel):
    schema_version: str = SCHEMA_VERSION
    match_status: MatchStatus
    checks: list[CheckResult] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    validation_errors: list[ValidationErrorItem] = Field(default_factory=list)
    retriever_degraded: bool = False
    plan_bindings: list[PlanBinding] = Field(default_factory=list)
    skills_considered: int = 0
    notes: list[str] = Field(default_factory=list)


class SkillMeta(BaseModel):
    skill_id: str
    name: str
    description: str
    body: str
    path: str
    depends_on: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    frontmatter: dict[str, Any] = Field(default_factory=dict)

