from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class MatchStatus(str, Enum):
    complete = "complete"
    partial = "partial"
    no_match = "no_match"


class Requirement(ContractModel):
    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required: bool = True
    search_query: str | None = None
    depends_on: list[str] = Field(default_factory=list)


class Constraints(ContractModel):
    skills_dir: str | None = None
    rerank: Literal["off", "qwen3-0.6b"] | None = None
    top_k: int | None = Field(
        default=None, ge=1,
        description="Evidence items per requirement; defaults to server setting (5). Does not change acceptance.",
    )
    min_score: float | None = Field(default=None, ge=0)


class DraftPlanStep(ContractModel):
    id: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    requirement_ids: list[str] = Field(default_factory=list)


class DraftPlan(ContractModel):
    steps: list[DraftPlanStep] = Field(default_factory=list)


class SkillInjectRequest(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    requirements: list[Requirement] = Field(default_factory=list)
    draft_plan: DraftPlan | None = None
    constraints: Constraints | None = None


class EvidenceCitation(ContractModel):
    field: Literal["description", "body"]
    quote: str = Field(min_length=1)


class CheckResult(ContractModel):
    requirement_id: str
    matched: bool
    skill_id: str | None = None
    candidate_skill_id: str | None = None
    ranking_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    reason: str | None = None
    assessment: Literal["supported", "partial", "unsupported", "unknown", "blocked"] = "unknown"
    missing_terms: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    citations: list[EvidenceCitation] = Field(default_factory=list)
    unmet_requirements: list[str] = Field(default_factory=list)
    verifier: Literal["semantic", "lexical"] | None = None
    layer: Literal["meta", "domain"] | None = None


class EvidenceItem(ContractModel):
    skill_id: str
    requirement_id: str
    ranking_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    snippet: str | None = None
    name: str | None = None
    description: str | None = None
    layer: Literal["meta", "domain"] | None = None


class GapItem(ContractModel):
    requirement_id: str
    description: str
    reason: str


class ValidationErrorItem(ContractModel):
    code: str
    message: str
    path: str | None = None


class PlanBinding(ContractModel):
    step_id: str
    requirement_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)


class VerificationDiagnostic(ContractModel):
    """Sanitized attempt metadata; never raw content, reasoning, or exception messages."""

    requirement_id: str | None = None
    code: Literal[
        "response_truncated", "response_filtered", "response_refused", "unexpected_finish_reason",
        "invalid_response_json", "invalid_response_envelope", "empty_response_content",
        "invalid_verdict_json", "invalid_verdict_schema", "candidate_set_mismatch",
        "invalid_source_reference", "empty_source_reference", "missing_evidence",
        "supported_with_unmet_requirements", "partial_without_unmet_requirements",
        "http_timeout", "http_error", "transport_error",
    ]
    attempt: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    retrying: bool = False
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls", "function_call", "error"] | None = None
    http_status: int | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    output_chars: int | None = Field(default=None, ge=0)


class SkillInjectResponse(ContractModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    match_status: MatchStatus
    checks: list[CheckResult] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    gaps: list[GapItem] = Field(default_factory=list)
    validation_errors: list[ValidationErrorItem] = Field(default_factory=list)
    retriever_degraded: bool = False
    verification_degraded: bool = False
    verification_mode: Literal["semantic", "lexical"] | None = None
    verification_diagnostics: list[VerificationDiagnostic] = Field(default_factory=list)
    plan_bindings: list[PlanBinding] = Field(default_factory=list)
    skills_considered: int = 0
    notes: list[str] = Field(default_factory=list)
    registry_snapshot: str | None = None


class SkillMeta(ContractModel):
    skill_id: str
    name: str
    description: str
    body: str
    path: str
    depends_on: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    content_hash: str = ""
    source_path: str | None = None
    layer: Literal["meta", "domain"] = "domain"
