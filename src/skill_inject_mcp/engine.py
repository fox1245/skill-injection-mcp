from __future__ import annotations

from pathlib import Path

from skill_inject_mcp.config import Settings, get_settings
from skill_inject_mcp.embed.embedder import build_embedder
from skill_inject_mcp.index.sparse import SparseIndex
from skill_inject_mcp.index.vector import build_vector_index
from skill_inject_mcp.registry.scan import SkillRegistry
from skill_inject_mcp.retrieve.checks import evaluate_candidate
from skill_inject_mcp.retrieve.hybrid import HybridRetriever
from skill_inject_mcp.schemas import (
    CheckResult,
    EvidenceItem,
    GapItem,
    MatchStatus,
    PlanBinding,
    SkillInjectRequest,
    SkillInjectResponse,
    ValidationErrorItem,
)


def expand_query(description: str, search_query: str | None) -> str:
    if search_query and search_query.strip():
        return search_query.strip()
    # Server-side expansion: keep description + light keyword hints
    return description.strip()


class SkillInjectEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.registry = SkillRegistry()
        self.sparse: SparseIndex | None = None
        self.dense = None
        self.dense_backend = "numpy"
        self.embedder = None
        self.embedder_degraded = False
        self.retriever: HybridRetriever | None = None
        self._indexed = False

    def ensure_index(self, skills_dir: Path | None = None) -> dict:
        return self.reindex(skills_dir=skills_dir)

    def reindex(self, skills_dir: Path | None = None) -> dict:
        s = self.settings
        root = Path(skills_dir) if skills_dir else Path(s.skills_dir)
        index_dir = Path(s.index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        skills = self.registry.load(root)
        self.embedder, self.embedder_degraded = build_embedder(
            api_key=s.resolve_api_key(),
            use_fake=s.use_fake_embedder,
            model=s.embedding_model,
            dim=s.embedding_dim,
            base_url=s.embedding_base_url,
        )
        self.dense, self.dense_backend = build_vector_index(index_dir, dim=s.embedding_dim)
        self.sparse = SparseIndex(index_dir / "sparse.sqlite")

        self.dense.clear()
        self.sparse.clear()

        docs = []
        ids = []
        for skill in skills:
            text = f"{skill.name}\n{skill.description}\n{skill.body}\n{' '.join(skill.tags)}"
            docs.append(text)
            ids.append(skill.skill_id)
            self.sparse.upsert(
                skill.skill_id,
                skill.name,
                skill.description,
                skill.body,
                skill.tags,
            )

        if docs:
            vectors = self.embedder.embed_documents(docs)
            for sid, vec in zip(ids, vectors):
                self.dense.upsert(sid, vec)

        self.retriever = HybridRetriever(
            sparse=self.sparse,
            dense=self.dense,
            embedder=self.embedder,
            skills=self.registry.skills,
            rrf_k=s.rrf_k,
            retrieve_top_k=s.retrieve_top_k,
        )
        self._indexed = True
        return {
            "skills_indexed": len(skills),
            "dense_backend": self.dense_backend,
            "embedder": type(self.embedder).__name__,
            "embedder_degraded": self.embedder_degraded,
            "validation_errors": [e.model_dump() for e in self.registry.validation_errors],
            "skills_dir": str(root.resolve()),
        }

    def get_skill_body(self, skill_id: str) -> dict:
        if not self._indexed:
            self.ensure_index()
        skill = self.registry.get(skill_id)
        if not skill:
            return {"found": False, "skill_id": skill_id, "body": None}
        return {
            "found": True,
            "skill_id": skill.skill_id,
            "name": skill.name,
            "description": skill.description,
            "path": skill.path,
            "body": skill.body,
            "depends_on": skill.depends_on,
            "tags": skill.tags,
        }

    def resolve(self, request: SkillInjectRequest) -> SkillInjectResponse:
        notes: list[str] = [
            "AGENTS.md nudge: only treat match_status=complete as skill-backed; "
            "if partial/no_match, fill gaps or reindex before proceeding."
        ]
        degraded = False
        validation_errors: list[ValidationErrorItem] = []

        constraints = request.constraints
        skills_dir = None
        rerank = self.settings.rerank
        top_k = 5
        min_score = None
        if constraints:
            if constraints.skills_dir:
                skills_dir = Path(constraints.skills_dir)
            rerank = constraints.rerank
            top_k = constraints.top_k
            min_score = constraints.min_score

        if rerank == "qwen3-0.6b":
            # Stub only: skip rerank and mark degraded
            degraded = True
            notes.append("rerank=qwen3-0.6b stub: skipped; retriever_degraded=true")

        info = self.ensure_index(skills_dir=skills_dir)
        if info.get("embedder_degraded"):
            # Using FakeEmbedder because no API key — acceptable for MVP/tests,
            # but mark degraded when user expected real embeddings.
            if not self.settings.use_fake_embedder:
                degraded = True
                notes.append("No OPENROUTER_API_KEY; using FakeEmbedder (retriever_degraded)")

        validation_errors.extend(self.registry.validation_errors)

        # Validate requirement DAG (depends_on among requirements)
        req_ids = {r.id for r in request.requirements}
        req_graph = {r.id: list(r.depends_on) for r in request.requirements}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {rid: WHITE for rid in req_graph}
        stack: list[str] = []

        def visit(u: str) -> None:
            color[u] = GRAY
            stack.append(u)
            for v in req_graph.get(u, []):
                if v not in color:
                    validation_errors.append(
                        ValidationErrorItem(
                            code="unknown_requirement_dependency",
                            message=f"Requirement '{u}' depends_on unknown '{v}'",
                            path=u,
                        )
                    )
                    continue
                if color[v] == GRAY:
                    cyc = stack[stack.index(v) :] + [v]
                    validation_errors.append(
                        ValidationErrorItem(
                            code="requirement_dependency_cycle",
                            message="Requirement dependency cycle: " + " -> ".join(cyc),
                            path=u,
                        )
                    )
                    return
                if color[v] == WHITE:
                    visit(v)
            stack.pop()
            color[u] = BLACK

        for rid in list(req_graph):
            if color[rid] == WHITE:
                visit(rid)

        # Duplicate requirement ids
        seen_req: set[str] = set()
        for r in request.requirements:
            if r.id in seen_req:
                validation_errors.append(
                    ValidationErrorItem(
                        code="duplicate_requirement_id",
                        message=f"Duplicate requirement id '{r.id}'",
                        path=r.id,
                    )
                )
            seen_req.add(r.id)

        checks: list[CheckResult] = []
        evidence: list[EvidenceItem] = []
        gaps: list[GapItem] = []
        bindings_by_req: dict[str, str] = {}

        assert self.retriever is not None


        def _runner_up_for(hit, hits):
            others = [h for h in hits if h.skill_id != hit.skill_id]
            dense_others = [h for h in others if h.dense_score is not None]
            if dense_others:
                return max(dense_others, key=lambda h: float(h.dense_score))
            return others[0] if others else None

        for req in request.requirements:
            query = expand_query(req.description, req.search_query)
            hits = self.retriever.retrieve(query, top_k=max(top_k, 1))

            # Optionally filter by min_score (RRF)
            if min_score is not None:
                hits = [h for h in hits if h.ranking_score >= min_score]

            accepted = None
            accepted_reason = None
            top_reject_reason = None

            for hit in hits:
                skill = self.registry.get(hit.skill_id)
                if skill is None:
                    continue
                assessment = evaluate_candidate(
                    query, skill, hit, _runner_up_for(hit, hits)
                )
                if top_reject_reason is None and not assessment.matched:
                    top_reject_reason = assessment.reason
                if assessment.matched:
                    accepted = hit
                    accepted_reason = assessment.reason
                    break

            # Evidence for top candidates whether or not a match was accepted
            for h in hits[:top_k]:
                sk = self.registry.get(h.skill_id)
                evidence.append(
                    EvidenceItem(
                        skill_id=h.skill_id,
                        requirement_id=req.id,
                        ranking_score=h.ranking_score,
                        dense_rank=h.dense_rank,
                        sparse_rank=h.sparse_rank,
                        snippet=(sk.description if sk else None),
                        name=(sk.name if sk else None),
                        description=(sk.description if sk else None),
                    )
                )

            if accepted is not None:
                checks.append(
                    CheckResult(
                        requirement_id=req.id,
                        matched=True,
                        skill_id=accepted.skill_id,
                        ranking_score=accepted.ranking_score,
                        dense_rank=accepted.dense_rank,
                        sparse_rank=accepted.sparse_rank,
                        reason=accepted_reason or "lexical+scores",
                    )
                )
                bindings_by_req[req.id] = accepted.skill_id
            else:
                reason = (
                    top_reject_reason
                    if hits
                    else "no hybrid candidates"
                )
                checks.append(
                    CheckResult(
                        requirement_id=req.id,
                        matched=False,
                        reason=reason,
                    )
                )
                if req.required:
                    gaps.append(
                        GapItem(
                            requirement_id=req.id,
                            description=req.description,
                            reason=reason or "no matching skill found",
                        )
                    )

        # Required unresolved ⇒ not complete
        required_ids = [r.id for r in request.requirements if r.required]
        unresolved_required = [rid for rid in required_ids if rid not in bindings_by_req]

        # If validation errors exist that are structural, force non-complete
        blocking_codes = {
            "duplicate_skill_id",
            "dependency_cycle",
            "requirement_dependency_cycle",
            "duplicate_requirement_id",
        }
        has_blocking = any(e.code in blocking_codes for e in validation_errors)

        if has_blocking:
            match_status = MatchStatus.no_match if not bindings_by_req else MatchStatus.partial
            notes.append("Blocking validation_errors present; cannot be complete")
        elif not request.requirements:
            match_status = MatchStatus.no_match
            gaps.append(
                GapItem(
                    requirement_id="*",
                    description="(none)",
                    reason="no requirements provided",
                )
            )
        elif unresolved_required:
            if bindings_by_req:
                match_status = MatchStatus.partial
            else:
                match_status = MatchStatus.no_match
        else:
            # all required matched
            optional_unmatched = [
                r.id
                for r in request.requirements
                if (not r.required) and r.id not in bindings_by_req
            ]
            match_status = MatchStatus.complete
            if optional_unmatched:
                notes.append(
                    f"Optional requirements unmatched (still complete): {optional_unmatched}"
                )

        # Hard rule: never complete if any required unmatched
        if unresolved_required and match_status == MatchStatus.complete:
            match_status = MatchStatus.partial

        plan_bindings: list[PlanBinding] = []
        if request.draft_plan:
            for step in request.draft_plan.steps:
                sids = []
                for rid in step.requirement_ids:
                    if rid in bindings_by_req:
                        sids.append(bindings_by_req[rid])
                plan_bindings.append(
                    PlanBinding(
                        step_id=step.id,
                        requirement_ids=list(step.requirement_ids),
                        skill_ids=sids,
                    )
                )
        else:
            for rid, sid in bindings_by_req.items():
                plan_bindings.append(
                    PlanBinding(step_id=f"auto:{rid}", requirement_ids=[rid], skill_ids=[sid])
                )

        return SkillInjectResponse(
            match_status=match_status,
            checks=checks,
            evidence=evidence,
            gaps=gaps,
            validation_errors=validation_errors,
            retriever_degraded=degraded,
            plan_bindings=plan_bindings,
            skills_considered=len(self.registry.skills),
            notes=notes,
        )
