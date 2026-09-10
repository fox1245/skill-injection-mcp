from __future__ import annotations

from graphlib import TopologicalSorter
from pathlib import Path
from threading import RLock, Thread
import asyncio
import time
import logging

import httpx

from skill_inject_mcp.config import Settings, get_settings
from skill_inject_mcp.embed.embedder import build_embedder
from skill_inject_mcp.index.snapshot import build_snapshot, embedding_identity, fingerprint
from skill_inject_mcp.registry.scan import SkillRegistry
from skill_inject_mcp.registry.layer import classify_requirement_layer, partition_hits
from skill_inject_mcp.registry.signature import source_signature
from skill_inject_mcp.retrieve.checks import evaluate_candidate
from skill_inject_mcp.retrieve.hybrid import HybridRetriever
from skill_inject_mcp.retrieve.multi_query import expand_queries
from skill_inject_mcp.retrieve.semantic import SemanticVerifier
from skill_inject_mcp.schemas import (
    CheckResult, EvidenceItem, GapItem, MatchStatus, PlanBinding,
    SkillInjectRequest, SkillInjectResponse,
)
from skill_inject_mcp.validation import validate_request


class SkillInjectEngine:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.registry = SkillRegistry()
        self.sparse = self.dense = self.embedder = self.retriever = None
        self.dense_backend = "numpy"
        self.embedder_degraded = False
        self._indexed = False
        self._snapshot = None
        # Serialize mutating tools and synchronous clients. Hooks never acquire this lock.
        self._lock = RLock()
        self._verifier = SemanticVerifier(cache_size=self.settings.verification_cache_size)
        self._source_signature = None
        self._embedders = {}
        self._chat_client = None
        # Publication/reader ownership only; no network or index construction under this lock.
        self._snapshot_guard = RLock()
        self._refresh_guard = RLock()
        self._refresh_pending = False
        self._next_refresh_after = 0.0
        self._last_refresh_error = None
        self._closed = False
        self._cleanup_jobs = set()

    def _selection(self, skills_dir: Path | None = None) -> tuple[str, Path]:
        if skills_dir is None and self.settings.skill_manifest is not None:
            return ("manifest", Path(self.settings.skill_manifest).resolve())
        return ("directory", Path(skills_dir if skills_dir is not None else self.settings.skills_dir).resolve())

    def _get_embedder(self):
        identity = (embedding_identity(self.settings), self.settings.embedding_timeout_s, self.settings.query_cache_size)
        with self._snapshot_guard:
            if identity not in self._embedders:
                self._embedders[identity] = build_embedder(
                    api_key=self.settings.resolve_api_key(), use_fake=self.settings.use_fake_embedder,
                    model=self.settings.embedding_model, dim=self.settings.embedding_dim,
                    base_url=self.settings.embedding_base_url, timeout_s=self.settings.embedding_timeout_s,
                    query_cache_size=self.settings.query_cache_size,
                )
            return self._embedders[identity]

    def _get_chat_client(self):
        if self._chat_client is None:
            self._chat_client = httpx.Client()
        return self._chat_client

    def ensure_index(self, skills_dir: Path | None = None) -> dict:
        with self._lock:
            return self._index(skills_dir, force=False)

    def reindex(self, skills_dir: Path | None = None) -> dict:
        with self._lock:
            return self._index(skills_dir, force=True)

    def _index(self, skills_dir: Path | None, *, force: bool) -> dict:
        if self._closed:
            raise RuntimeError("Skill engine is closed")
        selection = self._selection(skills_dir)
        signature = source_signature(selection, None if force else self._source_signature)
        if (not force and self._snapshot is not None and signature == self._source_signature
                and self._snapshot.identity == embedding_identity(self.settings)):
            return self._snapshot.info()
        registry = SkillRegistry()
        if skills_dir is None and self.settings.skill_manifest is not None:
            manifest = Path(self.settings.skill_manifest).resolve()
            registry.load_manifest(manifest)
            root = manifest.parent
        else:
            root = Path(skills_dir if skills_dir is not None else self.settings.skills_dir).resolve()
            if not root.is_dir():
                raise ValueError(f"Skills directory does not exist: {root}")
            registry.load(root)
        snapshot_id = fingerprint(registry, root, embedding_identity(self.settings))
        if not force and self._snapshot is not None and self._snapshot.snapshot_id == snapshot_id:
            self._source_signature = signature
            self._snapshot.source_signature = signature
            return self._snapshot.info()
        candidate = build_snapshot(self.settings, registry, root, snapshot_id, self._snapshot,
                                   embedder_state=self._get_embedder())
        retriever = HybridRetriever(
            sparse=candidate.sparse, dense=candidate.dense, embedder=candidate.embedder,
            skills=registry.skills, rrf_k=self.settings.rrf_k,
            retrieve_top_k=self.settings.retrieve_top_k,
        )
        candidate.source_signature = signature
        with self._snapshot_guard:
            previous = self._snapshot
            self._snapshot = candidate
            self._source_signature = signature
        self.registry, self.sparse, self.dense = registry, candidate.sparse, candidate.dense
        self.embedder = candidate.embedder
        self.embedder_degraded = candidate.embedder_degraded
        self.dense_backend, self.retriever = candidate.backend, retriever
        self._indexed = True
        if previous is not None:
            previous.close()
        return candidate.info()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            with self._snapshot_guard:
                snapshot, self._snapshot = self._snapshot, None
                embedders = list(self._embedders.values())
                self._embedders.clear()
            if snapshot is not None:
                snapshot.close()
            self.registry = SkillRegistry()
            self.sparse = self.dense = self.embedder = self.retriever = None
            self._indexed = False
            self._source_signature = None
            for embedder, _ in embedders:
                embedder.close()
            if self._chat_client is not None:
                self._chat_client.close()
                self._chat_client = None

    async def aclose_async_clients(self) -> None:
        loop = asyncio.get_running_loop()
        with self._snapshot_guard:
            embedders = list(self._embedders.values())
        for embedder, _ in embedders:
            if getattr(embedder, "_async_loop", None) not in (None, loop):
                continue  # A synchronous helper must not close another loop's pooled client.
            await embedder.aclose()

    async def aclose(self) -> None:
        self._closed = True
        await self.aclose_async_clients()
        if self._cleanup_jobs:
            await asyncio.gather(*tuple(self._cleanup_jobs), return_exceptions=True)
        await asyncio.to_thread(self.close)

    def defer_snapshot_cleanup(self, cleanup) -> None:
        job = asyncio.get_running_loop().run_in_executor(None, cleanup)
        self._cleanup_jobs.add(job)
        def completed(future):
            self._cleanup_jobs.discard(future)
            if not future.cancelled() and future.exception() is not None:
                logging.getLogger(__name__).warning("Retired index cleanup failed (%s)", type(future.exception()).__name__)
        job.add_done_callback(completed)

    def acquire_hook_snapshot(self):
        selection = self._selection()
        with self._snapshot_guard:
            snapshot = self._snapshot
            if (self._closed or snapshot is None or snapshot.source_signature is None
                    or snapshot.source_signature[0] != selection
                    or snapshot.identity != embedding_identity(self.settings)):
                return None
            return snapshot if snapshot.acquire() else None

    def hook_catalog_state(self, snapshot) -> str:
        try:
            current = source_signature(self._selection(), snapshot.source_signature)
            if current == snapshot.source_signature and snapshot.identity == embedding_identity(self.settings):
                return "current"
        except (OSError, ValueError, KeyError, TypeError):
            pass
        self.request_background_refresh()
        return "last-known"

    def request_background_refresh(self) -> None:
        with self._refresh_guard:
            if self._closed or self._refresh_pending or time.monotonic() < self._next_refresh_after:
                return
            self._refresh_pending = True
        def refresh():
            try:
                self.ensure_index()
                self._last_refresh_error = None
            except Exception as exc:
                self._last_refresh_error = type(exc).__name__
            finally:
                with self._refresh_guard:
                    self._refresh_pending = False
                    self._next_refresh_after = time.monotonic() + 1.0
        try:
            Thread(target=refresh, name="skill-index-refresh", daemon=True).start()
        except Exception:
            with self._refresh_guard:
                self._refresh_pending = False
            raise

    def get_skill_body(self, skill_id: str, registry_snapshot: str | None = None) -> dict:
        with self._lock:
            if not self._indexed:
                self.ensure_index()
            current = self._snapshot.snapshot_id
            if registry_snapshot is not None and registry_snapshot != current:
                return {"found": False, "skill_id": skill_id, "reason": "snapshot_mismatch",
                        "registry_snapshot": current}
            skill = self.registry.get(skill_id)
            if skill is None:
                return {"found": False, "skill_id": skill_id, "body": None}
            return {
                "found": True, "skill_id": skill.skill_id, "name": skill.name,
                "description": skill.description, "path": skill.path, "body": skill.body,
                "depends_on": skill.depends_on, "tags": skill.tags,
                "content_hash": skill.content_hash, "registry_snapshot": current,
                "source_path": skill.source_path,
                "skills_dir": str(self._snapshot.root),
                "layer": skill.layer,
            }

    def resolve(self, request: SkillInjectRequest) -> SkillInjectResponse:
        with self._lock:
            return self._resolve(request)

    def _resolve(self, request: SkillInjectRequest) -> SkillInjectResponse:
        notes = [
            "complete means all required requirements passed the reported verifier and dependency "
            "checks; it does not certify execution success. Inspect verification_mode and evidence."
        ]
        errors = validate_request(request)
        if errors:
            return SkillInjectResponse(
                match_status=MatchStatus.no_match, validation_errors=errors,
                checks=[CheckResult(requirement_id=r.id, matched=False, assessment="blocked",
                                    reason="invalid_request") for r in request.requirements],
                gaps=[GapItem(requirement_id=r.id, description=r.description, reason="invalid_request")
                      for r in request.requirements if r.required], notes=notes,
            )
        if not request.requirements:
            return SkillInjectResponse(
                match_status=MatchStatus.no_match,
                gaps=[GapItem(requirement_id="*", description="(none)", reason="no requirements provided")],
                notes=notes,
            )
        constraints = request.constraints
        root = Path(constraints.skills_dir) if constraints and constraints.skills_dir else None
        rerank = constraints.rerank if constraints and constraints.rerank is not None else self.settings.rerank
        top_k = constraints.top_k if constraints and constraints.top_k is not None else self.settings.resolve_top_k
        min_score = constraints.min_score if constraints else None
        info = self.ensure_index(skills_dir=root)
        degraded = self.embedder_degraded
        if degraded:
            notes.append("No OPENROUTER_API_KEY; using FakeEmbedder (retriever_degraded)")
        n_meta = sum(1 for skill in self.registry.all() if skill.layer == "meta")
        notes.append(
            f"skill layers: meta={n_meta} domain={len(self.registry.skills) - n_meta}; "
            "domain requirements search the domain layer first"
        )
        if rerank != "off":
            degraded = True
            notes.append("rerank=qwen3-0.6b stub: skipped; retriever_degraded=true")
        mode = self.settings.verification_mode
        verification_degraded = False
        verification_diagnostics = []
        if mode == "lexical":
            notes.append("Lexical fallback is not a cross-language semantic verifier.")
        errors = list(self.registry.validation_errors)
        checks, evidence = [], []
        bindings = {}
        assert self.retriever is not None
        for req in request.requirements:
            mq = expand_queries(
                req.description, req.search_query, api_key=self.settings.resolve_api_key(),
                enabled=self.settings.multi_query_enabled(), model=self.settings.multi_query_model,
                base_url=self.settings.embedding_base_url, timeout_s=self.settings.multi_query_timeout_s,
                client=self._get_chat_client() if self.settings.multi_query_enabled() else None,
            )
            if mq.skipped and self.settings.multi_query_enabled():
                degraded = True
                notes.append(f"multi_query_skipped for '{req.id}': {mq.reason or 'unknown'}")
            # Always retrieve the description as well as hints/expansions.
            queries = list(dict.fromkeys([req.description, *mq.queries]))
            # Preserve all channel candidates for verification and dense margins.
            all_hits = self.retriever.retrieve(
                req.description, top_k=max(len(self.registry.skills), 1), queries=queries,
            )
            wanted = classify_requirement_layer(req.description, req.search_query)
            primary, secondary = partition_hits(all_hits, self.registry.skills, wanted)
            search_hits = primary or secondary
            ranked_hits = primary + secondary
            semantic_assessments = {}
            if mode == "semantic":
                candidates = [
                    self.registry.get(hit.skill_id) for hit in search_hits
                    if (min_score is None or hit.ranking_score >= min_score)
                    and self.registry.get(hit.skill_id) is not None
                ][:self.settings.verification_top_k]
                verification = self._verifier.verify(
                    req.description, candidates, api_key=self.settings.resolve_api_key(),
                    model=self.settings.verification_model, base_url=self.settings.embedding_base_url,
                    timeout_s=self.settings.verification_timeout_s,
                    max_source_chars=self.settings.verification_max_source_chars,
                    max_tokens=self.settings.verification_max_tokens,
                    max_retries=self.settings.verification_max_retries,
                    client=self._get_chat_client(),
                )
                semantic_assessments = verification.assessments
                verification_diagnostics.extend(
                    diagnostic.model_copy(update={"requirement_id": req.id})
                    for diagnostic in verification.diagnostics
                )
                verification_degraded |= verification.degraded
                if verification.degraded:
                    notes.append(f"Verification degraded for '{req.id}': {verification.reason}")
            accepted = None
            accepted_assessment = None
            first_assessment = None
            first_candidate_id = None
            rejection_reason = None
            for hit in search_hits:
                if min_score is not None and hit.ranking_score < min_score:
                    continue
                skill = self.registry.get(hit.skill_id)
                if skill is None:
                    continue
                others = [h for h in search_hits if h.skill_id != hit.skill_id and h.dense_score is not None]
                runner_up = max(others, key=lambda h: h.dense_score) if others else None
                if mode == "semantic":
                    assessment = semantic_assessments.get(skill.skill_id)
                    if assessment is None:
                        continue
                else:
                    assessment = evaluate_candidate(req.description, skill, hit, runner_up)
                if first_assessment is None:
                    first_assessment = assessment
                    first_candidate_id = hit.skill_id
                if not assessment.matched:
                    continue
                closure, dependency_errors = self.registry.dependency_closure(skill.skill_id)
                if dependency_errors:
                    rejection_reason = "invalid_skill_dependencies"
                    continue
                accepted, accepted_assessment = hit, assessment
                bindings[req.id] = closure
                break
            for hit in ranked_hits[:top_k]:
                skill = self.registry.get(hit.skill_id)
                evidence.append(EvidenceItem(
                    skill_id=hit.skill_id, requirement_id=req.id, ranking_score=hit.ranking_score,
                    dense_rank=hit.dense_rank, sparse_rank=hit.sparse_rank,
                    snippet=skill.description if skill else None,
                    name=skill.name if skill else None, description=skill.description if skill else None,
                    layer=skill.layer if skill else None,
                ))
            assessment = accepted_assessment or first_assessment
            checks.append(CheckResult(
                requirement_id=req.id, matched=accepted is not None,
                skill_id=accepted.skill_id if accepted else None,
                candidate_skill_id=accepted.skill_id if accepted else first_candidate_id,
                ranking_score=accepted.ranking_score if accepted else None,
                dense_rank=accepted.dense_rank if accepted else None,
                sparse_rank=accepted.sparse_rank if accepted else None,
                reason=(accepted_assessment.reason if accepted else
                        rejection_reason or (assessment.reason if assessment else "no eligible candidates")),
                assessment=("supported" if accepted else "blocked" if rejection_reason else
                            assessment.assessment if assessment else "unknown"),
                missing_terms=assessment.missing_terms if assessment else [],
                evidence=assessment.evidence if assessment else [],
                citations=assessment.citations if assessment else [],
                unmet_requirements=assessment.unmet_requirements if assessment else [],
                verifier=mode,
                layer=(self.registry.get(accepted.skill_id).layer if accepted and self.registry.get(accepted.skill_id) else None),
            ))
        # Propagate unresolved dependencies even when the dependency was optional.
        by_id = {c.requirement_id: c for c in checks}
        graph = {r.id: r.depends_on for r in request.requirements}
        for rid in TopologicalSorter(graph).static_order():
            unmet = [dep for dep in graph[rid] if dep not in bindings]
            if unmet:
                bindings.pop(rid, None)
                check = by_id[rid]
                check.matched = False
                check.skill_id = None
                check.assessment = "blocked"
                check.reason = "unresolved_requirement_dependencies: " + ", ".join(unmet)
        gaps = [GapItem(requirement_id=r.id, description=r.description,
                        reason=by_id[r.id].reason or "unmatched")
                for r in request.requirements if r.required and r.id not in bindings]
        status = MatchStatus.complete if not gaps else MatchStatus.partial if bindings else MatchStatus.no_match

        plan_bindings = []
        if request.draft_plan is not None:
            for step in request.draft_plan.steps:
                skill_ids = list(dict.fromkeys(
                    sid for rid in step.requirement_ids for sid in bindings.get(rid, [])
                ))
                plan_bindings.append(PlanBinding(
                    step_id=step.id, requirement_ids=step.requirement_ids, skill_ids=skill_ids,
                ))
        else:
            for rid in TopologicalSorter(graph).static_order():
                if rid in bindings:
                    plan_bindings.append(PlanBinding(
                        step_id=f"auto:{rid}", requirement_ids=[rid], skill_ids=bindings[rid],
                    ))
        return SkillInjectResponse(
            match_status=status, checks=checks, evidence=evidence, gaps=gaps,
            validation_errors=errors, retriever_degraded=degraded, plan_bindings=plan_bindings,
            skills_considered=len(self.registry.skills), notes=notes,
            registry_snapshot=info.get("registry_snapshot"),
            verification_mode=mode, verification_degraded=verification_degraded,
            verification_diagnostics=verification_diagnostics,
        )
