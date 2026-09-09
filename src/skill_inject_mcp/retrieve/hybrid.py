from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from skill_inject_mcp.embed.embedder import Embedder
from skill_inject_mcp.index.sparse import SparseIndex
from skill_inject_mcp.index.vector import VectorIndex
from skill_inject_mcp.schemas import SkillMeta


@dataclass
class RRFResult:
    skill_id: str
    ranking_score: float
    dense_rank: int | None
    sparse_rank: int | None
    dense_score: float | None = None
    sparse_score: float | None = None


def reciprocal_rank_fusion(
    dense: Sequence[tuple[str, float]],
    sparse: Sequence[tuple[str, float, int]],
    *,
    k: int = 60,
) -> list[RRFResult]:
    """Fuse at skill_id. Tie-break: RRF desc, dense_rank asc, skill_id asc."""
    dense_rank: dict[str, int] = {}
    dense_score: dict[str, float] = {}
    for i, (sid, score) in enumerate(dense, start=1):
        if sid not in dense_rank:
            dense_rank[sid] = i
            dense_score[sid] = score

    sparse_rank: dict[str, int] = {}
    sparse_score: dict[str, float] = {}
    for sid, score, rank in sparse:
        if sid not in sparse_rank:
            sparse_rank[sid] = rank
            sparse_score[sid] = score

    all_ids = set(dense_rank) | set(sparse_rank)
    fused: list[RRFResult] = []
    for sid in all_ids:
        score = 0.0
        dr = dense_rank.get(sid)
        sr = sparse_rank.get(sid)
        if dr is not None:
            score += 1.0 / (k + dr)
        if sr is not None:
            score += 1.0 / (k + sr)
        fused.append(
            RRFResult(
                skill_id=sid,
                ranking_score=score,
                dense_rank=dr,
                sparse_rank=sr,
                dense_score=dense_score.get(sid),
                sparse_score=sparse_score.get(sid),
            )
        )

    fused.sort(
        key=lambda r: (
            -r.ranking_score,
            r.dense_rank if r.dense_rank is not None else 10**9,
            r.skill_id,
        )
    )
    return fused


def _merge_best_scores(
    dense_by_query: list[list[tuple[str, float]]],
    sparse_by_query: list[list[tuple[str, float, int]]],
) -> tuple[list[tuple[str, float]], list[tuple[str, float, int]]]:
    """Take max score per skill_id across queries, then re-rank once for RRF."""
    best_dense: dict[str, float] = {}
    for hits in dense_by_query:
        for sid, score in hits:
            prev = best_dense.get(sid)
            if prev is None or score > prev:
                best_dense[sid] = score

    best_sparse: dict[str, float] = {}
    for hits in sparse_by_query:
        for sid, score, _rank in hits:
            prev = best_sparse.get(sid)
            if prev is None or score > prev:
                best_sparse[sid] = score

    dense_list = sorted(best_dense.items(), key=lambda x: (-x[1], x[0]))
    sparse_sorted = sorted(best_sparse.items(), key=lambda x: (-x[1], x[0]))
    sparse_list = [(sid, score, i) for i, (sid, score) in enumerate(sparse_sorted, start=1)]
    return dense_list, sparse_list


class HybridRetriever:
    def __init__(
        self,
        *,
        sparse: SparseIndex,
        dense: VectorIndex,
        embedder: Embedder,
        skills: dict[str, SkillMeta],
        rrf_k: int = 60,
        retrieve_top_k: int = 20,
    ) -> None:
        self.sparse = sparse
        self.dense = dense
        self.embedder = embedder
        self.skills = skills
        self.rrf_k = rrf_k
        self.retrieve_top_k = retrieve_top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        queries: Sequence[str] | None = None,
    ) -> list[RRFResult]:
        """Hybrid retrieve with optional multi-query expansion.

        When ``queries`` is provided, sparse+dense run per query; best scores
        per skill_id are merged, then a single RRF pass ranks candidates.
        ``top_k`` truncates the fused list (agent-facing candidate width).
        Internal channel size remains ``retrieve_top_k`` per query/side.
        """
        top_k = top_k if top_k is not None else self.retrieve_top_k
        qlist = [q.strip() for q in (queries or [query]) if (q or "").strip()]
        if not qlist:
            qlist = [query]

        dense_by_query: list[list[tuple[str, float]]] = []
        sparse_by_query: list[list[tuple[str, float, int]]] = []
        for q in qlist:
            sparse_hits = self.sparse.search(q, top_k=self.retrieve_top_k)
            sparse_by_query.append(sparse_hits)
            qvec = self.embedder.embed_queries([q])[0]
            dense_hits = self.dense.search(qvec, top_k=self.retrieve_top_k)
            dense_by_query.append(dense_hits)

        if len(qlist) == 1:
            fused = reciprocal_rank_fusion(dense_by_query[0], sparse_by_query[0], k=self.rrf_k)
        else:
            dense_merged, sparse_merged = _merge_best_scores(dense_by_query, sparse_by_query)
            fused = reciprocal_rank_fusion(dense_merged, sparse_merged, k=self.rrf_k)
        return fused[: max(top_k, 1)]
