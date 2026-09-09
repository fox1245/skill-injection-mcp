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

    def retrieve(self, query: str, top_k: int | None = None) -> list[RRFResult]:
        top_k = top_k or self.retrieve_top_k
        sparse_hits = self.sparse.search(query, top_k=self.retrieve_top_k)
        qvec = self.embedder.embed_queries([query])[0]
        dense_hits = self.dense.search(qvec, top_k=self.retrieve_top_k)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=self.rrf_k)
        return fused[:top_k]
