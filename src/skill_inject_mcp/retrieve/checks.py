from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from skill_inject_mcp.schemas import SkillMeta

# Tiny English + light Korean function words (not content-bearing).
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "by",
        "from",
        "as",
        "at",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "into",
        "over",
        "under",
        "about",
        "than",
        "then",
        "so",
        "if",
        "not",
        "no",
        "nor",
        "but",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "may",
        "might",
        "will",
        "just",
        "also",
        "only",
        "very",
        "via",
        "per",
        "using",
        "use",
        "used",
        "when",
        "where",
        "what",
        "which",
        "who",
        "how",
        "you",
        "your",
        "we",
        "our",
        "they",
        "their",
        "i",
        "me",
        "my",
        "need",
        "needs",
        "get",
        "got",
        "make",
        "made",
        "이",
        "그",
        "저",
        "및",
        "또는",
        "은",
        "는",
        "을",
        "를",
        "에",
        "의",
        "가",
        "와",
        "과",
        "도",
        "으로",
        "로",
        "에서",
        "하다",
        "있는",
        "없는",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

# Lexical / score thresholds (similarity != fulfillment).
_OVERLAP_RATIO_MIN = 0.12
_SPARSE_MIN = 0.15
_DENSE_MIN = 0.45
_DENSE_SOLO_MIN = 0.55
_DENSE_MARGIN_MIN = 0.05
_BODY_PREFIX_CHARS = 400


@dataclass
class MatchAssessment:
    matched: bool
    reason: str
    lexical_overlap: float
    checks: list[str] = field(default_factory=list)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; drop tiny stopwords (en/ko-light)."""
    raw = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for tok in raw:
        if len(tok) < 2:
            continue
        if tok in _STOPWORDS:
            continue
        out.append(tok)
    return out


def _skill_card_text(skill: SkillMeta) -> str:
    parts = [
        skill.name or "",
        skill.description or "",
        " ".join(skill.tags or []),
    ]
    body = skill.body or ""
    if body:
        parts.append(body[:_BODY_PREFIX_CHARS])
    return "\n".join(parts)


def _overlap_ratio(query_toks: set[str], skill_toks: set[str]) -> float:
    if not query_toks:
        return 0.0
    inter = query_toks & skill_toks
    # Overlap coefficient vs query (how much of the query is covered).
    return len(inter) / len(query_toks)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def lexical_gate(query: str, skill: SkillMeta) -> tuple[bool, float, list[str]]:
    """Primary lexical evidence gate.

    - Single-token queries: that token must appear on the skill card.
    - Longer queries: at least one significant query token on the card, OR
      overlap_ratio/Jaccard >= ~0.12 with >=2 overlapping tokens.
    """
    notes: list[str] = []
    q_toks = tokenize(query)
    s_toks = set(tokenize(_skill_card_text(skill)))
    q_set = set(q_toks)
    inter = q_set & s_toks
    overlap = _overlap_ratio(q_set, s_toks)
    jac = _jaccard(q_set, s_toks)
    notes.append(f"overlap_tokens={sorted(inter)}")
    notes.append(f"overlap_ratio={overlap:.4f}")
    notes.append(f"jaccard={jac:.4f}")

    if not q_toks:
        notes.append("lexical=empty_query")
        return False, 0.0, notes

    if len(q_set) == 1:
        ok = bool(inter)
        notes.append("lexical=single_token_present" if ok else "lexical=single_token_missing")
        return ok, overlap, notes

    if inter:
        notes.append("lexical=token_hit")
        return True, overlap, notes

    if len(inter) >= 2 and (overlap >= _OVERLAP_RATIO_MIN or jac >= _OVERLAP_RATIO_MIN):
        notes.append("lexical=overlap_ratio")
        return True, overlap, notes

    # Longer-query overlap path when token_hit somehow empty but ratio ok
    # (kept explicit for the OR clause in the product gate).
    if (overlap >= _OVERLAP_RATIO_MIN or jac >= _OVERLAP_RATIO_MIN) and len(inter) >= 2:
        notes.append("lexical=jaccard")
        return True, overlap, notes

    notes.append("lexical=fail")
    return False, overlap, notes


def score_gate(
    hit: Any,
    runner_up_hit: Any | None,
) -> tuple[bool, str, list[str]]:
    """Sparse OR (dense with margin / solo threshold)."""
    notes: list[str] = []
    sparse = getattr(hit, "sparse_score", None)
    dense = getattr(hit, "dense_score", None)
    notes.append(f"sparse_score={sparse}")
    notes.append(f"dense_score={dense}")

    if sparse is not None and sparse >= _SPARSE_MIN:
        notes.append("score=sparse_ok")
        return True, "sparse", notes

    if dense is not None and dense >= _DENSE_MIN:
        if runner_up_hit is not None and getattr(runner_up_hit, "dense_score", None) is not None:
            ru = float(runner_up_hit.dense_score)
            margin = float(dense) - ru
            notes.append(f"dense_margin={margin:.4f}")
            if margin >= _DENSE_MARGIN_MIN:
                notes.append("score=dense_margin_ok")
                return True, "dense_margin", notes
            notes.append("score=dense_margin_weak")
            return False, "weak_dense_margin", notes
        # No runner-up (tiny / single-skill index): demand a higher absolute cosine.
        if dense >= _DENSE_SOLO_MIN:
            notes.append("score=dense_solo_ok")
            return True, "dense_solo", notes
        notes.append("score=dense_solo_weak")
        return False, "weak_dense_solo", notes

    notes.append("score=weak")
    return False, "weak_scores", notes


def evaluate_candidate(
    query: str,
    skill: SkillMeta,
    hit: Any,
    runner_up_hit: Any | None = None,
) -> MatchAssessment:
    """Decide whether a retrieval hit fulfills the requirement.

    Similarity (nearest-neighbor cosine in a tiny registry) is not fulfillment.
    Lexical evidence is required; scores are a secondary gate.
    """
    lex_ok, lex_overlap, lex_notes = lexical_gate(query, skill)
    score_ok, score_kind, score_notes = score_gate(hit, runner_up_hit)
    checks = [*lex_notes, *score_notes]

    if not lex_ok:
        return MatchAssessment(
            matched=False,
            reason="no_lexical_evidence",
            lexical_overlap=lex_overlap,
            checks=checks,
        )

    if not score_ok:
        return MatchAssessment(
            matched=False,
            reason="weak_retrieval_scores",
            lexical_overlap=lex_overlap,
            checks=checks,
        )

    if score_kind == "sparse":
        reason = "lexical+sparse"
    elif score_kind == "dense_margin":
        reason = "lexical+dense_margin"
    elif score_kind == "dense_solo":
        reason = "lexical+dense_solo"
    else:
        reason = f"lexical+{score_kind}"

    return MatchAssessment(
        matched=True,
        reason=reason,
        lexical_overlap=lex_overlap,
        checks=checks,
    )
