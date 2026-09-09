from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from skill_inject_mcp.schemas import SkillMeta
from skill_inject_mcp.text import concepts, tokenize  # re-export tokenize for callers

_DENSE_MIN = 0.45
_DENSE_SOLO_MIN = 0.55
_DENSE_MARGIN_MIN = 0.05
_NEGATIVE = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|doesn't|don't|unsupported|"
    r"non[- ]?goals?|out of scope)\b|"
    r"지원하지|지원하지\s*않|미지원|불가|금지|제외|않는|않습니다|없이",
    re.IGNORECASE,
)
_NEGATIVE_HEADING = re.compile(
    r"non[- ]?goals?|when not|not to use|out of scope|unsupported|"
    r"limitations?|exclusions?|제외|미지원|제한|사용하지",
    re.IGNORECASE,
)


@dataclass
class MatchAssessment:
    matched: bool
    reason: str
    lexical_overlap: float
    checks: list[str] = field(default_factory=list)
    assessment: str = "unknown"
    missing_terms: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def positive_statements(skill: SkillMeta) -> list[str]:
    """Exclude negative sections (including children) and negative sentences.

    This is a conservative textual evidence check, not a semantic entailment
    model or a guarantee that executing a skill will achieve the user's task.
    """
    statements = []
    # Names and tags help retrieval, but do not independently prove capabilities.
    if skill.description and not _NEGATIVE.search(skill.description):
        statements.append(skill.description)
    excluded_level = None
    in_code = False
    for line in skill.body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_code = not in_code
            continue
        if in_code:
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if heading:
            level, title = len(heading[1]), heading[2]
            if excluded_level is not None and level <= excluded_level:
                excluded_level = None
            if excluded_level is None and _NEGATIVE_HEADING.search(title):
                excluded_level = level
            continue
        if excluded_level is not None:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", stripped):
            sentence = sentence.strip(" -*")
            if sentence and not _NEGATIVE.search(sentence):
                statements.append(sentence)
    return statements


def lexical_gate(query: str, skill: SkillMeta) -> tuple[bool, float, list[str]]:
    required = concepts(query)
    supported = concepts("\n".join(positive_statements(skill)))
    overlap = len(required & supported) / len(required) if required else 0.0
    missing = sorted(required - supported)
    return bool(required) and not missing and not _NEGATIVE.search(query), overlap, [
        f"supported_terms={sorted(required & supported)}",
        f"missing_terms={missing}",
    ]


def score_gate(hit: Any, runner_up_hit: Any | None) -> tuple[bool, str, list[str]]:
    sparse = getattr(hit, "sparse_score", None)
    dense = getattr(hit, "dense_score", None)
    notes = [f"sparse_score={sparse}", f"dense_score={dense}"]
    # BM25 is corpus-dependent; it is not a calibrated confidence score.
    # Require complete positive textual coverage separately before using it.
    if sparse is not None and sparse > 0:
        return True, "sparse", notes
    if dense is not None and dense >= _DENSE_MIN:
        if runner_up_hit is not None and runner_up_hit.dense_score is not None:
            margin = float(dense) - float(runner_up_hit.dense_score)
            notes.append(f"dense_margin={margin:.4f}")
            return margin >= _DENSE_MARGIN_MIN, "dense_margin", notes
        return dense >= _DENSE_SOLO_MIN, "dense_solo", notes
    return False, "weak_scores", notes


def evaluate_candidate(
    query: str, skill: SkillMeta, hit: Any, runner_up_hit: Any | None = None,
) -> MatchAssessment:
    statements = positive_statements(skill)
    required = concepts(query)
    supported = concepts("\n".join(statements))
    missing = sorted(required - supported)
    lex_ok, overlap, notes = lexical_gate(query, skill)
    score_ok, kind, score_notes = score_gate(hit, runner_up_hit)
    notes.extend(score_notes)
    if _NEGATIVE.search(query):
        reason = "negative_requirement_needs_verification"
    elif not required or overlap == 0:
        reason = "no_lexical_evidence"
    elif not lex_ok:
        reason = "unverified_requirement_terms"
    elif not score_ok:
        reason = "weak_retrieval_scores"
    else:
        reason = f"lexical+{kind}"
    matched = lex_ok and score_ok
    return MatchAssessment(
        matched=matched, reason=reason, lexical_overlap=overlap, checks=notes,
        assessment="supported" if matched else "unknown", missing_terms=missing,
        evidence=[s for s in statements if concepts(s) & required],
    )
