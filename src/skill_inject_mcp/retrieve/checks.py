from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from skill_inject_mcp.schemas import SkillMeta
from skill_inject_mcp.text import concepts, tokenize  # re-export tokenize for callers

_DENSE_MIN = 0.45
_DENSE_SOLO_MIN = 0.55
_DENSE_MARGIN_MIN = 0.05
_MIN_PREFIX_LEN = 4
_MIN_PREFIX_RATIO = 0.65
_MULTIWORD_COVERAGE_MIN = 0.70

_NEGATIVE_MARKERS = (
    "not",
    "no",
    "never",
    "without",
    "cannot",
    "can't",
    "doesn't",
    "don't",
    "unsupported",
    "non-goal",
    "non-goals",
    "nongoal",
    "nongoals",
    "out of scope",
    "지원하지",
    "미지원",
    "불가",
    "금지",
    "제외",
    "않는",
    "않습니다",
    "없이",
)

_NEGATIVE_HEADING_MARKERS = (
    "non-goal",
    "non goal",
    "when not",
    "not to use",
    "out of scope",
    "unsupported",
    "limitation",
    "exclusion",
    "제외",
    "미지원",
    "제한",
    "사용하지",
)


def _has_negative_marker(text: str) -> bool:
    lowered = text.casefold()
    padded_words = set(tokenize(lowered))
    for marker in _NEGATIVE_MARKERS:
        if " " in marker or "-" in marker or "'" in marker or not marker.isascii():
            if marker in lowered:
                return True
        elif marker in padded_words:
            return True
    return False


def _is_negative_heading(title: str) -> bool:
    lowered = title.casefold()
    return any(marker in lowered for marker in _NEGATIVE_HEADING_MARKERS)


def _parse_heading(stripped: str) -> tuple[int, str] | None:
    if not stripped.startswith("#"):
        return None
    level = 0
    for ch in stripped:
        if ch == "#":
            level += 1
        else:
            break
    if 1 <= level <= 6 and len(stripped) > level and stripped[level].isspace():
        return level, stripped[level:].strip()
    return None


def _split_sentences(line: str) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    n = len(line)
    for i, ch in enumerate(line):
        buf.append(ch)
        if ch in ".!?" and (i + 1 == n or line[i + 1].isspace()):
            s = "".join(buf).strip(" -*")
            if s:
                out.append(s)
            buf.clear()
    if buf:
        s = "".join(buf).strip(" -*")
        if s:
            out.append(s)
    return out


def _partition_skill_statements(skill: SkillMeta) -> tuple[list[str], list[str]]:
    positive: list[str] = []
    negative: list[str] = []
    if skill.description:
        if _has_negative_marker(skill.description):
            negative.append(skill.description)
        else:
            positive.append(skill.description)
    if skill.tags:
        positive.append(" ".join(skill.tags))
    excluded_level: int | None = None
    in_code = False
    for line in skill.body.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_code = not in_code
            continue
        if in_code:
            continue
        heading = _parse_heading(stripped)
        if heading is not None:
            level, title = heading
            if excluded_level is not None and level <= excluded_level:
                excluded_level = None
            if excluded_level is None and _is_negative_heading(title):
                excluded_level = level
            continue
        if excluded_level is not None:
            if stripped:
                negative.append(stripped)
            continue
        for sentence in _split_sentences(stripped):
            if _has_negative_marker(sentence):
                negative.append(sentence)
            else:
                positive.append(sentence)
    return positive, negative


def positive_statements(skill: SkillMeta) -> list[str]:
    """Exclude negative sections (including children) and negative sentences."""
    pos, _ = _partition_skill_statements(skill)
    return pos


def _token_matches_supported(token: str, supported: set[str]) -> bool:
    if token in supported:
        return True
    if len(token) < _MIN_PREFIX_LEN:
        return False
    for cand in supported:
        if len(cand) < _MIN_PREFIX_LEN:
            continue
        lcp = 0
        for a, b in zip(token, cand):
            if a != b:
                break
            lcp += 1
        if lcp >= _MIN_PREFIX_LEN and (lcp / max(len(token), len(cand))) >= _MIN_PREFIX_RATIO:
            return True
    return False


@dataclass
class MatchAssessment:
    matched: bool
    reason: str
    lexical_overlap: float
    checks: list[str] = field(default_factory=list)
    assessment: str = "unknown"
    missing_terms: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    citations: list[dict[str, str]] = field(default_factory=list)
    unmet_requirements: list[str] = field(default_factory=list)
    verifier: str = "lexical"


def lexical_gate(query: str, skill: SkillMeta) -> tuple[bool, float, list[str]]:
    required = concepts(query)
    pos_stmts, neg_stmts = _partition_skill_statements(skill)
    supported = concepts("\n".join(pos_stmts))
    excluded = concepts("\n".join(neg_stmts)) - supported

    matched_terms = {t for t in required if _token_matches_supported(t, supported)}
    missing = sorted(required - matched_terms)
    excluded_hits = [t for t in missing if _token_matches_supported(t, excluded)]
    unmatched_identifiers = [t for t in missing if any(ch.isdigit() for ch in t)]
    overlap = len(matched_terms) / len(required) if required else 0.0
    threshold = _MULTIWORD_COVERAGE_MIN if len(required) >= 4 else 1.0
    lex_ok = (
        bool(required)
        and overlap >= threshold
        and not excluded_hits
        and not unmatched_identifiers
        and not _has_negative_marker(query)
    )
    return lex_ok, overlap, [
        f"supported_terms={sorted(matched_terms)}",
        f"missing_terms={missing}",
    ]


def score_gate(hit: Any, runner_up_hit: Any | None) -> tuple[bool, str, list[str]]:
    sparse = getattr(hit, "sparse_score", None)
    dense = getattr(hit, "dense_score", None)
    notes = [f"sparse_score={sparse}", f"dense_score={dense}"]
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
    matched_terms = {t for t in required if _token_matches_supported(t, supported)}
    missing = sorted(required - matched_terms)
    lex_ok, overlap, notes = lexical_gate(query, skill)
    score_ok, kind, score_notes = score_gate(hit, runner_up_hit)
    notes.extend(score_notes)
    if _has_negative_marker(query):
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
        evidence=[s for s in statements if any(_token_matches_supported(t, concepts(s)) for t in required)],
    )
