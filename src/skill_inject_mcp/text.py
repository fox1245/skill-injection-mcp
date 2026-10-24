"""Unicode character-property token segmentation without regexes, stemmer tables, or stopword lists."""

from __future__ import annotations

import unicodedata


def words(text: str) -> list[str]:
    """Split text into casefolded alphanumeric tokens using Unicode character properties."""
    if not text:
        return []
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    buf: list[str] = []
    for ch in normalized:
        if ch.isalnum():
            buf.append(ch)
        elif buf:
            tokens.append("".join(buf))
            buf.clear()
    if buf:
        tokens.append("".join(buf))
    return tokens


def tokenize(text: str) -> list[str]:
    """Return tokens of length >= 2 without hardcoded stopword filtering."""
    return [w for w in words(text) if len(w) >= 2]


def concepts(text: str) -> set[str]:
    """Return unique token set without hardcoded stemmers or suffix tables."""
    return set(tokenize(text))
