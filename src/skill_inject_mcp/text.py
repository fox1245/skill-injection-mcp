"""Shared Unicode tokenization for retrieval and conservative evidence checks."""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
STOPWORDS = frozenset("""
a an the and or of to in on for with is are was were be been being by from as at
it its this that these those into over under about than then so if but do does did
can could should would may might will just also only very via per using use used
when where what which who how you your we our they their i me my need needs get got
make made please skill skills implement implementing implementation locate find
이 그 저 및 또는 은 는 을 를 에 의 가 와 과 도 으로 로 에서 하다
""".split())


def words(text: str) -> list[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def tokenize(text: str) -> list[str]:
    return [word for word in words(text) if word not in STOPWORDS]


def concepts(text: str) -> set[str]:
    # Only small inflection normalization; no domain-specific synonym assumptions.
    result = set()
    derivations = {"installer": "install", "installation": "install"}
    for word in tokenize(text):
        word = derivations.get(word, word)
        if word.isascii() and len(word) > 4:
            if word.endswith("ies"):
                word = word[:-3] + "y"
            elif word.endswith(("xes", "ches", "shes", "sses")):
                word = word[:-2]
            elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
                word = word[:-1]
        result.add(word)
    return result
