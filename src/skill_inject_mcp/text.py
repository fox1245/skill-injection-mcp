"""Shared Unicode tokenization for retrieval and conservative evidence checks."""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_HANGUL_RE = re.compile(r"^[\uac00-\ud7a3]+$")
STOPWORDS = frozenset("""
a an the and or of to in on for with is are was were be been being by from as at
it its this that these those into over under about than then so if but do does did
can could should would may might will just also only very via per using use used
when where what which who how you your we our they their i me my need needs get got
make made please skill skills implement implementing implementation locate find
이 그 저 것 이것 그것 저것 이것은 그것은 저것은 이것을 그것을 저것을
및 또는 은 는 을 를 에 의 가 와 과 도 으로 로 에서 하다 해줘 해주세요 해봐
""".split())

# Ordered longest-first so compound verbal/request endings strip before single-char particles.
_KO_SUFFIXES = (
    "해주세요", "해주기", "해줘요", "해줘", "해봐요", "해봐",
    "하면서", "하거나", "하려고", "하려면", "하니까", "하지만",
    "합니다", "입니다", "습니다", "했다", "한다", "하는", "하기", "하고", "하며", "해서", "하면", "할", "한",
    "으로부터", "로부터", "에서부터", "에게서", "한테서", "이라도", "라든지",
    "으로", "에서", "에게", "한테", "보다", "처럼", "까지", "부터", "마다", "조차", "마저", "밖에", "대로",
    "은", "는", "이", "가", "을", "를", "와", "과", "의", "에", "로", "도", "만",
)
# Particles that grammatically attach only after a closed syllable (batchim != 0).
_KO_BATCHIM_REQUIRED = frozenset({"은", "이", "을", "과", "으로", "으로부터", "이라도"})
# Particles that grammatically attach only after an open syllable (batchim == 0).
_KO_OPEN_REQUIRED = frozenset({"는", "가", "를", "와"})
# Particles that attach after an open syllable (batchim == 0) or rieul batchim (batchim == 8).
_KO_OPEN_OR_RIEUL_REQUIRED = frozenset({"로", "로부터"})


def _hangul_batchim(ch: str) -> int:
    return (ord(ch) - 0xAC00) % 28


def _normalize_korean_token(word: str) -> str:
    if len(word) < 3 or not _HANGUL_RE.match(word):
        return word
    for suffix in _KO_SUFFIXES:
        if not word.endswith(suffix):
            continue
        stem = word[: -len(suffix)]
        if len(stem) < 2:
            continue
        batchim = _hangul_batchim(stem[-1])
        if suffix in _KO_BATCHIM_REQUIRED and batchim == 0:
            continue
        if suffix in _KO_OPEN_REQUIRED and batchim != 0:
            continue
        if suffix in _KO_OPEN_OR_RIEUL_REQUIRED and batchim not in (0, 8):
            continue
        return stem
    return word


def words(text: str) -> list[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


def tokenize(text: str) -> list[str]:
    return [word for word in words(text) if word not in STOPWORDS]


def concepts(text: str) -> set[str]:
    # Conservative inflection normalization; no domain-specific synonym assumptions.
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
        else:
            word = _normalize_korean_token(word)
        if word and word not in STOPWORDS:
            result.add(word)
    return result
