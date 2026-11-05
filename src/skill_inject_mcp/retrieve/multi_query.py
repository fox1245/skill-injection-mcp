from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import httpx

from skill_inject_mcp.timeouts import http_timeout

_SYSTEM = """You expand a skill-matching requirement into diverse search queries.
Return ONLY a JSON array of 3 to 5 short strings (no markdown, no commentary).
Include keyword variants, paraphrases, and skill-oriented phrasings.
Do not invent unrelated domains."""

_USER_TMPL = """Requirement description:
{description}

Optional search_query hint:
{search_query}

Return a JSON array of 3-5 diverse rewritten queries for hybrid skill retrieval."""


@dataclass
class MultiQueryResult:
    queries: list[str]
    skipped: bool
    reason: str | None = None


def _normalize_original(description: str, search_query: str | None) -> str:
    if search_query and search_query.strip():
        return search_query.strip()
    return (description or "").strip()


def _dedupe_preserve(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        q = (raw or "").strip()
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
    return out


def _extract_string_list(data: object) -> list[str] | None:
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        return None
    return [x.strip() for x in data if x.strip()]


def _parse_queries(content: str) -> list[str]:
    text = (content or "").strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    try:
        data, end = decoder.raw_decode(text)
        if end == len(text):
            # The whole reply is one JSON value: accept it or ignore it, never dig inside it.
            return _extract_string_list(data) or []
    except json.JSONDecodeError:
        pass
    pos = text.find("[")
    while pos != -1:
        try:
            data, _ = decoder.raw_decode(text, pos)
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return [x.strip() for x in data if x.strip()]
        except json.JSONDecodeError:
            pass
        pos = text.find("[", pos + 1)
    return []


def expand_queries(
    description: str,
    search_query: str | None = None,
    *,
    api_key: str | None,
    enabled: bool = True,
    model: str,
    base_url: str,
    timeout_s: float,
    connect_timeout_s: float,
    write_timeout_s: float,
    pool_timeout_s: float,
    provider_order: Sequence[str],
    client: httpx.Client | None = None,
) -> MultiQueryResult:
    """Expand a requirement into 3-5 retrieval queries via OpenRouter chat.

    Always returns the original as queries[0]. On disabled / no key / failure,
    returns [original] with skipped=True.
    """
    original = _normalize_original(description, search_query)
    if not original:
        return MultiQueryResult(queries=[""], skipped=True, reason="empty_query")

    if not enabled:
        return MultiQueryResult(queries=[original], skipped=True, reason="disabled")

    if not api_key:
        return MultiQueryResult(queries=[original], skipped=True, reason="no_api_key")

    order = list(provider_order)
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER_TMPL.format(
                    description=description or original,
                    search_query=(search_query or "").strip() or "(none)",
                ),
            },
        ],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    if order:
        payload["provider"] = {
            "order": order,
            "allow_fallbacks": True,
        }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fox1245/skill-injection-mcp",
        "X-Title": "skill-injection-mcp",
    }

    budget = http_timeout(
        timeout_s,
        connect_timeout_s=connect_timeout_s,
        write_timeout_s=write_timeout_s,
        pool_timeout_s=pool_timeout_s,
    )
    owns_client = client is None
    http = client
    try:
        if http is None:
            http = httpx.Client(timeout=budget)
        resp = http.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
            timeout=budget,
        )
        resp.raise_for_status()
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        parsed = _parse_queries(content)
        merged = _dedupe_preserve([original, *parsed])
        if len(merged) < 2:
            return MultiQueryResult(queries=[original], skipped=True, reason="empty_expansion")
        return MultiQueryResult(queries=merged[:5], skipped=False, reason=None)
    except Exception as exc:  # noqa: BLE001 — degrade on any transport/parse failure
        return MultiQueryResult(
            queries=[original],
            skipped=True,
            reason=f"error:{type(exc).__name__}",
        )
    finally:
        if owns_client and http is not None:
            http.close()
