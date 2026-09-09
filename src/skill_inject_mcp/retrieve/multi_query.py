from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

import httpx

DEFAULT_MULTI_QUERY_MODEL = "openai/gpt-oss-120b"
DEFAULT_TIMEOUT_S = 12.0
# Prefer fast inference hosts when the model is available there (OpenRouter provider routing).
DEFAULT_PROVIDER_ORDER = ["Cerebras", "Groq"]

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


def _parse_queries(content: str) -> list[str]:
    text = (content or "").strip()
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", text)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict) and "queries" in data:
        data = data["queries"]
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def expand_queries(
    description: str,
    search_query: str | None = None,
    *,
    api_key: str | None,
    enabled: bool = True,
    model: str = DEFAULT_MULTI_QUERY_MODEL,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    provider_order: Sequence[str] | None = None,
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

    order = list(provider_order) if provider_order is not None else list(DEFAULT_PROVIDER_ORDER)
    payload = {
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
        "provider": {
            "order": order,
            "allow_fallbacks": True,
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fox1245/skill-injection-mcp",
        "X-Title": "skill-injection-mcp",
    }

    owns_client = client is None
    http = client
    try:
        if http is None:
            http = httpx.Client(timeout=timeout_s)
        resp = http.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers,
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
