"""HTTP read budgets are separate from connection setup and outer MCP deadlines."""
from __future__ import annotations

import httpx

EMBEDDING_READ_TIMEOUT_S = 180.0
MULTI_QUERY_READ_TIMEOUT_S = 30.0
VERIFICATION_READ_TIMEOUT_S = 120.0


def http_timeout(read_timeout_s: float) -> httpx.Timeout:
    return httpx.Timeout(connect=10.0, read=read_timeout_s, write=30.0, pool=10.0)
