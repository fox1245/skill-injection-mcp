"""HTTP read budgets are separate from connection setup and outer MCP deadlines."""
from __future__ import annotations

import httpx

EMBEDDING_READ_TIMEOUT_S = 180.0
MULTI_QUERY_READ_TIMEOUT_S = 30.0
VERIFICATION_READ_TIMEOUT_S = 120.0


def http_timeout(
    read_timeout_s: float,
    connect_timeout_s: float,
    write_timeout_s: float,
    pool_timeout_s: float,
) -> httpx.Timeout:
    return httpx.Timeout(
        connect=connect_timeout_s,
        read=read_timeout_s,
        write=write_timeout_s,
        pool=pool_timeout_s,
    )
