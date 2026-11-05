from __future__ import annotations

import asyncio

import httpx
import pytest

from tests.embed_server import EmbeddingServer


def test_non_loopback_http_is_blocked_and_loopback_test_server_is_allowed() -> None:
    with httpx.Client() as client:
        with pytest.raises(AssertionError, match="Tests must use httpx.MockTransport, not real HTTP"):
            client.get("https://openrouter.ai/api/v1/models")

    async def check_async_blocked() -> None:
        async with httpx.AsyncClient() as async_client:
            with pytest.raises(AssertionError, match="Tests must use httpx.MockTransport, not real async HTTP"):
                await async_client.get("https://openrouter.ai/api/v1/models")

    asyncio.run(check_async_blocked())

    with EmbeddingServer() as server:
        with httpx.Client() as client:
            resp = client.post(
                f"{server.base_url}/embeddings",
                json={"model": "test", "input": ["hello"], "dimensions": 4},
            )
            assert resp.status_code == 200
            assert len(resp.json()["data"]) == 1

        async def check_async_loopback() -> None:
            async with httpx.AsyncClient() as async_client:
                async_resp = await async_client.post(
                    f"{server.base_url}/embeddings",
                    json={"model": "test", "input": ["world"], "dimensions": 4},
                )
                assert async_resp.status_code == 200
                assert len(async_resp.json()["data"]) == 1

        asyncio.run(check_async_loopback())
