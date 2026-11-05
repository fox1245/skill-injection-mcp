"""Child-process helper that builds Settings + OpenRouterEmbedder and embeds documents twice."""
from __future__ import annotations

from pathlib import Path
import os
import sys
import time

from skill_inject_mcp.config import Settings
from skill_inject_mcp.embed.admission import build_admission
from skill_inject_mcp.embed.embedder import build_embedder


def main(argv: list[str]) -> int:
    ready_file = Path(argv[1])
    start_file = Path(argv[2])
    disable_queue = "--no-admission" in argv[3:]

    ready_file.write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 15.0
    while not start_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for start file")
        time.sleep(0.01)

    settings = Settings(_env_file=None)
    admission = None if disable_queue else build_admission(settings)
    embedder, _ = build_embedder(
        api_key=settings.resolve_embedding_api_key(),
        use_fake=False,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        base_url=settings.embedding_base_url,
        timeout_s=settings.embedding_timeout_s,
        connect_timeout_s=settings.http_connect_timeout_s,
        write_timeout_s=settings.http_write_timeout_s,
        pool_timeout_s=settings.http_pool_timeout_s,
        query_cache_size=settings.query_cache_size,
        admission=admission,
    )
    try:
        pid = os.getpid()
        for call_idx in range(2):
            vectors = embedder.embed_documents([f"doc-{pid}-{call_idx}"])
            if len(vectors) != 1:
                raise RuntimeError("Unexpected vector count")
    finally:
        embedder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
