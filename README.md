# skill-injection-mcp

Python MCP server that indexes Agent Skills (SKILL.md) and resolves task requirements via hybrid retrieval (dense + sparse BM25 fused with RRF).

## Features

- **Dense retrieval**: prefers sqliteai/sqlite-vector when the extension is loadable; otherwise uses a numpy cosine VectorIndex fallback (default on Windows MVP).
- **Sparse retrieval**: SQLite FTS5 BM25.
- **Fusion**: Reciprocal Rank Fusion (k=60), top-20 each side, fuse by skill_id. Tie-break: RRF desc → dense_rank asc → skill_id asc.
- **Embeddings**: OpenRouter qwen/qwen3-embedding-8b MRL 1024 + L2 normalize; FakeEmbedder for offline tests (no API key).
- **Tools**: 
esolve_skills, 
eindex_skills, get_skill_body.
- **Never complete** if any required requirement is unmatched.
- **Rerank stub**: off | qwen3-0.6b (latter skips and sets 
etriever_degraded).

## Quick start (Windows)

Prefer a real CPython install (not LibreOffice's bundled interpreter):

`powershell
cd C:\junyeong\Codes\skill-injection-mcp
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
C:\junyeong\Codes\skill-injection-mcp\.pytest_tmp = "C:\junyeong\Codes\skill-injection-mcp\.pytest_tmp"; C:\junyeong\Codes\skill-injection-mcp\.pytest_tmp = C:\junyeong\Codes\skill-injection-mcp\.pytest_tmp
.\.venv\Scripts\python.exe -m pytest -q --basetemp="C:\junyeong\Codes\skill-injection-mcp\.pytest_tmp\bt"
`

Set OPENROUTER_API_KEY for real embeddings (optional). Without it, dense uses FakeEmbedder and responses may set 
etriever_degraded.

`powershell
 = "C:\path\to\skills"
.\.venv\Scripts\skill-inject-mcp.exe
`

## Layout

- src/skill_inject_mcp/ — server, schemas, config, registry, embed, index, retrieve
- ixtures/skills/ — sample Agent Skills for tests
- 	ests/ — pytest (no OpenRouter required)
- AGENTS.md — agent-oriented tool usage and schemas

## Caveats

- On Windows, sqlite-vector is often unavailable; the MVP automatically falls back to numpy cosine over an .npz store behind the same VectorIndex interface.
- Match acceptance requires meaningful BM25 (sparse_score >= 0.1) or strong dense cosine (>= 0.35) so weak OR-token FTS noise cannot false-complete.
