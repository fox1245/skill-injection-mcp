# skill-injection-mcp

Python MCP server that indexes Agent Skills (`SKILL.md`) and resolves task requirements via hybrid retrieval (dense + sparse BM25 fused with RRF).

Cross-platform: Windows and Linux (Python 3.11+).

## Configuration

Copy `.env.example` to `.env` and fill in values.

PowerShell:

```powershell
Copy-Item .env.example .env
```

bash:

```bash
cp .env.example .env
```

`.env` is gitignored. Without `OPENROUTER_API_KEY`, dense retrieval uses `FakeEmbedder`.

## Features

- **Dense retrieval**: prefers `sqliteai/sqlite-vector` when the optional native extension is loadable; otherwise uses a numpy cosine `VectorIndex` fallback (the default on most setups).
- **Sparse retrieval**: SQLite FTS5 BM25 (stdlib `sqlite3`).
- **Fusion**: Reciprocal Rank Fusion (k=60), top-20 each side, fuse by `skill_id`. Tie-break: RRF desc, dense_rank asc, skill_id asc.
- **Embeddings**: OpenRouter `qwen/qwen3-embedding-8b` MRL 1024 + L2 normalize; `FakeEmbedder` for offline tests (no API key).
- **Tools**: `resolve_skills`, `reindex_skills`, `get_skill_body`.
- **Never complete** if any required requirement is unmatched.
- **Rerank stub**: `off` | `qwen3-0.6b` (latter skips and sets `retriever_degraded`).
- **Portable skill paths**: registry stores skill file paths as POSIX-style paths relative to `skills_dir` (`Path.as_posix()`), so indexes remain portable when the skills tree layout matches across OS.

## Quick start

### Windows (PowerShell)

```powershell
cd path\to\skill-injection-mcp
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

Run the MCP server (stdio):

```powershell
$env:SKILL_INJECT_SKILLS_DIR = "C:\path\to\skills"
.\.venv\Scripts\skill-inject-mcp.exe
# or:
.\.venv\Scripts\python.exe -m skill_inject_mcp
```

Windows tip: if `py` / `python` unexpectedly points at LibreOffice's bundled interpreter, use an explicit CPython install (e.g. `py -3.12`) and put that ahead of LibreOffice on PATH.

### Linux (bash)

```bash
cd path/to/skill-injection-mcp
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pytest -q
```

Run the MCP server (stdio):

```bash
export SKILL_INJECT_SKILLS_DIR=/path/to/skills
skill-inject-mcp
# or:
python -m skill_inject_mcp
```

Set `OPENROUTER_API_KEY` for real embeddings (optional). Without it, dense uses `FakeEmbedder` and responses may set `retriever_degraded`.

## Layout

- `src/skill_inject_mcp/` — server, schemas, config, registry, embed, index, retrieve
- `fixtures/skills/` — sample Agent Skills for tests
- `tests/` — pytest (no OpenRouter required; uses `FakeEmbedder`)
- `AGENTS.md` — agent-oriented tool usage and schemas

## Entry points

Both of these work on Windows and Linux after `pip install -e .`:

- Console script: `skill-inject-mcp`
- Module: `python -m skill_inject_mcp`

## Caveats

- **sqlite-vector is optional**. When the native extension is not loadable (common on stock Windows Python builds and some Linux distro builds where extension loading is disabled), the MVP automatically falls back to numpy cosine over an `.npz` store behind the same `VectorIndex` interface.
- Stdlib SQLite FTS5 is used for sparse retrieval on both OS; no extra packages required.
- Match acceptance requires meaningful BM25 (`sparse_score >= 0.1`) or strong dense cosine (`>= 0.35`) so weak OR-token FTS noise cannot false-complete.
