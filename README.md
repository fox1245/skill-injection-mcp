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

- **Hybrid retrieval**: SQLite FTS5 BM25 and dense cosine, fused by RRF (k=60).
  Internal retrieval takes the top 20 per query/channel; ties are deterministic.
- **Conservative verification**: acceptance checks the original requirement, not its
  search hint. All significant terms need positive textual support. Negative sections,
  negative sentences, code blocks and names/tags alone do not prove capabilities.
  Incomplete support produces an unknown check rather than a false complete.
- **Unicode text**: sparse retrieval, fake embeddings and evidence tokenization retain
  Korean and other Unicode words.
- **Strict MCP contract**: typed input/output schemas, supported schema version only,
  and errors for unknown fields instead of silently discarding constraints.
- **Dependencies**: validate request references before indexing; verify selected skill
  dependencies transitively; include prerequisites in bindings and propagate unresolved needs.
- **Evidence width**: constraints.top_k (default 5) limits returned evidence only.
  It does not hide competitors from acceptance or dense-margin checks.
- **Incremental embeddings**: unchanged document vectors and live snapshots are reused
  within the server process. Changes rebuild independent index generations; failures
  preserve the previous generation. Embedding batches default to 32 documents.
- **Versioned reads**: resolve returns registry_snapshot; get_skill_body can require
  that snapshot and returns the indexed content hash.
- **Optional multi-query**: OpenRouter expansion widens retrieval, with original-description
  fallback and degraded status on failures. Expansion never substitutes for verification.
- **Embedding backend**: OpenRouter qwen/qwen3-embedding-8b (1024 dimensions), or the
  deterministic FakeEmbedder for offline development.
- **Rerank**: qwen3-0.6b remains a stub; requesting it sets retriever_degraded.

## Tool contract

The MCP tools/call arguments for resolve_skills include a request object:

```json
{
  "request": {
    "schema_version": "1.0",
    "requirements": [
      {
        "id": "sparse",
        "description": "BM25 FTS5 sparse full-text search in Python",
        "required": true
      }
    ],
    "constraints": {"top_k": 3}
  }
}
```

Requirements can include search_query and depends_on. The draft_plan is optional;
each step contains id, summary and requirement_ids. Unknown fields are rejected,
including unsupported task constraints. The constraints object configures retrieval,
not execution permissions.

Results include match_status, checks, evidence, gaps, validation_errors,
plan_bindings, retriever_degraded and registry_snapshot. Each check has an assessment
(supported, unknown or blocked), missing_terms and positive source evidence.
Complete means every required need has textual skill support and valid dependencies;
it is not a guarantee of semantic correctness or successful execution.

Use atomic requirements. The offline verifier is deliberately conservative: valid
synonyms, cross-language matches, negative constraints and nuanced conditions may
remain unknown. Review that evidence rather than removing conditions to force a match.
A no_match result means this registry did not establish support, not that the task is impossible.

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

- sqlite-vector is optional. When available it stores vectors in SQLite; this implementation
  computes exact cosine in Python, not native approximate nearest-neighbor search.
  Otherwise it uses numpy with non-pickle NPZ storage.
- BM25 scores depend on the corpus. They are ranking evidence, not calibrated probabilities.
  Sparse support requires a positive score after textual verification. Dense-only support
  requires cosine >=0.45 and a >=0.05 margin, or >=0.55 with no retrieved competitor.
- With two RRF channels and k=60, the maximum fused score is 2/61 (about 0.03279).
  Setting constraints.min_score above that excludes every candidate.
- Snapshot/vector reuse is currently process-local. A server restart rebuilds the index.
  Managed generations are cleaned up on replacement and normal server shutdown;
  an abruptly terminated process may leave its generation directory behind.
- Refresh failures are returned to the caller. The last successful snapshot remains
  available for indexed body reads, but a failed refresh is not presented as current.
- Tests run without real OpenRouter requests. Live model quality and provider limits
  require separate integration evaluation.
