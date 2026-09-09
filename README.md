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
- **Cross-language verification (opt-in)**: gpt-oss-120b compares the ORIGINAL requirement
  with each candidate's complete description and body, including non-goals. It checks
  meaning and constraints rather than shared words or embedding thresholds.
- **Source-grounded results**: supported / partial / unsupported / unknown verdicts
  cite permitted, candidate-specific source IDs. The server extracts exact original text;
  source ownership, IDs and verdict consistency are checked locally.
  Invalid responses and service errors remain unknown; they never fall through to lexical acceptance.
- **Offline fallback**: lexical mode retains conservative textual matching, explicitly
  labelled as lexical. An API key alone does not enable remote semantic verification.
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
plan_bindings, retriever_degraded, registry_snapshot, verification_mode and
verification_degraded. Each check reports its verifier, assessment, candidate_skill_id,
unmet_requirements, evidence and exact citations (description/body + quote).
Only supported candidates with resolved dependencies are bound. Partial, unsupported
and unknown candidates never produce complete on their own.

## Semantic verification (explicit opt-in)

Set these in .env to enable cross-language verification:

```dotenv
SKILL_INJECT_VERIFICATION_MODE=semantic
SKILL_INJECT_VERIFICATION_MODEL=openai/gpt-oss-120b
```

This sends each original requirement and the top candidate descriptions/bodies to
the configured OpenRouter chat endpoint. The default remains lexical; API-key presence
alone does not opt in. Semantic mode requires a key and returns unknown on failure.

The verifier checks up to SKILL_INJECT_VERIFICATION_TOP_K candidates (default 5),
independent of constraints.top_k, which controls displayed evidence only.
Whole candidate sources above SKILL_INJECT_VERIFICATION_MAX_SOURCE_CHARS (default
16000 characters) are not sent or silently truncated; those candidates remain unknown.
Validated verdicts are cached by original requirement, complete source, model, endpoint
and prompt version (256 entries by default).

Translation is not required. Dense retrieval directly embeds the original language;
BM25 and optional query expansion complement candidate recall. Semantic verification
does not require any lexical overlap, a fixed cosine threshold or a dense runner-up margin.
Lexical mode remains available for offline development, but cannot establish cross-language support.

Model judgement is not a proof of execution success. Exact quote validation establishes
that the cited text exists; it does not prove the model interpreted every condition correctly.

## Live multilingual evaluation

The opt-in evaluation uses only fixtures/skills and synthetic evals/*.json, never
the configured user's skill directory. It tests 18 positive requirements across
English, Korean, Japanese, Chinese, Vietnamese and Russian, plus six negative or
compound requirements. Translation and multi-query are disabled to isolate dense
retrieval and semantic verification.

```bash
python scripts/evaluate_multilingual.py --live --output work/multilingual-report.json
```

The --live flag authorizes fixture transmission and API usage charges for that run.
Results are written after each case; three consecutive service failures stop the run.
See evals/README.md for the measured sample and its limits.

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
  In lexical fallback mode, sparse support requires a positive score after textual verification. Dense-only support
  requires cosine >=0.45 and a >=0.05 margin, or >=0.55 with no retrieved competitor.
- With two RRF channels and k=60, the maximum fused score is 2/61 (about 0.03279).
  Setting constraints.min_score above that excludes every candidate.
- Live index snapshots are process-local. Set SKILL_INJECT_PERSISTENT_EMBEDDING_CACHE=true
  to reuse unchanged document embeddings across server restarts. The content-addressed
  SQLite cache includes model identity and validates vector shape/finiteness.
  Managed generations are cleaned up on replacement and normal server shutdown;
  an abruptly terminated process may leave its generation directory behind.
- Refresh failures are returned to the caller. The last successful snapshot remains
  available for indexed body reads, but a failed refresh is not presented as current.
- Pytest blocks real HTTP. The separate opt-in live evaluation checks model behaviour;
  a small fixture evaluation is not a general multilingual quality guarantee.

## Codex global integration

Codex's own skills/list API can provide the exact enabled global catalog, including
system and plugin skills. skill_inject_mcp.codex_adapter.sync_catalog writes that
catalog as a manifest of names and original SKILL.md paths. It does not start a task.

Set SKILL_INJECT_SKILL_MANIFEST to that manifest path to index it instead of
fixtures/skills. Explicit constraints.skills_dir still overrides the manifest.
get_skill_body returns source_path so relative resources can be loaded from the
original skill folder. Catalog refresh happens when the installed launcher starts.

Recommended settings for a large installed catalog:

```dotenv
SKILL_INJECT_VERIFICATION_MODE=semantic
SKILL_INJECT_VERIFICATION_TOP_K=5
SKILL_INJECT_VERIFICATION_MAX_SOURCE_CHARS=100000
SKILL_INJECT_PERSISTENT_EMBEDDING_CACHE=true
SKILL_INJECT_EMBEDDING_BATCH_SIZE=32
```

The optional codex_prompt_hook tool returns a UserPromptSubmit additionalContext
object containing up to three UNVERIFIED candidates. It never blocks the prompt,
skips simple acknowledgements, and leaves structured requirements and binding to
resolve_skills. Hook errors are advisory.

Configure a native MCP hook in Codex versions supporting mcp_tool handlers:

```json
{
  "type": "mcp_tool",
  "server": "skill-injection",
  "tool": "codex_prompt_hook",
  "input": {"prompt": "${prompt}"},
  "timeout": 240,
  "statusMessage": "Finding installed skill candidates"
}
```

Add it under UserPromptSubmit in hooks.json while preserving existing hooks.
Codex requires review/trust of the exact new definition. Initial indexing transmits
installed SKILL.md documents to the configured embedding provider; semantic mode
also transmits original requirements and candidate sources. Enable this only for
the catalogs and data the user has authorized.

## HTTP and Codex deadlines

HTTP response-read budgets are configurable:

| Setting | Default |
|---|---:|
| SKILL_INJECT_EMBEDDING_TIMEOUT_S | 180 seconds |
| SKILL_INJECT_MULTI_QUERY_TIMEOUT_S | 30 seconds |
| SKILL_INJECT_VERIFICATION_TIMEOUT_S | 120 seconds |

Connection/pool waits remain 10 seconds and request writes have a 30-second budget.
Timeouts also apply when a caller supplies an HTTP client. Embedding timeout errors
identify the timeout type and configured read budget without including credentials.

For Codex registration, use startup_timeout_sec=60, tool_timeout_sec=900 and a
240-second UserPromptSubmit hook timeout. The HTTP values are per-I/O budgets;
the Codex values are outer deadlines. Large plans can require multiple requests
per requirement, so split large plans into smaller batches rather than treating
these defaults as an unlimited end-to-end allowance. Restart MCP processes after
changing environment settings.

See AGENTS.md for the automatic installation/registration workflow. Installation
includes preserving existing config, exporting the actual global catalog, upserting
MCP and hook entries, updating global guidance, checking exact hook trust, and
verifying the registered connection. Ordinary code review and CI do not trigger
global configuration changes.
