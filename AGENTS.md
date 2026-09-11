# AGENTS.md — Skill Injection MCP

## Required operational defaults

Start normal installations with a real OpenRouter API key, `SKILL_INJECT_USE_FAKE_EMBEDDER=false`,
and `SKILL_INJECT_DENSE_BACKEND=sqlite-vector`. Provision and verify these before calling setup complete.
FakeEmbedder and NumPy are explicit offline/test options only; never silently substitute them
when a key is missing or a native extension fails. Explain the error and repair the setup.

Use the official [sqliteai/sqlite-vector release](https://github.com/sqliteai/sqlite-vector/releases/tag/1.1.0)
library directly. Python wheels and local compilation are not required: Windows uses vector.dll,
Linux vector.so, and macOS vector.dylib. After installing Python dependencies, run:

```bash
python scripts/setup_sqlite_vector.py --output-dir /absolute/path/to/native
```

The script pins version 1.1.0, verifies the official archive SHA-256, and checks native cosine search.
Set `SKILL_INJECT_SQLITE_VECTOR_PATH` to the resulting absolute library path.
To install without network access, pass `--archive /path/to/the-official-release.zip`.
Do not confuse this extension with the separate sqlite-vec project.

Keep credentials in an authorized, git-ignored .env file. Set
`SKILL_INJECT_OPENROUTER_API_KEY_FILE=/absolute/path/to/authorized/.env` to reuse its
OPENROUTER_API_KEY, including a shared file used by another MCP. The explicitly selected file is
authoritative; missing files/keys are errors. Never print or commit the key or copy it into Codex config.
The config contains only the path. Reuse existing user consent for remote embeddings and re-embedding;
obtain consent only when that data flow has not already been authorized. Unit tests remain offline.

Verify the actual configured process reports OpenRouterEmbedder, sqlite-vector, and no fallback.
Verify native vector_version()/vector_backend() and a successful retrieval; file existence is insufficient.
Preserve existing indexes and model-specific caches during migration. A new model must not reuse Fake
vectors. Reindex the authorized data and report any missing vectors or API failures.


## Purpose

Call `resolve_skills` before inventing workflows. Bind required capabilities to indexed Agent Skills.
Only `match_status == "complete"` means every required requirement passed the
reported verifier and dependency checks. Inspect `verification_mode`: semantic
verification and offline lexical matching have different capabilities. It does not mean the task has executed successfully.

## Platform

Python 3.11+, Windows and Linux. Install with `pip install -e ".[dev]"`.
Start with `skill-inject-mcp` or `python -m skill_inject_mcp`.
Registry paths are POSIX-relative to the selected skills root.

## Installation means Codex registration, not just pip install

When the user asks to install, bootstrap, register, or configure this project for
Codex, complete the MCP registration, global usage guidance, and prompt-hook setup
as part of that task. Act on already-authorized work; do not stop after installing
Python dependencies or merely suggest that the user register the server later.

This instruction is scoped to installation/integration work. Ordinary code reviews,
unit tests, CI runs, and unrelated edits must not silently rewrite global Codex
configuration. Do not require the not-yet-registered MCP to plan its own bootstrap.

### Registration workflow

1. **Inspect and preserve.** Resolve the actual repository path, user Codex home,
   Python interpreter, Codex desktop/CLI executable, and existing MCP/hooks config.
   Back up config.toml, hooks.json and global AGENTS.md before changing them.
   Preserve other MCP servers, hooks, style instructions and existing user choices.
   On Windows, a restricted process can resolve a different account's home; verify
   registration in the intended user's Codex host, not an empty sandbox profile.

2. **Install a working runtime.** Verify the interpreter and imports. Use a dedicated
   virtual environment with an editable installation of this repository. Use
   absolute executable/launcher paths and the repository as the MCP cwd so its
   .env can be loaded. Do not assume a copied .venv or the first python on PATH works.

3. **Use the real enabled catalog.** Call Codex's skills/list API and export enabled
   user/system/plugin entries with codex_adapter.sync_catalog. Retain the exact
   namespaced names and original SKILL.md paths. Configure SKILL_INJECT_SKILL_MANIFEST;
   fixtures/skills is for tests, never evidence of a successful global installation.
   Create/reuse a small launcher that refreshes the catalog before calling
   skill_inject_mcp.server.main. On refresh failure, identify any last-known catalog
   explicitly; do not silently substitute fixture skills.

4. **Upsert the MCP server.** Register the name skill-injection using Codex's MCP CLI
   or supported config API. Update an existing entry instead of creating duplicates.
   Set command/args/cwd and the environment listed below. Do not copy API keys into
   config.toml, hooks.json, AGENTS.md, logs, or git. Reuse the authorized .env/key source.

5. **Upsert global AGENTS.md guidance.** Use a uniquely marked managed section,
   preserving all other content. Direct agents to resolve atomic requirements before
   committing to substantive workflows; preserve user constraints; inspect assessment,
   citations and unmet requirements; read selected bodies with registry_snapshot;
   resolve resources from source_path. Explain that hook candidates are unverified
   and complete does not certify execution success. Do not force irrelevant skills
   or block all work just because no suitable skill exists.

6. **Upsert the UserPromptSubmit hook.** Use the native mcp_tool handler below when
   the installed Codex supports it. Inspect its schema/capabilities; 0.153.4 is a
   verified compatible version. Preserve existing matcher groups and avoid adding
   the same server/tool handler twice. The hook is advisory and must never turn
   discovery results into an execution approval or a forced binding.

7. **Complete the trust step correctly.** Codex must trust the exact hook definition.
   Review the concrete event, tool, input and timeout through the supported hook
   review/trust flow. Respect existing authorization and disabled states. If further
   user approval is required, prepare the complete registration first and ask only
   for the specific remaining trust/data-transmission decision. Never bypass hook
   trust or fabricate trust records. Changing a timeout can change the hook hash,
   so re-check the resulting trust status.

8. **Verify the registered setup.** Run codex mcp get skill-injection in the intended
   host; connect with the exact registered command/environment; list all four tools;
   test resolve_skills, get_skill_body and codex_prompt_hook. Check actual installed
   catalog counts, namespaces, source paths and errors. Use representative installed
   skills, including long and plugin-provided skills, rather than only fixtures.
   Report API failures/unknown results honestly and keep automated tests offline.

9. **Verify preservation and report state.** Confirm existing MCP behavior and hooks
   are preserved and repeat setup does not duplicate entries. Report configured,
   connected and trusted states separately. If the running app needs a reload,
   say so; do not claim that this task's already-loaded tool list has refreshed.

### Registered server defaults

Use machine-specific absolute paths instead of copying another user's paths:

```toml
[mcp_servers.skill-injection]
command = "<absolute-path-to-venv-python>"
args = ["-B", "-u", "<absolute-path-to-catalog-refresh-launcher>"]
cwd = "<absolute-path-to-this-repository>"
startup_timeout_sec = 60
tool_timeout_sec = 900

[mcp_servers.skill-injection.env]
PYTHONUTF8 = "1"
PYTHONUNBUFFERED = "1"
SKILL_INJECT_SKILL_MANIFEST = "<absolute-path-to-catalog.json>"
SKILL_INJECT_INDEX_DIR = "<absolute-path-to-local-index-cache>"
SKILL_INJECT_USE_FAKE_EMBEDDER = "false"
SKILL_INJECT_DENSE_BACKEND = "sqlite-vector"
SKILL_INJECT_SQLITE_VECTOR_PATH = "<absolute-path-to-native-vector-library>"
SKILL_INJECT_OPENROUTER_API_KEY_FILE = "<absolute-path-to-authorized-dotenv-file>"
SKILL_INJECT_PERSISTENT_EMBEDDING_CACHE = "true"
SKILL_INJECT_EMBEDDING_BATCH_SIZE = "32"
SKILL_INJECT_QUERY_CACHE_SIZE = "512"
SKILL_INJECT_HOOK_TIMEOUT_S = "2"
SKILL_INJECT_EMBEDDING_TIMEOUT_S = "180"
SKILL_INJECT_MULTI_QUERY_TIMEOUT_S = "30"
SKILL_INJECT_VERIFICATION_TIMEOUT_S = "120"
SKILL_INJECT_VERIFICATION_TOP_K = "5"
SKILL_INJECT_VERIFICATION_MAX_SOURCE_CHARS = "100000"
```

Enable SKILL_INJECT_VERIFICATION_MODE=semantic only within the user's authorized
scope. Initial indexing sends installed SKILL.md content to the embedding provider;
semantic verification sends original requirements and candidate sources to the chat
provider. Reuse existing consent for the same scope; do not assume unrelated
catalogs or data are authorized.

Add/update this handler under UserPromptSubmit in hooks.json:

```json
{
  "type": "mcp_tool",
  "server": "skill-injection",
  "tool": "codex_prompt_hook",
  "input": {"prompt": "${prompt}"},
  "timeout": 2,
  "statusMessage": "Finding installed skill candidates"
}
```

### Timeout policy

The HTTP read budgets are configurable: embeddings 180 seconds, query expansion
30 seconds, semantic verification 120 seconds. Connection and pool waits remain
10 seconds; writes have a 30-second budget. Injected/test HTTP clients must receive
the same per-request timeout, rather than silently keeping their own defaults.

Codex's tool timeout is an outer deadline (recommended 900 seconds); the advisory
prompt hook has a 2-second outer deadline and a separate cancellable async budget.
Cold hooks return unavailable; stale hooks label their last-known catalog while
one background refresh runs. These are not guarantees that an
arbitrary multi-requirement request will finish: each requirement can make several
HTTP calls. Split large plans into bounded batches when necessary. Keep failure
and unknown semantics intact; increasing a timeout is not permission to accept an
unverified result. Restart the server after environment timeout changes.


## resolve_skills

Pass a typed `SkillInjectRequest` in the MCP argument named `request`:

- `schema_version` must be `"1.0"`. Unknown fields are rejected.
- `requirements[]`: nonempty `id` and `description`, optional `required` (true),
  `depends_on`, and `search_query`.
- `search_query` is a retrieval hint. Acceptance always checks the original description.
- `draft_plan.steps[]`: unique `id`, `summary`, and valid `requirement_ids`.
- `constraints`: `skills_dir`, `rerank`, `top_k`, `min_score`.
  These are retrieval settings, not execution permissions or task constraints.
- `top_k` controls returned evidence only (server default 5). Lexical runner-up checks
  use the complete retrieved pool; semantic verification has a separate candidate budget.
  Internal retrieval channel width is 20.
- Optional multi-query expansion widens retrieval. It cannot certify fulfillment.

## Verification

Semantic verification is explicit opt-in via SKILL_INJECT_VERIFICATION_MODE=semantic.
It sends original requirements and top candidate descriptions/bodies to the configured
chat endpoint (gpt-oss-120b by default). A key alone does not enable it. Lexical is the
default offline fallback and does not establish cross-language semantic support.

In semantic mode:
- Compare meaning across languages, including all capabilities, exclusions and non-goals.
- Do not require shared words, a cosine cutoff or dense runner-up margin.
- `search_query` and expanded queries are retrieval hints, never acceptance requirements.
- `checks[].assessment`: supported, partial, unsupported, unknown, or dependency-blocked.
- Only supported candidates with valid dependencies are bound.
- Inspect `candidate_skill_id`, `unmet_requirements` and server-extracted `citations`.
- Invalid IDs, missing/duplicate results, invented quotes, inconsistent verdicts,
  timeouts and incomplete output remain unknown with `verification_degraded=true`.
- Do not silently fall back to lexical acceptance after a semantic failure.
- Inspect `verification_diagnostics` even after successful recovery. Stable codes distinguish
  truncated responses, invalid JSON/schema, candidate/source mismatches and HTTP failures.
- Output budget defaults to 8192 reasoning-plus-JSON tokens. Only `finish_reason=length`
  permits one retry with twice the budget, using the unchanged original request. Configure
  `SKILL_INJECT_VERIFICATION_MAX_TOKENS` and `SKILL_INJECT_VERIFICATION_MAX_RETRIES` (0 or 1).
  Every attempt must pass the original validation rules; partial JSON is never accepted.
- Full sources over the configured size limit remain unknown; no truncation.
- Verdict caches include original requirement, source content, model, endpoint and prompt version.

`verification_top_k` is a server setting (default 5), independent of returned evidence
width. Split compound requirements when appropriate, without dropping user constraints.
Exact quote checks verify source existence; model semantic judgement remains fallible.

Every requirement receives a check. A required requirement depending on an unmatched
optional requirement is still blocked. Bindings include prerequisite skills before
their dependents. Unrelated registry errors are reported without blocking valid candidates.

## reindex_skills

Rescan the selected skills root and rebuild a new index generation. Unchanged document
embeddings are reused during the process lifetime, and across restarts when
SKILL_INJECT_PERSISTENT_EMBEDDING_CACHE=true. Normal resolve calls automatically
detect content changes and reuse an unchanged snapshot. A failed refresh leaves the
previous generation intact; the failed request returns an error instead of claiming freshness.

## get_skill_body

Read `skill_id`; optionally pass `registry_snapshot` from resolve to reject a stale binding.
Returns the indexed body, root, relative path, content hash and snapshot identifier.

## Ranking and indexes

SQLite FTS5 BM25 + dense cosine, fused by RRF (k=60). Tie-break: RRF descending,
dense rank ascending, skill_id ascending. BM25 values are not confidence probabilities.
Only lexical fallback applies a positive sparse-score or dense-margin gate.
Semantic verification uses retrieval scores for ranking, not fulfillment.

sqlite-vector is the default and executes native exact cosine with vector_full_scan.
NumPy NPZ storage is an explicit offline option. Bulk writes persist once per generation.
Rerank `qwen3-0.6b` remains a stub and sets `retriever_degraded`.

## Validation

Run `python -m pytest -q`. Tests block real HTTP and use temporary indexes.
Cover false-complete requests, Korean text, dependency closure, MCP schemas,
candidate-width invariance, incremental embeddings, failed-refresh preservation,
semantic source-grounding and service-failure handling.

The separate scripts/evaluate_multilingual.py --live command runs a paid, opt-in
fixture-only cross-language evaluation; it is not part of pytest/CI.

## Installed catalogs and Codex hooks

SKILL_INJECT_SKILL_MANIFEST selects the actual enabled catalog exported from Codex
skills/list, retaining namespaced plugin names and original source_path.
Do not validate an installation using fixtures alone.

codex_prompt_hook is advisory UserPromptSubmit discovery. It does not bind skills
or certify support. Use resolve_skills with atomic requirements before substantive
work and load selected instructions from their original source_path.

For long installed skills, configure an adequate verification_max_source_chars;
do not silently truncate exclusions. Candidate source IDs are constrained to known
values and checked against the corresponding original document.

## Meta-skill layer

Skills-about-skills (authoring, installing, inspecting, maintaining, or cataloguing SKILL.md) are a separate meta layer. Domain work searches the domain layer first.

Classification uses frontmatter layer: meta|domain, tags meta/meta-skill, known meta skill ids, then conservative description heuristics. resolve_skills reports layer on checks and evidence. A meta-skill match is not domain-task coverage.

Canonical meta skills include `agentx-codex-conductor`, `skill-inspector`, `skill-creator`, `skill-installer`, `skills-maintain`, and `find-skills`. A conductor/orchestration requirement must resolve in the meta layer, not as a domain skill.

Before installing or binding an untrusted skill, follow skill-inspector / SkillSpector. APPROVE is not a Skill Epoch.
