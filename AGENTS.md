# AGENTS.md — Skill Injection MCP

## Purpose

Call `resolve_skills` before inventing workflows. Bind required capabilities to indexed Agent Skills.
Only `match_status == "complete"` means every required requirement passed the
reported verifier and dependency checks. Inspect `verification_mode`: semantic
verification and offline lexical matching have different capabilities. It does not mean the task has executed successfully.

## Platform

Python 3.11+, Windows and Linux. Install with `pip install -e ".[dev]"`.
Start with `skill-inject-mcp` or `python -m skill_inject_mcp`.
Registry paths are POSIX-relative to the selected skills root.

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

Optional sqlite-vector storage uses exact cosine in Python; it is not native ANN.
Otherwise numpy stores vectors in an NPZ without pickle. Bulk writes persist once per generation.
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
