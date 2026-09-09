# AGENTS.md — Skill Injection MCP

## Purpose

Call `resolve_skills` before inventing workflows. Bind required capabilities to indexed Agent Skills.
Only `match_status == "complete"` means every required requirement has conservative
textual support and valid dependencies. It does not mean the task has executed successfully.

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
- `top_k` controls returned evidence only (server default 5). Acceptance and dense
  runner-up checks use the complete retrieved candidate pool. Internal channel width is 20.
- Optional multi-query expansion widens retrieval. It cannot certify fulfillment.

Read `checks[].assessment`, `missing_terms`, and positive source `evidence`:

- `supported`: all significant requirement terms appear in positive skill statements,
  with retrieval support. Names/tags, negative statements, code blocks and excluded
  sections cannot independently prove support.
- `unknown`: textual support is incomplete or retrieval evidence is weak.
- `blocked`: references or required dependencies are unresolved.

The verifier is deliberately conservative. It can abstain on valid paraphrases,
cross-language requirements, negative constraints and nuanced semantic conditions.
Split compound requirements into atomic needs and review unknown checks; do not
discard requirements or invent search hints just to obtain complete.

Every requirement receives a check. A required requirement depending on an unmatched
optional requirement is still blocked. Bindings include prerequisite skills before
their dependents. Unrelated registry errors are reported without blocking valid candidates.

## reindex_skills

Rescan the selected skills root and rebuild a new index generation. Unchanged document
embeddings are reused during the process lifetime. Normal resolve calls automatically
detect content changes and reuse an unchanged snapshot. A failed refresh leaves the
previous generation intact; the failed request returns an error instead of claiming freshness.

## get_skill_body

Read `skill_id`; optionally pass `registry_snapshot` from resolve to reject a stale binding.
Returns the indexed body, root, relative path, content hash and snapshot identifier.

## Ranking and indexes

SQLite FTS5 BM25 + dense cosine, fused by RRF (k=60). Tie-break: RRF descending,
dense rank ascending, skill_id ascending. BM25 values are not confidence probabilities.
Sparse evidence must be positive; dense-only matches require cosine >=0.45 and margin
>=0.05 (or >=0.55 with truly no runner-up).

Optional sqlite-vector storage uses exact cosine in Python; it is not native ANN.
Otherwise numpy stores vectors in an NPZ without pickle. Bulk writes persist once per generation.
Rerank `qwen3-0.6b` remains a stub and sets `retriever_degraded`.

## Validation

Run `python -m pytest -q`. Tests block real HTTP and use temporary indexes.
Cover false-complete requests, Korean text, dependency closure, MCP schemas,
candidate-width invariance, incremental embeddings and failed-refresh preservation.
