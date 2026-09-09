# AGENTS.md — Skill Injection MCP

## Purpose

Call `resolve_skills` **before** inventing workflows. Bind required capabilities to indexed Agent Skills. Do **not** treat a response as done unless `match_status == "complete"` and every required requirement has a binding.

## Platform

Runs on Windows and Linux. Install with `pip install -e ".[dev]"` (PowerShell or bash). Start via `skill-inject-mcp` or `python -m skill_inject_mcp`. See README for OS-specific `.env` and PATH tips (LibreOffice PATH caveat is Windows-only).

Skill paths in the registry are POSIX-relative to `skills_dir` so the same skills tree layout indexes portably across OS.

## Tools

### resolve_skills

Input: `SkillInjectRequest` (schema_version `"1.0"`).

- Provide `requirements[]` with `id`, `description`, optional `required` (default true), optional `depends_on`, optional `search_query`.
- Server expands queries from descriptions when `search_query` is omitted.
- Optional `draft_plan`, `constraints` (`skills_dir`, `rerank`, `top_k`).
- Pass `constraints.top_k` (int >= 1) to control how many hybrid candidates are walked / returned as evidence; **default 5** when omitted. Internal retrieve channel stays `retrieve_top_k` (20).
- With `OPENROUTER_API_KEY`, resolve may expand each requirement into 3–5 queries (multi-query) before a single RRF; fulfillment still requires the lexical+score match gate.

Output: `SkillInjectResponse` with `match_status` (`complete` | `partial` | `no_match`), `checks`, `evidence`, `gaps`, `validation_errors`, `retriever_degraded`, `plan_bindings`.

**Nudge**: After `resolve_skills`, if status is not `complete`, stop and fill gaps or reindex — do not claim the plan is skill-backed.

### reindex_skills

Rescan `skills_dir` (or configured default), rebuild sparse + dense indexes.

### get_skill_body

Return full `SKILL.md` body for a `skill_id`.

## Ranking

Results ordered by fused `ranking_score` (RRF). Tie-break: RRF desc, dense_rank asc, skill_id asc. Rerank stub: `off` | `qwen3-0.6b` (latter skips and sets `retriever_degraded`).

## Validation

Duplicate skill ids or dependency DAG cycles → `validation_errors` and non-complete status.

## Indexes

- Sparse: SQLite FTS5 (stdlib) on both OS.
- Dense: optional `sqlite-vector` native extension when loadable; otherwise numpy `.npz` cosine fallback.
