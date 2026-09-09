# Multilingual evaluation

This is a small, synthetic regression evaluation over the four repository fixture skills.
It is not a broad multilingual benchmark or a guarantee about every language/domain.

## Models and settings

- Embedding: `qwen/qwen3-embedding-8b`, 1024 dimensions, L2 normalized.
- Verification: `openai/gpt-oss-120b`, structured output with exact source-quote validation.
- Documents: 4 English Agent Skills from `fixtures/skills`.
- Translation: disabled.
- Multi-query: disabled.
- Languages: English, Korean, Japanese, Chinese, Vietnamese and Russian.

## Observed sample

| Check | Result |
|---|---:|
| Raw dense retrieval: expected skill ranked first | 18 / 18 |
| Previous lexical gate: positive cases accepted using those dense results | 2 / 18 |
| Semantic pipeline: positive cases correctly bound | 18 / 18 |
| Negative/compound cases without false complete | 6 / 6 |
| API/response-validation degraded cases | 0 / 24 |

The negative cases include an unrelated chess-engine task, sparse+dense requirements
that no individual fixture fully supports, Kubernetes deployment, installation constrained
to CMake, and a no-network condition absent from the installer specification.

Inputs are checked into `multilingual.json` and `multilingual-negative.json`.
The live run was performed during this change after explicit approval for sending
fixture documents and synthetic requests to OpenRouter. A single run is not a statistical
estimate of accuracy; reruns can differ because models/providers may change.

## Reproduce

Install this repository and configure `OPENROUTER_API_KEY`, then run:

```bash
python scripts/evaluate_multilingual.py --live --output work/multilingual-report.json
```

`--live` is required. This sends only fixture skills and synthetic requirements to
OpenRouter and can incur API charges. It overrides the configured skill directory,
disables translation/multi-query and uses real embeddings plus semantic verification.
Each case is written to the report immediately; the process stops after three
consecutive service failures. Service-degraded results do not count as correct.
Pytest/CI do not run this live evaluation.
