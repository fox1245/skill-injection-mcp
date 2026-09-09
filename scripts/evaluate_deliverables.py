"""Opt-in semantic contrast checks using only small, synthetic bundled skill sources."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.retrieve.semantic import SemanticVerifier
from skill_inject_mcp.schemas import SkillMeta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Allow fixture transmission and OpenRouter API charges.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this sends synthetic test data to OpenRouter")
    root = Path(__file__).resolve().parents[1]
    fixtures = json.loads((root / "evals" / "deliverable-contrast.json").read_text(encoding="utf-8"))
    skills = [SkillMeta.model_validate(source) for source in fixtures["skills"]]
    settings = Settings(_env_file=root / ".env")
    if not settings.resolve_api_key():
        raise SystemExit("OPENROUTER_API_KEY is required")
    if settings.embedding_base_url != "https://openrouter.ai/api/v1":
        raise SystemExit("This fixture evaluation is scoped to https://openrouter.ai/api/v1")
    verifier = SemanticVerifier()
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in fixtures["cases"]:
        started = time.monotonic()
        result = verifier.verify(
            case["description"], skills, api_key=settings.resolve_api_key(),
            model=settings.verification_model, timeout_s=settings.verification_timeout_s,
            max_tokens=settings.verification_max_tokens, max_retries=settings.verification_max_retries,
        )
        matched = sorted(sid for sid, assessment in result.assessments.items() if assessment.matched)
        row = {"id": case["id"], "correct": not result.degraded and matched == sorted(case["expected_matched"]),
               "matched": matched, "degraded": result.degraded, "reason": result.reason,
               "diagnostics": [d.model_dump(mode="json") for d in result.diagnostics],
               "seconds": round(time.monotonic() - started, 3)}
        rows.append(row)
        report = {"model": settings.verification_model, "fixture_only": True,
                  "completed": len(rows), "total": len(fixtures["cases"]),
                  "correct": sum(r["correct"] for r in rows), "cases": rows}
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if len(rows) >= 3 and all(r["degraded"] for r in rows[-3:]):
            break
    return 0 if len(rows) == len(fixtures["cases"]) and all(r["correct"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
