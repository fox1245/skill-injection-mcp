"""Opt-in live evaluation using ONLY repository fixtures and synthetic requirements."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine
from skill_inject_mcp.schemas import Requirement, SkillInjectRequest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true",
                        help="Explicitly allow fixture/requirement transmission and API usage charges.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    args = parser.parse_args()
    if not args.live:
        parser.error("--live is required; this sends test data to OpenRouter")
    root = Path(__file__).resolve().parents[1]
    cases = json.loads((root / "evals" / "multilingual.json").read_text(encoding="utf-8"))
    cases += json.loads((root / "evals" / "multilingual-negative.json").read_text(encoding="utf-8"))
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="skill-eval-", dir=args.work_dir.resolve()) as directory:
        settings = Settings(
            _env_file=root / ".env", skills_dir=root / "fixtures" / "skills",
            index_dir=Path(directory), use_fake_embedder=False, multi_query=False,
            verification_mode="semantic",
        )
        if not settings.resolve_api_key():
            raise SystemExit("OPENROUTER_API_KEY is required; fake embeddings are not a live evaluation")
        if settings.embedding_base_url != "https://openrouter.ai/api/v1":
            raise SystemExit("This evaluation is scoped to https://openrouter.ai/api/v1")
        engine = SkillInjectEngine(settings)
        failures = 0
        try:
            for case in cases:
                start = time.monotonic()
                try:
                    result = engine.resolve(SkillInjectRequest(requirements=[
                        Requirement(id=case["id"], description=case["description"]),
                    ]))
                    check = result.checks[0]
                    correct = (
                        check.matched and result.match_status == "complete" and check.skill_id == case["expected_skill_id"]
                        if case["expected_skill_id"] is not None
                        else not check.matched and result.match_status != "complete"
                    )
                    correct = correct and not result.verification_degraded
                    row = {
                        **case, "correct": correct,
                        "response": result.model_dump(mode="json"),
                        "seconds": round(time.monotonic() - start, 3),
                    }
                    failures = failures + 1 if result.verification_degraded else 0
                except Exception as exc:
                    row = {**case, "correct": False, "error_type": type(exc).__name__}
                    failures += 1
                rows.append(row)
                report = {
                    "embedding_model": settings.embedding_model,
                    "verification_model": settings.verification_model,
                    "verification_mode": "semantic", "dimensions": settings.embedding_dim,
                    "translation": False, "multi_query": False,
                    "fixture_count": len(engine.registry.skills),
                    "total_planned": len(cases), "completed": len(rows),
                    "correct": sum(r["correct"] for r in rows), "cases": rows,
                }
                args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({
                    "id": case["id"], "language": case["language"], "correct": row["correct"],
                    "assessment": row.get("response", {}).get("checks", [{}])[0].get("assessment"),
                }), flush=True)
                if failures >= 3:
                    print("Stopped after three consecutive service failures.", flush=True)
                    break
        finally:
            engine.close()
    return 0 if len(rows) == len(cases) and all(r["correct"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
