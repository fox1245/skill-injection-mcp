from __future__ import annotations

from pathlib import Path

import pytest

from skill_inject_mcp.config import Settings
from skill_inject_mcp.engine import SkillInjectEngine

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "skills"


@pytest.fixture
def skills_dir() -> Path:
    return FIXTURES


@pytest.fixture
def engine(tmp_path: Path, skills_dir: Path) -> SkillInjectEngine:
    settings = Settings(
        skills_dir=skills_dir,
        index_dir=tmp_path / "index",
        use_fake_embedder=True,
        rerank="off",
    )
    eng = SkillInjectEngine(settings=settings)
    eng.reindex()
    return eng
