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
        multi_query=False,
        _env_file=None,
        rerank="off",
    )
    eng = SkillInjectEngine(settings=settings)
    eng.reindex()
    yield eng
    eng.close()


@pytest.fixture(autouse=True)
def offline_tests(monkeypatch):
    import httpx

    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("SKILL_INJECT_MULTI_QUERY", "false")
    monkeypatch.setenv("SKILL_INJECT_VERIFICATION_MODE", "lexical")

    def deny_network(*args, **kwargs):
        raise AssertionError("Tests must use httpx.MockTransport, not real HTTP")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", deny_network)
    async def deny_async_network(*args, **kwargs):
        raise AssertionError("Tests must use httpx.MockTransport, not real async HTTP")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", deny_async_network)


@pytest.fixture(autouse=True)
def explicit_test_backend(monkeypatch):
    monkeypatch.setenv("SKILL_INJECT_DENSE_BACKEND", "numpy")
    monkeypatch.delenv("SKILL_INJECT_OPENROUTER_API_KEY_FILE", raising=False)
