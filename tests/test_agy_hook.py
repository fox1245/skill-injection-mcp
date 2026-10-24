from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from skill_inject_mcp.config import load_hook_settings

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_agy_hook_module():
    script_path = REPO_ROOT / "scripts" / "agy_hook.py"
    spec = importlib.util.spec_from_file_location("agy_hook_script", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_hook_settings_resolves_relative_paths_against_repo_root_from_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_repo = tmp_path / "repo"
    (fake_repo / "custom_skills").mkdir(parents=True)
    (fake_repo / ".env").write_text(
        "SKILL_INJECT_SKILLS_DIR=custom_skills\n"
        "SKILL_INJECT_INDEX_DIR=.custom_idx\n"
        "SKILL_INJECT_DENSE_BACKEND=numpy\n",
        encoding="utf-8",
    )
    unrelated_cwd = tmp_path / "unrelated_workspace"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.delenv("SKILL_INJECT_SKILLS_DIR", raising=False)
    monkeypatch.delenv("SKILL_INJECT_INDEX_DIR", raising=False)
    monkeypatch.delenv("SKILL_INJECT_PERSISTENT_EMBEDDING_CACHE", raising=False)
    monkeypatch.delenv("SKILL_INJECT_HOOK_TIMEOUT_S", raising=False)

    settings = load_hook_settings(fake_repo)
    assert settings.skills_dir == (fake_repo / "custom_skills").resolve()
    assert settings.index_dir == (fake_repo / ".custom_idx").resolve()
    assert settings.persistent_embedding_cache is True
    assert settings.hook_timeout_s == 4.0
    assert settings.dense_backend == "numpy"


def test_extract_prompt_supports_last_user_input_and_strips_user_request_tags(tmp_path: Path):
    mod = _load_agy_hook_module()

    assert mod._extract_prompt({
        "lastUserInput": "<USER_REQUEST>\n파이썬 패키지를 설치하고 의존성을 관리해줘\n</USER_REQUEST>"
    }) == "파이썬 패키지를 설치하고 의존성을 관리해줘"

    assert mod._extract_prompt({"prompt": "  Install Python packages with pip  "}) == "Install Python packages with pip"

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"step_index": 0, "type": "USER_INPUT", "content": "<USER_REQUEST>Build BM25 index</USER_REQUEST>"}) + "\n",
        encoding="utf-8",
    )
    assert mod._extract_prompt({"transcriptPath": str(transcript)}) == "Build BM25 index"


def test_handle_pre_invocation_discovers_skills_from_unrelated_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_agy_hook_module()

    fake_repo = tmp_path / "repo"
    ko_skill = fake_repo / "skills" / "ko-pkg-manager" / "SKILL.md"
    ko_skill.parent.mkdir(parents=True)
    ko_skill.write_text(
        "---\nname: ko-pkg-manager\ndescription: 파이썬 패키지 설치와 의존성 관리\n---\n"
        "파이썬 패키지 설치와 의존성 관리 절차\n",
        encoding="utf-8",
    )
    (fake_repo / ".env").write_text(
        "SKILL_INJECT_SKILLS_DIR=skills\n"
        "SKILL_INJECT_INDEX_DIR=.idx\n"
        "SKILL_INJECT_USE_FAKE_EMBEDDER=true\n"
        "SKILL_INJECT_MULTI_QUERY=false\n",
        encoding="utf-8",
    )

    unrelated_cwd = tmp_path / "other_workspace"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)

    settings = load_hook_settings(fake_repo)
    ack_out = mod.handle_pre_invocation({"lastUserInput": "네"}, settings=settings)
    assert ack_out == {}

    out = mod.handle_pre_invocation(
        {"lastUserInput": "<USER_REQUEST>파이썬 패키지를 설치하고 의존성을 관리해줘</USER_REQUEST>"},
        settings=settings,
    )
    assert "injectSteps" in out
    msg = out["injectSteps"][0]["ephemeralMessage"]
    assert "ko-pkg-manager" in msg


def test_handle_pre_invocation_preserves_candidates_whose_description_contains_empty_brackets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_agy_hook_module()
    fake_repo = tmp_path / "repo"
    skill_md = fake_repo / "skills" / "array-parser" / "SKILL.md"
    skill_md.parent.mkdir(parents=True)
    skill_md.write_text(
        "---\nname: array-parser\ndescription: Parse JSON [] arrays in Python\n---\n"
        "Parse JSON [] arrays in Python.\n",
        encoding="utf-8",
    )
    (fake_repo / ".env").write_text(
        "SKILL_INJECT_SKILLS_DIR=skills\n"
        "SKILL_INJECT_INDEX_DIR=.idx\n"
        "SKILL_INJECT_USE_FAKE_EMBEDDER=true\n"
        "SKILL_INJECT_MULTI_QUERY=false\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    settings = load_hook_settings(fake_repo)
    out = mod.handle_pre_invocation({"lastUserInput": "Parse JSON [] arrays in Python"}, settings=settings)
    assert "injectSteps" in out
    assert "array-parser" in out["injectSteps"][0]["ephemeralMessage"]
