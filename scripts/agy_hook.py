#!/usr/bin/env python3
"""Antigravity (AGY) PreInvocation hook for skill-injection-mcp.

Extracts the user's latest prompt, performs hybrid retrieval against the
indexed skills catalog, and injects discovered skill suggestions via
ephemeralMessage before the model turn begins.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
venv_dir = REPO_ROOT / ".venv"
venv_python = venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
in_venv = Path(sys.prefix).resolve() == venv_dir.resolve()
if venv_python.is_file() and not in_venv and os.environ.get("SKILL_INJECT_NO_REEXEC") != "1":
    os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve())] + sys.argv[1:])

src_dir = REPO_ROOT / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

ACKNOWLEDGEMENTS = frozenset({
    "", "네", "응", "좋아", "고마워", "감사합니다", "ok", "okay", "thanks", "yes", "no",
    "ㄱㄱ", "ㅇㅇ", "ㅇㅋ", "ㄳ", "ㅎㅎ", "ㅋㅋㅋㅋ", "proceed", "continue", "next", "done",
})
_USER_REQUEST_RE = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.DOTALL)
_SAFE_SKILL_ID_RE = re.compile(r"[^A-Za-z0-9_.:/-]+")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_MAX_TRANSCRIPT_TAIL_BYTES = 65536


def _strip_user_request_tags(text: str) -> str:
    match = _USER_REQUEST_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _read_transcript_tail_lines(path: Path) -> list[str]:
    if path.suffix != ".jsonl" or path.is_symlink() or not path.is_file():
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        offset = max(0, size - _MAX_TRANSCRIPT_TAIL_BYTES)
        f.seek(offset, os.SEEK_SET)
        raw = f.read()
    text = raw.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    return lines[1:] if offset > 0 and len(lines) > 1 else lines


def _extract_prompt(payload: dict) -> str:
    for key in ("lastUserInput", "prompt", "userPrompt", "lastUserMessage"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            stripped = _strip_user_request_tags(val)
            if stripped:
                return stripped

    transcript_path = payload.get("transcriptPath")
    if isinstance(transcript_path, str) and transcript_path.strip():
        try:
            lines = _read_transcript_tail_lines(Path(transcript_path))
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if (
                        isinstance(record, dict)
                        and record.get("type") == "USER_INPUT"
                        and record.get("source") in (None, "USER_EXPLICIT")
                    ):
                        content = record.get("content", "")
                        if isinstance(content, str) and content.strip():
                            stripped = _strip_user_request_tags(content)
                            if stripped:
                                return stripped
                except Exception:
                    continue
        except Exception:
            pass

    return ""


def _sanitize_inline(text: str) -> str:
    return _CONTROL_RE.sub(" ", text).replace("`", "'").strip()


def handle_pre_invocation(payload: dict, *, settings=None, engine=None) -> dict:
    prompt = _extract_prompt(payload)
    if not prompt:
        return {}

    cleaned = prompt.strip().casefold().rstrip(".!? ")
    if cleaned in ACKNOWLEDGEMENTS or len(cleaned) < 3:
        return {}

    owned_engine = engine is None
    try:
        from skill_inject_mcp.codex_adapter import prompt_context
        from skill_inject_mcp.config import load_hook_settings
        from skill_inject_mcp.engine import SkillInjectEngine

        if owned_engine:
            engine = SkillInjectEngine(settings or load_hook_settings(REPO_ROOT))
        engine.ensure_index()
        hook_resp = prompt_context(engine, prompt)

        context_text = hook_resp.get("hookSpecificOutput", {}).get("additionalContext", "")
        if not context_text:
            return {}

        lines = context_text.split("\n")
        candidates = []
        for line in lines[1:]:
            line_str = line.strip()
            if line_str.startswith("[") and line_str.endswith("]"):
                try:
                    parsed = json.loads(line_str)
                    if isinstance(parsed, list):
                        candidates = parsed
                        break
                except Exception:
                    continue

        if not candidates:
            return {}

        out_lines = ["[Skill Injection MCP: Discovered Skills]"]
        for c in candidates[:3]:
            if not isinstance(c, dict):
                continue
            sid = _SAFE_SKILL_ID_RE.sub("", str(c.get("skill_id", "")))
            desc = _sanitize_inline(str(c.get("description", "")))
            if not sid:
                continue
            if len(desc) > 160:
                desc = desc[:157] + "..."
            out_lines.append(f"  • `{sid}`: {desc}")
        if len(out_lines) == 1:
            return {}
        out_lines.append(
            "If relevant, use `get_skill_body(skill_id)` to load instructions, "
            "or `resolve_skills` for full verification."
        )
        msg = "\n".join(out_lines)

        return {
            "injectSteps": [
                {
                    "ephemeralMessage": msg,
                }
            ]
        }
    except Exception:
        return {}
    finally:
        if owned_engine and engine is not None:
            try:
                engine.close()
            except Exception:
                pass


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    try:
        res = handle_pre_invocation(payload)
    except Exception:
        res = {}

    json.dump(res, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
