from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def test_real_stdio_hook_responds_while_resolve_is_running(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "slow_server.py"
    marker = tmp_path / "entered"
    script.write_text(textwrap.dedent('''
        import sys, time
        from pathlib import Path
        from skill_inject_mcp import server
        from skill_inject_mcp.config import Settings
        from skill_inject_mcp.engine import SkillInjectEngine
        server._engine = SkillInjectEngine(Settings(
            _env_file=None, skills_dir=Path(sys.argv[1]), skill_manifest=None,
            index_dir=Path(sys.argv[2]), use_fake_embedder=True,
            multi_query=False, verification_mode="lexical", OPENROUTER_API_KEY="",
        ))
        server._engine.ensure_index()
        original = server._engine.resolve
        def slow(request):
            with server._engine._lock:
                Path(sys.argv[3]).write_text("entered")
                time.sleep(.8)
                return original(request)
        server._engine.resolve = slow
        server.main()
    '''), encoding="utf-8")
    async def run():
        env = dict(os.environ, PYTHONUTF8="1", OPENROUTER_API_KEY="")
        params = StdioServerParameters(command=sys.executable,
            args=["-B", "-u", str(script), str(root / "fixtures" / "skills"), str(tmp_path / "index"), str(marker)],
            env=env, cwd=str(root))
        with (tmp_path / "stderr.log").open("w", encoding="utf-8") as log:
            async with stdio_client(params, errlog=log) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    slow = asyncio.create_task(session.call_tool("resolve_skills", {"request": {
                        "schema_version": "1.0", "requirements": [{"id": "r", "description": "Install Python packages with pip"}],
                    }}))
                    deadline = time.monotonic() + 3
                    while not marker.exists() and time.monotonic() < deadline:
                        await asyncio.sleep(.01)
                    assert marker.exists()
                    started = time.monotonic()
                    hook = await asyncio.wait_for(session.call_tool("codex_prompt_hook", {
                        "prompt": "Install Python packages with pip",
                    }), timeout=.5)
                    assert time.monotonic() - started < .5
                    assert not hook.isError and "package-installer" in str(hook.content)
                    assert not (await slow).isError
    asyncio.run(run())
