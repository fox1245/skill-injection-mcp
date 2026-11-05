"""Cheap source change detection. Explicit reindex still rereads every source."""
from __future__ import annotations

import json
import stat
from pathlib import Path


def _stat(path: Path) -> tuple:
    try:
        info = path.stat()
    except OSError:
        return (str(path), None)
    if not stat.S_ISREG(info.st_mode):
        return (str(path), None)
    return (str(path), info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def source_signature(selection: tuple[str, Path], previous: tuple | None = None) -> tuple:
    kind, path = selection
    if kind == "manifest":
        manifest = _stat(path)
        if previous is not None and previous[0] == selection and previous[1] == manifest:
            paths = [Path(entry[0]) for entry in previous[2]]
        else:
            entries = json.loads(path.read_text(encoding="utf-8-sig"))["skills"]
            paths = sorted({Path(e["path"]).absolute() for e in entries if e.get("enabled", True)})
        return (selection, manifest, tuple(_stat(source) for source in paths))
    if not path.is_dir():
        raise ValueError(f"Skills directory does not exist: {path}")
    return (selection, None, tuple(_stat(source) for source in sorted(path.rglob("SKILL.md"))))


def mark_unreadable(signature: tuple, paths: set[str]) -> tuple:
    """Record sources that the registry load could not read as missing.

    The signature is taken before the registry is loaded; a file that vanished or became unreadable in between
    would otherwise stay "present" in it, and the skill would not return when the file is readable again.
    """
    selection, manifest, sources = signature
    return (selection, manifest, tuple(
        (entry[0], None) if entry[0] in paths else entry for entry in sources
    ))
