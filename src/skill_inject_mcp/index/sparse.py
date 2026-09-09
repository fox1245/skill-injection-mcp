from __future__ import annotations

import sqlite3
from pathlib import Path

from skill_inject_mcp.text import words


def _fts_query(raw: str) -> str:
    """Build a safe FTS5 query from free text (OR of tokens)."""
    toks = words(raw)
    cleaned: list[str] = []
    for t in toks:
        # quote tokens to avoid FTS syntax issues
        cleaned.append(f'"{t}"')
    if not cleaned:
        return '""'
    return " OR ".join(cleaned)


class SparseIndex:
    """SQLite FTS5 BM25 sparse index over skill text."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                skill_id UNINDEXED,
                name,
                description,
                body,
                tags,
                tokenize = 'porter unicode61'
            )
            """
        )
        self._conn.commit()

    def clear(self) -> None:
        self._conn.execute("DELETE FROM skills_fts")
        self._conn.commit()

    def upsert(
        self,
        skill_id: str,
        name: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
    ) -> None:
        self._conn.execute("DELETE FROM skills_fts WHERE skill_id = ?", (skill_id,))
        self._conn.execute(
            "INSERT INTO skills_fts(skill_id, name, description, body, tags) VALUES (?,?,?,?,?)",
            (skill_id, name, description, body, " ".join(tags or [])),
        )
        self._conn.commit()

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float, int]]:
        """Return (skill_id, bm25_score, rank) where rank is 1-based.

        FTS5 bm25() returns lower (more negative) for better matches; we negate
        so higher is better for callers.
        """
        q = _fts_query(query)
        try:
            cur = self._conn.execute(
                """
                SELECT skill_id, bm25(skills_fts) AS score
                FROM skills_fts
                WHERE skills_fts MATCH ?
                ORDER BY score, skill_id
                LIMIT ?
                """,
                (q, top_k),
            )
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            return []
        out: list[tuple[str, float, int]] = []
        for i, row in enumerate(rows, start=1):
            # negate bm25 so higher is better
            out.append((row["skill_id"], float(-row["score"]), i))
        return out

    def upsert_many(self, skills) -> None:
        with self._conn:
            for skill in skills:
                self._conn.execute("DELETE FROM skills_fts WHERE skill_id = ?", (skill.skill_id,))
                self._conn.execute(
                    "INSERT INTO skills_fts(skill_id, name, description, body, tags) VALUES (?,?,?,?,?)",
                    (skill.skill_id, skill.name, skill.description, skill.body, " ".join(skill.tags)),
                )

    def close(self) -> None:
        self._conn.close()

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS c FROM skills_fts").fetchone()
        return int(row["c"]) if row else 0
