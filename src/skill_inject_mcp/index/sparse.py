from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

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
        # Engine writes/main reads are serialized; hooks use separate read-only connections.
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
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
        return self._search(self._conn, query, top_k)

    def search_readonly(self, query: str, top_k: int = 20) -> list[tuple[str, float, int]]:
        uri = "file:" + quote(self.db_path.resolve().as_posix(), safe="/:") + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            return self._search(connection, query, top_k)

    @staticmethod
    def _search(connection, query: str, top_k: int) -> list[tuple[str, float, int]]:
        """Return (skill_id, bm25_score, rank) where rank is 1-based.

        FTS5 bm25() returns lower (more negative) for better matches; we negate
        so higher is better for callers.
        """
        q = _fts_query(query)
        try:
            cur = connection.execute(
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
