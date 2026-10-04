"""Tiny database layer: SQLite (default, laptop) or PostgreSQL (server mode, optional ``psycopg``).

SQL is written with ``?`` placeholders; they are translated for PostgreSQL. Only the portable subset used by the
HYDRA stores (CREATE TABLE IF NOT EXISTS, INSERT, SELECT, UPDATE, DELETE) is relied upon."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.lock = threading.RLock()
        if url.startswith("sqlite:///"):
            self.kind = "sqlite"
            path = Path(url[len("sqlite:///"):])
            if str(path) != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)
            self.path = path
            self.con = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
            self.con.row_factory = sqlite3.Row
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.execute("PRAGMA foreign_keys=ON")
        elif url.startswith(("postgresql://", "postgres://")):
            self.kind = "postgres"
            import psycopg  # optional server-mode dependency

            self.con = psycopg.connect(url, autocommit=True, row_factory=psycopg.rows.dict_row)
            self.path = Path("")
        else:
            raise ValueError(f"unsupported database url {url!r}")

    def _sql(self, sql: str) -> str:
        return sql if self.kind == "sqlite" else sql.replace("?", "%s").replace("INSERT OR REPLACE", "INSERT")

    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self.lock:
            cur = self.con.execute(self._sql(sql), tuple(params))
            return int(cur.rowcount if cur.rowcount is not None else 0)

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self.lock:
            cur = self.con.execute(self._sql(sql), tuple(params))
            return [dict(r) for r in cur.fetchall()]

    def one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.lock:
            if self.kind == "sqlite":
                self.con.execute("BEGIN")
                try:
                    yield
                    self.con.execute("COMMIT")
                except Exception:
                    self.con.execute("ROLLBACK")
                    raise
            else:  # pragma: no cover - needs a server
                yield

    def backup(self, dest: Path) -> Path:
        """Consistent online backup (SQLite backup API). For PostgreSQL use ``pg_dump`` (documented in SECURITY.md)."""
        if self.kind != "sqlite":
            raise NotImplementedError("use pg_dump for PostgreSQL backups")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            out = sqlite3.connect(str(dest))
            self.con.backup(out)
            out.close()
        return dest

    def close(self) -> None:
        self.con.close()
