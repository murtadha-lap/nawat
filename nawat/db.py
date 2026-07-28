"""Durable state for the cache.

SQLite, one file under the state directory. Writes are synchronous because the
thing being tracked is which bytes are safe to delete; a lost write here is the
failure mode the whole product exists to prevent.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    key           TEXT PRIMARY KEY,
    bytes         INTEGER NOT NULL,
    files         INTEGER NOT NULL,
    fetched_at    REAL    NOT NULL,
    last_used     REAL    NOT NULL,
    pinned        INTEGER NOT NULL DEFAULT 0,
    replicated_at REAL
);
CREATE INDEX IF NOT EXISTS artifacts_lru ON artifacts (pinned, last_used);

CREATE TABLE IF NOT EXISTS leases (
    id          TEXT PRIMARY KEY,
    key         TEXT NOT NULL,
    pid         INTEGER NOT NULL,
    boot_id     TEXT NOT NULL,
    proc_start  REAL NOT NULL,
    holder      TEXT NOT NULL DEFAULT '',
    acquired_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS leases_key ON leases (key);
CREATE INDEX IF NOT EXISTS leases_pid ON leases (pid, boot_id);
"""


class Database:
    """Thread-local SQLite connections over one file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def connect(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        else:
            conn.execute("COMMIT")

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
