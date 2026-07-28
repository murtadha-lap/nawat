"""Leases, keyed to live processes rather than to timeouts.

A timeout-based lock is wrong in both directions: too short and it expires
during a six-hour run, permitting eviction of weights the trainer is reading;
too long and a crashed process deadlocks the cache until an operator clears it.

Liveness is decided by (boot id, pid, process start time). The boot id kills
every lease that predates a reboot. The start time defeats pid reuse — a new
process that happens to land on a dead trainer's pid does not inherit its lease.
Both checks are free, and neither needs tuning (PRD 8, FR-1.8, NFR-1.4).
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .db import Database
from .keys import Key
from .process import UNKNOWN_START as _UNKNOWN_START
from .process import process_alive as _pid_exists
from .process import process_start_time

_BOOT_ID_PATHS = ("/proc/sys/kernel/random/boot_id",)


def boot_id() -> str:
    """An identifier that changes on every reboot."""
    for path in _BOOT_ID_PATHS:
        try:
            return Path(path).read_text().strip()
        except OSError:
            continue
    # No /proc: fall back to boot time, which is stable within a boot.
    try:
        import psutil  # type: ignore

        return f"boottime-{int(psutil.boot_time())}"
    except Exception:
        return "unknown-boot"


@dataclass(frozen=True)
class LeaseRecord:
    id: str
    key: Key
    pid: int
    boot_id: str
    proc_start: float
    holder: str
    acquired_at: float

    def alive(self, current_boot: str | None = None) -> bool:
        """True while the process that took this lease is still running."""
        if self.boot_id != (current_boot if current_boot is not None else boot_id()):
            return False  # taken before the last reboot
        if not _pid_exists(self.pid):
            return False
        start = process_start_time(self.pid)
        if start is None:
            return False
        if start == _UNKNOWN_START or self.proc_start == _UNKNOWN_START:
            return True  # cannot compare start times; pid existence is all we have
        return start == self.proc_start

    def describe(self) -> str:
        who = self.holder or "process"
        return f"{who} (pid {self.pid})"


class LeaseRegistry:
    """Records which artifacts are in use, and by whom."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.boot = boot_id()

    def acquire(self, keys: Sequence[Key], holder: str = "", pid: int | None = None) -> list[LeaseRecord]:
        pid = os.getpid() if pid is None else pid
        start = process_start_time(pid)
        if start is None:
            start = _UNKNOWN_START
        now = time.time()
        records = [
            LeaseRecord(
                id=uuid.uuid4().hex,
                key=key,
                pid=pid,
                boot_id=self.boot,
                proc_start=start,
                holder=holder,
                acquired_at=now,
            )
            for key in keys
        ]
        with self.db.tx() as conn:
            conn.executemany(
                "INSERT INTO leases (id, key, pid, boot_id, proc_start, holder, acquired_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                [(r.id, str(r.key), r.pid, r.boot_id, r.proc_start, r.holder, r.acquired_at) for r in records],
            )
        return records

    def release(self, lease_ids: Iterable[str]) -> int:
        ids = [(lease_id,) for lease_id in lease_ids]
        if not ids:
            return 0
        with self.db.tx() as conn:
            cursor = conn.executemany("DELETE FROM leases WHERE id = ?", ids)
            return max(cursor.rowcount or 0, 0)

    def release_process(self, pid: int | None = None) -> int:
        pid = os.getpid() if pid is None else pid
        with self.db.tx() as conn:
            cursor = conn.execute("DELETE FROM leases WHERE pid = ? AND boot_id = ?", (pid, self.boot))
            return cursor.rowcount or 0

    def all(self) -> list[LeaseRecord]:
        rows = self.db.connect().execute("SELECT * FROM leases ORDER BY acquired_at").fetchall()
        return [self._row(row) for row in rows]

    def live(self) -> list[LeaseRecord]:
        """Every lease still held by a running process. Dead ones are reaped in passing."""
        self.reap()
        return [record for record in self.all() if record.alive(self.boot)]

    def live_for(self, key: Key) -> list[LeaseRecord]:
        rows = self.db.connect().execute("SELECT * FROM leases WHERE key = ?", (str(key),)).fetchall()
        alive, dead = [], []
        for row in rows:
            record = self._row(row)
            (alive if record.alive(self.boot) else dead).append(record)
        if dead:
            self.release(record.id for record in dead)
        return alive

    def reap(self) -> int:
        """Clear leases whose holders are gone. Runs without operator action."""
        dead = [record.id for record in self.all() if not record.alive(self.boot)]
        return self.release(dead)

    def _row(self, row) -> LeaseRecord:
        return LeaseRecord(
            id=row["id"],
            key=Key.parse(row["key"]),
            pid=row["pid"],
            boot_id=row["boot_id"],
            proc_start=row["proc_start"],
            holder=row["holder"],
            acquired_at=row["acquired_at"],
        )
