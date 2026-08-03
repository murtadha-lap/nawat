"""Run records — what was submitted, what happened, and what came out.

A run record is the answer to "what produced this adapter, and how did it
score?" weeks later (PRD §4, G4), so it is written durably as the run
progresses rather than assembled at the end (NFR-1.5), and it survives failure
(FR-2.11). Two copies exist: rows in SQLite for querying, and a ``run.json``
beside the log for recovering the history if the database is lost.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import checkpoints as checkpoint_store
from .checkpoints import CheckpointPolicy
from .db import Database
from .errors import InvalidKey, NotFound
from .keys import Key

RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    state        TEXT NOT NULL,
    spec         TEXT NOT NULL,
    submitted_at REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL,
    exit_code    INTEGER,
    pid          INTEGER,
    error        TEXT,
    artifacts    TEXT NOT NULL DEFAULT '[]',
    description  TEXT,
    checkpoint   TEXT,
    name         TEXT
);
CREATE INDEX IF NOT EXISTS runs_state ON runs (state, submitted_at);
"""

#: Columns added after the first release, applied to databases that predate them.
MIGRATIONS = (
    ("checkpoint", "ALTER TABLE runs ADD COLUMN checkpoint TEXT"),
    ("name", "ALTER TABLE runs ADD COLUMN name TEXT"),
)


class RunState(str, Enum):
    QUEUED = "queued"
    STAGING = "staging"
    RUNNING = "running"
    PUBLISHING = "publishing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (RunState.SUCCEEDED, RunState.FAILED, RunState.CANCELLED)

    @property
    def active(self) -> bool:
        return self in (RunState.STAGING, RunState.RUNNING, RunState.PUBLISHING)


def new_run_id() -> str:
    return f"{time.strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:4]}"


def validate_run_id(run_id: str) -> str:
    if not RUN_ID.match(run_id):
        raise InvalidKey(
            f"{run_id!r} is not a usable run id.",
            "Use letters, digits, and . _ - only, starting with a letter or digit.",
        )
    return run_id


def validate_run_name(name: str) -> str:
    """A run's name is the folder its artifacts occupy in object storage.

    So it has to be a usable key rather than free text — and it may contain a
    ``/``, which is how related runs end up filed together under one prefix:
    ``arabic-ocr/2-epochs`` publishes as ``runs/arabic-ocr/2-epochs/adapter``.
    """
    cleaned = name.strip().strip("/")
    if not cleaned:
        raise InvalidKey("A run name cannot be empty.", "Give it a short name, or leave it unset to use the run id.")
    Key.parse(f"runs/{cleaned}")  # raises with the segment rule if it is unusable
    return cleaned


@dataclass(frozen=True)
class RunSpec:
    """What was submitted. Enough to reproduce the run on its own (G4)."""

    script: str
    model: Key | None = None
    datasets: tuple[Key, ...] = ()
    inputs: tuple[Key, ...] = ()
    params: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""
    checkpoints: CheckpointPolicy = field(default_factory=CheckpointPolicy)

    def staged_keys(self) -> tuple[Key, ...]:
        """Every input to stage, in a stable order, without duplicates."""
        ordered: list[Key] = []
        for key in (*( (self.model,) if self.model else () ), *self.datasets, *self.inputs):
            if key not in ordered:
                ordered.append(key)
        return tuple(ordered)

    @property
    def is_notebook(self) -> bool:
        return self.script.endswith(".ipynb")

    @property
    def checkpoint_lineage(self) -> str:
        """Which checkpoint directory this run trains into.

        Derived from the script and its inputs unless it was named explicitly,
        so that two submissions of the same command share one lineage — that is
        what makes a resubmission resume rather than restart — while a different
        experiment gets a directory of its own.
        """
        if self.checkpoints.lineage:
            return checkpoint_store.validate(self.checkpoints.lineage)
        return checkpoint_store.lineage_name(self.script, (str(k) for k in self.staged_keys()))

    def to_json(self) -> dict[str, Any]:
        return {
            "script": self.script,
            "model": str(self.model) if self.model else None,
            "datasets": [str(k) for k in self.datasets],
            "inputs": [str(k) for k in self.inputs],
            "params": dict(self.params),
            "notes": self.notes,
            "checkpoints": self.checkpoints.to_json(),
            "checkpoint_lineage": self.checkpoint_lineage,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RunSpec":
        return cls(
            script=data["script"],
            model=Key.parse(data["model"]) if data.get("model") else None,
            datasets=tuple(Key.parse(k) for k in data.get("datasets") or ()),
            inputs=tuple(Key.parse(k) for k in data.get("inputs") or ()),
            params=dict(data.get("params") or {}),
            notes=data.get("notes", ""),
            checkpoints=CheckpointPolicy.from_json(data.get("checkpoints")),
        )


@dataclass(frozen=True)
class RunRecord:
    id: str
    spec: RunSpec
    state: RunState
    submitted_at: float
    started_at: float | None = None
    finished_at: float | None = None
    exit_code: int | None = None
    pid: int | None = None
    error: str | None = None
    artifacts: tuple[Key, ...] = ()
    description: str | None = None
    #: The newest checkpoint left on disk when the run ended, if any. Set on
    #: failure as well as success — especially on failure, since that is when
    #: it is the only thing standing between a crash and days of lost GPU time.
    checkpoint: str | None = None
    #: What this run is called. Chosen by the researcher rather than generated,
    #: and used as the run's folder in object storage, so a bucket reads as a
    #: list of experiments instead of a list of timestamps.
    name: str | None = None

    @property
    def publish_prefix(self) -> str:
        """The key prefix this run's artifacts are published under.

        The name when there is one, the id otherwise — so a run always has a
        home in object storage, named or not.
        """
        return self.name or self.id

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        return (self.finished_at or time.time()) - self.started_at

    @property
    def run_key(self) -> Key:
        """Where this run's own artifacts live."""
        return Key.parse(f"runs/{self.publish_prefix}")

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "state": self.state.value,
            "spec": self.spec.to_json(),
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
            "exit_code": self.exit_code,
            "pid": self.pid,
            "error": self.error,
            "artifacts": [str(k) for k in self.artifacts],
            "description": self.description,
            "checkpoint": self.checkpoint,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "RunRecord":
        return cls(
            id=data["id"],
            spec=RunSpec.from_json(data["spec"]),
            state=RunState(data["state"]),
            submitted_at=data["submitted_at"],
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            exit_code=data.get("exit_code"),
            pid=data.get("pid"),
            error=data.get("error"),
            artifacts=tuple(Key.parse(k) for k in data.get("artifacts") or ()),
            description=data.get("description"),
            checkpoint=data.get("checkpoint"),
            name=data.get("name"),
        )


class RunStore:
    """Durable run history."""

    def __init__(self, db: Database, root: Path) -> None:
        self.db = db
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        conn = self.db.connect()
        conn.executescript(SCHEMA)
        self._migrate(conn)

    def _migrate(self, conn) -> None:
        """Add columns a database written by an older version does not have."""
        present = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        for column, statement in MIGRATIONS:
            if column not in present:
                conn.execute(statement)

    # -- layout ------------------------------------------------------------

    def directory(self, run_id: str) -> Path:
        path = self.root / validate_run_id(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def log_path(self, run_id: str) -> Path:
        return self.directory(run_id) / "run.log"

    def metrics_path(self, run_id: str) -> Path:
        """The durable metric series — state directory, never the cache, so it
        outlives every artifact the run produced (FR-7.6)."""
        from .metrics import FILENAME

        return self.directory(run_id) / FILENAME

    def record_path(self, run_id: str) -> Path:
        return self.directory(run_id) / "run.json"

    # -- writing -----------------------------------------------------------

    def create(self, spec: RunSpec, run_id: str | None = None, name: str | None = None) -> RunRecord:
        run_id = validate_run_id(run_id) if run_id else new_run_id()
        if self.find(run_id) is not None:
            raise InvalidKey(f"Run {run_id} already exists.", "Choose another id, or omit it to have one generated.")
        record = RunRecord(
            id=run_id,
            spec=spec,
            state=RunState.QUEUED,
            submitted_at=time.time(),
            name=validate_run_name(name) if name else None,
        )
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO runs (id, state, spec, submitted_at, artifacts, name) VALUES (?, ?, ?, ?, '[]', ?)",
                (record.id, record.state.value, json.dumps(spec.to_json()), record.submitted_at, record.name),
            )
        self._write_record_file(record)
        return record

    def update(self, run_id: str, **changes: Any) -> RunRecord:
        """Apply a state transition and flush it to disk before returning."""
        record = self.get(run_id)
        if "artifacts" in changes:
            changes["artifacts"] = tuple(Key.parse(k) for k in changes["artifacts"])
        if "state" in changes:
            changes["state"] = RunState(changes["state"])
        if changes.get("name"):
            changes["name"] = validate_run_name(changes["name"])
        updated = replace(record, **changes)
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE runs SET state = ?, started_at = ?, finished_at = ?, exit_code = ?,"
                " pid = ?, error = ?, artifacts = ?, description = ?, checkpoint = ?, name = ?"
                " WHERE id = ?",
                (
                    updated.state.value,
                    updated.started_at,
                    updated.finished_at,
                    updated.exit_code,
                    updated.pid,
                    updated.error,
                    json.dumps([str(k) for k in updated.artifacts]),
                    updated.description,
                    updated.checkpoint,
                    updated.name,
                    updated.id,
                ),
            )
        self._write_record_file(updated)
        return updated

    def _write_record_file(self, record: RunRecord) -> None:
        path = self.record_path(record.id)
        temporary = path.with_suffix(".json.part")
        temporary.write_text(json.dumps(record.to_json(), indent=2) + "\n")
        temporary.replace(path)

    # -- reading -----------------------------------------------------------

    def find(self, run_id: str) -> RunRecord | None:
        row = self.db.connect().execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return self._row(row) if row else None

    def get(self, run_id: str) -> RunRecord:
        record = self.find(run_id)
        if record is None:
            raise NotFound(f"There is no run {run_id}.", "List them with: nawat runs")
        return record

    def list(self, *, state: RunState | None = None, limit: int = 100) -> list[RunRecord]:
        sql = "SELECT * FROM runs"
        args: list[Any] = []
        if state is not None:
            sql += " WHERE state = ?"
            args.append(state.value)
        sql += " ORDER BY submitted_at DESC LIMIT ?"
        args.append(limit)
        return [self._row(row) for row in self.db.connect().execute(sql, args).fetchall()]

    def active(self) -> list[RunRecord]:
        states = [s.value for s in RunState if s.active]
        placeholders = ",".join("?" for _ in states)
        rows = self.db.connect().execute(
            f"SELECT * FROM runs WHERE state IN ({placeholders}) ORDER BY submitted_at", states
        ).fetchall()
        return [self._row(row) for row in rows]

    def next_queued(self) -> RunRecord | None:
        """The head of the queue. Strictly FIFO — one GPU, so one run at a time."""
        row = self.db.connect().execute(
            "SELECT * FROM runs WHERE state = ? ORDER BY submitted_at LIMIT 1", (RunState.QUEUED.value,)
        ).fetchone()
        return self._row(row) if row else None

    def _row(self, row) -> RunRecord:
        return RunRecord(
            id=row["id"],
            spec=RunSpec.from_json(json.loads(row["spec"])),
            state=RunState(row["state"]),
            submitted_at=row["submitted_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            exit_code=row["exit_code"],
            pid=row["pid"],
            error=row["error"],
            artifacts=tuple(Key.parse(k) for k in json.loads(row["artifacts"])),
            description=row["description"],
            checkpoint=row["checkpoint"],
            name=row["name"],
        )

    # -- logs --------------------------------------------------------------

    def read_log(self, run_id: str, *, tail: int | None = None) -> str:
        path = self.log_path(run_id)
        if not path.exists():
            return ""
        text = path.read_text(errors="replace")
        if tail is None:
            return text
        return "".join(text.splitlines(keepends=True)[-tail:])

    def follow_log(self, run_id: str, *, poll: float = 0.25) -> Iterator[str]:
        """Yield log lines as they are written, ending when the run does.

        Tails the file rather than subscribing in memory, so a reader that
        attaches late still sees the whole run, and a reader in another process
        works identically (FR-2.6).
        """
        path = self.log_path(run_id)
        position = 0
        while True:
            if path.exists():
                with path.open("r", errors="replace") as handle:
                    handle.seek(position)
                    chunk = handle.read()
                    position = handle.tell()
                if chunk:
                    yield chunk
                    continue
            record = self.find(run_id)
            if record is None or record.state.terminal:
                if path.exists():  # one last read, for anything written as it exited
                    with path.open("r", errors="replace") as handle:
                        handle.seek(position)
                        remainder = handle.read()
                    if remainder:
                        yield remainder
                return
            time.sleep(poll)

    def reconcile(self, is_alive, checkpoint_root: Path | None = None) -> list[RunRecord]:
        """After a crash or reboot, no run is still running. Mark them failed.

        Takes a liveness predicate so the caller decides what "still running"
        means; a run whose process is genuinely alive is left alone.

        Given the checkpoint root, each recovered run also picks up whatever
        checkpoint it had reached. A run the platform lost track of is precisely
        the one nobody watched fail, so it is the one most likely to be
        discovered days later — with the record as the only clue that there is
        something to resume from.
        """
        recovered = []
        for record in self.active():
            if record.pid is not None and is_alive(record.pid):
                continue
            newest = None
            if checkpoint_root is not None:
                newest = checkpoint_store.latest(
                    checkpoint_store.lineage_path(checkpoint_root, record.spec.checkpoint_lineage)
                )
            recovered.append(
                self.update(
                    record.id,
                    state=RunState.FAILED,
                    finished_at=record.finished_at or time.time(),
                    error="The run did not survive a restart of the platform.",
                    checkpoint=str(newest.path) if newest else record.checkpoint,
                )
            )
        return recovered


def rebuild_from_disk(store: RunStore) -> int:
    """Re-import run.json files the database has forgotten."""
    restored = 0
    for path in sorted(store.root.glob("*/run.json")):
        try:
            record = RunRecord.from_json(json.loads(path.read_text()))
        except (OSError, ValueError, KeyError):
            continue
        if store.find(record.id) is not None:
            continue
        with store.db.tx() as conn:
            conn.execute(
                "INSERT INTO runs (id, state, spec, submitted_at, started_at, finished_at,"
                " exit_code, pid, error, artifacts, description, checkpoint, name)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.state.value,
                    json.dumps(record.spec.to_json()),
                    record.submitted_at,
                    record.started_at,
                    record.finished_at,
                    record.exit_code,
                    record.pid,
                    record.error,
                    json.dumps([str(k) for k in record.artifacts]),
                    record.description,
                    record.checkpoint,
                    record.name,
                ),
            )
        restored += 1
    return restored


__all__ = [
    "CheckpointPolicy",
    "RunRecord",
    "RunSpec",
    "RunState",
    "RunStore",
    "new_run_id",
    "rebuild_from_disk",
    "validate_run_id",
    "validate_run_name",
]
