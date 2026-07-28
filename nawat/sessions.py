"""Inference sessions.

One session at a time, because 16 GB of VRAM cannot arbitrate two of them and
failing predictably beats failing mid-request (PRD §8, FR-4.9). The inference
server is a supervised subprocess rather than something imported, which keeps
torch and vLLM out of this package and — more usefully — means the lease on the
served weights is keyed to the server's own pid, so the weights stay put even if
the control plane restarts underneath it (FR-4.3).

Adapters are loaded onto a running base at runtime, never merged: a LoRA is
~200 MB against a 16 GB base, so merging to test would cost two orders of
magnitude more disk per experiment and minutes per evaluation (PRD §8, FR-4.2).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .cache import Cache
from .db import Database
from .errors import NawatError, NotFound, Protected, StoreUnavailable
from .keys import Key
from .leases import boot_id
from .process import process_alive, process_start_time, terminate_group
from .runs import RunStore

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id           TEXT PRIMARY KEY,
    model        TEXT NOT NULL,
    pid          INTEGER,
    boot_id      TEXT,
    proc_start   REAL,
    port         INTEGER NOT NULL,
    state        TEXT NOT NULL,
    started_at   REAL NOT NULL,
    last_used    REAL NOT NULL,
    idle_timeout REAL NOT NULL,
    adapters     TEXT NOT NULL DEFAULT '{}',
    lease_ids    TEXT NOT NULL DEFAULT '[]',
    error        TEXT
);
"""


class SessionState(str, Enum):
    STARTING = "starting"
    READY = "ready"
    STOPPED = "stopped"
    FAILED = "failed"

    @property
    def live(self) -> bool:
        return self in (SessionState.STARTING, SessionState.READY)


@dataclass
class Session:
    id: str
    model: Key
    port: int
    state: SessionState
    started_at: float
    last_used: float
    idle_timeout: float
    pid: int | None = None
    boot: str | None = None
    proc_start: float | None = None
    adapters: dict[str, str] = field(default_factory=dict)
    lease_ids: tuple[str, ...] = ()
    error: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def idle_for(self) -> float:
        return time.time() - self.last_used

    def alive(self) -> bool:
        """Whether the server process is still running.

        A server that exited but has not been reaped is not running, however
        much its pid still answers.
        """
        if self.pid is None:
            return False
        if self.boot != boot_id():
            return False  # predates a reboot
        if not process_alive(self.pid):
            return False
        start = process_start_time(self.pid)
        if start is None:
            return False
        if self.proc_start is None or start < 0 or self.proc_start < 0:
            return True
        return start == self.proc_start  # not a different process on a reused pid

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "model": str(self.model),
            "port": self.port,
            "state": self.state.value,
            "started_at": self.started_at,
            "last_used": self.last_used,
            "idle_for": self.idle_for,
            "idle_timeout": self.idle_timeout,
            "pid": self.pid,
            "adapters": dict(self.adapters),
            "url": self.base_url,
            "error": self.error,
        }


# -- backends -----------------------------------------------------------------


class ServingBackend:
    """How to start an OpenAI-compatible server for a set of weights."""

    name = "backend"
    supports_adapters = False

    def command(self, *, model_path: Path, model_name: str, port: int, extra: list[str]) -> list[str]:
        raise NotImplementedError

    def environment(self) -> dict[str, str]:
        return {}

    def health_path(self) -> str:
        return "/health"


class VLLMBackend(ServingBackend):
    """vLLM's OpenAI-compatible server, with runtime LoRA loading enabled."""

    name = "vllm"
    supports_adapters = True

    def command(self, *, model_path: Path, model_name: str, port: int, extra: list[str]) -> list[str]:
        return [
            "vllm",
            "serve",
            str(model_path),
            "--served-model-name",
            model_name,
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--enable-lora",
            *extra,
        ]

    def environment(self) -> dict[str, str]:
        # Without this, vLLM refuses the runtime adapter endpoints and testing an
        # adapter would mean restarting the server.
        return {"VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}


def build_backend(name: str) -> ServingBackend:
    if name == "vllm":
        return VLLMBackend()
    raise NotFound(
        f"Unknown serving backend {name!r}.",
        "Set NAWAT_SERVE_BACKEND=vllm, or pass a backend when constructing the manager.",
    )


# -- manager ------------------------------------------------------------------


class SessionManager:
    """Starts, supervises and tears down the one inference session."""

    def __init__(
        self,
        cache: Cache,
        db: Database | None = None,
        backend: ServingBackend | None = None,
        runs: RunStore | None = None,
    ) -> None:
        self.cache = cache
        self.config = cache.config
        self.db = db or cache.db
        self.backend = backend or build_backend(self.config.serve_backend)
        self.runs = runs
        #: Handles for servers this process started, so their exit is reaped
        #: rather than left as a zombie that still answers to signals.
        self._children: dict[int, subprocess.Popen] = {}
        self.db.connect().executescript(SCHEMA)

    # -- state -------------------------------------------------------------

    def current(self) -> Session | None:
        """The live session, if there is one. Dead ones are cleaned up in passing."""
        row = self.db.connect().execute("SELECT * FROM sessions LIMIT 1").fetchone()
        if row is None:
            return None
        session = self._row(row)
        if session.state.live and not session.alive():
            self._release(session)
            self._forget(session.id)
            return None
        return session if session.state.live else None

    def _row(self, row) -> Session:
        return Session(
            id=row["id"],
            model=Key.parse(row["model"]),
            port=row["port"],
            state=SessionState(row["state"]),
            started_at=row["started_at"],
            last_used=row["last_used"],
            idle_timeout=row["idle_timeout"],
            pid=row["pid"],
            boot=row["boot_id"],
            proc_start=row["proc_start"],
            adapters=json.loads(row["adapters"]),
            lease_ids=tuple(json.loads(row["lease_ids"])),
            error=row["error"],
        )

    def _save(self, session: Session) -> Session:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM sessions WHERE id != ?", (session.id,))
            conn.execute(
                "INSERT INTO sessions (id, model, pid, boot_id, proc_start, port, state, started_at,"
                " last_used, idle_timeout, adapters, lease_ids, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(id) DO UPDATE SET pid = excluded.pid, boot_id = excluded.boot_id,"
                "   proc_start = excluded.proc_start, state = excluded.state, last_used = excluded.last_used,"
                "   idle_timeout = excluded.idle_timeout, adapters = excluded.adapters,"
                "   lease_ids = excluded.lease_ids, error = excluded.error",
                (
                    session.id,
                    str(session.model),
                    session.pid,
                    session.boot,
                    session.proc_start,
                    session.port,
                    session.state.value,
                    session.started_at,
                    session.last_used,
                    session.idle_timeout,
                    json.dumps(session.adapters),
                    json.dumps(list(session.lease_ids)),
                    session.error,
                ),
            )
        return session

    def _forget(self, session_id: str) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # -- lifecycle ---------------------------------------------------------

    def start(
        self,
        model: "str | Key",
        *,
        idle_timeout: float | None = None,
        wait: bool = True,
        extra_args: list[str] | None = None,
    ) -> Session:
        """Serve ``model``, stopping whatever was running first (FR-4.1, FR-4.9)."""
        model = Key.parse(model)
        existing = self.current()
        if existing is not None:
            if existing.model == model and existing.state is SessionState.READY:
                return self.touch(existing)
            self.stop()

        path = self.cache.resolve(model, lease=False)
        session = Session(
            id=uuid.uuid4().hex,
            model=model,
            port=self.config.serve_port,
            state=SessionState.STARTING,
            started_at=time.time(),
            last_used=time.time(),
            idle_timeout=self.config.serve_idle_timeout if idle_timeout is None else idle_timeout,
        )

        extra = list(extra_args or [])
        if not extra and self.config.serve_extra_args:
            extra = self.config.serve_extra_args.split()
        command = self.backend.command(
            model_path=path, model_name=str(model), port=session.port, extra=extra
        )
        env = {**os.environ, **self.backend.environment()}
        log_path = self._log_path(session.id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with log_path.open("ab", buffering=0) as log:
                log.write(f"[nawat] serving {model} from {path}\n[nawat] {' '.join(command)}\n\n".encode())
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
        except FileNotFoundError as exc:
            raise NotFound(
                f"{command[0]} is not installed, so no inference server can be started.",
                "Install the serving backend (pip install vllm), or set NAWAT_SERVE_BACKEND.",
            ) from exc

        self._children[process.pid] = process
        session.pid = process.pid
        session.boot = boot_id()
        session.proc_start = process_start_time(process.pid)
        # The lease is taken against the server's pid, not ours, so the weights
        # stay put for exactly as long as something is actually reading them.
        leases = self.cache.leases.acquire([model], holder=f"serving {model}", pid=process.pid)
        session.lease_ids = tuple(lease.id for lease in leases)
        self._save(session)

        if wait:
            session = self.wait_ready(session)
        return session

    def wait_ready(self, session: Session, timeout: float | None = None) -> Session:
        """Poll until the server answers, or give up with the log to look at."""
        timeout = self.config.serve_startup_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not session.alive():
                session.state = SessionState.FAILED
                session.error = f"The inference server exited during startup. See {self._log_path(session.id)}."
                self._save(session)
                self._release(session)
                raise NawatError(session.error, "Check the log for the cause, then start the session again.")
            if self._probe(session):
                session.state = SessionState.READY
                session.last_used = time.time()
                return self._save(session)
            time.sleep(0.25)
        self.stop()
        raise NawatError(
            f"The inference server did not become ready within {int(timeout)}s.",
            f"Check {self._log_path(session.id)}, or raise NAWAT_SERVE_STARTUP_TIMEOUT.",
        )

    def _probe(self, session: Session) -> bool:
        try:
            with urllib.request.urlopen(session.base_url + self.backend.health_path(), timeout=2) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def stop(self) -> Session | None:
        """Tear the session down, release the GPU, and give the disk back (FR-4.4)."""
        row = self.db.connect().execute("SELECT * FROM sessions LIMIT 1").fetchone()
        if row is None:
            return None
        session = self._row(row)
        if session.pid is not None and session.alive():
            self._terminate(session.pid)
        self._release(session)
        session.state = SessionState.STOPPED
        self._forget(session.id)
        try:
            self.cache.collect()
        except NawatError:
            pass  # a full disk is reported elsewhere; stopping still succeeded
        return session

    def _terminate(self, pid: int) -> None:
        child = self._children.pop(pid, None)
        terminate_group(pid, reap=(child.poll if child is not None else None))

    def _release(self, session: Session) -> None:
        if session.lease_ids:
            self.cache.leases.release(session.lease_ids)

    # -- use ---------------------------------------------------------------

    def touch(self, session: Session | None = None) -> Session:
        session = session or self.require()
        session.last_used = time.time()
        return self._save(session)

    def require(self) -> Session:
        session = self.current()
        if session is None:
            raise NotFound("No inference server is running.", "Start one with: nawat serve <model key>")
        return session

    def reap_if_idle(self) -> Session | None:
        """Stop a session nobody has used for its idle timeout."""
        session = self.current()
        if session is None or session.idle_timeout <= 0:
            return None
        if session.idle_for < session.idle_timeout:
            return None
        return self.stop()

    # -- adapters ----------------------------------------------------------

    def load_adapter(self, key: "str | Key", name: str | None = None) -> Session:
        """Hot-load a trained adapter onto the running base — no merge, no restart."""
        key = Key.parse(key)
        session = self.require()
        if not self.backend.supports_adapters:
            raise Protected(
                f"The {self.backend.name} backend cannot load adapters at runtime.",
                "Serve the merged weights instead, or use the vLLM backend.",
            )
        name = name or key.path.replace("/", "-")
        path = self.cache.resolve(key, lease=False)
        leases = self.cache.leases.acquire([key], holder=f"serving {key}", pid=session.pid or os.getpid())
        try:
            self._post("/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)}, session)
        except NawatError:
            self.cache.leases.release(lease.id for lease in leases)
            raise
        session.adapters[name] = str(key)
        session.lease_ids = (*session.lease_ids, *(lease.id for lease in leases))
        session.last_used = time.time()
        return self._save(session)

    def unload_adapter(self, name: str) -> Session:
        session = self.require()
        if name not in session.adapters:
            raise NotFound(f"No adapter named {name} is loaded.", "List them with: nawat serve status")
        self._post("/v1/unload_lora_adapter", {"lora_name": name}, session)
        session.adapters.pop(name, None)
        session.last_used = time.time()
        return self._save(session)

    def _post(self, path: str, payload: dict[str, Any], session: Session) -> Any:
        request = urllib.request.Request(
            session.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = response.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            raise NawatError(
                f"The inference server refused {path}: HTTP {exc.code}. {detail}".strip(),
                "Check the adapter is compatible with the base model, then try again.",
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise StoreUnavailable(
                f"The inference server did not answer {path}: {type(exc).__name__}.",
                "It may still be starting, or it may have died — check: nawat serve status",
            ) from exc
        try:
            return json.loads(body) if body.strip() else None
        except ValueError:
            return body

    def _log_path(self, session_id: str) -> Path:
        return self.config.state_dir / "sessions" / f"{session_id}.log"

    def log(self, tail: int = 200) -> str:
        session = self.current()
        if session is None:
            return ""
        path = self._log_path(session.id)
        if not path.exists():
            return ""
        return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-tail:])
