"""One GPU, one run: the lock every run takes before it stages, and who is waiting.

The host has a single GPU. Two trainers on it compete for memory, run out of it,
or slow each other down, so a run holds the **GPU lock** from the moment it
starts staging until it has finished. The lock is an ``flock`` on a file, which
gives it the property that matters most: the kernel drops it when the holder
exits, however it exits. A crashed run never leaves the GPU locked.

The file lives in the workspace rather than the state directory, because the
sandbox has a state directory of its own but shares the workspace (and the GPU)
with the host — so a run in the sandbox and a run on the host still take turns.

Two more locks, of the same kind, make the queue work without a control plane:

- an **owner lock** in a queued run's directory, held by the terminal that runs
  ``nawat exp run`` and is waiting to run it in the foreground. A queued run
  with no owner file was handed to the queue (``--queue``) and is executed by the
  queue worker; one whose owner file exists but is not locked lost its terminal
  before it started, and is cancelled rather than run behind the user's back;
- a **worker lock** in the state directory, held by whatever serves those
  handed-over runs — the control plane's queue thread, or the small worker
  ``nawat exp run --queue`` starts when the control plane is not running — so
  exactly one of them does.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

GPU_LOCK = "gpu.lock"
OWNER_LOCK = "owner.lock"
WORKER_LOCK = "queue-worker.lock"


def gpu_lock_path(config) -> Path:
    return Path(config.workspace_root) / ".nawat" / GPU_LOCK


def worker_lock_path(config) -> Path:
    return Path(config.state_dir) / WORKER_LOCK


class FileLock:
    """An exclusive ``flock`` held through an open file. Released on :meth:`release` or on exit."""

    def __init__(self, path: Path, handle) -> None:
        self.path = path
        self._handle = handle

    @property
    def held(self) -> bool:
        return self._handle is not None

    def write(self, info: dict[str, Any]) -> None:
        """Say who holds it, for whoever is waiting. Readable without taking the lock."""
        if self._handle is None:
            return
        self._handle.seek(0)
        self._handle.truncate()
        self._handle.write(json.dumps(info) + "\n")
        self._handle.flush()

    def release(self) -> None:
        if self._handle is None:
            return
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "FileLock":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def try_lock(path: Path) -> FileLock | None:
    """Take the lock if nobody holds it; None, at once, if somebody does."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return FileLock(path, handle)


def is_held(path: Path) -> bool:
    """Whether some process holds the lock right now."""
    if not path.exists():
        return False
    try:
        handle = open(path, "r")
    except OSError:
        return False
    with handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False


def holder(path: Path) -> dict[str, Any] | None:
    """What the current holder wrote about itself, if the lock is held."""
    if not is_held(path):
        return None
    try:
        info = json.loads(path.read_text() or "null")
    except (OSError, ValueError):
        return {}
    return info if isinstance(info, dict) else {}


# -- the GPU ------------------------------------------------------------------------


def take_gpu(config, *, run_id: str, name: str | None) -> FileLock | None:
    lock = try_lock(gpu_lock_path(config))
    if lock is not None:
        lock.write({"run": run_id, "name": name, "pid": os.getpid(), "since": time.time(),
                    "where": "sandbox" if os.environ.get("NAWAT_SANDBOX") else "host"})
    return lock


def gpu_holder(config) -> dict[str, Any] | None:
    return holder(gpu_lock_path(config))


# -- who runs a queued run ------------------------------------------------------------


def claim(directory: Path) -> FileLock:
    """Lock a run directory's owner file, created already locked.

    The file is locked under a temporary name and then renamed into place, so an
    owner file that exists but is not locked always means its owner has gone.
    """
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{OWNER_LOCK}.{os.getpid()}"
    lock = try_lock(temporary)
    if lock is None:  # pragma: no cover - a fresh, private file
        raise RuntimeError(f"could not lock {temporary}")
    lock.write({"pid": os.getpid(), "since": time.time()})
    temporary.replace(directory / OWNER_LOCK)
    lock.path = directory / OWNER_LOCK
    return lock


def owner_state(directory: Path) -> str:
    """``"waiting"`` (a live terminal will run it), ``"queue"`` (handed to the queue) or ``"gone"``."""
    path = directory / OWNER_LOCK
    if not path.exists():
        return "queue"
    return "waiting" if is_held(path) else "gone"
