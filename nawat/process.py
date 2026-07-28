"""Process liveness, in one place.

Leases, run records and inference sessions all decide things based on whether a
pid is still running, so they had better agree on what that means.

``os.kill(pid, 0)`` is not enough. A child that has exited but has not been
waited for is a zombie: its pid still exists, ``/proc/<pid>`` still exists, and
signalling it still succeeds. Believing that would keep a dead trainer's lease
on 16 GB of weights forever, which is precisely the failure the lease design
exists to avoid.
"""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Callable

#: /proc/<pid>/stat field 3. 'Z' is a process that has exited and is waiting to
#: be reaped; 'X' is dead.
DEAD_STATES = frozenset("ZX")

UNKNOWN_START = -1.0

DEFAULT_GRACE = 30.0


def read_stat(pid: int) -> tuple[str, float] | None:
    """``(state, start time)`` for ``pid``, or None if it is gone."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The comm field is parenthesised and may itself contain spaces and
    # parentheses, so everything is counted from the last ')'.
    try:
        tail = raw[raw.rindex(")") + 2 :].split()
        return tail[0], float(tail[19])
    except (ValueError, IndexError):
        return "?", UNKNOWN_START


def process_start_time(pid: int) -> float | None:
    """Start time in clock ticks since boot, or None if the process is gone."""
    stat = read_stat(pid)
    if stat is None:
        return None
    return stat[1]


def is_zombie(pid: int) -> bool:
    stat = read_stat(pid)
    return stat is not None and stat[0] in DEAD_STATES


def process_alive(pid: int) -> bool:
    """Whether ``pid`` is a process that is actually still doing something."""
    stat = read_stat(pid)
    if stat is not None:
        return stat[0] not in DEAD_STATES
    # No /proc on this platform; fall back to signalling.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def terminate_group(
    pid: int,
    *,
    grace: float = DEFAULT_GRACE,
    reap: Callable[[], None] | None = None,
) -> bool:
    """Ask a process group to stop, then insist. True if it is gone.

    The group, not the process: a trainer spawns dataloader workers and an
    inference server spawns its own children, and leaving those behind would
    hold both the GPU and the cache.

    ``reap`` is called while waiting so a caller holding the ``Popen`` can clear
    the zombie; without it the state check below does the same job.
    """
    try:
        group = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return True
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if reap is not None:
            reap()
        if not process_alive(pid):
            return True
        time.sleep(0.05)

    try:
        os.killpg(group, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return True
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if reap is not None:
            reap()
        if not process_alive(pid):
            return True
        time.sleep(0.05)
    return not process_alive(pid)
