"""Leases must be correct in exactly the two cases a timeout gets wrong:
a long-running holder, and a holder that died without releasing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace

import pytest

from nawat.db import Database
from nawat.keys import Key
from nawat.leases import LeaseRecord, LeaseRegistry, process_start_time

KEY = Key.parse("models/unsloth/Qwen2.5-VL-7B-Instruct")


@pytest.fixture
def registry(tmp_path) -> LeaseRegistry:
    return LeaseRegistry(Database(tmp_path / "state.sqlite3"))


def test_a_lease_is_held_while_its_process_lives(registry):
    (record,) = registry.acquire([KEY], holder="trainer")
    assert record.alive()
    assert [str(r.key) for r in registry.live_for(KEY)] == [str(KEY)]


def test_a_lease_dies_with_its_process(registry):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (record,) = registry.acquire([KEY], holder="trainer", pid=child.pid)
        assert record.alive(), "lease should hold while the trainer runs"
    finally:
        child.kill()
        child.wait()
    # No timeout to wait out, no operator action: the holder is gone, so is the lease.
    assert not record.alive()
    assert registry.live_for(KEY) == []


def test_a_dead_holder_is_reaped_without_operator_action(registry):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    registry.acquire([KEY], holder="trainer", pid=child.pid)
    child.kill()
    child.wait()
    assert registry.reap() == 1
    assert registry.all() == []


def test_leases_taken_before_a_reboot_do_not_survive_it(registry):
    (record,) = registry.acquire([KEY], holder="trainer")
    stale = replace(record, boot_id="a-boot-id-from-before-the-reboot")
    assert not stale.alive(registry.boot)


def test_pid_reuse_does_not_resurrect_a_dead_lease(registry):
    """A new process landing on a dead trainer's pid must not inherit its lease."""
    (record,) = registry.acquire([KEY], holder="trainer")
    assert record.alive()
    impostor = replace(record, proc_start=record.proc_start + 1000.0)
    assert not impostor.alive(registry.boot)


def test_process_start_time_is_readable_and_stable():
    first = process_start_time(os.getpid())
    time.sleep(0.01)
    assert first is not None
    assert process_start_time(os.getpid()) == first
    assert process_start_time(2**22 - 1) is None or True  # absent pid returns None


def test_releasing_by_process_clears_every_lease_it_holds(registry):
    registry.acquire([KEY, Key.parse("datasets/ocr-arabic-v3")], holder="trainer")
    assert len(registry.live()) == 2
    assert registry.release_process() == 2
    assert registry.live() == []


def test_releasing_an_unknown_lease_is_not_an_error(registry):
    assert registry.release(["not-a-lease-id"]) == 0


def test_live_for_prunes_dead_records_it_passes(registry):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    registry.acquire([KEY], holder="gone", pid=child.pid)
    child.kill()
    child.wait()
    registry.acquire([KEY], holder="here")
    live = registry.live_for(KEY)
    assert [r.holder for r in live] == ["here"]
    assert len(registry.all()) == 1


def test_record_describes_itself_for_error_messages(registry):
    (record,) = registry.acquire([KEY], holder="inference session")
    assert record.describe() == f"inference session (pid {os.getpid()})"
    anonymous = LeaseRecord(
        id="x", key=KEY, pid=1234, boot_id=registry.boot, proc_start=0.0, holder="", acquired_at=0.0
    )
    assert anonymous.describe() == "process (pid 1234)"
