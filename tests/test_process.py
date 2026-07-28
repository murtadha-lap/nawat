"""Process liveness.

The zombie case has its own tests because getting it wrong is silent: a
crashed trainer whose pid still answers keeps its lease on the weights, and the
cache fills up with something nothing is actually using.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from nawat.process import (
    is_zombie,
    process_alive,
    process_start_time,
    read_stat,
    terminate_group,
)


def test_this_process_is_alive():
    assert process_alive(os.getpid())
    assert not is_zombie(os.getpid())


def test_a_pid_that_never_existed_is_not_alive():
    assert not process_alive(2**22 - 1)
    assert process_start_time(2**22 - 1) is None


def test_a_child_that_exited_but_was_not_reaped_is_not_alive():
    """os.kill(pid, 0) succeeds on a zombie. Believing it would strand a lease."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not is_zombie(child.pid):
        time.sleep(0.01)

    try:
        assert is_zombie(child.pid), "the child should be a zombie until it is waited for"
        os.kill(child.pid, 0)  # the pid still answers...
        assert not process_alive(child.pid)  # ...but it is not running
    finally:
        child.wait()

    assert not process_alive(child.pid)


def test_start_time_is_stable_while_a_process_runs():
    first = process_start_time(os.getpid())
    time.sleep(0.01)
    assert first == process_start_time(os.getpid())


def test_read_stat_survives_a_command_name_with_spaces_and_parens():
    state, start = read_stat(os.getpid())
    assert state in "RSD"
    assert start > 0


def test_terminating_a_group_takes_the_children_with_it():
    """A trainer's dataloader workers must not outlive the trainer."""
    spawner = (
        "import subprocess, sys, time;"
        "kid = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        "print(kid.pid, flush=True);"
        "time.sleep(60)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", spawner], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    child_pid = int(parent.stdout.readline().strip())
    assert process_alive(child_pid)

    assert terminate_group(parent.pid, grace=10.0, reap=parent.poll)

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and process_alive(child_pid):
        time.sleep(0.05)
    assert not process_alive(child_pid)
    parent.wait()


def test_terminating_something_already_gone_reports_success():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert terminate_group(child.pid, grace=1.0)


def test_a_stubborn_process_is_killed_after_the_grace_period():
    ignores_sigterm = (
        "import signal, time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "print('ready', flush=True);"
        "time.sleep(60)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", ignores_sigterm], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    process.stdout.readline()

    started = time.monotonic()
    assert terminate_group(process.pid, grace=0.5, reap=process.poll)

    assert time.monotonic() - started >= 0.5, "it should have been given its grace first"
    assert not process_alive(process.pid)
    process.wait()
