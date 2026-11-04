"""One GPU: every run takes its turn, in order, with or without the control plane (issue #10)."""

from __future__ import annotations

import json
import threading
import time

import pytest

from nawat import cli, gpulock
from nawat.errors import Protected
from nawat.executor import RunQueue
from nawat.lab import experiment as E
from nawat.lab import report as rep
from nawat.lab.cli import experiment_spec
from nawat.runs import RunState, new_run_id


@pytest.fixture
def queue_lab(lab, monkeypatch):
    """The lab, with the background queue worker recorded instead of started."""
    started = []
    monkeypatch.setattr(cli, "start_queue_worker", lambda config: started.append(config) or True)
    lab.started_workers = started
    return lab


def _executor(lab):
    executor = cli._executor(lab.cache)
    executor.poll = 0.02
    return executor


def _submit(lab, executor, name, *, owned=False):
    experiment = lab.plan(name)
    lab.approve(experiment)
    run_id = new_run_id()
    owner = gpulock.claim(executor.runs.directory(run_id)) if owned else None
    record = executor.runs.create(experiment_spec(experiment), run_id, name=experiment.name)
    experiment.add_attempt(record.id)
    return experiment, record, owner


def _until(predicate, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_run_waits_while_the_gpu_is_busy_then_runs(queue_lab):
    lab = queue_lab
    executor = _executor(lab)
    experiment, record, owner = _submit(lab, executor, "waits", owned=True)
    busy = gpulock.take_gpu(lab.config, run_id="2026-01-01-aaaa", name="other/exp")
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("record", executor.execute(record.id)))
    worker.start()
    try:
        assert _until(lambda: "waiting for run 2026-01-01-aaaa (other/exp) to finish — 1 ahead"
                      in executor.runs.read_log(record.id))
        assert executor.runs.get(record.id).state is RunState.QUEUED
        assert rep.status(E.Experiment.load(experiment.dir), lab.config) == "queued"
    finally:
        busy.release()
    worker.join(60)
    owner.release()
    assert result["record"].state is RunState.SUCCEEDED, result["record"].error
    assert not gpulock.is_held(gpulock.gpu_lock_path(lab.config))  # given back when the run ended


def test_runs_start_in_the_order_they_were_submitted(queue_lab):
    lab = queue_lab
    executor = _executor(lab)
    _, first, _ = _submit(lab, executor, "first-in")  # handed to the queue
    _, second, owner = _submit(lab, executor, "second-in", owned=True)  # a terminal waits to run it
    result = {}
    foreground = threading.Thread(target=lambda: result.setdefault("record", executor.execute(second.id)))
    foreground.start()
    # Nothing serves the queue yet: the foreground run waits behind the first, and asks for a worker.
    assert _until(lambda: "to start first — 1 ahead" in executor.runs.read_log(second.id))
    assert executor.runs.get(second.id).state is RunState.QUEUED
    assert lab.started_workers
    queue = RunQueue(_executor(lab), poll=0.02)
    assert queue.serve_until_empty() == 1  # runs the first only; the second belongs to its terminal
    foreground.join(60)
    owner.release()
    done_first, done_second = executor.runs.get(first.id), result["record"]
    assert done_first.state is RunState.SUCCEEDED and done_second.state is RunState.SUCCEEDED
    assert done_first.finished_at <= done_second.started_at  # never on the GPU together


def test_a_queued_run_whose_terminal_went_away_is_dropped_not_run(queue_lab):
    lab = queue_lab
    executor = _executor(lab)
    experiment, record, owner = _submit(lab, executor, "orphan", owned=True)
    owner.release()  # the terminal was closed before the run started
    assert RunQueue(executor, poll=0.02).serve_until_empty() == 0
    final = executor.runs.get(record.id)
    assert final.state is RunState.CANCELLED and final.started_at is None
    assert rep.status(E.Experiment.load(experiment.dir), lab.config) == "cancelled"


def test_only_one_process_serves_the_queue(queue_lab):
    lab = queue_lab
    executor = _executor(lab)
    _, record, _ = _submit(lab, executor, "served-once")
    other = gpulock.try_lock(gpulock.worker_lock_path(lab.config))  # the control plane, say
    try:
        assert RunQueue(executor, poll=0.02).serve_until_empty() == 0
        assert executor.runs.get(record.id).state is RunState.QUEUED
    finally:
        other.release()


def test_exp_run_queue_runs_without_the_control_plane(queue_lab, capsys):
    lab = queue_lab
    experiment = lab.plan("no-api")
    lab.approve(experiment)
    assert cli.main(["exp", "run", experiment.name, "--queue"]) == 0
    assert lab.started_workers  # what `nawat exp run --queue` starts when nothing serves the queue
    assert "Queued run" in capsys.readouterr().out
    assert rep.status(E.Experiment.load(experiment.dir), lab.config) == "queued"
    assert cli.main(["exp", "queue", "--work"]) == 0  # what that worker runs
    assert rep.status(E.Experiment.load(experiment.dir), lab.config) == "succeeded"


def test_an_experiment_cannot_be_queued_twice(queue_lab):
    lab = queue_lab
    experiment = lab.plan("twice")
    lab.approve(experiment)
    assert cli.main(["exp", "run", experiment.name, "--queue"]) == 0
    assert cli.main(["exp", "run", experiment.name]) == Protected.exit_code
    assert cli.main(["exp", "cancel", experiment.name]) == 0  # takes it out of the queue
    assert rep.status(E.Experiment.load(experiment.dir), lab.config) == "cancelled"


def test_exp_queue_lists_the_runs_in_order(queue_lab, capsys):
    lab = queue_lab
    executor = _executor(lab)
    _, first, _ = _submit(lab, executor, "q-one")
    _, second, owner = _submit(lab, executor, "q-two", owned=True)
    busy = gpulock.take_gpu(lab.config, run_id="2026-01-01-bbbb", name="sandbox/exp")
    try:
        capsys.readouterr()
        assert cli.main(["exp", "queue", "--json"]) == 0
        rows = json.loads(capsys.readouterr().out)
        assert [r["run"] for r in rows] == ["2026-01-01-bbbb", first.id, second.id]
        assert [r["by"].split()[0] for r in rows] == ["running", "queue", "terminal"]
        assert cli.main(["exp", "queue"]) == 0
        assert "q-two" in capsys.readouterr().out
    finally:
        busy.release()
        owner.release()


def test_a_cancel_while_waiting_takes_the_run_out_of_the_queue(queue_lab):
    lab = queue_lab
    executor = _executor(lab)
    _, record, owner = _submit(lab, executor, "leaves", owned=True)
    busy = gpulock.take_gpu(lab.config, run_id="2026-01-01-cccc", name=None)
    result = {}
    worker = threading.Thread(target=lambda: result.setdefault("record", executor.execute(record.id)))
    worker.start()
    try:
        assert _until(lambda: "waiting for run 2026-01-01-cccc to finish" in executor.runs.read_log(record.id))
        executor.cancel(record.id)  # what Ctrl-C does
        worker.join(10)
    finally:
        busy.release()
        owner.release()
    assert not worker.is_alive()
    assert result["record"].state is RunState.CANCELLED and result["record"].started_at is None


def test_the_background_worker_really_starts_and_drains_the_queue(lab):
    experiment = lab.plan("background")
    lab.approve(experiment)
    assert cli.main(["exp", "run", experiment.name, "--queue"]) == 0  # starts `nawat exp queue --work`
    loaded = lambda: E.Experiment.load(experiment.dir)  # noqa: E731
    assert _until(lambda: rep.status(loaded(), lab.config) == "succeeded", timeout=120), \
        (lab.config.state_dir / "queue-worker.log").read_text()
    assert _until(lambda: not gpulock.is_held(gpulock.worker_lock_path(lab.config)), timeout=30)  # and it exits
