"""Run records: written as the run progresses, and readable long afterwards."""

from __future__ import annotations

import json
import threading
import time

import pytest

from nawat.errors import InvalidKey, NotFound
from nawat.keys import Key
from nawat.runs import RunSpec, RunState, RunStore, rebuild_from_disk

CACHE_CEILING = 50 * 10**6

SPEC = RunSpec(
    script="train.py",
    model=Key.parse("models/unsloth/Qwen2.5-VL-7B-Instruct"),
    datasets=(Key.parse("datasets/ocr-arabic-v3"),),
    params={"learning_rate": "2e-4"},
    notes="baseline",
)


def test_a_submitted_run_starts_queued(runs):
    record = runs.create(SPEC)
    assert record.state is RunState.QUEUED
    assert record.spec.model == SPEC.model
    assert runs.get(record.id).spec.params == {"learning_rate": "2e-4"}


def test_the_record_is_on_disk_before_anything_runs(runs):
    record = runs.create(SPEC)
    written = json.loads(runs.record_path(record.id).read_text())
    assert written["id"] == record.id
    assert written["state"] == "queued"
    assert written["spec"]["notes"] == "baseline"


def test_each_transition_is_flushed_to_disk(runs):
    record = runs.create(SPEC)
    runs.update(record.id, state=RunState.RUNNING, started_at=time.time(), pid=4321)

    written = json.loads(runs.record_path(record.id).read_text())
    assert written["state"] == "running"
    assert written["pid"] == 4321


def test_staged_keys_are_ordered_and_deduplicated():
    shared = Key.parse("datasets/ocr-arabic-v3")
    spec = RunSpec(script="t.py", model=Key.parse("models/base"), datasets=(shared,), inputs=(shared,))
    assert [str(k) for k in spec.staged_keys()] == ["models/base", "datasets/ocr-arabic-v3"]


def test_a_run_id_must_be_usable_as_a_key_segment(runs):
    with pytest.raises(InvalidKey):
        runs.create(SPEC, "../escape")
    with pytest.raises(InvalidKey):
        runs.create(SPEC, "has space")


def test_a_duplicate_run_id_is_refused(runs):
    runs.create(SPEC, "2026-07-28-a91f")
    with pytest.raises(InvalidKey, match="already exists"):
        runs.create(SPEC, "2026-07-28-a91f")


def test_an_unknown_run_says_how_to_find_the_real_ones(runs):
    with pytest.raises(NotFound, match="nawat runs"):
        runs.get("never-submitted")


def test_runs_list_newest_first_and_filter_by_state(runs):
    first = runs.create(SPEC, "run-one")
    time.sleep(0.01)
    second = runs.create(SPEC, "run-two")
    runs.update(second.id, state=RunState.SUCCEEDED)

    assert [r.id for r in runs.list()] == ["run-two", "run-one"]
    assert [r.id for r in runs.list(state=RunState.SUCCEEDED)] == ["run-two"]
    assert [r.id for r in runs.list(state=RunState.QUEUED)] == ["run-one"]


def test_the_queue_is_first_in_first_out(runs):
    runs.create(SPEC, "first")
    time.sleep(0.01)
    runs.create(SPEC, "second")

    assert runs.next_queued().id == "first"
    runs.update("first", state=RunState.RUNNING)
    assert runs.next_queued().id == "second"


def test_history_survives_losing_the_database(runs, cache):
    record = runs.create(SPEC, "2026-07-28-a91f")
    runs.update(record.id, state=RunState.SUCCEEDED, artifacts=["runs/2026-07-28-a91f/adapter"])

    with runs.db.tx() as conn:
        conn.execute("DELETE FROM runs")
    assert runs.list() == []

    assert rebuild_from_disk(runs) == 1
    recovered = runs.get("2026-07-28-a91f")
    assert recovered.state is RunState.SUCCEEDED
    assert [str(k) for k in recovered.artifacts] == ["runs/2026-07-28-a91f/adapter"]


def test_a_run_that_did_not_survive_a_restart_is_marked_failed(runs):
    record = runs.create(SPEC)
    runs.update(record.id, state=RunState.RUNNING, pid=999999, started_at=time.time())

    recovered = runs.reconcile(is_alive=lambda pid: False)

    assert [r.id for r in recovered] == [record.id]
    assert runs.get(record.id).state is RunState.FAILED
    assert "did not survive a restart" in runs.get(record.id).error


def test_a_run_still_genuinely_running_is_left_alone(runs):
    record = runs.create(SPEC)
    runs.update(record.id, state=RunState.RUNNING, pid=1234, started_at=time.time())

    assert runs.reconcile(is_alive=lambda pid: True) == []
    assert runs.get(record.id).state is RunState.RUNNING


# -- logs --------------------------------------------------------------------


def test_following_a_log_ends_when_the_run_does(runs):
    record = runs.create(SPEC)
    runs.update(record.id, state=RunState.RUNNING)
    log = runs.log_path(record.id)
    log.write_text("first line\n")

    chunks: list[str] = []

    def finish() -> None:
        time.sleep(0.3)
        with log.open("a") as handle:
            handle.write("second line\n")
        time.sleep(0.3)
        runs.update(record.id, state=RunState.SUCCEEDED)

    threading.Thread(target=finish, daemon=True).start()
    for chunk in runs.follow_log(record.id, poll=0.05):
        chunks.append(chunk)

    combined = "".join(chunks)
    assert "first line" in combined
    assert "second line" in combined


def test_following_a_finished_run_returns_its_whole_log(runs):
    record = runs.create(SPEC)
    runs.log_path(record.id).write_text("everything that happened\n")
    runs.update(record.id, state=RunState.SUCCEEDED)

    assert "everything that happened" in "".join(runs.follow_log(record.id, poll=0.01))


def test_reading_a_log_tail(runs):
    record = runs.create(SPEC)
    runs.log_path(record.id).write_text("".join(f"line {n}\n" for n in range(100)))

    assert runs.read_log(record.id, tail=3) == "line 97\nline 98\nline 99\n"


def test_a_run_with_no_log_yet_reads_empty(runs):
    record = runs.create(SPEC)
    assert runs.read_log(record.id) == ""
