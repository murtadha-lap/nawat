"""Running a job: stage under lease, execute offline, publish, reclaim.

The scripts here are real files run in real subprocesses, because the contract
being tested is exactly what a training script sees.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from nawat.errors import InvalidKey, NotFound, Protected
from nawat.keys import Key
from nawat.runs import RunSpec, RunState

CACHE_CEILING = 50 * 10**6

WRITES_AN_ADAPTER = """
import os, pathlib
out = pathlib.Path(os.environ["NAWAT_OUT_DIR"])
(out / "adapter").mkdir(parents=True, exist_ok=True)
(out / "adapter" / "adapter_model.safetensors").write_bytes(b"w" * 2048)
(out / "adapter" / "adapter_config.json").write_text('{"r": 16}')
print("trained")
"""

REPORTS_ITS_ENVIRONMENT = """
import json, os, pathlib
out = pathlib.Path(os.environ["NAWAT_OUT_DIR"])
out.mkdir(parents=True, exist_ok=True)
seen = {k: v for k, v in os.environ.items() if k.startswith(("NAWAT_", "HF_", "TRANSFORMERS_"))}
(out / "env.json").write_text(json.dumps(seen, indent=2))
print("reported")
"""


def submit(executor, script_name, **kwargs):
    spec = RunSpec(script=script_name, **kwargs)
    return executor.runs.create(spec)


# -- the happy path ----------------------------------------------------------


def test_a_run_publishes_each_artifact_class_under_its_own_key(executor, script, cached, store):
    cached("datasets/ocr-arabic-v3", 4096)
    name = script(WRITES_AN_ADAPTER)
    record = submit(executor, name, datasets=(Key.parse("datasets/ocr-arabic-v3"),))

    finished = executor.execute(record.id)

    assert finished.state is RunState.SUCCEEDED
    assert finished.exit_code == 0
    assert [str(k) for k in finished.artifacts] == [f"runs/{record.id}/adapter"]
    assert store.exists(Key.parse(f"runs/{record.id}/adapter"))
    assert "trained" in executor.runs.read_log(record.id)


def test_the_local_copy_is_reclaimed_once_the_upload_verifies(executor, script, cache):
    record = submit(executor, script(WRITES_AN_ADAPTER))
    executor.execute(record.id)

    assert not executor.output_dir(record.id).exists()
    assert cache.get(f"runs/{record.id}/adapter") is None


def test_loose_output_files_publish_together(executor, script, store):
    record = submit(executor, script(REPORTS_ITS_ENVIRONMENT))
    finished = executor.execute(record.id)

    assert finished.state is RunState.SUCCEEDED
    assert f"runs/{record.id}/output" in [str(k) for k in finished.artifacts]
    assert "env.json" in store.list_prefix(Key.parse(f"runs/{record.id}/output"))


def test_the_log_and_record_reach_object_storage(executor, script, store):
    record = submit(executor, script(WRITES_AN_ADAPTER))
    executor.execute(record.id)

    published = store.list_prefix(Key.parse(f"runs/{record.id}/record"))
    assert "run.log" in published and "run.json" in published


# -- the contract the script sees --------------------------------------------


def test_the_script_receives_its_inputs_through_the_environment(executor, script, cached, store, cache):
    cached("models/base", 4096)
    cached("datasets/ocr-arabic-v3", 4096)
    record = submit(
        executor,
        script(REPORTS_ITS_ENVIRONMENT),
        model=Key.parse("models/base"),
        datasets=(Key.parse("datasets/ocr-arabic-v3"),),
        params={"learning_rate": "2e-4", "max steps": "60"},
    )

    executor.execute(record.id)

    staged = store.list_prefix(Key.parse(f"runs/{record.id}/output"))
    assert "env.json" in staged
    path = cache.resolve(f"runs/{record.id}/output", lease=False)
    seen = json.loads((path / "env.json").read_text())

    assert seen["NAWAT_RUN_ID"] == record.id
    assert seen["NAWAT_MODEL_DIR"].endswith("models/base")
    assert seen["NAWAT_DATASET_DIR"].endswith("datasets/ocr-arabic-v3")
    assert json.loads(seen["NAWAT_DATASET_DIRS"]) == [seen["NAWAT_DATASET_DIR"]]
    assert json.loads(seen["NAWAT_INPUTS"])["models/base"] == seen["NAWAT_MODEL_DIR"]
    assert seen["NAWAT_PARAM_LEARNING_RATE"] == "2e-4"
    assert seen["NAWAT_PARAM_MAX_STEPS"] == "60"
    assert json.loads(seen["NAWAT_PARAMS"])["max steps"] == "60"


def test_the_run_cannot_reach_the_internet(executor, script, cache, store):
    """An undeclared input fails here rather than downloading silently (FR-2.3)."""
    record = submit(executor, script(REPORTS_ITS_ENVIRONMENT))
    executor.execute(record.id)

    path = cache.resolve(f"runs/{record.id}/output", lease=False)
    seen = json.loads((path / "env.json").read_text())

    assert seen["HF_HUB_OFFLINE"] == "1"
    assert seen["TRANSFORMERS_OFFLINE"] == "1"
    assert seen["HF_DATASETS_OFFLINE"] == "1"
    assert seen["NAWAT_OFFLINE"] == "1"


def test_inputs_are_held_while_the_run_uses_them_and_released_after(executor, script, cached, cache):
    cached("models/base", 4096)
    holding = script(
        """
import os, pathlib, json
out = pathlib.Path(os.environ["NAWAT_OUT_DIR"]); out.mkdir(parents=True, exist_ok=True)
(out / "leases.json").write_text(os.environ["NAWAT_INPUTS"])
"""
    )
    record = submit(executor, holding, model=Key.parse("models/base"))

    assert not cache.get("models/base").leased
    executor.execute(record.id)
    assert not cache.get("models/base").leased, "leases must not outlive the run"


def test_an_input_in_use_by_a_run_cannot_be_evicted(executor, cached, cache, monkeypatch):
    cached("models/base", 4096)
    seen: dict[str, bool] = {}

    real_run = executor._run_process

    def observe(record, plan):
        seen["leased"] = cache.get("models/base").leased
        return real_run(record, plan)

    monkeypatch.setattr(executor, "_run_process", observe)
    script_path = executor.config.workspace_root / "noop.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("print('ok')")
    record = submit(executor, "noop.py", model=Key.parse("models/base"))

    executor.execute(record.id)

    assert seen["leased"] is True


# -- failure -----------------------------------------------------------------


def test_a_failed_run_keeps_its_record_and_publishes_nothing(executor, script, store):
    record = submit(executor, script("import sys; print('boom'); sys.exit(3)"))

    finished = executor.execute(record.id)

    assert finished.state is RunState.FAILED
    assert finished.exit_code == 3
    assert finished.artifacts == ()
    assert "exited 3" in finished.error
    assert "boom" in executor.runs.read_log(record.id)
    # The record still reaches object storage, so the failure is reviewable later.
    assert "run.log" in store.list_prefix(Key.parse(f"runs/{record.id}/record"))


def test_a_script_outside_the_workspace_is_refused(executor, tmp_path):
    outside = tmp_path / "elsewhere.py"
    outside.write_text("print('hi')")
    with pytest.raises(InvalidKey, match="outside the workspace"):
        executor.resolve_script(str(outside))


def test_a_traversal_out_of_the_workspace_is_refused(executor):
    with pytest.raises(InvalidKey, match="outside the workspace"):
        executor.resolve_script("../../etc/passwd")


def test_a_missing_script_says_so(executor):
    with pytest.raises(NotFound):
        executor.resolve_script("no-such-script.py")


def test_a_script_of_the_wrong_kind_is_refused(executor, workspace):
    (workspace / "notes.txt").write_text("not a trainer")
    with pytest.raises(InvalidKey, match="neither a Python script nor a notebook"):
        executor.resolve_script("notes.txt")


def test_validation_refuses_an_input_that_exists_nowhere(executor, script):
    spec = RunSpec(script=script(WRITES_AN_ADAPTER), inputs=(Key.parse("runs/never/adapter"),))
    with pytest.raises(NotFound, match="not cached and not in object storage"):
        executor.validate(spec)


def test_a_finished_run_cannot_be_executed_again(executor, script):
    record = submit(executor, script(WRITES_AN_ADAPTER))
    executor.execute(record.id)
    with pytest.raises(Protected, match="already finished"):
        executor.execute(record.id)


# -- cancellation ------------------------------------------------------------


def test_cancelling_stops_the_trainer_and_marks_the_run(executor, script):
    import threading

    record = submit(executor, script("import time\nprint('starting', flush=True)\ntime.sleep(60)\n"))
    worker = threading.Thread(target=lambda: executor.execute(record.id), daemon=True)
    worker.start()

    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        current = executor.runs.get(record.id)
        if current.state is RunState.RUNNING and current.pid:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the run never started")

    pid = executor.runs.get(record.id).pid
    executor.cancel(record.id)
    worker.join(timeout=30)

    finished = executor.runs.get(record.id)
    assert finished.state is RunState.CANCELLED
    assert finished.artifacts == ()
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_cancelling_a_queued_run_never_starts_it(executor, script):
    record = submit(executor, script(WRITES_AN_ADAPTER))
    cancelled = executor.cancel(record.id)
    assert cancelled.state is RunState.CANCELLED
    assert "before it started" in cancelled.error


def test_cancelling_a_finished_run_says_there_is_nothing_to_do(executor, script):
    record = submit(executor, script(WRITES_AN_ADAPTER))
    executor.execute(record.id)
    with pytest.raises(Protected, match="already finished"):
        executor.cancel(record.id)


# -- notebooks ---------------------------------------------------------------

def test_a_notebook_is_executed_and_archived_as_the_record(executor, workspace):
    (workspace / "explore.ipynb").write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
    record = executor.runs.create(RunSpec(script="explore.ipynb"))
    plan = executor.plan(record, {})

    assert "nbconvert" in plan.command
    assert "--execute" in plan.command
    archived = plan.command[plan.command.index("--output") + 1]
    assert archived.endswith("explore.executed.ipynb")
    assert "/record/" in archived, "the executed notebook publishes with the run record"
