"""Metrics: captured at each step, streamed live, durable past eviction.

The last test is Phase 3's exit criterion, verbatim: a completed run's full
metric history survives eviction of its artifacts.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from nawat import metrics
from nawat.errors import NotFound
from nawat.keys import Key
from nawat.runs import RunSpec
from nawat.units import spark

CACHE_CEILING = 50 * 10**6


@pytest.fixture
def metrics_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv(metrics.ENV_VAR, str(path))
    return path


# -- writing -----------------------------------------------------------------


def test_log_writes_numeric_fields_with_step_and_time(metrics_file):
    written = metrics.log(step=12, loss=0.63, lr=2e-4)

    assert written["step"] == 12
    assert written["loss"] == 0.63
    assert written["t"] > 0
    on_disk = json.loads(metrics_file.read_text())
    assert on_disk == written


def test_non_numeric_and_non_finite_values_never_reach_the_series(metrics_file):
    metrics.log(step=1, loss=0.5, note="ignore me", ratio=float("nan"), flag=True, depth=float("inf"))

    (point,) = metrics.read_points(metrics_file)
    assert set(point) == {"t", "step", "loss"}


def test_events_are_kept_as_strings_for_annotation(metrics_file):
    metrics.log(step=30, event="epoch_end", epoch=1.0)

    points = metrics.read_points(metrics_file)
    assert metrics.events(points) == [{"step": 30, "t": points[0]["t"], "event": "epoch_end"}]


def test_without_the_platform_the_script_still_works(tmp_path, monkeypatch):
    """The variable unset means standalone, not broken (PRD principle 4)."""
    monkeypatch.delenv(metrics.ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)

    metrics.log(step=1, loss=1.0)

    assert metrics.read_points(tmp_path / metrics.FILENAME)


# -- reading -----------------------------------------------------------------


def test_series_regroups_points_per_metric(metrics_file):
    metrics.log(step=1, loss=2.0, lr=1e-4)
    metrics.log(step=2, loss=1.5, lr=2e-4)
    metrics.log(step=3, loss=1.1)

    grouped = metrics.series(metrics.read_points(metrics_file))

    assert [entry["value"] for entry in grouped["loss"]] == [2.0, 1.5, 1.1]
    assert [entry["step"] for entry in grouped["lr"]] == [1, 2]


def test_a_corrupt_line_is_skipped_not_fatal(metrics_file):
    metrics.log(step=1, loss=2.0)
    with metrics_file.open("a") as handle:
        handle.write("{torn line\n")
    metrics.log(step=2, loss=1.5)

    assert [point["step"] for point in metrics.read_points(metrics_file)] == [1, 2]


def test_follow_streams_live_points_and_ends_with_the_run(metrics_file):
    finished = threading.Event()

    def writer() -> None:
        for step in range(3):
            metrics.log(step=step, loss=2.0 - step * 0.5)
            time.sleep(0.05)
        finished.set()

    threading.Thread(target=writer, daemon=True).start()
    seen = [point["step"] for point in metrics.follow(metrics_file, lambda: not finished.is_set(), poll=0.02)]

    assert seen == [0, 1, 2]


def test_follow_leaves_a_torn_line_for_the_next_pass(metrics_file):
    with metrics_file.open("a") as handle:
        handle.write('{"step": 1, "loss": 2.0}\n{"step": 2, "lo')

    points = list(metrics.follow(metrics_file, lambda: False, poll=0.01))

    assert [point["step"] for point in points] == [1]


# -- the trainer callback ----------------------------------------------------


@pytest.fixture
def fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    module.TrainerCallback = type("TrainerCallback", (), {})
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def _state(step: int, epoch: float = 0.0):
    return types.SimpleNamespace(global_step=step, epoch=epoch)


def test_the_callback_logs_what_the_trainer_logs(metrics_file, fake_transformers):
    callback = metrics.trainer_callback()

    callback.on_log(None, _state(1), None, logs={"loss": 0.636, "learning_rate": 2e-4, "epoch": 0.01})

    (point,) = metrics.read_points(metrics_file)
    assert point["step"] == 1
    assert point["loss"] == 0.636
    assert point["learning_rate"] == 2e-4


def test_the_callback_computes_throughput_between_logs(metrics_file, fake_transformers):
    """The trainer only reports steps/second at the end; the trace needs it live."""
    callback = metrics.trainer_callback()

    callback.on_log(None, _state(10), None, logs={"loss": 1.0})
    time.sleep(0.05)
    callback.on_log(None, _state(20), None, logs={"loss": 0.9})

    first, second = metrics.read_points(metrics_file)
    assert "steps_per_second" not in first
    assert second["steps_per_second"] > 0


def test_the_callback_marks_epoch_boundaries(metrics_file, fake_transformers):
    callback = metrics.trainer_callback()
    callback.on_epoch_end(None, _state(30, epoch=1.0), None)

    (mark,) = metrics.events(metrics.read_points(metrics_file))
    assert mark == {"step": 30, "t": mark["t"], "event": "epoch_end"}


def test_without_transformers_the_error_names_the_alternative(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(NotFound, match="nawat.metrics.log"):
        metrics.trainer_callback()


# -- the sparkline -----------------------------------------------------------


def test_spark_draws_a_descending_loss():
    line = spark([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    assert line == "█▇▆▅▄▃▂▁"


def test_spark_buckets_long_series_to_width():
    assert len(spark([float(i) for i in range(1000)], width=40)) == 40


def test_spark_handles_flat_and_empty_series():
    assert spark([]) == ""
    assert spark([1.0, 1.0, 1.0]) == "▄▄▄"


# -- through the executor ----------------------------------------------------


LOGS_METRICS = """
import os, pathlib
import nawat.metrics as m
out = pathlib.Path(os.environ["NAWAT_OUT_DIR"])
(out / "adapter").mkdir(parents=True, exist_ok=True)
(out / "adapter" / "adapter_model.safetensors").write_bytes(b"w" * 1024)
for step in range(1, 6):
    m.log(step=step, loss=2.5 / step, lr=2e-4)
m.log(step=5, event="epoch_end", epoch=1.0)
print("trained with metrics")
"""


def test_a_run_writes_its_series_through_the_environment(executor, script):
    record = executor.runs.create(RunSpec(script=script(LOGS_METRICS)))
    finished = executor.execute(record.id)

    assert finished.state.value == "succeeded"
    points = metrics.read_points(executor.runs.metrics_path(record.id))
    grouped = metrics.series(points)
    assert [entry["value"] for entry in grouped["loss"]] == [2.5, 1.25, pytest.approx(0.8333, rel=1e-3), 0.625, 0.5]
    assert metrics.events(points)[0]["event"] == "epoch_end"


def test_the_series_is_published_with_the_run_record(executor, script, store):
    record = executor.runs.create(RunSpec(script=script(LOGS_METRICS)))
    executor.execute(record.id)

    published = store.list_prefix(Key.parse(f"runs/{record.id}/record"))
    assert metrics.FILENAME in published


def test_the_full_metric_history_survives_eviction_of_the_artifacts(executor, script, cache):
    """Phase 3's exit criterion, as an assertion."""
    record = executor.runs.create(RunSpec(script=script(LOGS_METRICS)))
    executor.execute(record.id)
    before = metrics.read_points(executor.runs.metrics_path(record.id))
    assert len(before) == 6

    # The adapter comes back and is evicted again; the cache ends the day empty.
    adapter = f"runs/{record.id}/adapter"
    cache.resolve(adapter, lease=False)
    cache.evict(adapter)
    assert cache.status().used == 0

    after = metrics.read_points(executor.runs.metrics_path(record.id))
    assert after == before, "the series must render identically with every artifact gone"


def test_a_failed_run_keeps_the_points_it_recorded(executor, script):
    failing = LOGS_METRICS + "\nimport sys; sys.exit(3)\n"
    record = executor.runs.create(RunSpec(script=script(failing)))
    finished = executor.execute(record.id)

    assert finished.state.value == "failed"
    assert len(metrics.read_points(executor.runs.metrics_path(record.id))) == 6
