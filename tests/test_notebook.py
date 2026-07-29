"""Runs driven from a kernel, as a notebook drives them.

A notebook run has no subprocess, so the properties the executor gets from
process boundaries have to be re-established in-process: inputs held under
lease, outputs published only on success, the record durable either way. These
tests assert that a kernel run and a submitted run are the same run.
"""

from __future__ import annotations

import json
import os

import pytest

import nawat
from nawat import notebook
from nawat.errors import NotFound, Protected
from nawat.runs import RunState, RunStore

CACHE_CEILING = 50 * 10**6


@pytest.fixture(autouse=True)
def no_leftover_run():
    """A run left open by a test must not leak its leases or environment."""
    notebook._active = None
    yield
    run = notebook.current_run()
    if run is not None:
        if not run.finished:
            run.cancel()
        else:
            run.close()
    notebook._active = None


@pytest.fixture
def seeded(cache, cached):
    cached("models/unsloth/Qwen3.5-0.8B", 8192)
    cached("datasets/unsloth/LaTeX_OCR", 4096)
    return cache


@pytest.fixture
def run(seeded):
    return nawat.begin_run(
        model="models/unsloth/Qwen3.5-0.8B",
        dataset="datasets/unsloth/LaTeX_OCR",
        params={"max_steps": 30, "learning_rate": 2e-4},
        notes="LaTeX OCR baseline",
        cache=seeded,
        script="explore.ipynb",
    )


def store_for(cache) -> RunStore:
    return RunStore(cache.db, cache.config.state_dir / "runs")


# -- staging -----------------------------------------------------------------


def test_beginning_a_run_stages_the_inputs_and_answers_with_paths(run, seeded):
    assert os.path.isdir(run.model_dir)
    assert os.path.isdir(run.dataset_dir)
    assert run.model_dir == str(seeded.local_path("models/unsloth/Qwen3.5-0.8B"))
    assert run.out_dir.is_dir()
    assert run.state is RunState.RUNNING


def test_the_inputs_are_held_against_eviction_for_as_long_as_the_run_is_open(run, seeded):
    """The lease is the kernel's own pid, so a live notebook cannot lose its weights."""
    holders = seeded.leases.live_for(nawat.Key.parse("models/unsloth/Qwen3.5-0.8B"))
    assert [lease.pid for lease in holders] == [os.getpid()]
    assert seeded.get("models/unsloth/Qwen3.5-0.8B").leased

    run.finish()
    assert not seeded.leases.live_for(nawat.Key.parse("models/unsloth/Qwen3.5-0.8B"))


def test_a_missing_input_fails_the_record_rather_than_leaving_it_open(seeded):
    with pytest.raises(NotFound):
        nawat.begin_run(model="models/nowhere/at-all", cache=seeded, script="x.ipynb")
    assert notebook.current_run() is None
    record = store_for(seeded).list()[0]
    assert record.state is RunState.FAILED and record.error


def test_two_open_runs_in_one_kernel_are_refused(run, seeded):
    with pytest.raises(Protected):
        nawat.begin_run(model="models/unsloth/Qwen3.5-0.8B", cache=seeded, script="second.ipynb")


def test_a_run_without_a_model_says_so_rather_than_returning_nothing(seeded):
    with nawat.begin_run(cache=seeded, script="bare.ipynb") as bare:
        with pytest.raises(NotFound):
            bare.model_dir
        with pytest.raises(NotFound):
            bare.dataset_dir


# -- publishing --------------------------------------------------------------


def test_finishing_publishes_every_artifact_class_and_reclaims_the_disk(run, seeded):
    (run.artifact_dir("adapter") / "adapter_model.safetensors").write_bytes(b"w" * 2048)
    (run.artifact_dir("gguf") / "model-q4_k_m.gguf").write_bytes(b"q" * 1024)

    record = run.finish()

    assert record.state is RunState.SUCCEEDED
    assert sorted(str(k) for k in record.artifacts) == [
        f"runs/{run.id}/adapter",
        f"runs/{run.id}/gguf",
    ]
    # Published means verified in object storage and gone from local disk.
    for key in record.artifacts:
        assert seeded.store.exists(key)
        assert not seeded.is_present(key)


def test_loose_files_publish_under_one_class_as_they_do_for_a_script(run, seeded):
    (run.out_dir / "notes.txt").write_text("scratch thought")
    record = run.finish()
    assert [str(k) for k in record.artifacts] == [f"runs/{run.id}/output"]


def test_a_run_that_wrote_nothing_publishes_nothing_and_still_succeeds(run):
    record = run.finish()
    assert record.state is RunState.SUCCEEDED and record.artifacts == ()


def test_scratch_space_is_neither_published_nor_left_behind(run):
    scratch = run.scratch_dir("trainer")
    (scratch / "checkpoint-500.bin").write_bytes(b"c" * 4096)
    assert run.out_dir not in scratch.parents, "scratch must sit outside the published tree"

    record = run.finish()
    assert record.artifacts == ()
    assert not scratch.exists()


def test_finishing_twice_is_refused_rather_than_silently_republishing(run):
    run.finish()
    with pytest.raises(Protected):
        run.finish()


# -- failure -----------------------------------------------------------------


def test_a_failed_run_publishes_no_artifacts_but_keeps_its_record(run, seeded):
    (run.artifact_dir("adapter") / "half-written.safetensors").write_bytes(b"w" * 512)

    record = run.fail("CUDA out of memory")

    assert record.state is RunState.FAILED
    assert record.artifacts == ()
    assert "CUDA out of memory" in record.error
    assert not seeded.store.exists(nawat.Key.parse(f"runs/{run.id}/adapter"))
    # The outputs stay on local disk for inspection, as they do for a script.
    assert (run.out_dir / "adapter" / "half-written.safetensors").exists()
    assert store_for(seeded).record_path(run.id).exists()


def test_the_record_and_log_reach_object_storage_even_when_the_run_failed(run, seeded):
    run.log("about to explode")
    run.fail("exploded")
    assert seeded.store.exists(nawat.Key.parse(f"runs/{run.id}/record"))


def test_failing_releases_the_inputs(run, seeded):
    run.fail("no good")
    assert not seeded.leases.live_for(nawat.Key.parse("models/unsloth/Qwen3.5-0.8B"))


def test_a_crashed_kernel_leaves_a_run_that_reconciles_to_failed(run, seeded):
    """The pid on the record is the kernel's, so recovery can tell it died."""
    record = store_for(seeded).get(run.id)
    assert record.pid == os.getpid()
    recovered = store_for(seeded).reconcile(is_alive=lambda pid: False)
    assert [r.id for r in recovered] == [run.id]
    assert store_for(seeded).get(run.id).state is RunState.FAILED


def test_a_run_ended_elsewhere_can_still_be_let_go_of(run, seeded):
    """`nawat cancel` from another shell, or a reconciliation: the kernel is
    holding leases against a run the platform already considers over."""
    store_for(seeded).update(run.id, state=RunState.CANCELLED, error="Cancelled elsewhere.")
    assert run.finished
    with pytest.raises(Protected):
        run.finish()

    run.close()

    assert not seeded.leases.live_for(nawat.Key.parse("models/unsloth/Qwen3.5-0.8B"))
    assert "HF_HUB_OFFLINE" not in os.environ
    assert notebook.current_run() is None
    run.close()  # idempotent


# -- the context manager -----------------------------------------------------


def test_the_context_manager_publishes_on_the_way_out(seeded):
    with nawat.begin_run(model="models/unsloth/Qwen3.5-0.8B", cache=seeded, script="w.ipynb") as run:
        (run.artifact_dir("adapter") / "w.safetensors").write_bytes(b"w" * 128)
        run_id = run.id
    assert store_for(seeded).get(run_id).state is RunState.SUCCEEDED
    assert seeded.store.exists(nawat.Key.parse(f"runs/{run_id}/adapter"))


def test_an_exception_in_the_block_fails_the_run_and_still_raises(seeded):
    with pytest.raises(ZeroDivisionError):
        with nawat.begin_run(model="models/unsloth/Qwen3.5-0.8B", cache=seeded, script="w.ipynb") as run:
            run_id = run.id
            1 / 0
    record = store_for(seeded).get(run_id)
    assert record.state is RunState.FAILED
    assert "ZeroDivisionError" in record.error


# -- metrics -----------------------------------------------------------------


def test_metrics_logged_in_the_kernel_land_in_the_runs_durable_series(run, seeded):
    nawat.metrics.log(step=1, loss=2.31)
    nawat.metrics.log(step=2, loss=1.87, lr=1.9e-4)
    nawat.metrics.log(step=2, event="epoch_end")

    series = nawat.trace(run.id, cache=seeded)
    assert [point["value"] for point in series["loss"]] == [2.31, 1.87]
    assert series["lr"][0]["value"] == pytest.approx(1.9e-4)

    run.finish()
    # The series outlives the artifacts: it lives beside the log, not in the cache.
    assert nawat.trace(run.id, cache=seeded)["loss"]


def test_the_metric_series_is_addressed_by_the_run_not_the_working_directory(run):
    assert os.environ["NAWAT_METRICS_PATH"] == str(run.metrics_path)
    assert run.metrics_path.parent == run.runs.directory(run.id)


# -- the environment ---------------------------------------------------------


def test_an_open_run_sets_what_a_submitted_script_would_have_been_given(run):
    """Cells copied from a training script read the environment; it is there."""
    assert os.environ["NAWAT_RUN_ID"] == run.id
    assert os.environ["NAWAT_MODEL_DIR"] == run.model_dir
    assert os.environ["NAWAT_DATASET_DIR"] == run.dataset_dir
    assert os.environ["NAWAT_OUT_DIR"] == str(run.out_dir)
    assert json.loads(os.environ["NAWAT_PARAMS"])["max_steps"] == "30"
    assert os.environ["NAWAT_PARAM_MAX_STEPS"] == "30"


def test_the_hub_is_switched_off_while_a_run_is_open_and_restored_after(run):
    """An undeclared download would be an input nothing accounted for."""
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    run.finish()
    assert "HF_HUB_OFFLINE" not in os.environ


def test_the_hub_switch_can_be_declined_for_a_kernel_that_needs_the_internet(seeded):
    with nawat.begin_run(cache=seeded, script="online.ipynb", offline=False):
        assert "HF_HUB_OFFLINE" not in os.environ


def test_nawats_own_fetch_is_exempt_from_the_switch_it_sets(run, seeded, hub, cache):
    """Staging must still work: it is the one accounted-for download path."""
    from nawat.hub import sanctioned_download

    os.environ["HF_HUB_OFFLINE"] = "1"
    with sanctioned_download():
        assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["HF_HUB_OFFLINE"] == "1"


# -- accessors that work in both worlds --------------------------------------


def test_the_accessors_answer_from_the_open_run(run):
    assert nawat.model_dir() == run.model_dir
    assert nawat.dataset_dir() == run.dataset_dir
    assert nawat.out_dir() == run.out_dir
    assert nawat.run_id() == run.id
    assert nawat.artifact_dir("adapter") == run.out_dir / "adapter"


def test_the_accessors_answer_from_the_environment_under_the_executor(monkeypatch, tmp_path):
    """The same cell, in a script the executor is running: no kernel, no run."""
    monkeypatch.setattr(notebook, "_active", None)
    monkeypatch.setenv("NAWAT_MODEL_DIR", "/staged/model")
    monkeypatch.setenv("NAWAT_DATASET_DIR", "/staged/data")
    monkeypatch.setenv("NAWAT_OUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("NAWAT_RUN_ID", "2026-07-28-a91f")
    monkeypatch.setenv("NAWAT_PARAM_MAX_STEPS", "60")

    assert nawat.model_dir() == "/staged/model"
    assert nawat.dataset_dir() == "/staged/data"
    assert nawat.out_dir() == tmp_path / "out"
    assert nawat.run_id() == "2026-07-28-a91f"
    assert nawat.param("max_steps", 30) == 60
    assert nawat.artifact_dir("adapter").is_dir()


def test_the_accessors_say_what_to_do_when_there_is_no_run_at_all(monkeypatch):
    monkeypatch.setattr(notebook, "_active", None)
    for name in ("NAWAT_MODEL_DIR", "NAWAT_OUT_DIR"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(NotFound) as caught:
        nawat.model_dir()
    assert "begin_run" in caught.value.remedy


# -- parameters --------------------------------------------------------------


def test_parameters_arrive_typed_from_the_default_written_beside_them(run):
    assert run.param("max_steps", 5) == 30
    assert isinstance(run.param("max_steps", 5), int)
    assert run.param("learning_rate", 1e-5) == pytest.approx(2e-4)
    assert run.param("unset_one", "fallback") == "fallback"


def test_a_boolean_parameter_reads_the_way_a_submit_flag_writes_it(seeded):
    with nawat.begin_run(cache=seeded, script="p.ipynb", params={"merge": "true", "quiet": "0"}) as run:
        assert run.param("merge", False) is True
        assert run.param("quiet", True) is False


# -- history -----------------------------------------------------------------


def test_a_kernel_run_is_an_ordinary_run_in_the_history(run, seeded):
    (run.artifact_dir("adapter") / "a.bin").write_bytes(b"a" * 64)
    run.finish()

    recorded = nawat.run_record(run.id, cache=seeded)
    assert recorded.spec.script == "explore.ipynb"
    assert recorded.spec.notes == "LaTeX OCR baseline"
    assert str(recorded.spec.model) == "models/unsloth/Qwen3.5-0.8B"
    assert dict(recorded.spec.params) == {"max_steps": "30", "learning_rate": "0.0002"}
    assert [r.id for r in nawat.history(cache=seeded)] == [run.id]


def test_the_notebook_itself_is_archived_beside_the_log_when_it_can_be_found(seeded, tmp_path):
    source = tmp_path / "explore.ipynb"
    source.write_text('{"cells": []}')
    with nawat.begin_run(cache=seeded, script=str(source)) as run:
        pass
    assert (store_for(seeded).directory(run.id) / "source-explore.ipynb").exists()
