"""The whole managed path on CPU: plan → approve → run → trace → measure → publish → report.

Uses the real executor (a subprocess, as in production), the real reward
blocks, the real eval harness and a local object store.
"""

from __future__ import annotations

import json

from nawat.lab import report as rep
from nawat.lab.storage import walk
from nawat.runs import RunState


def test_an_approved_experiment_runs_and_keeps_everything(lab, executor_run):
    experiment = lab.plan("e2e")
    lab.approve(experiment)
    record, _ = executor_run(lab, experiment)
    assert record.state is RunState.SUCCEEDED, record.error

    # every completion of every round was traced, with each reward and the group advantage
    rollouts = [json.loads(line) for f in experiment.path("traces").glob("rollouts-*.jsonl")
                for line in f.read_text().splitlines()]
    assert len(rollouts) == 4 * 4
    assert set(rollouts[0]["rewards"]) == {"format_exact", "answer_match"}
    assert rollouts[0]["advantage"] is not None

    # measured by the eval block, with its fingerprint
    report = rep.build(experiment, lab.config)
    measured = report["eval"]["heldout-generate"]
    assert measured["fingerprint"] == experiment.data["eval_fingerprints"]["heldout-generate"]
    assert measured["metrics"]["n"] == 4
    assert report["status"] == "succeeded"
    assert report["approval_state"] == "verified"

    # the experiment folder: environment, effective outputs, the report
    assert any(experiment.dir.glob("env-*.json"))
    assert experiment.path("report.md").is_file()

    # and all of it in object storage, under one prefix
    keys = {obj.key for obj in walk(lab.cache.store, f"runs/{experiment.name}")}
    parts = {key.split("/")[3] for key in keys}
    assert {"adapter", "experiment", "traces", "record"} <= parts
