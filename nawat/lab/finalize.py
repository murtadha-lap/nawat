"""After a run: gather everything into the experiment folder and put it in object storage.

Called by the executor when any run of an experiment's ``run.py`` ends —
success, failure or cancellation, from the CLI or the control-plane queue.

Object storage then holds the whole experiment under one prefix::

    runs/<group>/<experiment>/
        adapter/  merged/  gguf/     what the export blocks wrote
        checkpoint/                  the last trainer checkpoint
        record/                      run log, metrics, run.json of the latest attempt
        experiment/                  the experiment folder: plan, code, params, env, eval, report, attempts
        traces/                      every rollout and probe sample
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from ..keys import Key
from .experiment import FILE, Experiment
from .report import build, render

EXPERIMENT_CLASS = "experiment"
TRACES_CLASS = "traces"


def experiment_for(config, script: str) -> Experiment | None:
    path = Path(script).expanduser()
    if not path.is_absolute():
        path = Path(config.workspace_root) / path
    path = path.resolve()
    if path.name == "run.py" and (path.parent / FILE).is_file():
        return Experiment.load(path.parent)
    return None


def finalize(cache, runs, record, log: Callable[[str], None] = print) -> Experiment | None:
    experiment = experiment_for(cache.config, record.spec.script)
    if experiment is None:
        return None
    if record.id not in experiment.attempts():
        experiment.add_attempt(record.id)
    attempt = experiment.path("attempts", record.id)
    attempt.mkdir(parents=True, exist_ok=True)
    source = runs.directory(record.id)
    if source.is_dir():
        for entry in source.iterdir():
            if entry.is_file():
                shutil.copy2(entry, attempt / entry.name)
    (attempt / "record.json").write_text(json.dumps(record.to_json(), indent=2, default=str) + "\n")
    write_report(experiment)
    publish(cache, experiment, log)
    if record.state.value != "succeeded":
        log(f"run this experiment again (it continues from its last checkpoint): nawat exp run {experiment.name}")
    return experiment


def write_report(experiment: Experiment) -> None:
    report = build(experiment)
    experiment.path("report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    experiment.path("report.md").write_text(render(report))


def publish(cache, experiment: Experiment, log: Callable[[str], None] = print) -> list[str]:
    """Mirror the experiment folder (and, separately, its traces) into object storage."""
    store = cache.store
    prefix = f"runs/{experiment.name}"
    published: list[str] = []
    staging_root = Path(cache.config.staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(staging_root)) as scratch:
        copy = Path(scratch) / "experiment"
        shutil.copytree(experiment.dir, copy, ignore=shutil.ignore_patterns(TRACES_CLASS, "__pycache__"))
        key = Key.parse(f"{prefix}/{EXPERIMENT_CLASS}")
        store.delete_prefix(key)
        store.publish(copy, key)
        published.append(str(key))
        for path in copy.rglob("*"):  # read-only code/ files would otherwise survive the cleanup
            if path.is_file():
                path.chmod(0o644)
    traces = experiment.path(TRACES_CLASS)
    if traces.is_dir() and any(traces.iterdir()):
        key = Key.parse(f"{prefix}/{TRACES_CLASS}")
        store.publish(traces, key)
        published.append(str(key))
    log(f"experiment {experiment.name} published: {', '.join(published)}")
    return published
