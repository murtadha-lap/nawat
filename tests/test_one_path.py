"""Guarantee: an experiment is the only way to train.

An agent (or anyone) must not be able to turn ad-hoc training code into a
recorded run: no command runs an arbitrary script, the executor refuses
anything but an approved experiment, notebooks cannot open runs, and runs/ in
object storage only takes what experiments produce.
"""

from __future__ import annotations

import importlib.util

import pytest

from nawat import cli
from nawat.errors import Protected
from nawat.lab.cli import experiment_spec
from nawat.runs import RunSpec

BYPASS_COMMANDS = {"submit", "resume", "hold", "scripts", "eval"}


def _commands() -> set[str]:
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if a.__class__.__name__ == "_SubParsersAction")
    return set(sub.choices)


def test_no_command_runs_an_arbitrary_script():
    assert not (BYPASS_COMMANDS & _commands())
    assert {"exp", "preset", "block", "sweep"} <= _commands()


def test_the_ad_hoc_evaluator_is_gone():
    assert importlib.util.find_spec("nawat.evaluate") is None


def test_executor_refuses_a_plain_script(lab):
    from nawat.executor import Executor
    from nawat.runs import RunStore

    script = lab.config.workspace_root / "train.py"
    script.write_text("print('my own training loop')\n")
    executor = Executor(lab.cache, RunStore(lab.cache.db, lab.config.state_dir / "runs"))
    with pytest.raises(Protected, match="not an experiment"):
        executor.validate(RunSpec(script="train.py"))


def test_executor_refuses_an_unapproved_experiment(lab):
    from nawat.executor import Executor
    from nawat.lab.experiment import ApprovalRequired
    from nawat.runs import RunStore

    experiment = lab.plan()
    executor = Executor(lab.cache, RunStore(lab.cache.db, lab.config.state_dir / "runs"))
    with pytest.raises(ApprovalRequired):
        executor.validate(experiment_spec(experiment))
    lab.approve(experiment)
    executor.validate(experiment_spec(experiment))  # now accepted


def test_executor_rechecks_approval_when_the_run_starts(lab):
    """A run queued while approved must not start once the approval is gone."""
    from nawat.executor import Executor
    from nawat.runs import RunState, RunStore

    experiment = lab.plan()
    lab.approve(experiment)
    executor = Executor(lab.cache, RunStore(lab.cache.db, lab.config.state_dir / "runs"))
    spec = experiment_spec(experiment)
    executor.validate(spec)
    record = executor.runs.create(spec, None, name=experiment.name)
    (experiment.dir / "approval.json").unlink()
    finished = executor.execute(record.id)
    assert finished.state is RunState.FAILED
    assert "not been approved" in (finished.error or "")
    assert finished.started_at is None or finished.exit_code is None  # the script never ran


def test_notebooks_cannot_open_runs():
    import nawat

    with pytest.raises(Protected, match="only as experiments"):
        nawat.begin_run(model="models/x/y", dataset="datasets/x/y")


def test_runs_prefix_is_reserved_for_experiments(lab, tmp_path):
    import nawat

    source = tmp_path / "adapter"
    source.mkdir()
    (source / "w.bin").write_bytes(b"1")
    with pytest.raises(Protected, match="only experiments write"):
        nawat.publish(source, "runs/mine/adapter")
    assert cli.main(["publish", str(source), "runs/mine/adapter"]) == Protected.exit_code
    assert cli.main(["add", str(source), "runs/mine/adapter"]) == Protected.exit_code


def test_http_api_queues_experiments_not_scripts(lab):
    pytest.importorskip("fastapi")
    from nawat.api import create_app

    app = create_app(lab.config, lab.cache, start_queue=False, sweep_idle=False)
    routes = {(m, r.path) for r in app.routes for m in getattr(r, "methods", ())}
    assert ("POST", "/runs") not in routes
    assert ("GET", "/scripts") not in routes
    assert not any(path.endswith("/evaluate") for _, path in routes)
    assert ("POST", "/experiments/{name:path}/run") in routes


def test_agent_rules_say_so():
    from nawat.lab.sandbox import AGENTS_MD

    assert "exactly one way to train" in AGENTS_MD
