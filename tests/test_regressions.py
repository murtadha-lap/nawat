"""Regressions: bugs found in the review of 0.3.0b1, each pinned so it stays fixed."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import nawat
from nawat import cli
from nawat.errors import InvalidConfig, NotFound, Protected
from nawat.lab import experiment as E
from nawat.lab import report as rep
from nawat.lab.experiment import ApprovalRequired
from nawat.lab.storage import walk
from nawat.runs import RunState

REPO = Path(__file__).resolve().parents[1]


# -- the registry after a run -----------------------------------------------------


def test_registry_lists_artifacts_not_their_folders_after_a_run(lab, executor_run):
    experiment = lab.plan("registry")
    lab.approve(experiment)
    record, _ = executor_run(lab, experiment)
    assert record.state is RunState.SUCCEEDED, record.error
    uploaded = [o.key for o in walk(lab.cache.store, f"runs/{experiment.name}")]
    assert uploaded and not any("__pycache__" in key for key in uploaded)
    assert not (experiment.dir / "code" / "__pycache__").exists()
    keys = {str(k) for k in lab.cache.registry()}
    assert {f"runs/{experiment.name}/adapter", f"runs/{experiment.name}/experiment", "models/test/toy"} <= keys
    assert not any("/experiment/" in k for k in keys), keys  # code/, attempts/ … are contents, not artifacts
    assert cli.main(["registry"]) == 0


def test_bytecode_caches_are_never_artifact_content(tmp_path):
    from nawat.store import scan_local

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    assert list(scan_local(tmp_path)) == ["a.py"]


# -- commands that must work on their own --------------------------------------------


def test_estimate_does_not_need_the_api_extra(lab):
    code = ("import sys; sys.modules['fastapi'] = None; "
            "from nawat.cli import main; raise SystemExit(main(['estimate', '--params', '7B']))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ.copy())
    assert done.returncode in (0, 1), done.stderr  # 1 means "will not fit"; what matters is no ImportError
    assert "ImportError" not in done.stderr and "total" in done.stdout


def test_the_notebook_extra_brings_jupyterlab():
    import tomllib

    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    assert any(dep.startswith("jupyterlab") for dep in project["optional-dependencies"]["notebook"])


def test_bad_configuration_values_are_reported_not_tracebacks(lab, monkeypatch):
    from nawat.config import Config

    monkeypatch.setenv("NAWAT_TRANSFER_WORKERS", "eight")
    with pytest.raises(InvalidConfig, match="NAWAT_TRANSFER_WORKERS"):
        Config.from_env(env_file=None)
    monkeypatch.setenv("NAWAT_TRANSFER_WORKERS", "8")
    monkeypatch.setenv("NAWAT_CACHE_CEILING", "lots")
    with pytest.raises(InvalidConfig, match="NAWAT_CACHE_CEILING"):
        Config.from_env(env_file=None)
    monkeypatch.setenv("NAWAT_CACHE_CEILING", "1GB")
    assert cli.main(["--ceiling", "lots", "status"]) == InvalidConfig.exit_code


# -- the sandbox cannot start what it cannot verify ----------------------------------


def test_sandbox_refuses_an_approval_it_cannot_verify(lab, monkeypatch):
    experiment = lab.plan("sandboxed")
    # A hand-written approval: the right plan hash, no signature, and no key anywhere to check one with.
    (experiment.dir / "approval.json").write_text(json.dumps(
        {"plan_hash": experiment.plan_hash, "approved_at": 0, "approved_by": "agent", "host": "sandbox"}))
    assert not (lab.config.state_dir / E.KEY_FILE).exists()
    experiment.require_approved(lab.config)  # another machine without the key: unverified, allowed as before
    monkeypatch.setenv("NAWAT_SANDBOX", "1")
    with pytest.raises(ApprovalRequired, match="cannot be verified inside the sandbox"):
        experiment.require_approved(lab.config)
    monkeypatch.setenv(E.HOST_VERIFIED_ENV, "0" * 64)  # the host verified some other plan
    with pytest.raises(ApprovalRequired):
        experiment.require_approved(lab.config)
    monkeypatch.setenv(E.HOST_VERIFIED_ENV, experiment.plan_hash)  # what `nawat exp run` on the host sets
    experiment.require_approved(lab.config)


def test_host_launch_into_the_sandbox_carries_the_verified_hash(lab, monkeypatch):
    from nawat.lab import cli as lab_cli
    from nawat.lab import sandbox as sbx

    experiment = lab.plan("launch", uses={"runtime": "sandbox"})
    lab.approve(experiment)
    calls: list[list[str]] = []
    monkeypatch.setattr(sbx, "running", lambda name: True)
    monkeypatch.setattr(sbx, "docker", lambda: "docker")
    monkeypatch.setattr(sbx, "call", lambda argv, env=None: calls.append(argv) or 0)
    assert lab_cli.run_experiment(lab.cache, experiment) == 0
    assert calls and f"{E.HOST_VERIFIED_ENV}={experiment.plan_hash}" in calls[0]


def test_agent_rules_match():
    from nawat.lab.sandbox import AGENTS_MD

    assert "cannot start the run" in AGENTS_MD and "approval.json" in AGENTS_MD


# -- runs made elsewhere ---------------------------------------------------------------


def test_exp_logs_and_metrics_fall_back_to_the_attempt_folder(lab, capsys):
    """A run made in the sandbox is unknown to the host's run store, but its copy is in attempts/."""
    experiment = lab.plan("elsewhere")
    experiment.add_attempt("2026-11-02-dead")
    attempt = experiment.path("attempts", "2026-11-02-dead")
    attempt.mkdir(parents=True)
    (attempt / "run.log").write_text("hello from the sandbox\n")
    (attempt / "metrics.jsonl").write_text(json.dumps({"step": 1, "loss": 0.5}) + "\n")
    assert cli.main(["exp", "logs", experiment.name]) == 0
    assert "hello from the sandbox" in capsys.readouterr().out
    assert cli.main(["exp", "metrics", experiment.name]) == 0
    assert "loss" in capsys.readouterr().out
    assert cli.main(["exp", "cancel", experiment.name]) == NotFound.exit_code  # nothing here to stop


# -- Ctrl-C ----------------------------------------------------------------------------


def test_ctrl_c_cancels_the_run_instead_of_orphaning_it(lab, monkeypatch):
    from nawat.lab.cli import experiment_spec
    from nawat.process import process_alive

    experiment = lab.plan("interrupted")
    lab.approve(experiment)
    executor = cli._executor(lab.cache)
    spec = experiment_spec(experiment)
    executor.validate(spec)
    record = executor.runs.create(spec, None, name=experiment.name)
    experiment.add_attempt(record.id)

    def interrupted(run_id, **kwargs):
        deadline = time.time() + 30  # wait until the trainer process is up, then interrupt
        while time.time() < deadline:
            current = executor.runs.get(run_id)
            if current.pid is not None or current.state.terminal:
                break
            time.sleep(0.01)
        raise KeyboardInterrupt
        yield  # pragma: no cover - a generator, like follow_log

    monkeypatch.setattr(executor.runs, "follow_log", interrupted)
    assert cli._launch(executor, record, queue=False) == 130
    final = executor.runs.get(record.id)
    assert final.state is RunState.CANCELLED
    assert final.pid is None or not process_alive(final.pid)
    assert rep.status(E.Experiment.load(experiment.dir)) == "cancelled"  # finalised like any other run


def test_a_cancel_during_staging_never_starts_the_trainer(lab):
    from nawat.lab.cli import experiment_spec

    experiment = lab.plan("early")
    lab.approve(experiment)
    executor = cli._executor(lab.cache)
    spec = experiment_spec(experiment)
    executor.validate(spec)
    record = executor.runs.create(spec, None, name=experiment.name)
    executor._request_cancel(record.id)  # what `nawat cancel` does while the run is staging
    final = executor.execute(record.id)
    assert final.state is RunState.CANCELLED and final.pid is None and final.exit_code is None


# -- a live run in the report (issue #8) -----------------------------------------------------


def test_report_shows_a_live_run_as_running_not_interrupted(lab):
    from nawat.lab.cli import experiment_spec

    experiment = lab.plan("live")
    lab.approve(experiment)
    executor = cli._executor(lab.cache)
    record = executor.runs.create(experiment_spec(experiment), None, name=experiment.name)
    experiment.add_attempt(record.id)  # what `nawat exp run` does before the run starts
    loaded = lambda: E.Experiment.load(experiment.dir)  # noqa: E731

    assert rep.status(loaded()) == "queued"
    executor.runs.update(record.id, state=RunState.STAGING, started_at=time.time())
    assert rep.status(loaded(), lab.config) == "staging"
    executor.runs.update(record.id, state=RunState.RUNNING, pid=os.getpid())  # a trainer that is alive
    report = rep.build(loaded(), lab.config)
    assert report["status"] == "running"
    assert report["attempts"][0]["id"] == record.id and report["attempts"][0]["state"] == "running"
    assert "- **Status:** running" in rep.render(report)

    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    executor.runs.update(record.id, pid=gone.pid)  # its trainer died and nothing recorded the end
    assert rep.status(loaded(), lab.config) == "interrupted"


def test_report_still_says_interrupted_for_an_attempt_nobody_knows(lab):
    experiment = lab.plan("lost")
    experiment.add_attempt("2026-01-01-dead")  # no record.json, and no run of that id in the run store
    assert rep.status(E.Experiment.load(experiment.dir), lab.config) == "interrupted"


# -- smaller repairs ----------------------------------------------------------------------


def test_exp_rm_removes_the_experiment_folder_when_asked(lab):
    experiment = lab.plan("rm-local")
    assert cli.main(["exp", "rm", experiment.name, "--parts", "experiment", "--local", "--yes"]) == 0
    assert not experiment.dir.exists()


def test_save_preset_fails_before_an_experiment_is_planned(lab):
    from nawat.lab.experiment import list_experiments

    assert cli.main(["exp", "plan", "toy", "--name", "kept", "--save-preset", "toy"]) == Protected.exit_code
    assert not [e for e in list_experiments(lab.config) if e.name.endswith("/kept")]


def test_health_hides_details_without_the_token(lab, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nawat.api import create_app
    from nawat.config import Config

    monkeypatch.setenv("NAWAT_API_TOKEN", "secret")
    app = create_app(Config.from_env(env_file=None), start_queue=False, sweep_idle=False)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "version": nawat.__version__}
        detailed = client.get("/health", headers={"Authorization": "Bearer secret"}).json()
        assert "cache" in detailed and "gpu" in detailed
        assert client.get("/config").status_code == 401
        assert client.get("/config", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_readme_installs_what_the_sandbox_installs():
    from nawat.lab.sandbox import TRAINING_STACK

    def commands(text: str) -> set[str]:
        joined = re.sub(r"\\\n\s*", " ", text)
        return {" ".join(line.split()) for line in joined.splitlines() if line.strip().startswith("uv pip install")}

    missing = commands(TRAINING_STACK) - commands((REPO / "README.md").read_text())
    assert not missing, f"README.md's installation differs from the sandbox's training stack: {missing}"


def test_no_message_names_a_removed_command():
    stale = re.compile(r"nawat (submit|resume|scripts|serve status)\b")
    offenders = [str(p.relative_to(REPO)) for p in (REPO / "nawat").rglob("*.py") if stale.search(p.read_text())]
    offenders += [name for name in ("README.md", "docs/lab.md", ".env.example") if stale.search((REPO / name).read_text())]
    assert not offenders
