"""Guarantee: nothing runs until a person approved exactly this plan.

The approval covers the frozen code, every parameter and the inputs; it needs
the right code, a terminal, the host (not the sandbox), and it is signed so a
hand-written approval.json is caught.
"""

from __future__ import annotations

import json

import pytest

from nawat import cli
from nawat.errors import Protected
from nawat.lab import experiment as E
from nawat.lab.experiment import ApprovalRequired


def test_wrong_code_is_refused(lab):
    experiment = lab.plan()
    with pytest.raises(ApprovalRequired):
        E.approve(experiment, "ZZZZZZ", lab.config)
    assert not experiment.approved


def test_right_code_approves_and_signs(lab):
    experiment = lab.plan()
    lab.approve(experiment)
    experiment.require_approved(lab.config)
    assert E.approval_state(lab.config, experiment.approval) == "verified"


def test_no_approval_inside_the_sandbox(lab, monkeypatch):
    experiment = lab.plan()
    monkeypatch.setenv("NAWAT_SANDBOX", "1")
    with pytest.raises(Protected, match="sandbox"):
        E.approve(experiment, experiment.code, lab.config)


def test_cli_approval_needs_a_terminal(lab, monkeypatch):
    experiment = lab.plan()
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert cli.main(["exp", "approve", experiment.name]) == Protected.exit_code
    assert not experiment.approved


def test_forged_approval_is_refused(lab):
    experiment = lab.plan()
    lab.approve(experiment)  # creates the host key
    approval = json.loads((experiment.dir / "approval.json").read_text())
    approval["signature"] = "0" * 64
    (experiment.dir / "approval.json").write_text(json.dumps(approval))
    with pytest.raises(ApprovalRequired, match="not signed"):
        experiment.require_approved(lab.config)


def test_unsigned_approval_is_refused_where_the_key_exists(lab):
    experiment = lab.plan()
    lab.approve(experiment)
    approval = json.loads((experiment.dir / "approval.json").read_text())
    approval.pop("signature")
    (experiment.dir / "approval.json").write_text(json.dumps(approval))
    with pytest.raises(ApprovalRequired):
        experiment.require_approved(lab.config)


def test_editing_frozen_code_voids_the_plan(lab):
    experiment = lab.plan()
    lab.approve(experiment)
    target = experiment.dir / "code" / "backbone" / "cpu-grpo.py"
    target.chmod(0o644)
    target.write_text(target.read_text() + "\n# a quiet change\n")
    with pytest.raises(Protected, match="differs from the plan"):
        experiment.require_approved(lab.config)


def test_editing_a_parameter_voids_the_approval(lab):
    experiment = lab.plan()
    lab.approve(experiment)
    data = json.loads((experiment.dir / "experiment.json").read_text())
    data["params"]["backbone"]["max_steps"] = 999
    (experiment.dir / "experiment.json").write_text(json.dumps(data))
    with pytest.raises(Protected, match="changed after it was planned"):
        E.Experiment.load(experiment.dir).require_approved(lab.config)


def test_approval_key_lives_outside_the_workspace(lab):
    experiment = lab.plan()
    lab.approve(experiment)
    key = lab.config.state_dir / E.KEY_FILE
    assert key.is_file()
    assert lab.config.workspace_root not in key.parents
    assert oct(key.stat().st_mode & 0o777) == "0o600"
