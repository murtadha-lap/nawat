"""Guarantee: no hidden parameters.

Every value a backbone or block uses is declared, shown in the plan and part of
the approval hash. Checked three ways: reading an undeclared parameter fails,
setting an unknown one fails, and the built-in library's code is scanned for
numbers written straight into calls.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nawat.lab import experiment as E
from nawat.lab.blocks import Library, library_root
from nawat.lab.params import P, ParamError, Params, resolve
from nawat.lab.presets import Presets

LIBRARY = library_root()
#: Call keywords that take a number for structural reasons (tensor axes), not as a setting.
STRUCTURAL_KEYWORDS = {"axis", "dim"}
#: Literal (non-parameter) values allowed in a trainer config: they switch off
#: integrations or describe data layout, they do not tune training.
STRUCTURAL_CONFIG = {"report_to", "log_completions", "remove_unused_columns", "label_names", "dataset_text_field",
                     "save_strategy", "output_dir", "fp16", "bf16"}


def library_files() -> list[Path]:
    return [p for p in sorted(LIBRARY.rglob("*.py")) if "_templates" not in p.parts]


def test_reading_an_undeclared_parameter_fails():
    p = Params("model/x", {"r": 16})
    assert p.r == 16
    with pytest.raises(ParamError, match="does not declare"):
        p.lora_alpha


def test_parameters_are_read_only():
    with pytest.raises(ParamError):
        Params("model/x", {"r": 16}).r = 32


def test_setting_an_unknown_parameter_fails():
    with pytest.raises(ParamError, match="Did you mean"):
        resolve("train", {"learning_rate": P(1e-4, "lr")}, [("--set", {"learnig_rate": 1})])


def test_values_are_coerced_to_their_declared_type():
    spec = P(16, "rank")
    assert spec.coerce("model.r", "32") == 32
    with pytest.raises(ParamError):
        spec.coerce("model.r", "lots")


def test_plan_rejects_unknown_settings(lab):
    with pytest.raises(ParamError):
        E.plan_experiment(lab.config, "toy", name="typo", sets={"train.max_stpes": "3"})


@pytest.mark.parametrize("path", library_files(), ids=lambda p: str(p.relative_to(LIBRARY)))
def test_no_numbers_hard_coded_in_calls(path):
    found = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            value = kw.value.operand if isinstance(kw.value, ast.UnaryOp) else kw.value
            if (isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
                    and not isinstance(value.value, bool) and kw.arg not in STRUCTURAL_KEYWORDS):
                found.append(f"line {node.lineno}: {kw.arg}={value.value}")
    assert not found, f"declare these in PARAMS instead: {found}"


@pytest.mark.parametrize("path", sorted((LIBRARY / "backbones").glob("*.py")), ids=lambda p: p.stem)
def test_trainer_config_values_come_from_parameters(path):
    """In every backbone's `values = dict(...)`, a literal is allowed only for structural fields."""
    offenders = []
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "values" for t in node.targets)
                and isinstance(node.value, ast.Call)):
            for kw in node.value.keywords:
                if kw.arg and isinstance(kw.value, ast.Constant) and kw.arg not in STRUCTURAL_CONFIG:
                    offenders.append(f"{kw.arg}={kw.value.value!r}")
    assert not offenders, f"{path.stem}: literal trainer settings must be parameters: {offenders}"


def test_every_builtin_block_loads_and_documents_its_parameters(lab):
    blocks = Library(lab.config).list()
    broken = [b.ref for b in blocks if not b.sha256]
    assert not broken, f"blocks that fail to import or miss their contract: {broken}"
    undocumented = [f"{b.ref}.{name}" for b in blocks if b.source == "builtin"
                    for name, spec in b.params.items() if not spec.help.strip()]
    assert not undocumented


def test_every_parameter_appears_in_the_plan(lab):
    experiment = lab.plan()
    shown = {(slot, name) for slot, name, *_ in E.param_rows(experiment)}
    for slot, values in experiment.data["params"].items():
        label = "train" if slot == "backbone" else slot
        for name in values:
            assert (label, name) in shown
    plan = experiment.path("plan.md").read_text()
    assert "## Every parameter" in plan


def test_every_builtin_preset_plans(lab):
    """All presets resolve against the library (llm-cpt has no default dataset by design)."""
    for preset in Presets(lab.config).list():
        if preset.source != "builtin":
            continue
        datasets = ["datasets/test/toy"] if not preset.datasets else None
        experiment = E.plan_experiment(lab.config, preset.name, name=f"check-{preset.name}", datasets=datasets)
        assert experiment.data["backbone"]["name"] == preset.backbone
