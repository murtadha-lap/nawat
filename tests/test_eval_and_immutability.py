"""Guarantees: one fair source of measurements, and nothing edited after use.

Evals are blocks with fingerprints; comparisons only put numbers side by side
when they were measured the same way. Blocks and presets are never changed in
place once used, so an old experiment stays reproducible from its preset.
"""

from __future__ import annotations

import json

import pytest

from nawat.errors import Protected
from nawat.lab import report as rep
from nawat.lab.blocks import Library


def _fake_result(experiment, block, metrics):
    run = "r-" + experiment.name.split("/")[-1]
    experiment.add_attempt(run)
    path = experiment.path("eval", f"{block}-{run}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = experiment.data["eval_fingerprints"][block]
    path.write_text(json.dumps({"block": block, "fingerprint": fingerprint, "metrics": metrics}))


def test_eval_settings_change_the_fingerprint_training_settings_do_not(lab):
    base = lab.plan("base")
    lr = lab.plan("lr", sets={"train.max_steps": "9"})
    longer = lab.plan("longer", sets={"eval.heldout-generate.max_items": "50"})
    fp = lambda e: e.data["eval_fingerprints"]["heldout-generate"]  # noqa: E731
    assert fp(base) == fp(lr)
    assert fp(base) != fp(longer)


def test_heldout_evals_depend_on_the_data_split(lab):
    base = lab.plan("base")
    other_split = lab.plan("split", sets={"dataset.held": "6"})
    assert (base.data["eval_fingerprints"]["heldout-generate"]
            != other_split.data["eval_fingerprints"]["heldout-generate"])


def test_compare_shows_fair_numbers_and_withholds_the_rest(lab):
    a, b, c = lab.plan("a"), lab.plan("b", sets={"train.max_steps": "9"}), lab.plan(
        "c", sets={"eval.heldout-generate.max_items": "50"})
    _fake_result(a, "heldout-generate", {"accuracy": 0.5})
    _fake_result(b, "heldout-generate", {"accuracy": 0.7})
    _fake_result(c, "heldout-generate", {"accuracy": 0.9})

    fair = rep.compare([a, b])
    assert not fair["warnings"]
    assert ("heldout-generate · accuracy", [0.5, 0.7]) in fair["results"]
    assert [label for label, _ in fair["differences"]] == ["train.max_steps"]

    unfair = rep.compare([a, c])
    assert unfair["warnings"] and "measured differently" in unfair["warnings"][0]
    assert ("heldout-generate", ["≠ fingerprint", "≠ fingerprint"]) in unfair["results"]
    assert not any(label.startswith("heldout-generate ·") for label, _ in unfair["results"])

    forced = rep.compare([a, c], force=True)
    assert any(label.endswith("⚠") for label, _ in forced["results"])


def test_a_used_block_cannot_be_edited_in_place(lab):
    lab.plan()
    library = Library(lab.config)
    block = library.get("model/toy-model")
    assert library.is_locked(block)
    block.path.chmod(0o644)
    block.path.write_text(block.path.read_text().replace('P(0.5,', 'P(0.9,'))
    with pytest.raises(Protected, match="edited after"):
        lab.plan("after-edit")
    restored, created = library.save_as("model/toy-model", "toy-model-v2")
    assert "P(0.5," in restored.path.read_text()
    assert "P(0.9," in created.path.read_text()
    lab.plan("after-restore")  # the original is usable again


def test_blocks_and_presets_are_never_overwritten(lab):
    from nawat.lab.presets import Presets

    library = Library(lab.config)
    with pytest.raises(Protected):
        library.fork("dataset/grpo-prompts", "grpo-prompts")  # a built-in name
    library.fork("dataset/grpo-prompts", "my-prompts")
    with pytest.raises(Protected):
        library.fork("dataset/grpo-prompts", "my-prompts")
    presets = Presets(lab.config)
    with pytest.raises(Protected):
        presets.fork("toy", "toy")
    forked = presets.fork("toy", "toy-2", sets={"train.max_steps": "2"})
    assert forked.parent == "toy"


def test_frozen_code_is_what_runs(lab):
    """The experiment keeps its own copy of every block; the library can change without touching it."""
    experiment = lab.plan()
    frozen = experiment.dir / "code" / "dataset" / "toy-data.py"
    assert frozen.read_text() == (lab.ws / "blocks" / "dataset" / "toy-data.py").read_text()
    assert not frozen.stat().st_mode & 0o222  # read-only
