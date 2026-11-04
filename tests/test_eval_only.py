"""Evaluating a model as stored, without training it (issue #9)."""

from __future__ import annotations

import json

from nawat.lab import experiment as E
from nawat.lab import report as rep
from nawat.lab import support
from nawat.runs import RunState

EVAL_PRESET = '''
[preset]
backbone = "eval-only"
model    = "models/test/toy"
dataset  = "datasets/test/toy"

[blocks]
model   = "toy-model"
dataset = "toy-data"
eval    = ["heldout-generate"]

[params."eval.heldout-generate"]
metric  = "number"
extract = "solution-tags"
'''


def test_eval_only_scores_the_stored_model_without_training(lab, executor_run):
    (lab.ws / "presets" / "toy-eval.toml").write_text(EVAL_PRESET)
    experiment = E.plan_experiment(lab.config, "toy-eval", name="baseline", store=lab.cache.store)
    assert experiment.data["backbone"]["trains"] is False
    assert "**Training:** none" in experiment.path("plan.md").read_text()
    lab.approve(experiment)

    record, _ = executor_run(lab, experiment)
    assert record.state is RunState.SUCCEEDED, record.error
    experiment = E.Experiment.load(experiment.dir)
    results = json.loads(next(experiment.dir.glob(f"eval/heldout-generate-{record.id}.json")).read_text())
    assert results["metrics"]["n"] == 4 and results["metrics"]["accuracy"] == 0.5  # the toy model, as stored
    assert not list(lab.config.checkpoint_root.rglob("checkpoint-*"))  # no optimizer step, nothing to resume

    report = rep.build(experiment, lab.config)
    assert report["trains"] is False and report["status"] == "succeeded"
    text = rep.render(report)
    assert "**Training:** none" in text and "## Training" not in text
    assert "## Evaluation" in text


def test_a_finished_run_and_a_base_model_compare_when_only_the_model_differs(lab):
    (lab.ws / "presets" / "toy-eval.toml").write_text(EVAL_PRESET)
    base = E.plan_experiment(lab.config, "toy-eval", name="base")
    tuned = E.plan_experiment(lab.config, "toy-eval", name="tuned", model="runs/toy/first/adapter")
    assert base.data["eval_fingerprints"] == tuned.data["eval_fingerprints"]
    result = rep.compare([base, tuned])
    assert not result["warnings"]
    assert ("model", ["models/test/toy", "runs/toy/first/adapter"]) in result["differences"]


def test_chat_template_kwargs_reach_the_template_and_the_fingerprint(lab):
    seen = {}

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.update(kwargs)
            return "prompt"

    text, templated = support.render_prompt(Tokenizer(), [{"role": "user", "content": "hi"}], {"enable_thinking": False})
    assert templated and text == "prompt"
    assert seen == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}

    direct = E.plan_experiment(lab.config, "llm-eval", name="direct")
    thinking = E.plan_experiment(lab.config, "llm-eval", name="thinking", sets={"model.chat_template_kwargs": {}})
    assert direct.data["params"]["model"]["chat_template_kwargs"] == {"enable_thinking": False}
    # A generation prompt with and without thinking is not the same measurement.
    assert direct.data["eval_fingerprints"]["gsm8k"] != thinking.data["eval_fingerprints"]["gsm8k"]
