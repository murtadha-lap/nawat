"""Fixtures: an isolated Nawāt — its own workspace, cache and local object store.

Nothing here touches the real configuration, bucket or GPU. The executor runs
experiments as subprocesses, so the configuration travels through the
environment exactly as it does in production.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from nawat.cache import open_cache
from nawat.config import Config
from nawat.lab import experiment as E

#: A CPU stand-in for a training loop, in the GRPO family so the real reward blocks fit it.
FAKE_BACKBONE = '''
"""TEST ONLY: drives the real tracer, metrics, eval and export plumbing without a GPU."""

import random

from nawat.lab import P

BLOCK = {"kind": "backbone", "family": "grpo", "summary": "CPU stand-in for GRPO (tests)", "models": ("llm",),
         "slots": {"model": "required", "dataset": "required", "rewards": "required", "eval": "optional",
                   "export": "required", "runtime": "optional"}}
PARAMS = {"max_steps": P(4, "steps"), "num_generations": P(4, "group size"), "seed": P(3407, "seed")}


def run(ctx, p):
    from nawat.lab import support

    rng = random.Random(p.seed)
    model, tok = ctx.block("model").call(ctx, "load")
    data = ctx.block("dataset").call(ctx, "load", model, tok)
    ctx.state["tags"] = dict(support.DEFAULT_TAGS)
    funcs = []
    for b in ctx.blocks("rewards"):
        f = b.call(ctx, "reward")
        f.__name__ = b.name.replace("-", "_")
        funcs.append(f)
    tracer = ctx.rollout_tracer(num_generations=p.num_generations, weights=[b.p.weight for b in ctx.blocks("rewards")],
                                scale_rewards="group")
    traced = tracer.wrap(funcs)
    for step in range(1, p.max_steps + 1):
        row = data["train"][step % len(data["train"])]
        prompts = [row["prompt"]] * p.num_generations
        answers = [row["answer"]] * p.num_generations
        completions = [[{"role": "assistant", "content":
                         f"x<end_working_out><SOLUTION>{row['answer'] if rng.random() < 0.5 else 0}</SOLUTION>"}]
                       for _ in range(p.num_generations)]

        class State:
            global_step = step

        for f in traced:
            f(prompts=prompts, completions=completions, answer=answers, trainer_state=State)
        ctx.log_metric(step=step, loss=1.0 / step)
    return {"train": {"rollout_rounds": tracer.round}, **support.finish(ctx, model, tok, data)}
'''

FAKE_MODEL = '''
"""TEST ONLY: a model that answers from a lookup and saves a few bytes."""

from nawat.lab import P

BLOCK = {"kind": "model", "model_type": "llm", "summary": "toy model (tests)"}
PARAMS = {"skill": P(0.5, "share answered right")}


class Model:
    def save_pretrained(self, directory):
        import pathlib
        (pathlib.Path(directory) / "adapter_model.safetensors").write_bytes(b"0" * 64)


class Tok:
    def save_pretrained(self, directory):
        import pathlib
        (pathlib.Path(directory) / "tokenizer.json").write_text("{}")


def load(ctx, p):
    return Model(), Tok()


def add_adapters(ctx, p, model):
    return model


def generate(ctx, p, model, tokenizer, items, max_new_tokens=8, temperature=0.0, top_p=1.0, batch_size=8):
    return [f"<SOLUTION>{item['reference']}</SOLUTION>" if i % 2 == 0 else "<SOLUTION>0</SOLUTION>"
            for i, item in enumerate(items)]
'''

FAKE_DATASET = '''
"""TEST ONLY: prompts with answers, no `datasets` dependency."""

import json

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "toy prompts (tests)", "backbones": ("grpo",)}
PARAMS = {"held": P(4, "rows held out")}


def load(ctx, p, model, tokenizer):
    import pathlib

    rows = [json.loads(line) for line in (pathlib.Path(ctx.dataset_dir) / "train.jsonl").read_text().splitlines()]
    items = [{"prompt": [{"role": "user", "content": r["q"]}], "answer": r["a"]} for r in rows]
    held = items[: p.held]
    return {"train": items[p.held:], "eval": None, "probes": [],
            "heldout": [{"input": r["prompt"], "reference": r["answer"]} for r in held]}
'''

PRESET = '''
[preset]
backbone = "cpu-grpo"
model    = "models/test/toy"
dataset  = "datasets/test/toy"

[blocks]
model   = "toy-model"
dataset = "toy-data"
rewards = ["format-exact", "answer-match"]
eval    = ["heldout-generate"]
export  = ["adapter"]

[params."eval.heldout-generate"]
metric  = "number"
extract = "solution-tags"
'''


@pytest.fixture
def lab(tmp_path, monkeypatch):
    """A configured, empty Nawāt with a toy library and its inputs already in the store."""
    env = {
        "NAWAT_WORKSPACE": str(tmp_path / "ws"),
        "NAWAT_CACHE_ROOT": str(tmp_path / "cache"),
        "NAWAT_STORE_BACKEND": "local",
        "NAWAT_LOCAL_STORE_ROOT": str(tmp_path / "store"),
        "NAWAT_OFFLINE": "1",
        "NAWAT_CACHE_CEILING": "1GB",
        "NAWAT_MIN_FREE": "1MB",
    }
    for key in [k for k in list(__import__("os").environ) if k.startswith("NAWAT_") or k.startswith("UNSLOTH_")]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    (tmp_path / "empty.env").write_text("")  # never the real .env
    monkeypatch.setenv("NAWAT_ENV_FILE", str(tmp_path / "empty.env"))
    monkeypatch.chdir(tmp_path)

    ws = tmp_path / "ws" / "lab"
    for kind, name, body in (("backbones", "cpu-grpo", FAKE_BACKBONE), ("model", "toy-model", FAKE_MODEL),
                             ("dataset", "toy-data", FAKE_DATASET)):
        target = ws / "blocks" / kind / f"{name}.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(body))
    (ws / "presets").mkdir(parents=True, exist_ok=True)
    (ws / "presets" / "toy.toml").write_text(PRESET)

    config = Config.from_env(env_file=None)
    cache = open_cache(config)
    data = tmp_path / "seed" / "toy"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("\n".join(json.dumps({"q": f"{i}+{i}?", "a": str(2 * i)}) for i in range(12)))
    cache.publish(data, "datasets/test/toy", keep_local=True)
    model = tmp_path / "seed" / "model"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    cache.publish(model, "models/test/toy", keep_local=True)

    class Lab:
        pass

    lab = Lab()
    lab.config, lab.cache, lab.root, lab.ws = config, cache, tmp_path, ws
    lab.plan = lambda name="first", **kw: E.plan_experiment(config, "toy", name=name, store=cache.store, **kw)
    lab.approve = lambda exp: E.approve(exp, exp.code, config)
    return lab


def run_executor(lab, experiment):
    """Run an experiment through the real executor, as `nawat exp run` does."""
    from nawat.executor import Executor
    from nawat.lab.cli import experiment_spec
    from nawat.runs import RunStore

    executor = Executor(lab.cache, RunStore(lab.cache.db, lab.config.state_dir / "runs"))
    spec = experiment_spec(experiment)
    executor.validate(spec)
    record = executor.runs.create(spec, None, name=experiment.name)
    experiment.add_attempt(record.id)
    return executor.execute(record.id), executor


@pytest.fixture
def executor_run():
    return run_executor
