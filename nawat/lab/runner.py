"""Running an approved experiment, inside the process the executor starts.

``run.py`` in every experiment folder calls :func:`main`. It checks the
approval gate again — the executor may be run by hand — loads the backbone and
blocks from the experiment's own ``code/`` snapshot, records the environment,
and hands control to the backbone. Everything it produces lands either in the
run's output directory (artifacts: adapter, merged, gguf …) or in the
experiment folder (traces, measurements, the effective configuration).
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .. import metrics as nawat_metrics
from ..errors import NawatError
from .blocks import LIST_SLOTS, load_module
from .experiment import Experiment
from .params import Params
from .trace import JsonlWriter, RolloutTracer, sample_callback

#: Recorded in env.json so a result can be tied to the exact library versions.
PACKAGES = (
    "nawat", "torch", "transformers", "trl", "peft", "accelerate", "datasets", "unsloth", "unsloth_zoo",
    "bitsandbytes", "xformers", "triton", "vllm", "sentence-transformers", "snac", "librosa", "torchaudio",
    "flash-linear-attention", "causal-conv1d", "tokenizers", "safetensors", "numpy",
)


class Block:
    """A loaded block as its functions see it: the module and its parameters."""

    def __init__(self, slot: str, name: str, module: ModuleType, params: Params) -> None:
        self.slot, self.name, self.module, self.p = slot, name, module, params

    def has(self, fn: str) -> bool:
        return callable(getattr(self.module, fn, None))

    def call(self, ctx: "Context", fn: str, *args, **kwargs):
        return getattr(self.module, fn)(ctx, self.p, *args, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.slot}/{self.name}>"


class Context:
    """What the backbone and every block receive as ``ctx``."""

    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment
        self.dir = experiment.dir
        self.name = experiment.name
        self.run_id = os.environ.get("NAWAT_RUN_ID", "local")
        #: Scratch space blocks share: a dataset block can leave probes or a
        #: formatting function here for the template and eval blocks.
        self.state: dict[str, Any] = {}
        self._writers: dict[str, JsonlWriter] = {}
        self._slots: dict[str, list[Block]] = {}
        data = experiment.data
        use_frozen_support(experiment)
        backbone_module = load_module(experiment.code_path(data["backbone"]), f"backbone/{data['backbone']['name']}")
        self.backbone = Block("backbone", data["backbone"]["name"], backbone_module,
                              Params(f"backbone/{data['backbone']['name']}", data["params"]["backbone"]))
        self.backbone.family = str(getattr(backbone_module, "BLOCK", {}).get("family") or data["backbone"]["name"])
        for slot, entries in data["blocks"].items():
            for entry in entries:
                key = f"{slot}.{entry['name']}" if slot in LIST_SLOTS else slot
                module = load_module(experiment.code_path(entry), f"{slot}/{entry['name']}")
                self._slots.setdefault(slot, []).append(
                    Block(slot, entry["name"], module, Params(f"{slot}/{entry['name']}", data["params"].get(key, {})))
                )

    # -- blocks ----------------------------------------------------------------

    def has(self, slot: str) -> bool:
        return bool(self._slots.get(slot))

    def block(self, slot: str) -> Block:
        blocks = self._slots.get(slot)
        if not blocks:
            raise NawatError(f"This experiment has no {slot} block.")
        return blocks[0]

    def blocks(self, slot: str) -> list[Block]:
        return list(self._slots.get(slot, ()))

    def call(self, slot: str, fn: str, *args, **kwargs):
        """Call ``fn`` on the slot's block; None when the slot is empty or the block lacks it."""
        blocks = self._slots.get(slot)
        if not blocks or not blocks[0].has(fn):
            return None
        return blocks[0].call(self, fn, *args, **kwargs)

    # -- inputs and outputs ----------------------------------------------------

    @property
    def model_dir(self) -> str:
        from .. import model_dir

        return model_dir()

    @property
    def dataset_dir(self) -> str:
        from .. import dataset_dir

        return dataset_dir()

    @property
    def dataset_dirs(self) -> list[str]:
        raw = os.environ.get("NAWAT_DATASET_DIRS")
        return json.loads(raw) if raw else ([self.dataset_dir] if os.environ.get("NAWAT_DATASET_DIR") else [])

    def input_dir(self, key: str) -> str:
        inputs = json.loads(os.environ.get("NAWAT_INPUTS", "{}"))
        if key not in inputs:
            raise NawatError(f"{key} is not an input of this experiment.", "Add it to the preset's inputs and plan again.")
        return inputs[key]

    def out(self, name: str) -> Path:
        """An artifact directory: published as runs/<experiment>/<name> when the run ends."""
        from .. import artifact_dir

        return artifact_dir(name)

    def checkpoint_args(self, save_steps: int, save_total_limit: int) -> dict[str, Any]:
        from .. import checkpoint_args

        return checkpoint_args(save_steps=save_steps, save_total_limit=save_total_limit)

    @property
    def resume_from(self) -> str | None:
        from .. import resume_from

        return resume_from()

    def path(self, *parts: str) -> Path:
        target = self.dir.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write_json(self, relpath: str, data: Any) -> Path:
        target = self.path(relpath)
        target.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False) + "\n")
        return target

    def log(self, message: str) -> None:
        print(f"[nawat] {message}", flush=True)

    # -- measurement -----------------------------------------------------------

    def metrics_callback(self):
        return nawat_metrics.trainer_callback()

    def log_metric(self, step: int | None = None, event: str | None = None, **values: Any) -> None:
        nawat_metrics.log(step=step, event=event, **values)

    def trace_writer(self, kind: str, max_mb: float = 256.0) -> JsonlWriter:
        if kind not in self._writers:
            self._writers[kind] = JsonlWriter(self.dir / "traces", f"{kind}-{self.run_id}", max_mb)
        return self._writers[kind]

    def rollout_tracer(self, *, num_generations: int, weights, scale_rewards: str, max_mb: float = 256.0) -> RolloutTracer:
        def on_round(summary: dict[str, Any]) -> None:
            fields = {k: v for k, v in summary.items() if k not in ("step", "round") and isinstance(v, (int, float))}
            nawat_metrics.log(step=summary["step"], event="rollout", **fields)

        return RolloutTracer(self.trace_writer("rollouts", max_mb), run_id=self.run_id, num_generations=num_generations,
                             weights=weights, scale_rewards=scale_rewards, on_round=on_round)

    def sample_callback(self, generate: Callable, probes: list[dict[str, Any]], every: int):
        return sample_callback(self, generate, probes, every)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()


def use_frozen_support(experiment: Experiment) -> None:
    """Make ``from nawat.lab import support`` mean the copy planned with this experiment."""
    entry = experiment.data.get("support")
    if not entry:
        return
    import nawat.lab as lab

    module = load_module(experiment.code_path(entry), "support")
    sys.modules["nawat.lab.support"] = module
    lab.support = module


def environment_snapshot() -> dict[str, Any]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    snap: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": versions,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("CUDA_", "PYTORCH_", "FLA_", "UNSLOTH_", "HF_HUB_OFFLINE"))},
    }
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            snap["gpus"] = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        import torch

        snap["torch_cuda"] = torch.version.cuda
        snap["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap["device"] = torch.cuda.get_device_name(0)
            snap["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        pass
    try:
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=60)
        if freeze.returncode == 0:
            snap["pip_freeze"] = freeze.stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    return snap


def main(run_py: str) -> None:
    experiment = Experiment.load(Path(run_py).resolve().parent)
    from ..config import Config

    try:
        config = Config.from_env()
    except Exception:  # noqa: BLE001 - verification is best-effort where no config is readable
        config = None
    experiment.require_approved(config)
    ctx = Context(experiment)
    ctx.log(f"experiment {ctx.name} · run {ctx.run_id} · backbone {ctx.backbone.name} · plan {experiment.code}")
    ctx.write_json(f"env-{ctx.run_id}.json", environment_snapshot())
    started = time.time()
    try:
        result = ctx.backbone.module.run(ctx, ctx.backbone.p) or {}
        ctx.write_json(f"result-{ctx.run_id}.json", {"status": "succeeded", "seconds": time.time() - started, **result})
    except BaseException as exc:
        ctx.write_json(f"result-{ctx.run_id}.json", {
            "status": "failed", "seconds": time.time() - started,
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        })
        raise
    finally:
        ctx.close()
