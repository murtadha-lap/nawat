"""Sweeps — one preset, several values, one approval.

    nawat sweep plan llm-grpo --model models/unsloth/Qwen3-4B-Base,models/unsloth/Qwen3-1.7B-Base \\
        --vary train.learning_rate=1e-6,5e-6

plans the four experiments of that grid, each a normal experiment with its own
folder and plan, filed under the preset's group. Approving the sweep approves
all of them at once: the code shown is the head of a hash over every plan in it,
so the table a person reads is exactly what all of them run. ``nawat sweep run``
then trains them one after another (the GPU runs one at a time) or queues them
for the control plane.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import NotFound, Protected
from .blocks import lab_root
from .experiment import (APPROVAL, SANDBOX_ENV, ApprovalRequired, Experiment, _write_json, approval_key, find,
                         plan_experiment, sign)

FILE = "sweep.json"


def sweeps_root(config) -> Path:
    return lab_root(config) / "sweeps"


def split_values(text: str) -> list[str]:
    """``1e-4,2e-4`` → two values; commas inside [], {} or quotes stay put."""
    out, depth, quote, current = [], 0, None, []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
        elif ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    if current:
        out.append("".join(current).strip())
    return [v for v in out if v]


@dataclass
class Sweep:
    dir: Path
    data: dict[str, Any]

    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def combined_hash(self) -> str:
        return self.data["combined_hash"]

    @property
    def code(self) -> str:
        return self.combined_hash[:6].upper()

    def experiments(self, config) -> list[Experiment]:
        return [find(config, name) for name in self.data["experiments"]]


def plan_sweep(
    config,
    preset: str,
    *,
    vary: Mapping[str, str],
    models: Sequence[str] = (),
    sets: Mapping[str, Any] = {},
    uses: Mapping[str, str] = {},
    datasets: list[str] | None = None,
    notes: str = "",
    group: str | None = None,
    store=None,
) -> Sweep:
    axes: list[tuple[str, list[Any]]] = []
    if models:
        axes.append(("model", list(models)))
    for key, raw in vary.items():
        values = split_values(raw)
        if len(values) < 1:
            raise Protected(f"--vary {key} has no values.")
        axes.append((key, values))
    if not axes:
        raise Protected("A sweep needs something to vary.", "Use --vary train.learning_rate=1e-4,2e-4 and/or several --model keys.")
    sweep_id = f"{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:4]}"
    combos = list(itertools.product(*[values for _, values in axes]))
    names: list[str] = []
    planned: list[Experiment] = []
    for index, combo in enumerate(combos, 1):
        chosen = dict(zip([k for k, _ in axes], combo))
        model = chosen.pop("model", None)
        label = ", ".join(f"{k}={v}" for k, v in zip([k for k, _ in axes], combo))
        experiment = plan_experiment(
            config, preset, name=f"sweep-{sweep_id}-{index:02d}", group=group, sets={**sets, **chosen}, uses=uses,
            model=model, datasets=datasets, notes=(notes + " · " if notes else "") + f"sweep {sweep_id}: {label}",
            store=store, sweep=sweep_id,
        )
        experiment.data["sweep_values"] = {k: v for k, v in zip([k for k, _ in axes], combo)}
        _write_json(experiment.dir / "experiment.json", experiment.data)
        planned.append(experiment)
        names.append(experiment.name)
    combined = combined_hash(planned)
    directory = sweeps_root(config) / sweep_id
    directory.mkdir(parents=True)
    data = {"id": sweep_id, "preset": preset, "axes": [[k, v] for k, v in axes], "experiments": names,
            "combined_hash": combined, "created_at": time.time(), "notes": notes}
    _write_json(directory / FILE, data)
    return Sweep(directory, data)


def combined_hash(experiments: Sequence[Experiment]) -> str:
    material = sorted(e.recompute_hash() for e in experiments)
    return hashlib.sha256(json.dumps(material).encode()).hexdigest()


def load(config, sweep_id: str) -> Sweep:
    directory = sweeps_root(config) / sweep_id
    if not (directory / FILE).is_file():
        raise NotFound(f"No sweep {sweep_id!r}.", "List them with: nawat sweep ls")
    return Sweep(directory, json.loads((directory / FILE).read_text()))


def is_sweep(config, name: str) -> bool:
    return (sweeps_root(config) / name / FILE).is_file()


def list_sweeps(config) -> list[Sweep]:
    root = sweeps_root(config)
    if not root.is_dir():
        return []
    return sorted((Sweep(p.parent, json.loads(p.read_text())) for p in root.glob(f"*/{FILE}")),
                  key=lambda s: s.data.get("created_at", 0))


def approve(config, sweep: Sweep, typed: str) -> None:
    if os.environ.get(SANDBOX_ENV):
        raise Protected("Sweeps cannot be approved from inside the sandbox.", f"Approve it on the host: nawat exp approve {sweep.id}")
    experiments = sweep.experiments(config)
    current = combined_hash(experiments)
    if current != sweep.combined_hash:
        raise Protected(f"An experiment of sweep {sweep.id} changed after it was planned.", "Plan the sweep again.")
    if typed.strip().upper() != current[:6].upper():
        raise ApprovalRequired("That is not the approval code for this sweep.", "Nothing was approved.")
    import getpass
    import socket

    key = approval_key(config, create=True)
    for experiment in experiments:
        approval = {"plan_hash": experiment.recompute_hash(), "approved_at": time.time(), "approved_by": getpass.getuser(),
                    "host": socket.gethostname(), "sweep": sweep.id, "sweep_hash": current}
        approval["signature"] = sign(key, approval)
        _write_json(experiment.dir / APPROVAL, approval)
