"""Traces — every output the model produced while it trained.

Two kinds, both plain JSONL so an agent can read them with no special tooling:

``traces/rollouts-<run>-NNN.jsonl``
    GRPO / DAPO. One line per completion of every generation round: the prompt,
    the completion, the score from each reward function, the weighted total, its
    advantage within its group, and the step it belongs to. Collected by wrapping
    the reward functions, which see exactly what the trainer scored — so the
    trace is the training signal, not a re-generation of it.

``traces/samples-<run>-NNN.jsonl``
    Every other method. Every ``trace.sample_every`` steps the model answers a
    fixed set of probe inputs, so how its outputs change across training can be
    read step by step.

Files rotate at ``max_mb`` so no single file becomes unreadable. Everything is
kept; ``nawat exp traces <name> --clear`` deletes them, locally and in object
storage, when they are no longer wanted.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


class JsonlWriter:
    def __init__(self, directory: Path, stem: str, max_mb: float = 256.0) -> None:
        self.directory = directory
        self.stem = stem
        self.max_bytes = int(max_mb * 1e6)
        self.index = 0
        self.count = 0
        self._handle = None
        self._lock = threading.Lock()
        directory.mkdir(parents=True, exist_ok=True)
        while self._path(self.index).exists():
            self.index += 1

    def _path(self, index: int) -> Path:
        return self.directory / f"{self.stem}-{index:03d}.jsonl"

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=_jsonable) + "\n"
        with self._lock:
            if self._handle is None:
                self._handle = self._path(self.index).open("a", encoding="utf-8")
            elif self._handle.tell() + len(line) > self.max_bytes:
                self._handle.close()
                self.index += 1
                self._handle = self._path(self.index).open("a", encoding="utf-8")
            self._handle.write(line)
            self.count += 1

    def flush(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


def _jsonable(value: Any) -> Any:
    for attr in ("tolist", "item"):
        if hasattr(value, attr):
            try:
                return getattr(value, attr)()
            except Exception:  # noqa: BLE001
                pass
    if hasattr(value, "size") and hasattr(value, "mode"):  # a PIL image
        return f"<image {getattr(value, 'size', '')}>"
    return repr(value)


def text_of(value: Any) -> Any:
    """A completion as text: the assistant message content for conversational data."""
    if isinstance(value, list) and value and isinstance(value[-1], dict) and "content" in value[-1]:
        content = value[-1]["content"]
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content
    return value


class RolloutTracer:
    """Wraps GRPO reward functions so every scored completion is written down."""

    #: Reward kwargs that are machinery rather than data.
    SKIP = frozenset({"trainer_state", "completion_ids", "prompts", "completions", "prompt", "completion"})

    def __init__(
        self,
        writer: JsonlWriter,
        *,
        run_id: str,
        num_generations: int,
        weights: Sequence[float],
        scale_rewards: str = "group",
        on_round: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.writer = writer
        self.run_id = run_id
        self.G = max(1, int(num_generations))
        self.weights = list(weights)
        self.scale = scale_rewards
        self.on_round = on_round
        self.step = 0
        self.round = 0
        self._pending: dict[int, dict[str, Any]] = {}
        self._names: list[str] = []

    def wrap(self, funcs: Sequence[Callable]) -> list[Callable]:
        self._names = [getattr(f, "__name__", f"reward_{i}") for i, f in enumerate(funcs)]
        if len(self.weights) != len(funcs):
            self.weights = [1.0] * len(funcs)
        return [self._wrap_one(i, f) for i, f in enumerate(funcs)]

    def _wrap_one(self, index: int, func: Callable) -> Callable:
        name = self._names[index]

        def traced(prompts=None, completions=None, **kwargs):
            scores = func(prompts=prompts, completions=completions, **kwargs)
            try:
                self._collect(index, prompts, completions, scores, kwargs)
            except Exception as exc:  # noqa: BLE001 - tracing must never break training
                print(f"[nawat] rollout trace skipped a batch: {type(exc).__name__}: {exc}", flush=True)
            return scores

        traced.__name__ = name
        traced.__qualname__ = name
        traced.__doc__ = func.__doc__
        return traced

    def _collect(self, index, prompts, completions, scores, kwargs) -> None:
        key = id(completions)
        batch = self._pending.setdefault(key, {"prompts": prompts, "completions": completions, "scores": {}, "kwargs": kwargs})
        batch["scores"][index] = list(scores) if scores is not None else [None] * len(completions or [])
        state = kwargs.get("trainer_state")
        if state is not None and getattr(state, "global_step", None) is not None:
            self.step = int(state.global_step)
        if len(batch["scores"]) == len(self._names):
            self._pending.pop(key, None)
            self._flush(batch)

    def _flush(self, batch: dict[str, Any]) -> None:
        prompts, completions = batch["prompts"] or [], batch["completions"] or []
        n = len(completions)
        per_func = [batch["scores"][i] for i in range(len(self._names))]
        totals: list[float | None] = []
        for j in range(n):
            vals = [per_func[i][j] for i in range(len(per_func))]
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
                present = [(w, v) for w, v in zip(self.weights, vals) if v is not None and not (isinstance(v, float) and math.isnan(v))]
                totals.append(sum(w * v for w, v in present) if present else None)
            else:
                totals.append(sum(w * float(v) for w, v in zip(self.weights, vals)))
        advantages = self._advantages(totals)
        extras = {k: v for k, v in batch["kwargs"].items() if k not in self.SKIP}
        now = time.time()
        for j in range(n):
            self.writer.write({
                "run": self.run_id,
                "step": self.step,
                "round": self.round,
                "group": j // self.G,
                "member": j % self.G,
                "prompt": prompts[j] if j < len(prompts) else None,
                "completion": text_of(completions[j]),
                "rewards": {name: per_func[i][j] for i, name in enumerate(self._names)},
                "total": totals[j],
                "advantage": advantages[j],
                "extra": {k: (v[j] if isinstance(v, (list, tuple)) and len(v) == n else v) for k, v in extras.items()},
                "t": now,
            })
        self.writer.flush()
        if self.on_round is not None:
            valid = [t for t in totals if t is not None]
            self.on_round({
                "step": self.step,
                "round": self.round,
                "completions": n,
                "reward_mean": sum(valid) / len(valid) if valid else None,
                "reward_max": max(valid) if valid else None,
                **{f"reward_{name}": _mean(per_func[i]) for i, name in enumerate(self._names)},
            })
        self.round += 1

    def _advantages(self, totals: list[float | None]) -> list[float | None]:
        out: list[float | None] = [None] * len(totals)
        valid_all = [t for t in totals if t is not None]
        batch_std = _std(valid_all)
        for start in range(0, len(totals), self.G):
            group = totals[start:start + self.G]
            valid = [t for t in group if t is not None]
            if not valid:
                continue
            mean = sum(valid) / len(valid)
            std = {"group": _std(valid), "batch": batch_std}.get(str(self.scale), None)
            for k, t in enumerate(group):
                if t is None:
                    continue
                out[start + k] = (t - mean) / (std + 1e-4) if std is not None else t - mean
        return out


def _mean(values: Sequence[Any]) -> float | None:
    valid = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(valid) / len(valid) if valid else None


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def sample_callback(ctx, generate: Callable[[list[dict[str, Any]]], list[str]], probes: list[dict[str, Any]], every: int):
    """A trainer callback that answers the probes every ``every`` steps and at the end."""
    from transformers import TrainerCallback

    writer = ctx.trace_writer("samples")

    def record(step: int, event: str) -> None:
        try:
            outputs = generate(probes)
        except Exception as exc:  # noqa: BLE001 - a failed sample must not stop training
            print(f"[nawat] sample trace failed at step {step}: {type(exc).__name__}: {exc}", flush=True)
            return
        now = time.time()
        for i, (probe, output) in enumerate(zip(probes, outputs)):
            writer.write({"run": ctx.run_id, "step": step, "event": event, "probe": i,
                          "input": probe.get("input"), "reference": probe.get("reference"),
                          "output": output, "t": now})
        writer.flush()

    class SampleTrace(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            record(state.global_step, "begin")

        def on_step_end(self, args, state, control, **kwargs):
            if every > 0 and state.global_step % every == 0:
                record(state.global_step, "step")

        def on_train_end(self, args, state, control, **kwargs):
            record(state.global_step, "end")

    return SampleTrace()


def read(directory: Path, kind: str | None = None) -> Iterator[dict[str, Any]]:
    """Every trace record under an experiment's traces folder, file by file."""
    pattern = f"{kind}-*.jsonl" if kind else "*.jsonl"
    for path in sorted(directory.glob(pattern)):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except ValueError:
                        continue
