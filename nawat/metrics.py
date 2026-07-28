"""Step-level training metrics.

The contract with the training script is one JSONL file, named by
``NAWAT_METRICS_PATH``: the script appends a line per logging step, the platform
tails it live (FR-7.1), and the file itself is the durable series (FR-7.6). It
lives beside ``run.log`` in the state directory — never in the cache — so it
survives eviction of every artifact the run produced, and it is published to
object storage with the run record.

A line is a JSON object. Numeric fields are metrics; ``step`` and ``t`` place
the point; an optional ``event`` string marks run events — epoch boundaries,
LR phase changes — for trace annotation (FR-7.4). Anything else is dropped
rather than stored, so a chart never meets a string.

Three ways to write it, in increasing convenience:

    echo '{"step": 12, "loss": 0.63}' >> "$NAWAT_METRICS_PATH"   # any language

    import nawat.metrics
    nawat.metrics.log(step=12, loss=0.63, lr=2e-4)               # any script

    trainer = SFTTrainer(..., callbacks=[nawat.metrics.trainer_callback()])
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from .errors import NotFound

ENV_VAR = "NAWAT_METRICS_PATH"
FILENAME = "metrics.jsonl"

#: Keys that place a point rather than measure something.
_POSITIONAL = ("step", "t", "event")


def metrics_path() -> Path:
    """Where this process should write metrics.

    Under the platform the executor sets the variable; outside it, a file in
    the working directory — the script runs standalone either way (PRD
    principle 4).
    """
    return Path(os.environ.get(ENV_VAR) or FILENAME)


def _numeric(fields: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            continue
        out[name] = float(value)
    return out


def log(step: int | None = None, event: str | None = None, **fields: Any) -> dict[str, Any]:
    """Append one point. Returns what was written."""
    point: dict[str, Any] = {"t": time.time()}
    if step is not None:
        point["step"] = int(step)
    if event:
        point["event"] = str(event)
    point.update(_numeric(fields))
    path = metrics_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(point) + "\n")  # one write per line: appends interleave, not tear
    return point


# -- reading ------------------------------------------------------------------


def _parse(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        point = json.loads(line)
    except ValueError:
        return None
    return point if isinstance(point, dict) else None


def read_points(path: Path) -> list[dict[str, Any]]:
    """Every point in the file, in write order. Bad lines are skipped, not fatal."""
    if not path.exists():
        return []
    points = []
    for line in path.read_text(errors="replace").splitlines():
        point = _parse(line)
        if point is not None:
            points.append(point)
    return points


def series(points: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Points regrouped per metric name, each carrying its step and time."""
    out: dict[str, list[dict[str, Any]]] = {}
    for index, point in enumerate(points):
        step = point.get("step", index)
        for name, value in point.items():
            if name in _POSITIONAL or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            out.setdefault(name, []).append({"step": step, "t": point.get("t"), "value": float(value)})
    return out


def events(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run events, for annotating the trace."""
    return [
        {"step": point.get("step", index), "t": point.get("t"), "event": point["event"]}
        for index, point in enumerate(points)
        if point.get("event")
    ]


def follow(
    path: Path,
    still_running: Callable[[], bool],
    *,
    poll: float = 0.25,
) -> Iterator[dict[str, Any]]:
    """Yield points as they are appended, ending when the run does.

    Tails the file rather than subscribing in memory, so a late reader sees the
    whole series and a reader in another process works identically — the same
    shape as log following.
    """
    position = 0
    draining = False
    while True:
        if path.exists():
            with path.open(errors="replace") as handle:
                handle.seek(position)
                chunk = handle.read()
            # Only complete lines: a writer mid-append is left for the next pass.
            end = chunk.rfind("\n")
            if end >= 0:
                for line in chunk[: end + 1].splitlines():
                    point = _parse(line)
                    if point is not None:
                        yield point
                position += end + 1
        if draining:
            return
        if not still_running():
            draining = True  # one final read for anything written as it exited
            continue
        time.sleep(poll)


# -- the trainer integration --------------------------------------------------


def trainer_callback():
    """A ``transformers`` TrainerCallback that logs every step the trainer logs.

    Captures whatever the trainer reports — loss, learning rate, grad norm —
    and computes steps/second between logs, which the trainer itself only
    reports at the end (FR-7.1). Epoch boundaries are recorded as events.
    """
    try:
        from transformers import TrainerCallback
    except ImportError as exc:
        raise NotFound(
            "transformers is not installed, so no trainer callback can be built.",
            "Install it in the training environment, or call nawat.metrics.log() directly.",
        ) from exc

    class NawatMetricsCallback(TrainerCallback):
        def __init__(self) -> None:
            self._last: tuple[float, int] | None = None

        def on_log(self, args, state, control, logs=None, **kwargs) -> None:
            fields = _numeric(logs or {})
            now = time.time()
            step = int(getattr(state, "global_step", 0) or 0)
            if self._last is not None and step > self._last[1]:
                elapsed = now - self._last[0]
                if elapsed > 0:
                    fields.setdefault("steps_per_second", (step - self._last[1]) / elapsed)
            self._last = (now, step)
            log(step=step, **fields)

        def on_epoch_end(self, args, state, control, **kwargs) -> None:
            log(step=int(getattr(state, "global_step", 0) or 0), event="epoch_end",
                epoch=float(getattr(state, "epoch", 0.0) or 0.0))

    return NawatMetricsCallback()


__all__ = [
    "ENV_VAR",
    "FILENAME",
    "events",
    "follow",
    "log",
    "metrics_path",
    "read_points",
    "series",
    "trainer_callback",
]
