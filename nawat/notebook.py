"""Reading runs from a notebook, and the accessors a backbone or block uses.

Training is recorded only as experiments, so a kernel cannot open a run of its
own: :func:`begin_run` refuses and says what to do instead. What a notebook
*is* for is looking — at data staged with ``nawat.resolve()``, and at results::

    import nawat

    nawat.history()                      # recent runs, like `nawat runs`
    nawat.run_record("2026-11-02-ab12")  # one run: inputs, outcome, artifacts
    nawat.trace("2026-11-02-ab12")       # its metric series, ready to plot

The accessors — :func:`model_dir`, :func:`dataset_dir`, :func:`out_dir`,
:func:`param`, :func:`checkpoint_dir`, :func:`resume_from` — read what the
executor put in the environment, so the code of a backbone or block sees its
inputs without knowing anything else about the platform.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from . import checkpoints as checkpoint_store
from . import metrics
from .cache import Cache
from .errors import InvalidKey, NotFound, Protected
from .executor import param_env_name
from .runs import RunRecord, RunStore


def begin_run(*args: Any, **kwargs: Any) -> None:
    """Recorded training is experiments only; a notebook cannot open a run of its own.

    A run opened from a kernel would train whatever code the cells hold —
    outside the fixed backbones, without declared parameters, without a
    person's approval — and still be filed as a result. So it refuses, and says
    what to do instead. A notebook remains the place to look at data and
    results: ``nawat.resolve()`` stages inputs read-only, and ``nawat exp
    report`` / ``nawat exp traces`` read experiments back.
    """
    raise Protected(
        "Training runs are recorded only as experiments; nawat.begin_run() is not available.",
        "Plan one from a preset (nawat preset ls; nawat exp plan <preset>), approve it at a terminal, then nawat exp run.",
    )


# -- what the executor set --------------------------------------------------------


def _required(name: str, what: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise NotFound(
            f"{name} is not set, so there is no {what} here.",
            "This code runs inside an experiment, where `nawat exp run` sets it.",
        )
    return value


def model_dir() -> str:
    """The staged model directory."""
    return _required("NAWAT_MODEL_DIR", "staged model")


def dataset_dir(index: int = 0) -> str:
    """A staged dataset directory; the first one by default."""
    if index == 0 and os.environ.get("NAWAT_DATASET_DIR"):
        return os.environ["NAWAT_DATASET_DIR"]
    listed = os.environ.get("NAWAT_DATASET_DIRS")
    if listed:
        return json.loads(listed)[index]
    return _required("NAWAT_DATASET_DIR", "staged dataset")


def out_dir() -> Path:
    """Where artifacts are written to be published."""
    return Path(_required("NAWAT_OUT_DIR", "output directory"))


def artifact_dir(name: str = "adapter") -> Path:
    """A directory that publishes as ``runs/<experiment>/<name>``.

    ``adapter``, ``merged`` and ``gguf`` are the conventional names; any other
    is equally valid and publishes under whatever it is called.
    """
    if not name or "/" in name or name.startswith("."):
        raise InvalidKey(
            f"{name!r} is not usable as an artifact class.",
            "Use a plain directory name such as adapter, merged or gguf.",
        )
    path = out_dir() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_dir() -> Path:
    """Where to write trainer checkpoints so that a failed run is not a lost one.

    Durable and shared with the next run of the same experiment, which is what
    makes resuming possible at all. Answers outside the platform too — code run
    by hand still gets a real directory.
    """
    value = os.environ.get("NAWAT_CHECKPOINT_DIR")
    if value:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path
    from . import default_cache

    config = default_cache().config
    lineage = os.environ.get("NAWAT_CHECKPOINT_LINEAGE") or checkpoint_store.lineage_name(sys.argv[0] or "adhoc")
    path = checkpoint_store.lineage_dir(config, lineage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resume_from() -> str | None:
    """The checkpoint to continue from, or None to start at step 0.

        trainer.train(resume_from_checkpoint=nawat.resume_from())
    """
    value = os.environ.get("NAWAT_RESUME_FROM")
    if value:
        return value
    if "NAWAT_RESUME_FROM" in os.environ:
        return None  # the executor looked and found nothing; do not look again
    newest = checkpoint_store.latest(checkpoint_dir())
    return str(newest.path) if newest else None


def checkpoint_args(save_steps: int = 250, save_total_limit: int = 3, **overrides: Any) -> dict[str, Any]:
    """Trainer arguments that make a long run survivable.

        args = SFTConfig(**nawat.checkpoint_args(save_steps=250), ...)
        trainer.train(resume_from_checkpoint=nawat.resume_from())

    Saving every N *steps* rather than once an epoch is the whole difference:
    an epoch over a large corpus can be a day wide, and a crash lands between
    two of them. ``save_total_limit`` bounds what that costs in disk.
    """
    args: dict[str, Any] = {
        "output_dir": str(checkpoint_dir()),
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
    }
    args.update(overrides)
    return args


def param(name: str, default: Any = None, cast: Callable[[str], Any] | None = None) -> Any:
    """A run parameter from the environment, falling back to the value written here."""
    raw = os.environ.get(param_env_name(name))
    return _coerce(raw, default, cast) if raw is not None else default


def run_id() -> str | None:
    """The id of the run this code is part of, if any."""
    return os.environ.get("NAWAT_RUN_ID") or None


# -- reading the history from a kernel ----------------------------------------


def _store(cache: Cache | None) -> RunStore:
    from . import default_cache

    cache = cache or default_cache()
    return RunStore(cache.db, cache.config.state_dir / "runs")


def history(limit: int = 20, cache: Cache | None = None) -> list[RunRecord]:
    """Recent runs, newest first — the notebook equivalent of ``nawat runs``."""
    return _store(cache).list(limit=limit)


def run_record(identifier: str | None = None, cache: Cache | None = None) -> RunRecord:
    """One run's record: what ran, what happened, what came out."""
    return _store(cache).get(_named(identifier))


def trace(identifier: str | None = None, cache: Cache | None = None) -> dict[str, list[dict[str, Any]]]:
    """A run's metric series, grouped by name, ready to plot.

    ``matplotlib.pyplot.plot([p["step"] for p in loss], [p["value"] for p in loss])``
    — the series is read from the durable file, so it plots identically long
    after the weights themselves have been reclaimed.
    """
    store = _store(cache)
    return metrics.series(metrics.read_points(store.metrics_path(_named(identifier))))


def _named(identifier: str | None) -> str:
    resolved = identifier or run_id()
    if resolved is None:
        raise NotFound("No run was named.", "Pass a run id; list them with: nawat runs")
    return resolved


# -- helpers ------------------------------------------------------------------


def _coerce(raw: str, default: Any, cast: Callable[[str], Any] | None) -> Any:
    if cast is not None:
        return cast(raw)
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    for kind in (int, float):
        if isinstance(default, kind):
            try:
                return kind(raw)
            except ValueError as exc:
                raise InvalidKey(
                    f"Parameter {raw!r} is not a {kind.__name__}.",
                    f"Pass a {kind.__name__} value.",
                ) from exc
    return raw


__all__ = [
    "artifact_dir",
    "begin_run",
    "checkpoint_args",
    "checkpoint_dir",
    "dataset_dir",
    "history",
    "resume_from",
    "model_dir",
    "out_dir",
    "param",
    "run_id",
    "run_record",
    "trace",
]
