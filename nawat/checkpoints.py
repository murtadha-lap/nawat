"""Trainer checkpoints — the one thing on this host that cannot be re-fetched.

Everything else in the cache is disposable because it can be brought back:
a model from the hub, a dataset from object storage, an adapter from the run
that made it. A half-trained model cannot. Bringing it back means spending the
GPU hours again, and on this host that is measured in days.

So checkpoints get the opposite treatment from artifacts. They are not a key,
not published, and not evictable; they are written to a durable directory of
their own, kept whenever a run does *not* succeed, and handed to the next run of
the same shape so training continues from the last step instead of the first.

They are grouped into **lineages**. The run id cannot be the grouping — the
whole point is to be picked up by a *later* run — so a lineage is "this script,
trained on these inputs"::

    <cache root>/checkpoints/<lineage>/checkpoint-9522/

The name is derived from the script and its staged inputs, which gives the
property that matters after a crash at hour 60: *resubmitting the same command
resumes it*, and submitting a different experiment does not collide with it.

The layout inside a lineage is the ``transformers`` one, unchanged, because the
trainer writes it directly: ``checkpoint-<global step>`` directories, each
holding ``trainer_state.json``. That file is written last, after the weights and
the optimizer state, so its presence is what distinguishes a checkpoint that can
be resumed from the torso of one that was interrupted mid-save.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .errors import InvalidKey
from .units import human_bytes

#: What ``transformers`` calls a checkpoint directory.
PREFIX = "checkpoint-"

#: And what it calls one it is still writing.
TEMP_PREFIX = "tmp-checkpoint-"

#: Written last inside a checkpoint, so it is the completeness signal.
STATE_FILE = "trainer_state.json"

#: Which run last wrote to a lineage. Informational; nothing depends on it.
OWNER_FILE = "lineage.json"

LINEAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_STEP = re.compile(r"^" + PREFIX + r"(\d+)$")


# -- policy -------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointPolicy:
    """What becomes of a run's checkpoints. Decided at submit, applied at the end.

    The defaults are the ones that lose the least: resume if there is something
    to resume from, and keep whatever a failed run leaves behind. Only success
    removes them, and only because a successful run has already published the
    outputs that make them redundant.
    """

    #: Empty derives one from the script and the staged inputs.
    lineage: str = ""
    #: Continue from the newest checkpoint of this lineage, if there is one.
    resume: bool = True
    #: Keep the checkpoints on local disk even after the run succeeds.
    keep: bool = False
    #: Replicate the last checkpoint to object storage when the run ends,
    #: whether it ended well or badly. On by default: the disk it sits on is
    #: the one thing here with no second copy.
    publish: bool = True

    def to_json(self) -> dict[str, object]:
        return {"lineage": self.lineage, "resume": self.resume, "keep": self.keep, "publish": self.publish}

    @classmethod
    def from_json(cls, data: object) -> "CheckpointPolicy":
        if not isinstance(data, dict):
            return cls()
        return cls(
            lineage=str(data.get("lineage") or ""),
            resume=bool(data.get("resume", True)),
            keep=bool(data.get("keep", False)),
            publish=bool(data.get("publish", True)),
        )


# -- what is on disk ----------------------------------------------------------


@dataclass(frozen=True)
class Checkpoint:
    """One saved training step."""

    path: Path
    step: int
    written_at: float
    bytes: int
    #: Whether ``trainer_state.json`` is there, i.e. whether it can be resumed.
    complete: bool

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def age(self) -> float:
        return max(0.0, time.time() - self.written_at)

    def describe(self) -> str:
        from .units import human_age

        state = "" if self.complete else ", incomplete"
        return f"{self.name} (step {self.step}, {human_bytes(self.bytes)}, {human_age(self.age)} old{state})"

    def to_json(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "step": self.step,
            "written_at": self.written_at,
            "bytes": self.bytes,
            "complete": self.complete,
        }


@dataclass(frozen=True)
class Lineage:
    """A checkpoint directory: everything one experiment can resume from."""

    name: str
    path: Path
    checkpoints: tuple[Checkpoint, ...] = ()
    run_id: str | None = None
    updated_at: float | None = None

    @property
    def latest(self) -> Checkpoint | None:
        """The newest checkpoint that can actually be resumed from."""
        usable = [c for c in self.checkpoints if c.complete]
        return usable[-1] if usable else None

    @property
    def partial(self) -> tuple[Checkpoint, ...]:
        return tuple(c for c in self.checkpoints if not c.complete)

    @property
    def bytes(self) -> int:
        return sum(c.bytes for c in self.checkpoints)

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": str(self.path),
            "bytes": self.bytes,
            "run_id": self.run_id,
            "updated_at": self.updated_at,
            "latest": self.latest.to_json() if self.latest else None,
            "checkpoints": [c.to_json() for c in self.checkpoints],
        }


# -- naming -------------------------------------------------------------------


def sanitize(text: str) -> str:
    cleaned = _UNSAFE.sub("-", text).strip("-._")
    return cleaned[:64] or "run"


def validate(name: str) -> str:
    if not LINEAGE.match(name):
        raise InvalidKey(
            f"{name!r} is not a usable checkpoint lineage.",
            "Use letters, digits, and . _ - only, starting with a letter or digit.",
        )
    return name


def lineage_name(script: str, inputs: Iterable[str] = ()) -> str:
    """A stable name for "this script trained on these inputs".

    The digest covers the inputs rather than the parameters, so that correcting
    a learning rate and resubmitting still resumes, while pointing the same
    script at a different corpus does not silently continue someone else's run.
    """
    stem = sanitize(Path(script).stem)
    material = "\n".join(sorted(str(item) for item in inputs))
    digest = hashlib.sha256(f"{Path(script).name}\n{material}".encode()).hexdigest()[:8]
    return validate(f"{stem}-{digest}")


def lineage_path(root: Path, name: str) -> Path:
    return root / validate(name)


def lineage_dir(config, name: str) -> Path:
    """Where a lineage lives. Created on demand, never inside the run's output."""
    return lineage_path(config.checkpoint_root, name)


# -- reading ------------------------------------------------------------------


def directory_bytes(path: Path) -> int:
    total = 0
    stack = [path]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
            except OSError:
                continue
    return total


def _step_from_name(name: str) -> int:
    match = _STEP.match(name)
    return int(match.group(1)) if match else 0


def _read_one(path: Path) -> Checkpoint:
    state = path / STATE_FILE
    step = _step_from_name(path.name)
    complete = False
    try:
        written_at = path.stat().st_mtime
    except OSError:
        written_at = 0.0
    try:
        data = json.loads(state.read_text())
        complete = "global_step" in data
        step = int(data.get("global_step", step))
        written_at = state.stat().st_mtime
    except (OSError, ValueError, TypeError):
        pass
    return Checkpoint(path=path, step=step, written_at=written_at, bytes=directory_bytes(path), complete=complete)


def read(directory: Path) -> list[Checkpoint]:
    """Every checkpoint in a lineage directory, oldest step first."""
    if not directory.is_dir():
        return []
    found: list[Checkpoint] = []
    try:
        entries = list(os.scandir(directory))
    except OSError:
        return []
    for entry in entries:
        if entry.is_dir(follow_symlinks=False) and entry.name.startswith(PREFIX):
            found.append(_read_one(Path(entry.path)))
    return sorted(found, key=lambda c: (c.step, c.written_at))


def latest(directory: Path) -> Checkpoint | None:
    """The newest checkpoint that can be resumed from, or None."""
    usable = [c for c in read(directory) if c.complete]
    return usable[-1] if usable else None


def read_lineage(directory: Path, name: str | None = None) -> Lineage:
    owner: dict[str, object] = {}
    try:
        owner = json.loads((directory / OWNER_FILE).read_text())
    except (OSError, ValueError):
        pass
    return Lineage(
        name=name or directory.name,
        path=directory,
        checkpoints=tuple(read(directory)),
        run_id=str(owner["run_id"]) if isinstance(owner.get("run_id"), str) else None,
        updated_at=float(owner["updated_at"]) if isinstance(owner.get("updated_at"), (int, float)) else None,
    )


def lineages(root: Path) -> list[Lineage]:
    """Every lineage under the checkpoint root, most recently written first."""
    if not root.is_dir():
        return []
    found: list[Lineage] = []
    try:
        entries = list(os.scandir(root))
    except OSError:
        return []
    for entry in entries:
        if not entry.is_dir(follow_symlinks=False) or entry.name.startswith("."):
            continue
        found.append(read_lineage(Path(entry.path)))
    return sorted(found, key=lambda l: l.updated_at or (l.latest.written_at if l.latest else 0.0), reverse=True)


def usage_bytes(root: Path) -> int:
    """Disk held by checkpoints. Reported by ``nawat status``, because disk that
    nothing in the cache will ever reclaim has to be visible somewhere."""
    return directory_bytes(root) if root.is_dir() else 0


def note_owner(directory: Path, run_id: str, script: str = "") -> None:
    """Record which run last wrote here, so a stale lineage can be identified."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "script": script, "updated_at": time.time()}
    try:
        (directory / OWNER_FILE).write_text(json.dumps(payload, indent=2) + "\n")
    except OSError:
        pass


# -- removal ------------------------------------------------------------------


def _remove(checkpoint: Checkpoint) -> int:
    shutil.rmtree(checkpoint.path, ignore_errors=True)
    return checkpoint.bytes


def sweep_partial(directory: Path) -> list[Checkpoint]:
    """Remove checkpoints that were interrupted mid-save — but never the last resort.

    A kill during a save leaves a directory with weights and no
    ``trainer_state.json``, which no trainer will resume from. It is still only
    swept when a complete checkpoint survives it: an unusable checkpoint is
    worth more than nothing at all, and deciding it is worthless is the user's
    call, not this function's.
    """
    found = read(directory)
    if not any(c.complete for c in found):
        return []
    removed = [c for c in found if not c.complete]
    for checkpoint in removed:
        _remove(checkpoint)
    for entry in _temporaries(directory):
        shutil.rmtree(entry, ignore_errors=True)
    return removed


def _temporaries(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    try:
        return [
            Path(entry.path)
            for entry in os.scandir(directory)
            if entry.is_dir(follow_symlinks=False) and entry.name.startswith(TEMP_PREFIX)
        ]
    except OSError:
        return []


def prune(directory: Path, keep: int = 1) -> list[Checkpoint]:
    """Keep the newest ``keep`` resumable checkpoints; remove the rest.

    Partial checkpoints go too, but only once something complete is being kept,
    for the reason in :func:`sweep_partial`.
    """
    keep = max(0, keep)
    found = read(directory)
    complete = [c for c in found if c.complete]
    keeping = {c.path for c in (complete[-keep:] if keep else ())}
    if keep and not keeping:
        return []  # nothing resumable to keep, so nothing here is spare
    removed = [c for c in found if c.path not in keeping]
    for checkpoint in removed:
        _remove(checkpoint)
    if keeping:
        for entry in _temporaries(directory):
            shutil.rmtree(entry, ignore_errors=True)
    return removed


def discard(directory: Path) -> int:
    """Remove a lineage outright. Returns the bytes freed."""
    if not directory.is_dir():
        return 0
    freed = directory_bytes(directory)
    shutil.rmtree(directory, ignore_errors=True)
    return freed


# -- the rule -----------------------------------------------------------------


def settle(
    directory: Path,
    *,
    succeeded: bool,
    keep: bool,
    replicated: bool = False,
    log: Callable[[str], None] = lambda message: None,
) -> Checkpoint | None:
    """Apply the keep-on-failure rule once a run is over, and say what happened.

    Returns the newest checkpoint that survived, which is what the run record
    stores and what the next run of this lineage will resume from.

    Checkpoints are removed from local disk in exactly one case — the run
    succeeded and did not ask for them to be kept — because that is the only
    case in which the thing they protect against has already been ruled out:
    the outputs are published and verified in object storage.
    """
    for checkpoint in sweep_partial(directory):
        log(f"discarded {checkpoint.name}, which was interrupted before it finished being written")
    newest = latest(directory)
    if succeeded and not keep:
        freed = discard(directory)
        if freed:
            where = "it is in object storage" if replicated else "the run's outputs are published"
            log(f"checkpoints removed from local disk, freeing {human_bytes(freed)} — {where}")
        return None
    if newest is None:
        return None
    if succeeded:
        log(f"checkpoints kept at {newest.path.parent} — {newest.describe()}")
    else:
        log(f"checkpoint kept: {newest.path} — nothing up to step {newest.step} was lost")
    return newest


__all__ = [
    "Checkpoint",
    "CheckpointPolicy",
    "Lineage",
    "OWNER_FILE",
    "PREFIX",
    "STATE_FILE",
    "directory_bytes",
    "discard",
    "latest",
    "lineage_dir",
    "lineage_name",
    "lineages",
    "note_owner",
    "prune",
    "read",
    "read_lineage",
    "sanitize",
    "settle",
    "sweep_partial",
    "usage_bytes",
    "validate",
]
