"""Nawāt — storage and orchestration core for a storage-constrained GPU host.

Inside a notebook or a block the whole cache is one import (FR-3.2)::

    import nawat

    model = nawat.resolve("models/unsloth/Qwen3.5-0.8B")   # local cache → object storage → hub, once
    data  = nawat.resolve("datasets/unsloth/LaTeX_OCR")
    nawat.publish("./prepared", "datasets/my-lab/latex-ocr-v1")

Leases and the cache ceiling are handled underneath; nothing here names a byte
count or deletes a file by hand.

Training is recorded only as experiments — a fixed backbone, declared
parameters, a person's approval — planned and run from the CLI::

    nawat exp plan vlm-sft --set train.max_steps=30
    nawat exp approve vlm-sft/<name>
    nawat exp run vlm-sft/<name>

A run that fails is not a run that is lost: every backbone writes checkpoints
to a durable lineage, and running the experiment again continues from the last
one (see :mod:`nawat.checkpoints`). Results are read back with ``nawat exp
report`` or, from a notebook, :func:`history`, :func:`run_record` and
:func:`trace` (see :mod:`nawat.notebook`).
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

from . import checkpoints, metrics
from .cache import ArtifactStatus, Cache, CacheStatus, CollectResult, PublishResult, open_cache
from .checkpoints import Checkpoint, CheckpointPolicy
from .config import Config
from .errors import (
    InsufficientSpace,
    InvalidKey,
    NawatError,
    NotFound,
    Offline,
    Protected,
    StoreUnavailable,
    VerificationFailed,
)
from .keys import KINDS, Key
from .notebook import (
    artifact_dir,
    begin_run,
    checkpoint_args,
    checkpoint_dir,
    dataset_dir,
    history,
    model_dir,
    out_dir,
    param,
    resume_from,
    run_id,
    run_record,
    trace,
)
from .store import VerificationResult
from .units import human_bytes, parse_size

__version__ = "0.3.0b1"

_default: Cache | None = None
_default_lock = threading.Lock()


def default_cache() -> Cache:
    """The cache described by this process's environment, built once."""
    global _default
    with _default_lock:
        if _default is None:
            _default = open_cache()
        return _default


def use_cache(cache: Cache | None) -> None:
    """Replace the process-wide cache. Mainly for tests and embedding."""
    global _default
    with _default_lock:
        _default = cache


def resolve(key: str | Key, **kwargs) -> Path:
    """Make an artifact present locally and return its path."""
    return default_cache().resolve(key, **kwargs)


def holding(*keys: str | Key, **kwargs) -> Iterator[dict[str, Path]]:
    """Stage artifacts for the duration of a ``with`` block."""
    return default_cache().holding(*keys, **kwargs)


def publish(source: str | Path, key: str | Key, **kwargs) -> PublishResult:
    """Upload, verify, then reclaim the local copy. Not to runs/: only experiments write there."""
    from .keys import require_not_experiment_output

    require_not_experiment_output(key)
    return default_cache().publish(source, key, **kwargs)


def keep(key: str | Key) -> ArtifactStatus:
    """Keep on disk until released."""
    return default_cache().pin(key)


def release(key: str | Key) -> ArtifactStatus:
    """Stop keeping on disk."""
    return default_cache().unpin(key)


def free_space(need: int = 0, **kwargs) -> CollectResult:
    """Reclaim local disk down to the ceiling."""
    return default_cache().collect(need, **kwargs)


def status() -> CacheStatus:
    return default_cache().status()


def artifacts() -> list[ArtifactStatus]:
    return default_cache().list()


def verify(key: str | Key) -> VerificationResult:
    return default_cache().verify(key)


__all__ = [
    "ArtifactStatus",
    "Cache",
    "CacheStatus",
    "Checkpoint",
    "CheckpointPolicy",
    "CollectResult",
    "Config",
    "InsufficientSpace",
    "InvalidKey",
    "KINDS",
    "Key",
    "NawatError",
    "NotFound",
    "Offline",
    "Protected",
    "PublishResult",
    "StoreUnavailable",
    "VerificationFailed",
    "VerificationResult",
    "__version__",
    "artifact_dir",
    "artifacts",
    "begin_run",
    "checkpoint_args",
    "checkpoint_dir",
    "checkpoints",
    "dataset_dir",
    "default_cache",
    "free_space",
    "history",
    "holding",
    "human_bytes",
    "keep",
    "metrics",
    "model_dir",
    "open_cache",
    "out_dir",
    "param",
    "parse_size",
    "publish",
    "release",
    "resolve",
    "resume_from",
    "run_id",
    "run_record",
    "status",
    "trace",
    "use_cache",
    "verify",
]
