"""Nawāt — storage and orchestration core for a storage-constrained GPU host.

Inside a notebook or a training script the whole cache is one import (FR-3.2)::

    import nawat

    model = nawat.resolve("models/unsloth/Qwen2.5-VL-7B-Instruct")
    data  = nawat.resolve("datasets/ocr-arabic-v3")
    ...
    nawat.publish(out_dir, "runs/2026-07-28-a91f/adapter")

Both leases and the cache ceiling are handled underneath; the script never
names a byte count or deletes a file.

For a notebook that should also leave a run record — the provenance a submitted
script gets for free — open a run instead, and the staging, leasing, metric
series and verified publish happen around your cells::

    run = nawat.begin_run(
        model   = "models/unsloth/Qwen3.5-0.8B",
        dataset = "datasets/unsloth/LaTeX_OCR",
        params  = {"max_steps": 30},
    )
    model, tokenizer = FastVisionModel.from_pretrained(run.model_dir)
    ...
    model.save_pretrained(run.artifact_dir("adapter"))
    run.finish()

See :mod:`nawat.notebook`.

A run that fails is not a run that is lost. Checkpoints go to a durable
directory of their own, are kept when anything but success ends the run, and are
handed back to the next run of the same script and inputs::

    args = SFTConfig(**nawat.checkpoint_args(save_steps=250), ...)
    trainer.train(resume_from_checkpoint=nawat.resume_from())

Two lines, and a crash at hour 60 costs the last few minutes rather than the
whole run. See :mod:`nawat.checkpoints`.
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
    Run,
    artifact_dir,
    begin_run,
    checkpoint_args,
    checkpoint_dir,
    current_run,
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

__version__ = "0.2.0"

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
    """Upload, verify, then reclaim the local copy."""
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
    "Run",
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
    "current_run",
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
