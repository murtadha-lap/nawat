"""Nawāt — storage and orchestration core for a storage-constrained GPU host.

Inside a notebook or a training script the whole cache is one import (FR-3.2)::

    import nawat

    model = nawat.resolve("models/unsloth/Qwen2.5-VL-7B-Instruct")
    data  = nawat.resolve("datasets/ocr-arabic-v3")
    ...
    nawat.publish(out_dir, "runs/2026-07-28-a91f/adapter")

Both leases and the cache ceiling are handled underneath; the script never
names a byte count or deletes a file.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

from .cache import ArtifactStatus, Cache, CacheStatus, CollectResult, PublishResult, open_cache
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
from .store import VerificationResult
from .units import human_bytes, parse_size

__version__ = "0.1.0"

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
    "artifacts",
    "default_cache",
    "free_space",
    "holding",
    "human_bytes",
    "keep",
    "open_cache",
    "parse_size",
    "publish",
    "release",
    "resolve",
    "status",
    "use_cache",
    "verify",
]
