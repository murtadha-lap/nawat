"""Configuration, injected through the environment and never embedded.

Training scripts, the CLI and (later) the API all read the same variables, so a
script that runs under the platform runs unmodified outside it (PRD principle 4).
Credentials are held in fields excluded from ``repr`` and stripped by
:meth:`Config.redacted`, which is the only form allowed near a log or the UI
(NFR-4.1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Mapping

from .units import parse_size

_SECRET_FIELDS = frozenset({"access_key", "secret_key", "hub_token"})

DEFAULT_CACHE_ROOT = "~/nawat/cache"
DEFAULT_WORKSPACE = "~/nawat/workspace"
DEFAULT_CEILING = "120GB"
DEFAULT_MIN_FREE = "10GB"


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    """Everything the storage core needs to know about this host."""

    cache_root: Path
    workspace_root: Path
    state_dir: Path

    #: Local cache may not exceed this (FR-1.4).
    cache_ceiling: int
    #: Filesystem headroom kept free beneath the ceiling, for scratch and checkpoints.
    min_free: int

    store_backend: str = "s3"
    bucket: str = "nawat"
    store_prefix: str = ""
    endpoint: str | None = None
    region: str = "us-east-1"
    local_store_root: Path | None = None

    access_key: str | None = field(default=None, repr=False)
    secret_key: str | None = field(default=None, repr=False)
    hub_token: str | None = field(default=None, repr=False)

    #: No outbound access; an upstream fetch fails instead of reaching the internet.
    offline: bool = False

    transfer_workers: int = 8
    multipart_threshold: int = 64 * 10**6
    multipart_chunk: int = 32 * 10**6

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Config":
        env = os.environ if env is None else env
        cache_root = Path(env.get("NAWAT_CACHE_ROOT", DEFAULT_CACHE_ROOT)).expanduser().resolve()
        state_dir = Path(env.get("NAWAT_STATE_DIR", str(cache_root / ".nawat"))).expanduser().resolve()
        local_store = env.get("NAWAT_LOCAL_STORE_ROOT")
        return cls(
            cache_root=cache_root,
            workspace_root=Path(env.get("NAWAT_WORKSPACE", DEFAULT_WORKSPACE)).expanduser().resolve(),
            state_dir=state_dir,
            cache_ceiling=parse_size(env.get("NAWAT_CACHE_CEILING", DEFAULT_CEILING)),
            min_free=parse_size(env.get("NAWAT_MIN_FREE", DEFAULT_MIN_FREE)),
            store_backend=env.get("NAWAT_STORE_BACKEND", "s3").strip().lower(),
            bucket=env.get("NAWAT_S3_BUCKET", "nawat"),
            store_prefix=env.get("NAWAT_S3_PREFIX", "").strip("/"),
            endpoint=env.get("NAWAT_S3_ENDPOINT") or None,
            region=env.get("NAWAT_S3_REGION", "us-east-1"),
            local_store_root=Path(local_store).expanduser().resolve() if local_store else None,
            access_key=env.get("NAWAT_S3_ACCESS_KEY") or env.get("AWS_ACCESS_KEY_ID") or None,
            secret_key=env.get("NAWAT_S3_SECRET_KEY") or env.get("AWS_SECRET_ACCESS_KEY") or None,
            hub_token=env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or None,
            offline=_flag(env, "NAWAT_OFFLINE", False),
            transfer_workers=int(env.get("NAWAT_TRANSFER_WORKERS", "8")),
            multipart_threshold=parse_size(env.get("NAWAT_MULTIPART_THRESHOLD", "64MB")),
            multipart_chunk=parse_size(env.get("NAWAT_MULTIPART_CHUNK", "32MB")),
        )

    @property
    def database_path(self) -> Path:
        return self.state_dir / "nawat.sqlite3"

    @property
    def lock_path(self) -> Path:
        return self.state_dir / "cache.lock"

    @property
    def staging_root(self) -> Path:
        """Downloads land here and are renamed into place, so a partial transfer
        is never mistaken for a complete artifact (NFR-1.3)."""
        return self.cache_root / ".nawat-staging"

    def ensure_dirs(self) -> None:
        for path in (self.cache_root, self.state_dir, self.staging_root):
            path.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, object]:
        """The only representation safe to log, store in a run record, or show in the UI."""
        out: dict[str, object] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name in _SECRET_FIELDS:
                out[f.name] = "set" if value else None
            elif isinstance(value, Path):
                out[f.name] = str(value)
            else:
                out[f.name] = value
        return out
