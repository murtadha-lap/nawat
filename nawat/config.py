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

from .dotenv import find as find_dotenv
from .dotenv import load as load_dotenv
from .errors import NotFound
from .units import parse_size

_SECRET_FIELDS = frozenset({"access_key", "secret_key", "hub_token", "api_token"})

#: Distinguishes "discover a .env" from "use no .env at all" (None).
UNSET = object()

DEFAULT_CACHE_ROOT = "~/nawat/cache"
DEFAULT_WORKSPACE = "~/nawat/workspace"
DEFAULT_CEILING = "120GB"
DEFAULT_MIN_FREE = "10GB"


def _flag(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _merge_dotenv(base: Mapping[str, str], env_file) -> tuple[Mapping[str, str], Path | None]:
    """Layer a .env underneath the real environment, which always wins.

    ``NAWAT_ENV_FILE`` names a file explicitly; set it empty to use none. With
    it unset, the nearest .env above the working directory is used if there is
    one, and its absence is not an error.
    """
    if env_file is None:
        return base, None
    if env_file is UNSET:
        named = base.get("NAWAT_ENV_FILE")
        if named is not None:
            if named.strip() == "":
                return base, None
            path = Path(named).expanduser()
            if not path.is_file():
                raise NotFound(
                    f"NAWAT_ENV_FILE points at {path}, which does not exist.",
                    "Correct the path, or unset NAWAT_ENV_FILE to use the nearest .env.",
                )
        else:
            path = find_dotenv()
            if path is None:
                return base, None
    else:
        path = Path(env_file).expanduser()
        if not path.is_file():
            raise NotFound(f"{path} does not exist.", "Check the --env-file path.")
    return {**load_dotenv(path), **base}, path.resolve()


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

    #: The control-plane API. Bound to the local network only; external exposure
    #: is a separately configured reverse proxy (NFR-4.3).
    api_host: str = "127.0.0.1"
    api_port: int = 8080
    api_token: str | None = field(default=None, repr=False)

    #: The inference server this session manager supervises.
    serve_port: int = 8001
    serve_idle_timeout: float = 900.0
    serve_startup_timeout: float = 600.0
    serve_backend: str = "vllm"
    serve_extra_args: str = ""

    #: The .env these values were layered from, if any. Recorded so the CLI can
    #: show which file is in force — misconfiguration is usually the wrong file.
    env_file: Path | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, env_file=UNSET) -> "Config":
        env = os.environ if env is None else env
        env, loaded = _merge_dotenv(env, env_file)
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
            api_host=env.get("NAWAT_API_HOST", "127.0.0.1"),
            api_port=int(env.get("NAWAT_API_PORT", "8080")),
            api_token=env.get("NAWAT_API_TOKEN") or None,
            serve_port=int(env.get("NAWAT_SERVE_PORT", "8001")),
            serve_idle_timeout=float(env.get("NAWAT_SERVE_IDLE_TIMEOUT", "900")),
            serve_startup_timeout=float(env.get("NAWAT_SERVE_STARTUP_TIMEOUT", "600")),
            serve_backend=env.get("NAWAT_SERVE_BACKEND", "vllm").strip().lower(),
            serve_extra_args=env.get("NAWAT_SERVE_EXTRA_ARGS", ""),
            env_file=loaded,
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
