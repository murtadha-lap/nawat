"""Shared fixtures.

The object store used here is the real ``LocalObjectStore``, not a mock, so the
download/upload/verify paths under test are the same ones the S3 backend runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from nawat.cache import Cache
from nawat.config import Config
from nawat.errors import StoreUnavailable
from nawat.hub import Hub
from nawat.keys import Key
from nawat.store import LocalObjectStore


class RecordingHub(Hub):
    """An upstream hub holding a fixed set of artifacts, counting every fetch.

    The count is the assertion behind G2: an artifact is downloaded from the
    public internet exactly once, ever.
    """

    name = "test hub"

    def __init__(self) -> None:
        self.contents: dict[str, dict[str, bytes]] = {}
        self.fetches: list[str] = []

    def offer(self, key: str, files: dict[str, bytes]) -> None:
        self.contents[key] = files

    def size(self, key: Key) -> int | None:
        files = self.contents.get(str(key))
        return sum(len(blob) for blob in files.values()) if files else None

    def fetch(self, key: Key, dest: Path) -> None:
        from nawat.errors import NotFound

        files = self.contents.get(str(key))
        if files is None:
            raise NotFound(f"{key} is not on {self.name}.", "Seed it first.")
        self.fetches.append(str(key))
        dest.mkdir(parents=True, exist_ok=True)
        for rel, blob in files.items():
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)


class UnavailableStore(LocalObjectStore):
    """A store that has gone away mid-session."""

    def list_prefix(self, key: Key):
        raise StoreUnavailable("Object storage did not answer while listing.", "Check the endpoint.")


@pytest.fixture
def config(tmp_path: Path, request) -> Config:
    # Eviction tests want a cache measured in bytes; execution tests need room
    # for real files. A module sets CACHE_CEILING to say which it is.
    return Config(
        cache_root=tmp_path / "cache",
        workspace_root=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        cache_ceiling=getattr(request.module, "CACHE_CEILING", 1000),
        min_free=0,
        store_backend="local",
        local_store_root=tmp_path / "store",
        transfer_workers=2,
    )


@pytest.fixture
def store(config: Config) -> LocalObjectStore:
    assert config.local_store_root is not None
    return LocalObjectStore(config.local_store_root, workers=2)


@pytest.fixture
def hub() -> RecordingHub:
    return RecordingHub()


@pytest.fixture
def cache(config: Config, store: LocalObjectStore, hub: RecordingHub) -> Cache:
    return Cache(config, store=store, hub=hub)


@pytest.fixture
def workspace(config: Config) -> Path:
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    return config.workspace_root


@pytest.fixture
def runs(cache: Cache):
    from nawat.runs import RunStore

    return RunStore(cache.db, config_state(cache) / "runs")


def config_state(cache: Cache) -> Path:
    return cache.config.state_dir


@pytest.fixture
def executor(cache: Cache, runs):
    from nawat.executor import Executor

    return Executor(cache, runs)


@pytest.fixture
def script(workspace: Path):
    """Write a training script into the workspace and return its relative name."""

    def build(body: str, name: str = "train.py") -> str:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        return name

    return build


class FakeBackend:
    """Runs tests/fake_server.py the way the vLLM backend runs vllm."""

    name = "fake"
    supports_adapters = True

    def __init__(self, ready_after: float = 0.0, exit_after: float = 0.0) -> None:
        self.ready_after = ready_after
        self.exit_after = exit_after

    def command(self, *, model_path: Path, model_name: str, port: int, extra: list[str]) -> list[str]:
        import sys

        command = [
            sys.executable,
            "-m",
            "tests.fake_server",
            "--port",
            str(port),
            "--model",
            model_name,
            "--ready-after",
            str(self.ready_after),
        ]
        if self.exit_after:
            command += ["--exit-after", str(self.exit_after)]
        return command + list(extra)

    def environment(self) -> dict[str, str]:
        return {"PYTHONPATH": str(Path(__file__).resolve().parent.parent)}

    def health_path(self) -> str:
        return "/health"


@pytest.fixture
def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def sessions(cache: Cache, free_port: int):
    """A session manager wired to the fake server, on a port nothing else uses."""
    from dataclasses import replace as dataclass_replace

    from nawat.sessions import SessionManager

    cache.config = dataclass_replace(cache.config, serve_port=free_port, serve_startup_timeout=20.0)
    manager = SessionManager(cache, backend=FakeBackend())
    yield manager
    manager.stop()


@pytest.fixture
def make_artifact(tmp_path: Path):
    """Build a scratch directory of a known total size."""
    counter = {"n": 0}

    def build(size: int, files: int = 1, name: str | None = None) -> Path:
        counter["n"] += 1
        directory = tmp_path / "scratch" / (name or f"artifact-{counter['n']}")
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        per_file, remainder = divmod(size, files)
        for index in range(files):
            blob = b"x" * (per_file + (remainder if index == files - 1 else 0))
            (directory / f"part-{index}.bin").write_bytes(blob)
        return directory

    return build


@pytest.fixture
def cached(cache: Cache):
    """Put an artifact straight onto local disk at a known size.

    Deliberately bypasses ``Cache.add``, which would reclaim space as it went:
    eviction tests need to set up a cache that is already over its ceiling.
    """

    def build(key: str, size: int, *, replicated: bool = True, files: int = 1):
        import time

        from nawat.store import manifest_bytes, scan_local

        parsed = Key.parse(key)
        target = parsed.local_path(cache.config.cache_root)
        target.mkdir(parents=True, exist_ok=True)
        per_file, remainder = divmod(size, files)
        for index in range(files):
            blob = b"x" * (per_file + (remainder if index == files - 1 else 0))
            (target / f"part-{index}.bin").write_bytes(blob)
        if replicated:
            cache.store.publish(target, parsed)
        manifest = scan_local(target)
        cache._write_marker(parsed, target, manifest_bytes(manifest), len(manifest), time.time())
        cache._register(parsed, size=manifest_bytes(manifest), files=len(manifest), replicated=replicated)
        return cache.get(parsed)

    return build
