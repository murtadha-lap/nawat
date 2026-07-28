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
def config(tmp_path: Path) -> Config:
    return Config(
        cache_root=tmp_path / "cache",
        workspace_root=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        cache_ceiling=1000,
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
