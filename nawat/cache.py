"""The cache — local disk as a disposable mirror of object storage.

Every decision in this module derives from one rule: object storage is truth and
local disk is disposable (PRD principle 1). Concretely that means nothing is
deleted until it has been verified present in the store, at the moment of
deletion; when verification cannot be completed the platform keeps the data and
reports a full disk instead (principles 2 and 3).

The interesting entry points are :meth:`Cache.resolve`, which makes an artifact
appear locally by whatever means, and :meth:`Cache.collect`, which makes it go
away again once it is safe.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from . import checkpoints
from .config import Config
from .db import Database
from .errors import InsufficientSpace, NotFound, Protected, StoreUnavailable, VerificationFailed
from .hub import Hub, build_hub
from .keys import Key
from .leases import LeaseRecord, LeaseRegistry
from .store import (
    ObjectStore,
    Progress,
    VerificationResult,
    build_store,
    manifest_bytes,
    scan_local,
)
from .units import human_bytes

MARKER_NAME = ".nawat-artifact.json"
STAGING_TTL = 24 * 3600
MAX_WALK_DEPTH = 6


@dataclass(frozen=True)
class ArtifactStatus:
    key: Key
    bytes: int
    files: int
    fetched_at: float
    last_used: float
    pinned: bool
    replicated_at: float | None
    leases: tuple[LeaseRecord, ...] = ()

    @property
    def leased(self) -> bool:
        return bool(self.leases)

    @property
    def replicated(self) -> bool:
        return self.replicated_at is not None

    @property
    def evictable(self) -> bool:
        """Whether eviction may even be *considered*. Replication is still
        re-verified against the store before anything is deleted."""
        return not self.pinned and not self.leased

    def flags(self) -> str:
        return "".join(
            (
                "K" if self.pinned else "·",
                "L" if self.leased else "·",
                "R" if self.replicated else "·",
            )
        )


@dataclass(frozen=True)
class CacheStatus:
    used: int
    ceiling: int
    artifacts: int
    pinned_bytes: int
    leased_bytes: int
    unreplicated_bytes: int
    disk_free: int
    disk_total: int
    #: Disk held by trainer checkpoints. Outside the ceiling and outside
    #: eviction — nothing here reclaims it — so it is reported instead, because
    #: disk the cache will never free still has to be accounted for somewhere.
    checkpoint_bytes: int = 0

    @property
    def headroom(self) -> int:
        return self.ceiling - self.used

    @property
    def fraction(self) -> float:
        return (self.used / self.ceiling) if self.ceiling else 0.0


@dataclass
class CollectResult:
    freed: int = 0
    evicted: list[Key] = field(default_factory=list)
    skipped: list[tuple[Key, str]] = field(default_factory=list)
    target: int = 0

    @property
    def satisfied(self) -> bool:
        return self.freed >= self.target


@dataclass
class PublishResult:
    key: Key
    verification: VerificationResult
    local_removed: bool
    retained_reason: str | None = None

    @property
    def bytes(self) -> int:
        return self.verification.local_bytes


class Cache:
    """Resolves keys to local paths, and reclaims local paths when it is safe."""

    def __init__(
        self,
        config: Config,
        store: ObjectStore | None = None,
        hub: Hub | None = None,
        db: Database | None = None,
    ) -> None:
        self.config = config
        config.ensure_dirs()
        self.db = db or Database(config.database_path)
        self.leases = LeaseRegistry(self.db)
        self._store = store
        self._hub = hub
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._lock_file = None

    # -- lazily built collaborators ---------------------------------------

    @property
    def store(self) -> ObjectStore:
        if self._store is None:
            self._store = build_store(self.config)
        return self._store

    @property
    def hub(self) -> Hub:
        if self._hub is None:
            self._hub = build_hub(self.config)
        return self._hub

    # -- cross-process locking --------------------------------------------

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialise mutating operations across threads and across processes.

        Read-only status calls deliberately do not take this, so `nawat status`
        stays responsive while a 16 GB model is staging in another shell.
        """
        with self._thread_lock:
            first = self._depth == 0
            if first:
                self.config.state_dir.mkdir(parents=True, exist_ok=True)
                self._lock_file = open(self.config.lock_path, "w")
                fcntl.flock(self._lock_file, fcntl.LOCK_EX)
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._lock_file is not None:
                    fcntl.flock(self._lock_file, fcntl.LOCK_UN)
                    self._lock_file.close()
                    self._lock_file = None

    # -- reading state -----------------------------------------------------

    def _row(self, key: Key):
        return self.db.connect().execute("SELECT * FROM artifacts WHERE key = ?", (str(key),)).fetchone()

    def _status_from_row(self, row, leases: dict[str, list[LeaseRecord]] | None = None) -> ArtifactStatus:
        key = Key.parse(row["key"])
        if leases is None:
            held = tuple(self.leases.live_for(key))
        else:
            held = tuple(leases.get(row["key"], ()))
        return ArtifactStatus(
            key=key,
            bytes=row["bytes"],
            files=row["files"],
            fetched_at=row["fetched_at"],
            last_used=row["last_used"],
            pinned=bool(row["pinned"]),
            replicated_at=row["replicated_at"],
            leases=held,
        )

    def _live_lease_index(self) -> dict[str, list[LeaseRecord]]:
        index: dict[str, list[LeaseRecord]] = {}
        for record in self.leases.live():
            index.setdefault(str(record.key), []).append(record)
        return index

    def list(self) -> list[ArtifactStatus]:
        """Everything currently on local disk, newest use first (FR-1.10)."""
        leases = self._live_lease_index()
        rows = self.db.connect().execute("SELECT * FROM artifacts ORDER BY last_used DESC").fetchall()
        return [self._status_from_row(row, leases) for row in rows]

    def get(self, key: "str | Key") -> ArtifactStatus | None:
        key = Key.parse(key)
        row = self._row(key)
        return self._status_from_row(row) if row else None

    def status(self) -> CacheStatus:
        artifacts = self.list()
        usage = shutil.disk_usage(self.config.cache_root)
        return CacheStatus(
            used=sum(a.bytes for a in artifacts),
            ceiling=self.config.cache_ceiling,
            artifacts=len(artifacts),
            pinned_bytes=sum(a.bytes for a in artifacts if a.pinned),
            leased_bytes=sum(a.bytes for a in artifacts if a.leased),
            unreplicated_bytes=sum(a.bytes for a in artifacts if not a.replicated),
            disk_free=usage.free,
            disk_total=usage.total,
            checkpoint_bytes=checkpoints.usage_bytes(self.config.checkpoint_root),
        )

    def is_present(self, key: "str | Key") -> bool:
        key = Key.parse(key)
        return self._row(key) is not None and key.local_path(self.config.cache_root).exists()

    def local_path(self, key: "str | Key") -> Path:
        return Key.parse(key).local_path(self.config.cache_root)

    # -- registration ------------------------------------------------------

    def _write_marker(self, key: Key, path: Path, size: int, files: int, fetched_at: float) -> None:
        marker = {
            "key": str(key),
            "bytes": size,
            "files": files,
            "fetched_at": fetched_at,
            "version": 1,
        }
        (path / MARKER_NAME).write_text(json.dumps(marker, indent=2) + "\n")

    def _register(self, key: Key, *, size: int, files: int, replicated: bool, fetched_at: float | None = None) -> None:
        now = time.time()
        fetched_at = now if fetched_at is None else fetched_at
        with self.db.tx() as conn:
            conn.execute(
                "INSERT INTO artifacts (key, bytes, files, fetched_at, last_used, pinned, replicated_at)"
                " VALUES (?, ?, ?, ?, ?, 0, ?)"
                " ON CONFLICT(key) DO UPDATE SET"
                "   bytes = excluded.bytes, files = excluded.files,"
                "   last_used = excluded.last_used, replicated_at = excluded.replicated_at",
                (str(key), size, files, fetched_at, now, now if replicated else None),
            )

    def touch(self, key: "str | Key") -> None:
        """Record use, which is what LRU ordering reads."""
        with self.db.tx() as conn:
            conn.execute("UPDATE artifacts SET last_used = ? WHERE key = ?", (time.time(), str(Key.parse(key))))

    def _forget(self, key: Key) -> None:
        with self.db.tx() as conn:
            conn.execute("DELETE FROM artifacts WHERE key = ?", (str(key),))

    # -- reconciliation ----------------------------------------------------

    def reconcile(self) -> None:
        """Bring the record and the filesystem back into agreement.

        Runs after a reboot, after a crash, and before anything that consults
        occupancy. Cheap enough to call often: it walks only as far as the
        artifact markers.
        """
        self.leases.reap()
        root = self.config.cache_root
        rows = self.db.connect().execute("SELECT key FROM artifacts").fetchall()
        known = {row["key"] for row in rows}
        for name in known:
            if not Key.parse(name).local_path(root).exists():
                self._forget(Key.parse(name))
        for marker in self._walk_markers(root):
            try:
                data = json.loads(marker.read_text())
                key = Key.parse(data["key"])
            except (OSError, ValueError, KeyError):
                continue
            if str(key) in known:
                continue
            # Recovered from the marker: the cache describes itself on disk, so
            # losing the database does not lose track of what is cached.
            directory = marker.parent
            manifest = scan_local(directory)
            self._register(
                key,
                size=manifest_bytes(manifest),
                files=len(manifest),
                replicated=data.get("replicated", False),
                fetched_at=data.get("fetched_at"),
            )
        self._sweep_staging()

    def _walk_markers(self, root: Path) -> Iterator[Path]:
        # Checkpoints are not artifacts and hold no marker, so the walk skips
        # that subtree rather than scanning thousands of shard files for one it
        # will never find.
        skip = {self.config.staging_root, self.config.checkpoint_root}
        stack: list[tuple[Path, int]] = [(root, 0)]
        while stack:
            directory, depth = stack.pop()
            marker = directory / MARKER_NAME
            if marker.is_file():
                yield marker
                continue  # an artifact is a leaf; do not descend into its files
            if depth >= MAX_WALK_DEPTH:
                continue
            try:
                entries = list(os.scandir(directory))
            except OSError:
                continue
            for entry in entries:
                if not entry.is_dir(follow_symlinks=False) or entry.name.startswith(".nawat"):
                    continue
                path = Path(entry.path)
                if path not in skip:
                    stack.append((path, depth + 1))

    def _sweep_staging(self) -> None:
        staging = self.config.staging_root
        if not staging.is_dir():
            return
        cutoff = time.time() - STAGING_TTL
        for entry in os.scandir(staging):
            try:
                if entry.stat().st_mtime < cutoff:
                    shutil.rmtree(entry.path, ignore_errors=True)
            except OSError:
                continue

    # -- resolution --------------------------------------------------------

    def resolve(
        self,
        key: "str | Key",
        *,
        lease: bool = True,
        holder: str = "",
        allow_hub: bool = True,
        progress: Progress | None = None,
    ) -> Path:
        """Make ``key`` present locally and return its path (FR-1.2).

        Local cache first, then object storage, then the upstream hub — and a
        hub fetch writes through to object storage before returning, so it
        happens exactly once in the artifact's life (FR-1.3, G2).
        """
        key = Key.parse(key)
        with self._locked():
            self.reconcile()
            path = key.local_path(self.config.cache_root)
            if self._row(key) is not None and path.exists():
                self.touch(key)
                if lease:
                    self.leases.acquire([key], holder)
                return path
            self._fetch(key, progress=progress, allow_hub=allow_hub)
            if lease:
                self.leases.acquire([key], holder)
            return path

    def _fetch(self, key: Key, *, progress: Progress | None, allow_hub: bool) -> None:
        target = key.local_path(self.config.cache_root)
        from_store = self._store_has(key)

        if from_store:
            expected = self.store.total_size(key)
        elif allow_hub:
            expected = self.hub.size(key) or 0
        else:
            raise NotFound(
                f"{key} is not in object storage and the upstream hub was not consulted.",
                "Seed it into object storage, or call resolve() without allow_hub=False.",
            )
        # An unknown upstream size still gets headroom reserved, and whatever the
        # estimate missed is reclaimed by the collect() below once it is known.
        self.ensure_space(expected or self.config.min_free)

        staging = self.config.staging_root / f"{key.kind}-{os.getpid()}-{time.time_ns():x}"
        try:
            if from_store:
                self.store.download(key, staging, progress=progress)
                replicated = True
            else:
                self.hub.fetch(key, staging)
                # Write through before returning, so the next resolution is offline.
                self.store.publish(staging, key, progress=progress)
                replicated = True
            manifest = scan_local(staging)
            size, files = manifest_bytes(manifest), len(manifest)
            self._write_marker(key, staging, size, files, time.time())
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, target)  # the artifact appears atomically or not at all
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self._register(key, size=size, files=files, replicated=replicated)
        if expected and size > expected:
            self.collect()  # the estimate was low; give the difference back

    def _store_has(self, key: Key) -> bool:
        try:
            return self.store.exists(key)
        except StoreUnavailable:
            raise

    @contextmanager
    def holding(self, *keys: "str | Key", holder: str = "", progress: Progress | None = None) -> Iterator[dict[str, Path]]:
        """Stage several artifacts and hold them for the duration of the block."""
        parsed = [Key.parse(k) for k in keys]
        paths: dict[str, Path] = {}
        records: list[LeaseRecord] = []
        try:
            for key in parsed:
                paths[str(key)] = self.resolve(key, lease=False, progress=progress)
                records.extend(self.leases.acquire([key], holder))
            yield paths
        finally:
            self.leases.release(record.id for record in records)

    # -- pinning -----------------------------------------------------------

    def pin(self, key: "str | Key") -> ArtifactStatus:
        """Keep on disk until explicitly released (FR-1.6)."""
        return self._set_pinned(key, True)

    def unpin(self, key: "str | Key") -> ArtifactStatus:
        return self._set_pinned(key, False)

    def _set_pinned(self, key: "str | Key", pinned: bool) -> ArtifactStatus:
        key = Key.parse(key)
        with self._locked():
            self.reconcile()
            if self._row(key) is None:
                raise NotFound(
                    f"{key} is not on local disk, so it cannot be kept there.",
                    f"Stage it first with: nawat resolve {key}",
                )
            with self.db.tx() as conn:
                conn.execute("UPDATE artifacts SET pinned = ? WHERE key = ?", (int(pinned), str(key)))
            status = self.get(key)
            assert status is not None
            return status

    # -- verification ------------------------------------------------------

    def verify(self, key: "str | Key") -> VerificationResult:
        """Compare the local copy against its replica, and record the outcome."""
        key = Key.parse(key)
        path = key.local_path(self.config.cache_root)
        if not path.exists():
            raise NotFound(f"{key} is not on local disk.", "Nothing to verify.")
        result = self.store.verify(path, key)
        with self.db.tx() as conn:
            conn.execute(
                "UPDATE artifacts SET replicated_at = ? WHERE key = ?",
                (time.time() if result.ok else None, str(key)),
            )
        return result

    def _verify_for_eviction(self, key: Key) -> tuple[bool, str]:
        """The gate in front of every automatic deletion (FR-1.5, NFR-1.1).

        Verification happens here, against the store, at the moment of
        deletion — not read from a cached flag that may be stale.
        """
        try:
            result = self.verify(key)
        except StoreUnavailable as exc:
            return False, f"not verified — {exc.cause.rstrip('.').lower()}"
        except NotFound:
            return False, "not verified — not present in object storage"
        if not result.ok:
            return False, f"not verified — {result.reason()}"
        return True, "verified"

    # -- reclamation -------------------------------------------------------

    def collect(self, need: int = 0, *, dry_run: bool = False) -> CollectResult:
        """Free space, least recently used first (FR-1.4).

        Never raises for want of space: it reports what it freed and what it
        refused to touch, and why. :meth:`ensure_space` turns that into an error
        when the caller actually needed the room.
        """
        with self._locked():
            self.reconcile()
            result = CollectResult(target=self._deficit(need))
            if result.target <= 0:
                return result
            for artifact in sorted(self.list(), key=lambda a: a.last_used):
                if result.freed >= result.target:
                    break
                if artifact.pinned:
                    result.skipped.append((artifact.key, "kept on disk"))
                    continue
                if artifact.leased:
                    holders = ", ".join(record.describe() for record in artifact.leases)
                    result.skipped.append((artifact.key, f"in use by {holders}"))
                    continue
                ok, reason = self._verify_for_eviction(artifact.key)
                if not ok:
                    result.skipped.append((artifact.key, reason))
                    continue
                if dry_run:
                    result.evicted.append(artifact.key)
                    result.freed += artifact.bytes
                    continue
                self._delete_local(artifact.key)
                result.evicted.append(artifact.key)
                result.freed += artifact.bytes
            return result

    def _deficit(self, need: int) -> int:
        """Bytes that must go, to fit ``need`` under both the ceiling and the disk."""
        status = self.status()
        over_ceiling = status.used + need - status.ceiling
        over_disk = need + self.config.min_free - status.disk_free
        return max(0, over_ceiling, over_disk)

    def ensure_space(self, need: int) -> CollectResult:
        """Make room for ``need`` bytes, or refuse with an actionable message."""
        if need <= 0:
            return CollectResult()
        result = self.collect(need)
        if result.satisfied:
            return result
        short = result.target - result.freed
        held: list[str] = []
        for key, reason in result.skipped:
            status = self.get(key)
            size = human_bytes(status.bytes) if status else "?"
            held.append(f"{key} ({size}) — {reason}")
        kept = checkpoints.usage_bytes(self.config.checkpoint_root)
        if kept:
            held.append(f"checkpoints ({human_bytes(kept)}) — kept so failed runs can be resumed")
        leading = result.skipped[0] if result.skipped else None
        if leading is not None:
            key, reason = leading
            status = self.get(key)
            size = human_bytes(status.bytes) if status else "unknown size"
            remedy = f"{key} ({size}) is {reason} — release it, or raise the ceiling in Settings."
        elif kept >= short:
            # Nothing cached can go, but the checkpoints can — and the user is
            # the only one who can say a half-trained model is expendable.
            remedy = f"Checkpoints hold {human_bytes(kept)}. Free them with: nawat checkpoints --prune"
        else:
            remedy = "Nothing on disk can be freed — raise the ceiling in Settings."
        raise InsufficientSpace(
            f"Not enough space for {human_bytes(need)}; {human_bytes(short)} short.",
            remedy,
            held=held,
        )

    def evict(self, key: "str | Key", *, force: bool = False) -> int:
        """Remove one artifact from local disk. Returns the bytes freed."""
        key = Key.parse(key)
        with self._locked():
            self.reconcile()
            status = self.get(key)
            if status is None:
                raise NotFound(f"{key} is not on local disk.", "Nothing to remove.")
            if status.pinned and not force:
                raise Protected(f"{key} is kept on disk.", f"Release it first with: nawat release {key}")
            if status.leased and not force:
                holders = ", ".join(record.describe() for record in status.leases)
                raise Protected(f"{key} is in use by {holders}.", "Stop it first, then remove.")
            if not force:
                ok, reason = self._verify_for_eviction(key)
                if not ok:
                    raise VerificationFailed(
                        f"{key} is {reason} — removing it would lose the only copy.",
                        f"Publish it first with: nawat publish {self.local_path(key)} {key}",
                    )
            self._delete_local(key)
            return status.bytes

    def _delete_local(self, key: Key) -> None:
        path = key.local_path(self.config.cache_root)
        shutil.rmtree(path, ignore_errors=True)
        self._prune_empty_parents(path.parent)
        self._forget(key)

    def _prune_empty_parents(self, directory: Path) -> None:
        root = self.config.cache_root
        while directory != root and root in directory.parents:
            try:
                directory.rmdir()
            except OSError:
                return
            directory = directory.parent

    # -- publication -------------------------------------------------------

    def publish(
        self,
        source: "str | Path",
        key: "str | Key",
        *,
        keep_local: bool = False,
        progress: Progress | None = None,
    ) -> PublishResult:
        """Upload, verify, then reclaim the local copy (FR-2.7, FR-2.8).

        The verification is not optional and not deferred: if the replica does
        not match file for file, the local copy stays and the call raises.
        """
        source = Path(source).expanduser()
        key = Key.parse(key)
        if not source.exists():
            raise NotFound(f"{source} does not exist, so there is nothing to publish.", "Check the output directory.")
        with self._locked():
            verification = self.store.publish(source, key, progress=progress)
            target = key.local_path(self.config.cache_root)
            same_place = source.resolve() == target.resolve()

            if keep_local:
                if not same_place:
                    self.ensure_space(verification.local_bytes)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        shutil.rmtree(target)
                    shutil.move(str(source), str(target))
                self._write_marker(key, target, verification.local_bytes, verification.local_files, time.time())
                self._register(key, size=verification.local_bytes, files=verification.local_files, replicated=True)
                return PublishResult(key, verification, local_removed=False, retained_reason="kept at your request")

            held = self.leases.live_for(key) if same_place else []
            if held:
                holders = ", ".join(record.describe() for record in held)
                self._register(key, size=verification.local_bytes, files=verification.local_files, replicated=True)
                return PublishResult(key, verification, local_removed=False, retained_reason=f"in use by {holders}")

            shutil.rmtree(source, ignore_errors=True)
            self._prune_empty_parents(source.parent)
            if same_place:
                self._forget(key)
            return PublishResult(key, verification, local_removed=True)

    def add(self, source: "str | Path", key: "str | Key", *, publish: bool = True) -> ArtifactStatus:
        """Adopt a directory already on disk as a cached artifact under ``key``."""
        source = Path(source).expanduser().resolve()
        key = Key.parse(key)
        if not source.is_dir():
            raise NotFound(f"{source} is not a directory.", "Point at the directory holding the artifact.")
        with self._locked():
            target = key.local_path(self.config.cache_root)
            manifest = scan_local(source)
            if source != target.resolve():
                self.ensure_space(manifest_bytes(manifest))
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise Protected(f"{key} is already on local disk.", f"Remove it first with: nawat rm {key}")
                shutil.move(str(source), str(target))
            replicated = False
            if publish:
                self.store.publish(target, key)
                replicated = True
            self._write_marker(key, target, manifest_bytes(manifest), len(manifest), time.time())
            self._register(key, size=manifest_bytes(manifest), files=len(manifest), replicated=replicated)
            status = self.get(key)
            assert status is not None
            return status

    # -- registry ----------------------------------------------------------

    def registry(self, kinds: Sequence[str] | None = None) -> list[Key]:
        """Keys present in object storage, whether or not they are cached (FR-5.2)."""
        from .keys import KINDS

        wanted = tuple(kinds) if kinds else KINDS
        found: list[Key] = []
        store = self.store
        if hasattr(store, "root"):  # local backend: walk the tree
            root = Path(getattr(store, "root"))
            for kind in wanted:
                base = root / kind
                if not base.is_dir():
                    continue
                for path in sorted(base.rglob("*")):
                    if path.is_dir() and any(child.is_file() for child in path.iterdir()):
                        found.append(Key.parse(f"{kind}/{path.relative_to(base).as_posix()}"))
            return found
        client = getattr(store, "client")
        bucket, prefix = getattr(store, "bucket"), getattr(store, "prefix")
        seen: set[str] = set()
        paginator = client.get_paginator("list_objects_v2")
        for kind in wanted:
            base = "/".join(p for p in (prefix, kind) if p) + "/"
            try:
                pages = paginator.paginate(Bucket=bucket, Prefix=base)
                for page in pages:
                    for obj in page.get("Contents", ()):
                        rel = obj["Key"][len(base) :]
                        if "/" not in rel:
                            continue
                        name = f"{kind}/{rel.rsplit('/', 1)[0]}"
                        if name not in seen:
                            seen.add(name)
                            found.append(Key.parse(name))
            except Exception as exc:
                raise StoreUnavailable(
                    f"Could not list {kind} in object storage: {type(exc).__name__}.",
                    "Check that the store is reachable and the credentials are current.",
                ) from exc
        return found


def open_cache(config: Config | None = None) -> Cache:
    """The cache described by the environment."""
    return Cache(config or Config.from_env())
