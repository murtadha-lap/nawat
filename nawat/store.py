"""Object storage — the source of truth.

Two backends behind one interface: RustFS (or any S3-compatible endpoint) for
real deployments, and a plain directory tree for development on a host without
a store running. Both are exercised by the same tests, so the safety properties
that matter — atomic writes, name-and-size verification — are properties of the
base class rather than of either backend.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .config import Config
from .errors import NotFound, StoreUnavailable, VerificationFailed
from .keys import Key
from .units import human_bytes

#: Anything the platform writes alongside an artifact. Never uploaded, never
#: counted towards an artifact's size, never compared during verification.
INTERNAL_PREFIX = ".nawat"
PART_SUFFIX = ".nawat-part"

Progress = Callable[[str, int, int], None]


@dataclass(frozen=True)
class ObjectStat:
    size: int
    etag: str | None = None
    modified: float | None = None


#: Relative path -> stat, for one artifact.
Manifest = dict[str, ObjectStat]


def is_internal(rel: str) -> bool:
    return any(part.startswith(INTERNAL_PREFIX) for part in Path(rel).parts) or rel.endswith(PART_SUFFIX)


def scan_local(root: Path) -> Manifest:
    """Manifest of a local directory, or of a single file addressed by key."""
    if root.is_file():
        stat = root.stat()
        return {root.name: ObjectStat(size=stat.st_size, modified=stat.st_mtime)}
    manifest: Manifest = {}
    if not root.exists():
        return manifest
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if is_internal(rel):
            continue
        stat = path.stat()
        manifest[rel] = ObjectStat(size=stat.st_size, modified=stat.st_mtime)
    return manifest


def manifest_bytes(manifest: Manifest) -> int:
    return sum(stat.size for stat in manifest.values())


@dataclass(frozen=True)
class VerificationResult:
    """Outcome of comparing a local copy against its replica (NFR-1.2)."""

    key: Key
    missing: tuple[str, ...] = ()
    size_mismatch: tuple[str, ...] = ()
    remote_only: tuple[str, ...] = ()
    local_files: int = 0
    local_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing and not self.size_mismatch

    def reason(self) -> str:
        parts = []
        if self.missing:
            parts.append(f"{len(self.missing)} file(s) absent from object storage ({self.missing[0]}…)")
        if self.size_mismatch:
            parts.append(f"{len(self.size_mismatch)} file(s) differ in size ({self.size_mismatch[0]}…)")
        return "; ".join(parts) or "verified"


def verify_manifests(key: Key, local: Manifest, remote: Manifest) -> VerificationResult:
    """Compare every file by name and size. Nothing is deleted on a mismatch."""
    missing = tuple(sorted(rel for rel in local if rel not in remote))
    mismatch = tuple(sorted(rel for rel in local if rel in remote and remote[rel].size != local[rel].size))
    extra = tuple(sorted(rel for rel in remote if rel not in local))
    return VerificationResult(
        key=key,
        missing=missing,
        size_mismatch=mismatch,
        remote_only=extra,
        local_files=len(local),
        local_bytes=manifest_bytes(local),
    )


class ObjectStore(ABC):
    """Per-file primitives; the directory-level safety logic lives on the base."""

    name = "object store"

    def __init__(self, workers: int = 8) -> None:
        self.workers = max(1, workers)

    # -- primitives implemented per backend -------------------------------

    @abstractmethod
    def list_prefix(self, key: Key) -> Manifest: ...

    @abstractmethod
    def _get(self, key: Key, rel: str, dest: Path) -> None: ...

    @abstractmethod
    def _put(self, key: Key, rel: str, src: Path) -> None: ...

    @abstractmethod
    def delete_prefix(self, key: Key) -> int: ...

    # -- shared behaviour --------------------------------------------------

    def exists(self, key: Key) -> bool:
        return bool(self.list_prefix(key))

    def total_size(self, key: Key) -> int:
        return manifest_bytes(self.list_prefix(key))

    def download(self, key: Key, dest: Path, progress: Progress | None = None) -> Manifest:
        """Fetch an artifact into ``dest``.

        Each file is written to a ``.nawat-part`` sidecar and renamed on
        completion, so an interrupted transfer never presents as a finished
        file (NFR-1.3). Callers stage into a scratch directory and rename the
        directory itself, which extends the same guarantee to the artifact.
        """
        remote = self.list_prefix(key)
        if not remote:
            raise NotFound(f"{key} is not in {self.name}.", "Check the key, or fetch it from the upstream hub first.")
        dest.mkdir(parents=True, exist_ok=True)
        total = manifest_bytes(remote)
        done = 0

        def fetch(item: tuple[str, ObjectStat]) -> int:
            rel, stat = item
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + PART_SUFFIX)
            self._get(key, rel, part)
            os.replace(part, target)
            return stat.size

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for size in pool.map(fetch, sorted(remote.items())):
                done += size
                if progress:
                    progress(str(key), done, total)
        return remote

    def upload(self, source: Path, key: Key, progress: Progress | None = None) -> Manifest:
        """Publish a local directory under ``key``. Returns the local manifest sent."""
        local = scan_local(source)
        if not local:
            raise NotFound(
                f"{source} holds no files to publish under {key}.",
                "Check that the run wrote its outputs to this directory.",
            )
        total = manifest_bytes(local)
        done = 0

        def send(item: tuple[str, ObjectStat]) -> int:
            rel, stat = item
            self._put(key, rel, source / rel if source.is_dir() else source)
            return stat.size

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for size in pool.map(send, sorted(local.items())):
                done += size
                if progress:
                    progress(str(key), done, total)
        return local

    def verify(self, source: Path, key: Key) -> VerificationResult:
        """Compare a local copy against its replica without changing either."""
        return verify_manifests(key, scan_local(source), self.list_prefix(key))

    def publish(self, source: Path, key: Key, progress: Progress | None = None) -> VerificationResult:
        """Upload, then verify. Raises rather than reporting success on any mismatch."""
        self.upload(source, key, progress=progress)
        result = self.verify(source, key)
        if not result.ok:
            raise VerificationFailed(
                f"Publish of {key} did not verify: {result.reason()}.",
                "The local copy has been kept. Retry the publish, then check object storage.",
            )
        return result


class LocalObjectStore(ObjectStore):
    """A directory tree standing in for the object store.

    Real for development and for the test suite — not a mock. It exercises the
    same download/upload/verify paths the S3 backend uses.
    """

    name = "local object store"

    def __init__(self, root: Path, workers: int = 8) -> None:
        super().__init__(workers)
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, key: Key) -> Path:
        return self.root.joinpath(*key.segments)

    def list_prefix(self, key: Key) -> Manifest:
        return scan_local(self._dir(key))

    def _get(self, key: Key, rel: str, dest: Path) -> None:
        shutil.copyfile(self._dir(key) / rel, dest)

    def _put(self, key: Key, rel: str, src: Path) -> None:
        target = self._dir(key) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + PART_SUFFIX)
        shutil.copyfile(src, part)
        os.replace(part, target)

    def delete_prefix(self, key: Key) -> int:
        directory = self._dir(key)
        count = len(scan_local(directory))
        shutil.rmtree(directory, ignore_errors=True)
        return count


class S3ObjectStore(ObjectStore):
    """RustFS, MinIO, or any other S3-compatible endpoint."""

    name = "object storage"

    def __init__(self, config: Config) -> None:
        super().__init__(config.transfer_workers)
        self.bucket = config.bucket
        self.prefix = config.store_prefix
        self._config = config
        self._client = None
        self._transfer = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                from boto3.s3.transfer import TransferConfig
                from botocore.config import Config as BotoConfig
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise StoreUnavailable(
                    "boto3 is not installed, so object storage cannot be reached.",
                    "Install it with: pip install boto3",
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self._config.endpoint,
                region_name=self._config.region,
                aws_access_key_id=self._config.access_key,
                aws_secret_access_key=self._config.secret_key,
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    retries={"max_attempts": 5, "mode": "standard"},
                    max_pool_connections=max(16, self.workers * 2),
                ),
            )
            # Parallel multipart, so transfers saturate the local link (NFR-2.2).
            self._transfer = TransferConfig(
                multipart_threshold=self._config.multipart_threshold,
                multipart_chunksize=self._config.multipart_chunk,
                max_concurrency=self.workers,
                use_threads=True,
            )
        return self._client

    def _object_key(self, key: Key, rel: str = "") -> str:
        parts = [p for p in (self.prefix, key.object_prefix.rstrip("/"), rel) if p]
        return "/".join(parts)

    def _wrap(self, exc: Exception, what: str) -> StoreUnavailable:
        endpoint = self._config.endpoint or f"s3://{self.bucket}"
        return StoreUnavailable(
            f"Object storage did not answer while {what}: {type(exc).__name__}.",
            f"Check that {endpoint} is reachable and the credentials in the environment are current.",
        )

    def list_prefix(self, key: Key) -> Manifest:
        prefix = self._object_key(key) + "/"
        manifest: Manifest = {}
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for obj in page.get("Contents", ()):
                    rel = obj["Key"][len(prefix) :]
                    if not rel or rel.endswith("/") or is_internal(rel):
                        continue
                    manifest[rel] = ObjectStat(
                        size=obj["Size"],
                        etag=obj.get("ETag", "").strip('"') or None,
                        modified=obj["LastModified"].timestamp() if obj.get("LastModified") else None,
                    )
        except Exception as exc:
            if type(exc).__name__ == "NoSuchBucket":
                raise StoreUnavailable(
                    f"Bucket {self.bucket!r} does not exist.",
                    "Create it, or set NAWAT_S3_BUCKET to the right name.",
                ) from exc
            raise self._wrap(exc, f"listing {key}") from exc
        return manifest

    def _get(self, key: Key, rel: str, dest: Path) -> None:
        self.client  # force construction so self._transfer is set
        try:
            self.client.download_file(self.bucket, self._object_key(key, rel), str(dest), Config=self._transfer)
        except Exception as exc:
            raise self._wrap(exc, f"downloading {key}/{rel}") from exc

    def _put(self, key: Key, rel: str, src: Path) -> None:
        self.client
        try:
            self.client.upload_file(str(src), self.bucket, self._object_key(key, rel), Config=self._transfer)
        except Exception as exc:
            raise self._wrap(exc, f"uploading {key}/{rel}") from exc

    def delete_prefix(self, key: Key) -> int:
        manifest = self.list_prefix(key)
        if not manifest:
            return 0
        objects = [{"Key": self._object_key(key, rel)} for rel in manifest]
        try:
            for batch in _chunks(objects, 1000):
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
        except Exception as exc:
            raise self._wrap(exc, f"deleting {key}") from exc
        return len(objects)


def _chunks(items: list, size: int) -> Iterable[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_store(config: Config) -> ObjectStore:
    """The store this host is configured to use."""
    if config.store_backend == "local":
        root = config.local_store_root or (config.state_dir / "object-store")
        return LocalObjectStore(root, workers=config.transfer_workers)
    if config.store_backend != "s3":
        raise StoreUnavailable(
            f"Unknown store backend {config.store_backend!r}.",
            "Set NAWAT_STORE_BACKEND to s3 or local.",
        )
    if not config.endpoint and not config.access_key:
        where = f"{config.env_file} does not set them" if config.env_file else "no .env was found"
        raise StoreUnavailable(
            f"No object-storage endpoint or credentials are configured — {where}.",
            "Set NAWAT_S3_ENDPOINT and the S3 credentials in .env beside the project,"
            " or point at one with --env-file.",
        )
    return S3ObjectStore(config)


__all__ = [
    "ObjectStat",
    "ObjectStore",
    "LocalObjectStore",
    "S3ObjectStore",
    "Manifest",
    "VerificationResult",
    "build_store",
    "scan_local",
    "manifest_bytes",
    "verify_manifests",
    "human_bytes",
]
