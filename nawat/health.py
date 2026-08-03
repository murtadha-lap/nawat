"""Bring-up checks.

The operator needs failure states to be legible (PRD §4), so each check reports
what it tried, whether it worked, and what to do about it (NFR-3.3). The store
checks exercise the real upload/list/verify/delete path rather than pinging the
endpoint, because a reachable endpoint with write-only credentials still cannot
run this platform.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .cache import Cache
from .errors import NawatError
from .keys import Key
from .units import human_bytes

PROBE = Key.parse("exports/nawat-healthcheck")


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    remedy: str | None = None


def _writable(directory: Path) -> tuple[bool, str]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".nawat-write-probe-{os.getpid()}"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc.strerror or exc}"
    return True, str(directory)


def run_checks(cache: Cache, *, create_bucket: bool = False) -> list[Check]:
    config = cache.config
    checks: list[Check] = []

    ok, detail = _writable(config.cache_root)
    checks.append(
        Check(
            "cache root",
            ok,
            detail,
            None if ok else f"Create {config.cache_root} and give this user write access, or set NAWAT_CACHE_ROOT.",
        )
    )

    ok, detail = _writable(config.state_dir)
    checks.append(
        Check("state directory", ok, detail, None if ok else f"Give this user write access to {config.state_dir}.")
    )

    # A checkpoint directory that cannot be written is a long run with nothing
    # to show for it if anything goes wrong, so it is worth knowing at bring-up
    # rather than at hour 60.
    ok, detail = _writable(config.checkpoint_root)
    checks.append(
        Check(
            "checkpoint directory",
            ok,
            detail,
            None if ok else f"Give this user write access to {config.checkpoint_root}, or set NAWAT_CHECKPOINT_ROOT.",
        )
    )

    try:
        cache.db.connect().execute("SELECT count(*) FROM artifacts").fetchone()
        checks.append(Check("state database", True, str(config.database_path)))
    except Exception as exc:
        checks.append(
            Check(
                "state database",
                False,
                f"{type(exc).__name__}: {exc}",
                f"Remove {config.database_path} to rebuild it; the cache re-reads itself from disk.",
            )
        )

    usage = shutil.disk_usage(config.cache_root if config.cache_root.exists() else Path.cwd())
    fits = config.cache_ceiling + config.min_free <= usage.total
    checks.append(
        Check(
            "cache ceiling",
            fits,
            f"{human_bytes(config.cache_ceiling)} ceiling + {human_bytes(config.min_free)} reserve"
            f" on a {human_bytes(usage.total)} filesystem ({human_bytes(usage.free)} free)",
            None
            if fits
            else "The ceiling plus the reserve exceeds the disk. Lower NAWAT_CACHE_CEILING or NAWAT_MIN_FREE.",
        )
    )

    if create_bucket:
        checks.append(_create_bucket(cache))

    endpoint = config.endpoint or config.store_backend
    try:
        cache.store.list_prefix(PROBE)
        checks.append(Check("object storage reachable", True, f"{endpoint} · bucket {config.bucket}"))
    except NawatError as exc:
        return [*checks, Check("object storage reachable", False, exc.cause, exc.remedy)]

    checks.append(_round_trip(cache))
    return checks


def _create_bucket(cache: Cache) -> Check:
    store = cache.store
    if not hasattr(store, "client"):
        return Check("bucket", True, "the local backend needs no bucket")
    try:
        store.client.create_bucket(Bucket=cache.config.bucket)
        return Check("bucket created", True, cache.config.bucket)
    except Exception as exc:
        name = type(exc).__name__
        if name in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            return Check("bucket", True, f"{cache.config.bucket} already exists")
        return Check("bucket", False, f"{name}: {exc}", "Create the bucket with the RustFS console or an S3 client.")


def _round_trip(cache: Cache) -> Check:
    """Write, list, verify and delete a probe object — the four operations the
    cache depends on, proving the credentials permit all of them."""
    with tempfile.TemporaryDirectory() as scratch:
        source = Path(scratch) / "probe"
        source.mkdir()
        (source / "probe.txt").write_bytes(b"nawat health check\n")
        try:
            cache.store.publish(source, PROBE)
            cache.store.delete_prefix(PROBE)
        except NawatError as exc:
            return Check("object storage round trip", False, exc.cause, exc.remedy)
        except Exception as exc:
            return Check(
                "object storage round trip",
                False,
                f"{type(exc).__name__}: {exc}",
                "Check that the credentials permit PutObject, ListBucket and DeleteObject on this bucket.",
            )
    return Check("object storage round trip", True, "write, list, verify and delete all succeeded")
