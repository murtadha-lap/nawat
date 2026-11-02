"""Direct control of what object storage holds: list, measure, download, delete.

The cache commands manage the local disk; these manage the bucket itself — the
durable library — so the researcher decides what is kept there and what goes.
Deleting is explicit and confirmed, refuses anything a live run is using, and is
not available inside the sandbox.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..errors import NawatError, Protected
from .experiment import SANDBOX_ENV


@dataclass(frozen=True)
class Obj:
    key: str  # relative to the bucket's Nawāt prefix
    size: int
    modified: float


def _s3(store):
    return getattr(store, "client", None) is not None and hasattr(store, "bucket")


def walk(store, prefix: str = "") -> Iterator[Obj]:
    """Every object under ``prefix`` (``""`` = the whole bucket)."""
    prefix = prefix.strip("/")
    if hasattr(store, "root"):
        root = Path(store.root)
        base = root / prefix if prefix else root
        if base.is_file():
            st = base.stat()
            yield Obj(prefix, st.st_size, st.st_mtime)
            return
        if not base.is_dir():
            return
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.name.startswith(".nawat"):
                st = path.stat()
                yield Obj(path.relative_to(root).as_posix(), st.st_size, st.st_mtime)
        return
    client, bucket, root = store.client, store.bucket, getattr(store, "prefix", "") or ""
    full = "/".join(p for p in (root, prefix) if p)
    paginator = client.get_paginator("list_objects_v2")
    exact = [full] if full else []
    for candidate in ([full + "/"] if full else [""]) + exact:
        found = False
        for page in paginator.paginate(Bucket=bucket, Prefix=candidate):
            for item in page.get("Contents", ()):
                key = item["Key"][len(root) + 1:] if root else item["Key"]
                if exact and candidate == full and key != prefix:
                    continue  # second pass: an exact object, not a longer key that shares the prefix
                found = True
                yield Obj(key, int(item["Size"]), item["LastModified"].timestamp())
        if found:
            return


def usage(store, prefix: str = "", depth: int = 2) -> list[tuple[str, int, int]]:
    """(folder, objects, bytes) grouped at ``depth`` path segments below the bucket root."""
    totals: dict[str, list[int]] = {}
    for obj in walk(store, prefix):
        parts = obj.key.split("/")
        folder = "/".join(parts[:depth]) if len(parts) > depth else "/".join(parts[:-1]) or parts[0]
        entry = totals.setdefault(folder, [0, 0])
        entry[0] += 1
        entry[1] += obj.size
    return sorted((k, v[0], v[1]) for k, v in totals.items())


def delete(store, prefix: str, *, live_keys: set[str] = frozenset()) -> tuple[int, int]:
    """Delete every object under ``prefix``. Returns (objects, bytes)."""
    if os.environ.get(SANDBOX_ENV):
        raise Protected("Deleting from object storage is not allowed inside the sandbox.", "Delete it from the host.")
    prefix = prefix.strip("/")
    if not prefix or prefix.count("/") < 1:
        raise Protected(f"Refusing to delete {prefix or 'the whole bucket'!r}: too broad.",
                        "Name at least <kind>/<name>, e.g. runs/llm-grpo/20261102-ab12.")
    for key in live_keys:
        if key == prefix or key.startswith(prefix + "/") or prefix.startswith(key + "/"):
            raise Protected(f"{key} is in use by a live run or session.", "Wait for it to finish, or cancel it first.")
    objects = list(walk(store, prefix))
    if not objects:
        return 0, 0
    if hasattr(store, "root"):
        target = Path(store.root) / prefix
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    else:
        client, bucket, root = store.client, store.bucket, getattr(store, "prefix", "") or ""
        keys = [{"Key": "/".join(p for p in (root, o.key) if p)} for o in objects]
        try:
            for start in range(0, len(keys), 1000):
                client.delete_objects(Bucket=bucket, Delete={"Objects": keys[start:start + 1000]})
        except Exception as exc:  # noqa: BLE001
            raise NawatError(f"Deleting {prefix} failed part-way: {exc}", "Run the same command again to finish.") from exc
    return len(objects), sum(o.size for o in objects)


def download(store, prefix: str, dest: Path) -> tuple[int, int]:
    """Copy everything under ``prefix`` into ``dest``, keeping the relative layout."""
    prefix = prefix.strip("/")
    count = total = 0
    for obj in walk(store, prefix):
        rel = obj.key[len(prefix):].lstrip("/") or Path(obj.key).name
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(store, "root"):
            shutil.copy2(Path(store.root) / obj.key, target)
        else:
            root = getattr(store, "prefix", "") or ""
            store.client.download_file(store.bucket, "/".join(p for p in (root, obj.key) if p), str(target))
        count += 1
        total += obj.size
    return count, total
