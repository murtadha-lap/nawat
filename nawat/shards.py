"""Sharded datasets — many small files packed into few large ones.

Streaming a corpus of a hundred thousand small images out of object storage
issues a hundred thousand GETs, and the dataloader starves the GPU waiting on
round trips (PRD §12, NFR-2.5). Packing the corpus into fixed-size tar shards
turns that into a few dozen large sequential reads.

Shards are plain POSIX tars — WebDataset-compatible by construction, and
`datasets`/`webdataset` both stream them — with an ``index.json`` recording
what went where, so membership questions never require opening a shard.
"""

from __future__ import annotations

import json
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .errors import NawatError, NotFound
from .units import human_bytes

INDEX_NAME = "index.json"
DEFAULT_SHARD_SIZE = 512 * 10**6


@dataclass
class ShardIndex:
    shard_size: int
    total_files: int = 0
    total_bytes: int = 0
    shards: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "format": "tar-shards-v1",
            "created_at": time.time(),
            "shard_size": self.shard_size,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "shards": self.shards,
        }


def pack(
    source: Path,
    dest: Path,
    *,
    shard_size: int = DEFAULT_SHARD_SIZE,
    progress: Callable[[int, int], None] | None = None,
) -> ShardIndex:
    """Pack every file under ``source`` into tar shards in ``dest``.

    Files are placed in sorted order, so samples that sort together (an image
    beside its label file) land in the same shard — the property WebDataset
    training loops rely on. A shard closes when adding the next file would
    push it past ``shard_size``, so shards may run slightly under, never
    wildly over.
    """
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise NotFound(f"{source} is not a directory.", "Point at the corpus to shard.")
    files = sorted(p for p in source.rglob("*") if p.is_file() and not p.is_symlink())
    if not files:
        raise NotFound(f"{source} holds no files.", "Nothing to shard.")
    dest.mkdir(parents=True, exist_ok=True)

    index = ShardIndex(shard_size=shard_size)
    writer: tarfile.TarFile | None = None
    current: dict[str, Any] = {}

    def close_current() -> None:
        nonlocal writer
        if writer is None:
            return
        writer.close()
        writer = None
        current["bytes"] = (dest / current["name"]).stat().st_size
        index.shards.append(dict(current))

    def open_next() -> None:
        nonlocal writer
        name = f"shard-{len(index.shards):05d}.tar"
        current.clear()
        current.update(name=name, files=0, content_bytes=0)
        writer = tarfile.open(dest / name, "w")

    for number, path in enumerate(files, start=1):
        size = path.stat().st_size
        if size > shard_size:
            raise NawatError(
                f"{path.name} is {human_bytes(size)}, larger than the {human_bytes(shard_size)} shard size.",
                "Raise --shard-size above the largest file.",
            )
        if writer is None or current["content_bytes"] + size > shard_size:
            close_current()
            open_next()
        writer.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
        current["files"] += 1
        current["content_bytes"] += size
        index.total_files += 1
        index.total_bytes += size
        if progress:
            progress(number, len(files))
    close_current()

    (dest / INDEX_NAME).write_text(json.dumps(index.to_json(), indent=2) + "\n")
    return index


def read_index(directory: Path) -> dict[str, Any]:
    path = directory / INDEX_NAME
    if not path.is_file():
        raise NotFound(f"{directory} carries no {INDEX_NAME}.", "Is this a sharded dataset?")
    return json.loads(path.read_text())


def verify(directory: Path) -> dict[str, Any]:
    """Re-open every shard and confirm the index still tells the truth."""
    index = read_index(directory)
    counted_files = 0
    for entry in index["shards"]:
        shard_path = directory / entry["name"]
        if not shard_path.is_file():
            raise NawatError(f"{entry['name']} is missing from {directory}.", "Re-pack the dataset.")
        if shard_path.stat().st_size != entry["bytes"]:
            raise NawatError(
                f"{entry['name']} is {human_bytes(shard_path.stat().st_size)} but the index says "
                f"{human_bytes(entry['bytes'])}.",
                "The shard changed after packing — re-pack the dataset.",
            )
        with tarfile.open(shard_path) as reader:
            members = [m for m in reader.getmembers() if m.isfile()]
        if len(members) != entry["files"]:
            raise NawatError(f"{entry['name']} holds {len(members)} files, index says {entry['files']}.",
                             "Re-pack the dataset.")
        counted_files += len(members)
    if counted_files != index["total_files"]:
        raise NawatError("Shard contents do not add up to the index total.", "Re-pack the dataset.")
    return index


def unpack(directory: Path, dest: Path) -> int:
    """Restore the original tree — the inverse of :func:`pack`, for inspection."""
    index = read_index(directory)
    dest.mkdir(parents=True, exist_ok=True)
    restored = 0
    for entry in index["shards"]:
        with tarfile.open(directory / entry["name"]) as reader:
            reader.extractall(dest, filter="data")  # refuses traversal and device nodes
            restored += sum(1 for m in reader.getmembers() if m.isfile())
    return restored
