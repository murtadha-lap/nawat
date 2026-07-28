"""Artifact keys.

A key is the single name for a thing. It maps 1:1 onto an object-storage prefix
and onto a local cache path (FR-1.1), so no component ever has to translate
between two naming schemes.

    models/unsloth/Qwen2.5-VL-7B-Instruct
    datasets/ocr-arabic-v3
    runs/2026-07-28-a91f/adapter

The first segment is the kind. For models and datasets the remainder doubles as
the upstream hub repo id, which is what makes "fetch it once, ever" (G2) fall
out without a mapping table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .errors import InvalidKey

KINDS = ("models", "datasets", "runs", "exports")
HUB_KINDS = ("models", "datasets")

_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@+-]*$")
_MAX_LENGTH = 512


@dataclass(frozen=True, order=True)
class Key:
    """An addressable model, dataset or run artifact."""

    kind: str
    path: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise InvalidKey(
                f"{self.kind!r} is not an artifact kind.",
                f"Use one of: {', '.join(KINDS)}.",
            )
        if not self.path:
            raise InvalidKey(f"Key {self.kind!r} names a kind but no artifact.", "Write it as <kind>/<name>.")

    @classmethod
    def parse(cls, raw: "str | Key") -> "Key":
        """Parse and validate a key string. Rejects anything that could escape the cache root."""
        if isinstance(raw, Key):
            return raw
        text = raw.strip().strip("/")
        if not text:
            raise InvalidKey("Empty key.", "Write it as <kind>/<name>, e.g. models/unsloth/Qwen2.5-VL-7B-Instruct.")
        if len(text) > _MAX_LENGTH:
            raise InvalidKey(f"Key is longer than {_MAX_LENGTH} characters.", "Shorten the name.")
        if "\\" in text:
            raise InvalidKey(f"Key {raw!r} contains a backslash.", "Separate segments with /.")
        segments = text.split("/")
        for segment in segments:
            if not segment:
                raise InvalidKey(f"Key {raw!r} has an empty segment.", "Remove the doubled /.")
            if segment in (".", ".."):
                raise InvalidKey(
                    f"Key {raw!r} contains a relative path segment.",
                    "Keys address artifacts, not filesystem locations — remove it.",
                )
            if not _SEGMENT.match(segment):
                raise InvalidKey(
                    f"Segment {segment!r} in key {raw!r} is not a valid name.",
                    "Use letters, digits, and . _ - + @, starting with a letter or digit.",
                )
        if len(segments) < 2:
            raise InvalidKey(
                f"Key {raw!r} names a kind but no artifact.",
                f"Write it as <kind>/<name>, with kind one of: {', '.join(KINDS)}.",
            )
        return cls(kind=segments[0], path="/".join(segments[1:]))

    def __str__(self) -> str:
        return f"{self.kind}/{self.path}"

    @property
    def segments(self) -> tuple[str, ...]:
        return (self.kind, *self.path.split("/"))

    @property
    def object_prefix(self) -> str:
        """Object-storage prefix, always trailing-slashed so it cannot match a sibling."""
        return f"{self.kind}/{self.path}/"

    @property
    def hub_repo_id(self) -> str | None:
        """The upstream hub repo id, when this kind of key has one."""
        return self.path if self.kind in HUB_KINDS else None

    def local_path(self, cache_root: Path) -> Path:
        """The one local location for this artifact."""
        return cache_root.joinpath(*self.segments)

    def child(self, *segments: str) -> "Key":
        """Derive a sub-key, e.g. ``runs/<id>`` -> ``runs/<id>/adapter``."""
        return Key.parse("/".join((str(self), *segments)))
