"""Blocks — the swappable parts of an experiment, and the fixed backbones they plug into.

A block is one ``.py`` file with three things at the top level::

    BLOCK  = {"kind": "dataset", "summary": "...", "backbones": ("sft",), "models": ("llm", "vlm")}
    PARAMS = {"split": P("train", "Dataset split to train on")}
    def load(ctx, p, tokenizer): ...      # the function its kind requires

Kinds and the function each must provide:

=========  ==========================================================
backbone   ``run(ctx, p)`` — the fixed training flow for one method
model      ``load(ctx, p)`` and ``add_adapters(ctx, p, model)``
dataset    ``load(ctx, p, model, tokenizer) -> {"train": ..., "eval": ...}``
template   ``apply(ctx, p, tokenizer, data) -> (tokenizer, data)``
rewards    ``reward(ctx, p) -> callable`` (GRPO / DAPO), one per block
eval       ``evaluate(ctx, p, model, tokenizer, data) -> dict``
export     ``export(ctx, p, model, tokenizer)``
runtime    ``RUNTIME = "host" | "sandbox"`` — where the experiment runs
=========  ==========================================================

Heavy imports (torch, unsloth, trl) belong inside those functions, so reading a
block's metadata to plan an experiment costs nothing.

Two libraries are searched: the built-in one shipped with Nawāt, and the
workspace's ``lab/blocks/<kind>/``. A workspace block may not reuse a built-in
name — a fork always gets a name of its own, so a preset never silently picks up
a different file than the one it was written against.

**Blocks are never edited after use.** The first time an experiment uses a
workspace block its content hash is recorded and the file made read-only. Any
later plan that finds different content refuses, and the change has to be saved
as a new block (``nawat block save-as``), which also puts the original back.
That is the rule that keeps an old experiment reproducible from its preset.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from ..errors import InvalidKey, NawatError, NotFound, Protected
from .params import P

KINDS = ("backbone", "model", "dataset", "template", "rewards", "eval", "export", "runtime")
#: Slots that take a list of blocks rather than one.
LIST_SLOTS = frozenset({"rewards", "eval", "export"})
#: The function each kind must define.
CONTRACT = {
    "backbone": ("run",),
    "model": ("load", "add_adapters"),
    "dataset": ("load",),
    "template": ("apply",),
    "rewards": ("reward",),
    "eval": ("evaluate",),
    "export": ("export",),
    "runtime": (),
}
#: Folder names on disk; ``rewards`` is plural already.
_DIR = {kind: ("backbones" if kind == "backbone" else kind) for kind in KINDS}

LOCKS_FILE = ".locks.json"
ORIGINALS_DIR = ".originals"


class BlockError(NawatError):
    exit_code = 2


def library_root() -> Path:
    return Path(__file__).resolve().parent / "library"


def lab_root(config) -> Path:
    """The researcher's lab inside the workspace: blocks, presets, experiments."""
    return Path(config.workspace_root) / "lab"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned or not all(c.isalnum() or c in "-_." for c in cleaned) or not cleaned[0].isalnum():
        raise InvalidKey(
            f"{name!r} is not a usable block or preset name.",
            "Use letters, digits, '-', '_' and '.', starting with a letter or digit.",
        )
    return cleaned


def parse_ref(ref: str, kind: str | None = None) -> tuple[str, str]:
    """``dataset/chat-messages`` or, when the kind is known, ``chat-messages``."""
    if "/" in ref:
        found_kind, name = ref.split("/", 1)
        found_kind = "backbone" if found_kind == "backbones" else found_kind
        if found_kind not in KINDS:
            raise InvalidKey(f"{found_kind!r} is not a block kind.", f"Use one of: {', '.join(KINDS)}.")
        if kind is not None and found_kind != kind:
            raise InvalidKey(f"{ref} is a {found_kind} block, but this slot takes a {kind} block.")
        return found_kind, validate_name(name)
    if kind is None:
        raise InvalidKey(f"Say which kind of block {ref!r} is.", f"Write it as <kind>/{ref}, e.g. dataset/{ref}.")
    return kind, validate_name(ref)


@dataclass
class Block:
    """A block file and what it declares."""

    kind: str
    name: str
    path: Path
    source: str  # "builtin" | "workspace"
    sha256: str
    meta: dict[str, Any] = field(default_factory=dict)
    params: dict[str, P] = field(default_factory=dict)
    doc: str = ""

    @property
    def ref(self) -> str:
        return f"{self.kind}/{self.name}"

    @property
    def summary(self) -> str:
        return str(self.meta.get("summary") or (self.doc.strip().splitlines() or [""])[0])

    @property
    def family(self) -> str:
        """For a backbone: the method it implements. Blocks written for that method fit it."""
        return str(self.meta.get("family") or self.name)

    def fits(self, backbone: "str | Block") -> bool:
        wanted = self.meta.get("backbones")
        names = {backbone.name, backbone.family} if isinstance(backbone, Block) else {backbone}
        return not wanted or bool(names & set(wanted))

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "name": self.name, "source": self.source, "sha256": self.sha256}


def load_module(path: Path, label: str) -> ModuleType:
    """Import a block file by path, under a name that cannot collide with a package."""
    module_name = "nawat_block_" + hashlib.sha1(f"{label}:{path}".encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise BlockError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise BlockError(f"{label} failed to import: {type(exc).__name__}: {exc}",
                         "Keep heavy imports inside the block's functions, and fix the error above.") from exc
    return module


def inspect_block(kind: str, name: str, path: Path, source: str) -> Block:
    module = load_module(path, f"{kind}/{name}")
    meta = dict(getattr(module, "BLOCK", {}) or {})
    declared_kind = meta.get("kind", kind)
    if declared_kind != kind:
        raise BlockError(f"{path} declares kind {declared_kind!r} but sits in the {kind} folder.")
    params = dict(getattr(module, "PARAMS", {}) or {})
    for pname, spec in params.items():
        if not isinstance(spec, P):
            raise BlockError(f"{kind}/{name}: PARAMS[{pname!r}] must be P(default, help).")
    missing = [fn for fn in CONTRACT[kind] if not callable(getattr(module, fn, None))]
    if kind == "runtime" and not getattr(module, "RUNTIME", None):
        missing.append("RUNTIME")
    if missing:
        raise BlockError(f"{kind}/{name} is missing {', '.join(missing)}, which every {kind} block must define.")
    return Block(kind, name, path, source, sha256_file(path), meta, params, module.__doc__ or "")


class Library:
    """Finds blocks across the built-in library and the workspace lab."""

    def __init__(self, config) -> None:
        self.config = config
        self.builtin = library_root()
        self.workspace = lab_root(config)

    # -- discovery -----------------------------------------------------------

    def _dirs(self, kind: str) -> list[tuple[str, Path]]:
        return [("workspace", self.workspace / "blocks" / _DIR[kind]), ("builtin", self.builtin / _DIR[kind])]

    def path_for(self, kind: str, name: str) -> tuple[str, Path] | None:
        for source, directory in reversed(self._dirs(kind)):  # builtin first: names cannot be shadowed
            candidate = directory / f"{name}.py"
            if candidate.is_file():
                return source, candidate
        return None

    def get(self, ref: str, kind: str | None = None) -> Block:
        kind, name = parse_ref(ref, kind)
        found = self.path_for(kind, name)
        if found is None:
            names = [b.name for b in self.list(kind)]
            import difflib

            close = difflib.get_close_matches(name, names, n=3)
            raise NotFound(
                f"No {kind} block named {name!r}.",
                (f"Did you mean: {', '.join(close)}? " if close else "") + f"List them with: nawat block ls {kind}",
            )
        source, path = found
        return inspect_block(kind, name, path, source)

    def list(self, kind: str | None = None) -> list[Block]:
        out: list[Block] = []
        for k in ([kind] if kind else KINDS):
            seen: set[str] = set()
            for source, directory in reversed(self._dirs(k)):
                if not directory.is_dir():
                    continue
                for path in sorted(directory.glob("*.py")):
                    if path.name.startswith("_") or path.stem in seen:
                        continue
                    seen.add(path.stem)
                    try:
                        out.append(inspect_block(k, path.stem, path, source))
                    except NawatError as exc:
                        out.append(Block(k, path.stem, path, source, "", {"summary": f"BROKEN: {exc.cause}"}))
        return out

    # -- immutability ----------------------------------------------------------

    @property
    def locks_path(self) -> Path:
        return self.workspace / "blocks" / LOCKS_FILE

    def _locks(self) -> dict[str, Any]:
        try:
            return json.loads(self.locks_path.read_text())
        except (OSError, ValueError):
            return {}

    def _save_locks(self, locks: Mapping[str, Any]) -> None:
        self.locks_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.locks_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(locks, indent=2, sort_keys=True))
        os.replace(tmp, self.locks_path)

    def check_unchanged(self, block: Block) -> None:
        """Refuse a workspace block whose content differs from when it was first used."""
        if block.source != "workspace":
            return
        lock = self._locks().get(block.ref)
        if lock and lock["sha256"] != block.sha256:
            raise Protected(
                f"{block.ref} was edited after experiment {lock['first_used_by']!r} used it.",
                f"Blocks are never changed in place. Keep your edit as a new block and restore the original with: "
                f"nawat block save-as {block.ref} <new-name>",
            )

    def lock(self, block: Block, experiment: str) -> None:
        """Record a workspace block's content at its first use, and keep a copy of it."""
        if block.source != "workspace":
            return
        locks = self._locks()
        if block.ref in locks:
            return
        originals = self.workspace / "blocks" / ORIGINALS_DIR
        originals.mkdir(parents=True, exist_ok=True)
        shutil.copy2(block.path, originals / f"{block.sha256}.py")
        locks[block.ref] = {"sha256": block.sha256, "first_used_by": experiment, "at": time.time()}
        self._save_locks(locks)
        _read_only(block.path)

    def is_locked(self, block: Block) -> bool:
        return block.source == "workspace" and block.ref in self._locks()

    # -- making new blocks ---------------------------------------------------

    def _target(self, kind: str, name: str) -> Path:
        name = validate_name(name)
        if self.path_for(kind, name) is not None:
            raise Protected(f"A {kind} block named {name!r} already exists.", "Blocks are never overwritten; pick a new name.")
        directory = self.workspace / "blocks" / _DIR[kind]
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{name}.py"

    def fork(self, ref: str, new_name: str) -> Block:
        """Copy a block under a new name, ready to edit. The source is untouched."""
        source = self.get(ref)
        original = self._original_content(source)
        target = self._target(source.kind, new_name)
        target.write_bytes(original if original is not None else source.path.read_bytes())
        _writable(target)
        return self.get(f"{source.kind}/{new_name}")

    def save_as(self, ref: str, new_name: str) -> tuple[Block, Block]:
        """Move an in-place edit of a locked block to a new block and restore the original."""
        edited = self.get(ref)
        original = self._original_content(edited)
        if original is None:
            raise Protected(f"{edited.ref} has no recorded original to restore.",
                            f"It is not locked, so edit it freely, or copy it with: nawat block fork {edited.ref} <new-name>")
        if hashlib.sha256(original).hexdigest() == edited.sha256:
            raise Protected(f"{edited.ref} is unchanged since it was first used; there is nothing to save.",
                            f"To start an edit, copy it with: nawat block fork {edited.ref} <new-name>")
        target = self._target(edited.kind, new_name)
        target.write_bytes(edited.path.read_bytes())
        _writable(target)
        _writable(edited.path)
        edited.path.write_bytes(original)
        _read_only(edited.path)
        return self.get(edited.ref), self.get(f"{edited.kind}/{new_name}")

    def new(self, kind: str, name: str) -> Block:
        """Start a block from the template for its kind."""
        template = self.builtin / "_templates" / f"{kind}.py"
        if not template.is_file():
            raise NotFound(f"There is no starter template for {kind} blocks.", f"Fork an existing one: nawat block ls {kind}")
        target = self._target(kind, name)
        target.write_text(template.read_text())
        _writable(target)
        return self.get(f"{kind}/{name}")

    def _original_content(self, block: Block) -> bytes | None:
        lock = self._locks().get(block.ref)
        if not lock or block.source != "workspace":
            return None
        path = self.workspace / "blocks" / ORIGINALS_DIR / f"{lock['sha256']}.py"
        return path.read_bytes() if path.is_file() else None


def _read_only(path: Path) -> None:
    try:
        mode = path.stat().st_mode
        path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    except OSError:
        pass


def _writable(path: Path) -> None:
    try:
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    except OSError:
        pass
