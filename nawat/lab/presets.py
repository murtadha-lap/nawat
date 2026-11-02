"""Presets — a named choice of backbone, blocks, inputs and parameter values.

A preset is a small TOML file::

    [preset]
    backbone    = "grpo"
    description = "GRPO with format + answer rewards"
    model       = "models/unsloth/Qwen3-4B-Base"
    dataset     = "datasets/openai/gsm8k"

    [blocks]
    model    = "llm-lora"
    dataset  = "grpo-prompts"
    template = "reasoning-tags"
    rewards  = ["format-tags", "exact-answer"]
    eval     = ["heldout-generate"]
    export   = ["adapter"]
    runtime  = "host"

    [params.train]
    learning_rate = 5e-6

Like blocks, presets are never overwritten. Changing one means saving a new
preset — ``nawat preset fork llm-grpo --as my-grpo --set train.beta=0.04`` —
which records the preset it came from, so the history of an idea reads as a
chain of files rather than as edits nobody can see any more.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..errors import NotFound, Protected
from .blocks import KINDS, LIST_SLOTS, BlockError, lab_root, library_root, sha256_file, validate_name
from .params import _parse_text

SLOTS = tuple(k for k in KINDS if k != "backbone")


@dataclass
class Preset:
    name: str
    backbone: str
    description: str = ""
    model: str | None = None
    datasets: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    blocks: dict[str, Any] = field(default_factory=dict)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    parent: str | None = None
    path: Path | None = None
    source: str = "workspace"

    @property
    def sha256(self) -> str:
        return sha256_file(self.path) if self.path and self.path.is_file() else ""

    def slot_blocks(self, slot: str) -> list[str]:
        value = self.blocks.get(slot)
        if value is None:
            return []
        return list(value) if isinstance(value, (list, tuple)) else [value]

    def to_toml(self) -> str:
        head: dict[str, Any] = {"backbone": self.backbone}
        if self.description:
            head["description"] = self.description
        if self.parent:
            head["parent"] = self.parent
        if self.model:
            head["model"] = self.model
        if len(self.datasets) == 1:
            head["dataset"] = self.datasets[0]
        elif self.datasets:
            head["datasets"] = self.datasets
        if self.inputs:
            head["inputs"] = self.inputs
        lines = [f"# Nawāt preset {self.name!r}. Never edit in place: nawat preset fork {self.name} --as <new>", ""]
        lines += _table("preset", head)
        blocks = {slot: self.blocks[slot] for slot in SLOTS if slot in self.blocks}
        lines += _table("blocks", blocks)
        for slot, values in self.params.items():
            if values:
                lines += _table(f"params.{_toml_key(slot)}", values)
        return "\n".join(lines).rstrip() + "\n"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backbone": self.backbone,
            "description": self.description,
            "parent": self.parent,
            "model": self.model,
            "datasets": self.datasets,
            "inputs": self.inputs,
            "blocks": self.blocks,
            "params": self.params,
            "source": self.source,
        }


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items()) + " }"
    raise BlockError(f"Cannot write {value!r} to a preset.", "Use strings, numbers, booleans, lists or tables.")


def _toml_key(key: str) -> str:
    return key if all(c.isalnum() or c in "-_" for c in key) else json.dumps(key)


def _table(name: str, values: Mapping[str, Any]) -> list[str]:
    if not values:
        return []
    width = max(len(_toml_key(k)) for k in values)
    lines = [f"[{name}]"]
    for key, value in values.items():
        if value is None:
            continue
        lines.append(f"{_toml_key(key).ljust(width)} = {_toml_value(value)}")
    return lines + [""]


def parse(text: str, name: str, path: Path | None = None, source: str = "workspace") -> Preset:
    data = tomllib.loads(text)
    head = data.get("preset", {})
    if "backbone" not in head:
        raise BlockError(f"Preset {name!r} does not say which backbone it uses.", 'Add backbone = "sft" (or cpt, dpo, grpo, ...) under [preset].')
    datasets = head.get("datasets") or ([head["dataset"]] if head.get("dataset") else [])
    unknown = set(data.get("blocks", {})) - set(SLOTS)
    if unknown:
        raise BlockError(f"Preset {name!r} names unknown slots: {', '.join(sorted(unknown))}.", f"Slots are: {', '.join(SLOTS)}.")
    return Preset(
        name=name,
        backbone=head["backbone"],
        description=head.get("description", ""),
        model=head.get("model"),
        datasets=list(datasets),
        inputs=list(head.get("inputs") or []),
        blocks=dict(data.get("blocks", {})),
        params={slot: dict(values) for slot, values in data.get("params", {}).items()},
        parent=head.get("parent"),
        path=path,
        source=source,
    )


class Presets:
    def __init__(self, config) -> None:
        self.builtin = library_root() / "presets"
        self.workspace = lab_root(config) / "presets"

    def _path(self, name: str) -> tuple[str, Path] | None:
        for source, directory in (("builtin", self.builtin), ("workspace", self.workspace)):
            path = directory / f"{name}.toml"
            if path.is_file():
                return source, path
        return None

    def get(self, name: str) -> Preset:
        found = self._path(name)
        if found is None:
            path = Path(name).expanduser()
            if path.suffix == ".toml" and path.is_file():
                return parse(path.read_text(), path.stem, path, "file")
            import difflib

            close = difflib.get_close_matches(name, [p.name for p in self.list()], n=3)
            raise NotFound(f"No preset named {name!r}.",
                           (f"Did you mean: {', '.join(close)}? " if close else "") + "List them with: nawat preset ls")
        source, path = found
        return parse(path.read_text(), name, path, source)

    def list(self) -> list[Preset]:
        out: list[Preset] = []
        for source, directory in (("builtin", self.builtin), ("workspace", self.workspace)):
            if directory.is_dir():
                for path in sorted(directory.glob("*.toml")):
                    try:
                        out.append(parse(path.read_text(), path.stem, path, source))
                    except Exception as exc:  # noqa: BLE001 - one broken file must not hide the rest
                        out.append(Preset(path.stem, "?", f"BROKEN: {exc}", path=path, source=source))
        return out

    def save_new(self, preset: Preset) -> Preset:
        name = validate_name(preset.name)
        if self._path(name) is not None:
            raise Protected(f"A preset named {name!r} already exists.", "Presets are never overwritten; save under a new name.")
        self.workspace.mkdir(parents=True, exist_ok=True)
        path = self.workspace / f"{name}.toml"
        path.write_text(preset.to_toml())
        return self.get(name)

    def fork(
        self,
        source: str,
        new_name: str,
        *,
        sets: Mapping[str, str] = {},
        uses: Mapping[str, str] = {},
        model: str | None = None,
        datasets: list[str] | None = None,
        description: str | None = None,
    ) -> Preset:
        base = self.get(source)
        preset = Preset(
            name=new_name,
            backbone=base.backbone,
            description=description if description is not None else base.description,
            model=model or base.model,
            datasets=list(datasets) if datasets else list(base.datasets),
            inputs=list(base.inputs),
            blocks=dict(base.blocks),
            params={slot: dict(v) for slot, v in base.params.items()},
            parent=base.name,
        )
        apply_uses(preset, uses)
        apply_sets(preset, sets)
        return self.save_new(preset)


def apply_uses(preset: Preset, uses: Mapping[str, str]) -> None:
    """``--use dataset=my-data`` / ``--use rewards=format-tags,exact-answer``."""
    for slot, value in uses.items():
        if slot not in SLOTS:
            raise BlockError(f"{slot!r} is not a slot.", f"Slots are: {', '.join(SLOTS)}.")
        names = [v.strip() for v in value.split(",") if v.strip()]
        if slot in LIST_SLOTS:
            preset.blocks[slot] = names
        elif not names or names == ["none"]:
            preset.blocks.pop(slot, None)
        else:
            preset.blocks[slot] = names[0]


def apply_sets(preset: Preset, sets: Mapping[str, str]) -> None:
    """``--set train.learning_rate=1e-4`` → ``[params.backbone] learning_rate``.

    ``train`` is accepted as the name of the backbone's slot, because that is
    what its parameters are about.
    """
    for dotted, raw in sets.items():
        slot, name = split_param(dotted)
        preset.params.setdefault(slot, {})[name] = _parse_text(raw) if isinstance(raw, str) else raw


def split_param(dotted: str) -> tuple[str, str]:
    """``train.lr`` → (backbone, lr); ``rewards.exact-answer.weight`` → (rewards.exact-answer, weight).

    Slots that hold several blocks (rewards, eval, export) name the block too,
    because two reward functions can both have a ``weight``.
    """
    if "." not in dotted:
        raise BlockError(f"{dotted!r} does not say which block it belongs to.",
                         "Write it as <slot>.<name>, e.g. train.learning_rate or model.rank.")
    slot, name = dotted.split(".", 1)
    slot = "backbone" if slot in ("train", "backbone", "trainer") else slot
    if slot not in ("backbone", *SLOTS):
        raise BlockError(f"{slot!r} in {dotted!r} is not a slot.", f"Use train, or one of: {', '.join(SLOTS)}.")
    if slot in LIST_SLOTS:
        if "." not in name:
            raise BlockError(f"{dotted!r} must name the {slot} block as well.",
                             f"Write it as {slot}.<block>.<name>, e.g. rewards.exact-answer.weight.")
        block, name = name.split(".", 1)
        slot = f"{slot}.{block}"
    return slot, name
