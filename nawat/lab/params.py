"""Declared parameters — the only knobs an experiment has.

A hidden parameter is a number written inline in training code: nobody is asked
about it, the run record does not show it, and two experiments that differ only
in it look identical afterwards. So every value a backbone or block uses is
declared up front, with its default and a line of help, and read back through a
namespace that refuses names it was not given::

    PARAMS = {
        "learning_rate": P(2e-4, "Peak learning rate"),
        "lr_scheduler_type": P("linear", "LR schedule", choices=("linear", "cosine", "constant")),
    }

    def run(ctx, p):
        ... learning_rate=p.learning_rate ...   # p.lr would raise, not default

The plan shows every declared value, where it came from (default, preset, or
``--set``), and the experiment cannot start until someone has read that table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..errors import NawatError

_MISSING = object()


class ParamError(NawatError):
    exit_code = 2


@dataclass(frozen=True)
class P:
    """One declared parameter."""

    default: Any
    help: str = ""
    choices: tuple | None = None
    #: The type a ``--set name=value`` string is converted to. Inferred from the
    #: default when omitted; give it when the default is None.
    type: type | None = None

    @property
    def kind(self) -> type | None:
        if self.type is not None:
            return self.type
        if self.default is None:
            return None
        return type(self.default)

    def coerce(self, name: str, value: Any) -> Any:
        """Bring a value from TOML or the command line to the declared type."""
        kind = self.kind
        if isinstance(value, str) and kind is not str:
            value = _parse_text(value)
        if value is None:
            return None
        if kind is bool:
            if isinstance(value, bool):
                pass
            elif isinstance(value, (int, float)) and value in (0, 1):
                value = bool(value)
            else:
                raise ParamError(f"{name} must be true or false, not {value!r}.")
        elif kind is int:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                raise ParamError(f"{name} must be a whole number, not {value!r}.")
            value = int(value)
        elif kind is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParamError(f"{name} must be a number, not {value!r}.")
            value = float(value)
        elif kind is str:
            value = str(value)
        elif kind in (list, tuple):
            if not isinstance(value, (list, tuple)):
                value = [value]
            value = list(value)
        elif kind is dict:
            if not isinstance(value, dict):
                raise ParamError(f"{name} must be a table / JSON object, not {value!r}.")
        if self.choices is not None and value not in self.choices:
            raise ParamError(
                f"{name}={value!r} is not one of the allowed values.",
                f"Choose from: {', '.join(map(str, self.choices))}.",
            )
        return value


def _parse_text(text: str) -> Any:
    """Read a command-line value the way a person means it."""
    stripped = text.strip()
    lowered = stripped.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null"):
        return None
    try:
        return json.loads(stripped)
    except ValueError:
        return stripped


class Params:
    """The values one block runs with. Reading an undeclared name is an error."""

    def __init__(self, owner: str, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_values", dict(values))

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        if name.startswith("__"):
            raise AttributeError(name)
        owner = object.__getattribute__(self, "_owner")
        raise ParamError(
            f"{owner} read a parameter it does not declare: {name!r}.",
            f"Add {name!r} to PARAMS in {owner} so it appears in the plan, rather than reading it silently.",
        )

    def __setattr__(self, name: str, value: Any) -> None:
        raise ParamError("Parameters are read-only while an experiment runs.", "Change them in the preset and plan again.")

    def __getitem__(self, name: str) -> Any:
        return self.__getattr__(name)

    def __contains__(self, name: str) -> bool:
        return name in object.__getattribute__(self, "_values")

    def get(self, name: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_values").get(name, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(object.__getattribute__(self, "_values"))

    def __repr__(self) -> str:
        return f"Params({object.__getattribute__(self, '_owner')}, {self.as_dict()!r})"


@dataclass
class Resolved:
    """A slot's parameters after defaults, the preset and ``--set`` are applied."""

    values: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)


def resolve(
    owner: str,
    declared: Mapping[str, P],
    layers: Iterable[tuple[str, Mapping[str, Any]]],
) -> Resolved:
    """Apply each layer of overrides on top of the declared defaults.

    Every override must name a declared parameter: a typo in a preset or on the
    command line is caught here instead of being silently ignored.
    """
    out = Resolved()
    for name, spec in declared.items():
        out.values[name] = spec.coerce(f"{owner}.{name}", spec.default) if spec.default is not None else None
        out.sources[name] = "default"
    for source, overrides in layers:
        for name, value in overrides.items():
            if name not in declared:
                raise ParamError(
                    f"{owner} has no parameter {name!r} (set by {source}).",
                    _suggest(name, declared) or "List every parameter with: nawat preset show <preset>",
                )
            out.values[name] = declared[name].coerce(f"{owner}.{name}", value)
            out.sources[name] = source
    return out


def _suggest(name: str, declared: Mapping[str, P]) -> str:
    import difflib

    close = difflib.get_close_matches(name, list(declared), n=3)
    return f"Did you mean: {', '.join(close)}?" if close else ""


def format_value(value: Any) -> str:
    if isinstance(value, str):
        return value if value else '""'
    if isinstance(value, float):
        return f"{value:g}"
    return json.dumps(value) if isinstance(value, (list, dict)) else str(value)
