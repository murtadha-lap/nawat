"""Reading configuration out of a .env file.

Configuration is injected through the environment (PRD principle 4), but a file
named .env sitting in the project root should be part of that environment
without the researcher having to remember `set -a && . ./.env`. This module is
the bridge, and it is deliberately small: no dependency, no interpolation, no
surprises.

Real environment variables always win over the file, so an explicit `export`
still overrides for one command.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

FILENAME = ".env"
MAX_DEPTH = 6

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        body = value[1:-1]
        if value[0] == "'":
            return body  # single quotes are literal
        out, index = [], 0
        while index < len(body):
            char = body[index]
            if char == "\\" and index + 1 < len(body):
                out.append(_ESCAPES.get(body[index + 1], "\\" + body[index + 1]))
                index += 2
            else:
                out.append(char)
                index += 1
        return "".join(out)
    # Unquoted: an inline comment needs whitespace before the #, so that a value
    # like a URL fragment or a password containing # survives.
    cut = value.find(" #")
    if cut != -1:
        value = value[:cut]
    return value.rstrip()


def parse(text: str) -> dict[str, str]:
    """Parse .env content. Blank lines and comments are skipped; bad lines too."""
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE.match(line)
        if match:
            values[match.group(1)] = _unquote(match.group(2))
    return values


def find(start: Path | None = None) -> Path | None:
    """The nearest .env, walking up from ``start`` the way git finds its root."""
    directory = (start or Path.cwd()).resolve()
    for _ in range(MAX_DEPTH):
        candidate = directory / FILENAME
        if candidate.is_file():
            return candidate
        if directory.parent == directory:
            break
        directory = directory.parent
    return None


def load(path: Path) -> dict[str, str]:
    return parse(path.read_text(encoding="utf-8"))
