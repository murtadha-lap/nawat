"""Size and duration formatting.

Decimal suffixes (KB, MB, GB) are powers of 1000; binary suffixes (KiB, MiB,
GiB) are powers of 1024. A bare number is bytes. Output is decimal, matching
how model and dataset sizes are quoted everywhere else.
"""

from __future__ import annotations

import re

_DECIMAL = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12, "PB": 10**15}
_BINARY = {"KIB": 2**10, "MIB": 2**20, "GIB": 2**30, "TIB": 2**40, "PIB": 2**50}
_SUFFIXES = {**_DECIMAL, **_BINARY, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12}

_SIZE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([A-Za-z]*)\s*$")


def parse_size(text: str | int | float) -> int:
    """Parse ``"150GB"``, ``"12 GiB"`` or ``"1024"`` into a byte count."""
    if isinstance(text, (int, float)):
        return int(text)
    match = _SIZE.match(text)
    if not match:
        raise ValueError(f"cannot read {text!r} as a size — write it like 150GB or 12GiB")
    number, suffix = match.groups()
    unit = _SUFFIXES.get(suffix.upper() or "B")
    if unit is None:
        raise ValueError(f"unknown size unit {suffix!r} — use B, KB, MB, GB, TB, or the KiB/MiB/GiB forms")
    return int(float(number) * unit)


def human_bytes(count: int | float) -> str:
    """Format a byte count for display: ``16.0 GB``, ``212 MB``, ``0 B``."""
    count = float(count)
    sign = "-" if count < 0 else ""
    count = abs(count)
    for unit, scale in (("PB", 10**15), ("TB", 10**12), ("GB", 10**9), ("MB", 10**6), ("KB", 10**3)):
        if count >= scale:
            value = count / scale
            precision = 1 if value < 100 else 0
            return f"{sign}{value:.{precision}f} {unit}"
    return f"{sign}{int(count)} B"


def human_age(seconds: float) -> str:
    """Format an elapsed time compactly: ``42s``, ``17m``, ``3h``, ``12d``."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def bar(fraction: float, width: int = 24) -> str:
    """A plain occupancy bar. No colour, no animation — it is a reading."""
    fraction = min(max(fraction, 0.0), 1.0)
    filled = int(round(fraction * width))
    return "█" * filled + "░" * (width - filled)
