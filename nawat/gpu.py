"""One reading of the GPU, from nvidia-smi.

Kept apart from the control plane so that ``nawat estimate`` and the run
record can ask for it without importing FastAPI.
"""

from __future__ import annotations

import subprocess
import time
from typing import Any

_CACHE: dict[str, Any] = {"at": 0.0, "value": None}


def gpu_info() -> dict[str, Any] | None:
    """One reading from nvidia-smi, cached briefly. None when there is no GPU."""
    now = time.time()
    if now - _CACHE["at"] < 2.0:
        return _CACHE["value"]
    value = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode == 0 and out.stdout.strip():
            name, used, total, util = [part.strip() for part in out.stdout.strip().splitlines()[0].split(",")]
            value = {
                "name": name,
                "memory_used": int(used) * 2**20,
                "memory_total": int(total) * 2**20,
                "utilization": int(util),
            }
    except (OSError, ValueError, subprocess.TimeoutExpired):
        value = None
    _CACHE.update(at=now, value=value)
    return value


__all__ = ["gpu_info"]
