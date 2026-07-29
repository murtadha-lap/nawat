"""VRAM and budget estimation — the gate in front of wasted GPU hours.

The point is not precision; it is catching the proposal that obviously cannot
run before it costs a review, or hours (FR-6.6, PRD §8). Everything here is an
estimate and says so. The heuristics favour over-estimating, because "it fit
after all" costs nothing and "it OOMed at step 40" costs the evening.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .units import human_bytes

GB = 10**9

#: CUDA context, allocator slack, kernels.
FIXED_OVERHEAD = 1.5 * GB

#: Fraction of parameters a typical LoRA (r=16, most modules) trains.
LORA_FRACTION = 0.02

#: AdamW: fp32 master + two moments per trainable parameter.
OPTIMIZER_BYTES_PER_PARAM = 12

_SIZE_SUFFIX = re.compile(r"([\d.]+)\s*([bm])\b", re.IGNORECASE)


def parse_params(text: str) -> int:
    """``"7B"`` or ``"850M"`` as a parameter count."""
    match = _SIZE_SUFFIX.search(text)
    if not match:
        raise ValueError(f"cannot read {text!r} as a parameter count — write it like 7B or 850M")
    scale = 10**9 if match.group(2).lower() == "b" else 10**6
    return int(float(match.group(1)) * scale)


def params_from_bytes(weight_bytes: int, *, quantized: bool = False) -> int:
    """Parameter count inferred from checkpoint size (fp16/bf16 ≈ 2 B/param)."""
    return int(weight_bytes / (0.55 if quantized else 2.0))


@dataclass(frozen=True)
class Estimate:
    params: int
    method: str
    bits: int
    batch: int
    seq_len: int
    grad_checkpointing: bool
    weights: int
    optimizer: int
    gradients: int
    activations: int
    overhead: int
    vram_total: int
    vram_available: int | None

    @property
    def fits(self) -> bool | None:
        if self.vram_available is None:
            return None
        return self.vram_total <= self.vram_available

    def to_json(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "method": self.method,
            "bits": self.bits,
            "batch": self.batch,
            "seq_len": self.seq_len,
            "vram_total": self.vram_total,
            "vram_total_human": human_bytes(self.vram_total),
            "vram_available": self.vram_available,
            "fits": self.fits,
            "breakdown": {
                "weights": self.weights,
                "optimizer": self.optimizer,
                "gradients": self.gradients,
                "activations": self.activations,
                "overhead": self.overhead,
            },
        }

    def verdict(self) -> str:
        total = human_bytes(self.vram_total)
        if self.vram_available is None:
            return f"Estimated {total} of VRAM. No GPU reading available to compare against."
        available = human_bytes(self.vram_available)
        if self.fits:
            return f"Estimated {total} of VRAM against {available} available — should fit."
        return (
            f"Estimated {total} of VRAM against {available} available — will not fit. "
            f"Lower the batch size or sequence length, quantise to 4-bit, or enable gradient checkpointing."
        )


def estimate(
    *,
    params: int,
    method: str = "lora",
    bits: int = 16,
    batch: int = 2,
    seq_len: int = 2048,
    grad_checkpointing: bool = True,
    vram_available: int | None = None,
) -> Estimate:
    """A defensible upper-ish bound on training VRAM."""
    weights = int(params * bits / 8)
    trainable = params if method == "full" else int(params * LORA_FRACTION)
    optimizer = trainable * OPTIMIZER_BYTES_PER_PARAM
    gradients = trainable * 2  # bf16 grads

    # Activations scale with tokens in flight and model size; checkpointing
    # trades most of them for recompute. Calibrated loosely against observed
    # peaks for 1–8B models at seq 2048.
    tokens = batch * seq_len
    per_token = (params ** 0.5) * 55
    activations = int(tokens * per_token * (0.35 if grad_checkpointing else 1.0))

    total = weights + optimizer + gradients + activations + int(FIXED_OVERHEAD)
    return Estimate(
        params=params, method=method, bits=bits, batch=batch, seq_len=seq_len,
        grad_checkpointing=grad_checkpointing,
        weights=weights, optimizer=optimizer, gradients=gradients,
        activations=activations, overhead=int(FIXED_OVERHEAD),
        vram_total=total, vram_available=vram_available,
    )


#: Trainer kwargs a proposed training script is scanned for, so an unrunnable one is
#: flagged before it reaches review.
_KWARG = {
    "batch": re.compile(r"per_device_train_batch_size\s*=\s*(\d+)"),
    "seq_len": re.compile(r"max_(?:seq_)?length\s*=\s*(\d+)"),
    "load_in_4bit": re.compile(r"load_in_4bit\s*=\s*(True|False)"),
    "grad_checkpointing": re.compile(r"use_gradient_checkpointing\s*=\s*[\"']?(\w+)"),
}


def scan_script(source: str) -> dict[str, Any]:
    """Best-effort read of the training configuration inside a script."""
    found: dict[str, Any] = {}
    for name, pattern in _KWARG.items():
        match = pattern.search(source)
        if not match:
            continue
        value = match.group(1)
        if name in ("batch", "seq_len"):
            found[name] = int(value)
        elif name == "load_in_4bit":
            found[name] = value == "True"
        else:
            found[name] = value not in ("False", "None")
    return found


def estimate_script(source: str, *, params: int, vram_available: int | None = None) -> Estimate:
    scanned = scan_script(source)
    return estimate(
        params=params,
        bits=4 if scanned.get("load_in_4bit") else 16,
        batch=scanned.get("batch", 2),
        seq_len=scanned.get("seq_len", 2048),
        grad_checkpointing=scanned.get("grad_checkpointing", True),
        vram_available=vram_available,
    )
