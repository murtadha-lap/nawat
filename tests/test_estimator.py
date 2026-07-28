"""The estimator: catch the unrunnable proposal before it costs GPU hours."""

from __future__ import annotations

import pytest

from nawat.estimator import (
    estimate,
    estimate_script,
    params_from_bytes,
    parse_params,
    scan_script,
)

GB = 10**9


def test_parameter_counts_parse_like_model_names():
    assert parse_params("7B") == 7 * 10**9
    assert parse_params("850M") == 850 * 10**6
    assert parse_params("Qwen 0.8b vision") == int(0.8 * 10**9)
    with pytest.raises(ValueError):
        parse_params("large")


def test_params_infer_from_checkpoint_size():
    assert params_from_bytes(16 * GB) == 8 * GB  # fp16: 2 bytes per param
    assert params_from_bytes(1 * GB, quantized=True) > params_from_bytes(1 * GB)


def test_a_small_lora_fits_a_16gb_card():
    """The notebook's case: 0.8B model, 16-bit LoRA, batch 2, seq 2048."""
    result = estimate(params=800_000_000, method="lora", bits=16, batch=2, seq_len=2048,
                      vram_available=16 * GB)
    assert result.fits is True
    assert result.vram_total < 8 * GB, "estimate should be sane, not just under the card"


def test_a_full_finetune_of_7b_does_not_fit_16gb():
    result = estimate(params=7 * 10**9, method="full", bits=16, batch=2, seq_len=2048,
                      vram_available=16 * GB)
    assert result.fits is False
    assert "will not fit" in result.verdict()
    assert "batch size" in result.verdict(), "the verdict names the remedies"


def test_the_estimate_is_monotonic_in_batch_and_sequence():
    smaller = estimate(params=10**9, batch=1, seq_len=1024)
    bigger = estimate(params=10**9, batch=8, seq_len=4096)
    assert bigger.vram_total > smaller.vram_total


def test_checkpointing_and_quantisation_reduce_the_estimate():
    base = estimate(params=10**9, grad_checkpointing=False, bits=16)
    checkpointed = estimate(params=10**9, grad_checkpointing=True, bits=16)
    quantized = estimate(params=10**9, grad_checkpointing=True, bits=4)
    assert checkpointed.vram_total < base.vram_total
    assert quantized.vram_total < checkpointed.vram_total


def test_no_gpu_reading_gives_an_estimate_without_a_verdict():
    result = estimate(params=10**9)
    assert result.fits is None
    assert "No GPU reading" in result.verdict()


def test_scan_reads_the_training_configuration_out_of_a_script():
    source = """
model, tok = FastVisionModel.from_pretrained(MODEL_DIR, load_in_4bit = True,
    use_gradient_checkpointing = "unsloth")
args = SFTConfig(per_device_train_batch_size = 4, max_length = 4096)
"""
    found = scan_script(source)
    assert found == {"batch": 4, "seq_len": 4096, "load_in_4bit": True, "grad_checkpointing": True}


def test_estimate_script_combines_scan_and_size():
    tight = estimate_script("per_device_train_batch_size = 64\nmax_length = 8192\n",
                            params=8 * 10**9, vram_available=16 * GB)
    assert tight.fits is False
