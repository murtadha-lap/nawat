from __future__ import annotations

from pathlib import Path

import pytest

from nawat.errors import InvalidKey
from nawat.keys import Key


def test_key_maps_one_to_one_onto_storage_and_disk():
    key = Key.parse("models/unsloth/Qwen2.5-VL-7B-Instruct")
    assert key.kind == "models"
    assert str(key) == "models/unsloth/Qwen2.5-VL-7B-Instruct"
    assert key.object_prefix == "models/unsloth/Qwen2.5-VL-7B-Instruct/"
    assert key.local_path(Path("/cache")) == Path("/cache/models/unsloth/Qwen2.5-VL-7B-Instruct")


def test_model_and_dataset_keys_double_as_hub_repo_ids():
    assert Key.parse("models/unsloth/Qwen2.5-VL-7B-Instruct").hub_repo_id == "unsloth/Qwen2.5-VL-7B-Instruct"
    assert Key.parse("datasets/HuggingFaceM4/DocumentVQA").hub_repo_id == "HuggingFaceM4/DocumentVQA"
    assert Key.parse("runs/2026-07-28-a91f/adapter").hub_repo_id is None


def test_parse_is_idempotent_and_tolerates_surrounding_slashes():
    key = Key.parse("/models/gemma-3-4b/")
    assert str(key) == "models/gemma-3-4b"
    assert Key.parse(key) is key


@pytest.mark.parametrize(
    "raw",
    [
        "models/../../etc/passwd",
        "models/./weights",
        "models//weights",
        "models\\weights",
        "../models/x",
        "models",
        "",
        "   ",
        "weights/foo",  # not a known kind
        "models/-leading-dash",
        "models/with space",
    ],
)
def test_keys_that_could_escape_or_confuse_are_refused(raw):
    with pytest.raises(InvalidKey):
        Key.parse(raw)


def test_traversal_cannot_escape_the_cache_root():
    # Belt and braces: even if a bad key slipped through parsing, resolution of
    # a valid key never leaves the root.
    root = Path("/cache")
    key = Key.parse("runs/2026-07-28-a91f/adapter")
    assert root in key.local_path(root).parents


def test_child_derives_sub_keys():
    run = Key.parse("runs/2026-07-28-a91f")
    assert str(run.child("adapter")) == "runs/2026-07-28-a91f/adapter"
    assert str(run.child("exports", "gguf")) == "runs/2026-07-28-a91f/exports/gguf"


def test_keys_sort_stably_for_display():
    keys = [Key.parse("models/b"), Key.parse("datasets/a"), Key.parse("models/a")]
    assert [str(k) for k in sorted(keys)] == ["datasets/a", "models/a", "models/b"]
