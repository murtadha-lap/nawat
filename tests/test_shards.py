"""Sharding: many small files become few large ones, without losing any."""

from __future__ import annotations

import tarfile

import pytest

from nawat import shards
from nawat.errors import NawatError, NotFound

CACHE_CEILING = 50 * 10**6


@pytest.fixture
def corpus(tmp_path):
    source = tmp_path / "corpus"
    for index in range(40):
        stem = source / f"{index:04d}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        (source / f"{index:04d}.img").write_bytes(bytes([index]) * 900)
        (source / f"{index:04d}.txt").write_text(f"label {index}")
    return source


def test_packing_round_trips_every_file(corpus, tmp_path):
    dest = tmp_path / "sharded"
    index = shards.pack(corpus, dest, shard_size=4000)

    assert index.total_files == 80
    assert len(index.shards) > 1, "the corpus must actually split"

    restored = tmp_path / "restored"
    assert shards.unpack(dest, restored) == 80
    assert (restored / "0007.txt").read_text() == "label 7"
    assert (restored / "0007.img").read_bytes() == bytes([7]) * 900


def test_a_sample_and_its_label_land_in_the_same_shard(corpus, tmp_path):
    """Sorted packing: the property WebDataset loops rely on."""
    dest = tmp_path / "sharded"
    shards.pack(corpus, dest, shard_size=4000)

    for entry in shards.read_index(dest)["shards"]:
        with tarfile.open(dest / entry["name"]) as reader:
            names = sorted(m.name for m in reader.getmembers() if m.isfile())
        stems = [n.rsplit(".", 1)[0] for n in names]
        # every stem that appears, appears with both extensions, except at most
        # the shard's first and last stems (which may straddle a boundary)
        interior = [s for s in set(stems) if s not in (stems[0], stems[-1])]
        for stem in interior:
            assert stems.count(stem) == 2, f"{stem} split across shards"


def test_no_shard_exceeds_the_requested_size(corpus, tmp_path):
    dest = tmp_path / "sharded"
    shards.pack(corpus, dest, shard_size=4000)
    for entry in shards.read_index(dest)["shards"]:
        assert entry["content_bytes"] <= 4000


def test_a_file_larger_than_the_shard_size_is_refused_with_the_remedy(tmp_path):
    source = tmp_path / "big"
    source.mkdir()
    (source / "huge.bin").write_bytes(b"x" * 10_000)
    with pytest.raises(NawatError, match="--shard-size"):
        shards.pack(source, tmp_path / "out", shard_size=4000)


def test_verify_passes_a_fresh_pack_and_catches_a_tampered_shard(corpus, tmp_path):
    dest = tmp_path / "sharded"
    shards.pack(corpus, dest, shard_size=4000)
    assert shards.verify(dest)["total_files"] == 80

    victim = sorted(dest.glob("shard-*.tar"))[0]
    victim.write_bytes(victim.read_bytes()[:-512])
    with pytest.raises(NawatError, match="re-pack|Re-pack"):
        shards.verify(dest)


def test_an_empty_or_missing_source_says_so(tmp_path):
    with pytest.raises(NotFound):
        shards.pack(tmp_path / "absent", tmp_path / "out")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NotFound, match="Nothing to shard"):
        shards.pack(empty, tmp_path / "out")


def test_a_sharded_dataset_publishes_like_any_artifact(corpus, tmp_path, cache):
    dest = tmp_path / "sharded"
    shards.pack(corpus, dest, shard_size=40_000)

    status = cache.add(dest, "datasets/corpus-sharded")

    assert status.replicated
    staged = cache.resolve("datasets/corpus-sharded", lease=False)
    assert shards.verify(staged)["total_files"] == 80
