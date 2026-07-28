"""Publication: upload, verify, then reclaim — in that order, every time
(FR-2.7, FR-2.8, NFR-1.2).
"""

from __future__ import annotations

import pytest

from nawat.errors import NotFound, VerificationFailed
from nawat.keys import Key

ADAPTER = "runs/2026-07-28-a91f/adapter"


def test_publishing_uploads_verifies_then_frees_the_local_copy(cache, make_artifact, store):
    source = make_artifact(600, files=3)

    result = cache.publish(source, ADAPTER)

    assert result.verification.ok
    assert result.bytes == 600
    assert result.local_removed
    assert not source.exists()
    assert len(store.list_prefix(Key.parse(ADAPTER))) == 3


def test_a_published_artifact_comes_back_by_key(cache, make_artifact):
    source = make_artifact(600, files=2)
    original = {path.name: path.read_bytes() for path in source.iterdir()}

    cache.publish(source, ADAPTER)
    restored = cache.resolve(ADAPTER, lease=False)

    assert {p.name: p.read_bytes() for p in restored.iterdir() if p.name != ".nawat-artifact.json"} == original


def test_a_failed_verification_keeps_the_local_copy_and_raises(cache, make_artifact, store, monkeypatch):
    source = make_artifact(600, files=2)
    monkeypatch.setattr(store, "upload", lambda *a, **k: {})  # the transfer silently wrote nothing

    with pytest.raises(VerificationFailed) as caught:
        cache.publish(source, ADAPTER)

    assert "has been kept" in caught.value.remedy
    assert source.exists()
    assert len(list(source.iterdir())) == 2


def test_keeping_a_published_artifact_registers_it_as_cached_and_replicated(cache, make_artifact):
    source = make_artifact(600, files=2)

    result = cache.publish(source, ADAPTER, keep_local=True)

    assert not result.local_removed
    assert cache.is_present(ADAPTER)
    status = cache.get(ADAPTER)
    assert status.replicated and status.bytes == 600
    assert cache.local_path(ADAPTER).exists()


def test_publishing_something_in_use_retains_it_rather_than_pulling_it_away(cache, make_artifact):
    """Serving may be reading the adapter that just finished training."""
    cache.publish(make_artifact(300), ADAPTER, keep_local=True)
    cache.leases.acquire([Key.parse(ADAPTER)], holder="inference session")

    result = cache.publish(cache.local_path(ADAPTER), ADAPTER)

    assert not result.local_removed
    assert "in use by inference session" in result.retained_reason
    assert cache.is_present(ADAPTER)


def test_publishing_frees_space_for_the_next_run(cache, cached, make_artifact):
    cached("models/base", 700)
    source = make_artifact(200, files=1)
    cache.add(source, "runs/x/adapter", publish=False)
    assert cache.status().used == 900

    cache.publish(cache.local_path("runs/x/adapter"), "runs/x/adapter")

    assert cache.status().used == 700
    assert not cache.is_present("runs/x/adapter")


def test_publishing_a_directory_that_is_not_there_says_so(cache, tmp_path):
    with pytest.raises(NotFound):
        cache.publish(tmp_path / "no-such-output", ADAPTER)


def test_publishing_an_empty_output_directory_says_so(cache, tmp_path):
    empty = tmp_path / "outputs"
    empty.mkdir()
    with pytest.raises(NotFound, match="nothing to publish|no files to publish"):
        cache.publish(empty, ADAPTER)


def test_adding_a_directory_caches_and_publishes_it(cache, make_artifact, store):
    source = make_artifact(400, files=2)

    status = cache.add(source, "datasets/ocr-arabic-v3")

    assert status.bytes == 400 and status.replicated
    assert cache.is_present("datasets/ocr-arabic-v3")
    assert store.exists(Key.parse("datasets/ocr-arabic-v3"))


def test_adding_without_publishing_leaves_it_unreplicated_and_therefore_safe(cache, make_artifact):
    status = cache.add(make_artifact(400), "datasets/scratch", publish=False)
    assert not status.replicated
    assert cache.collect(need=1000).evicted == []
