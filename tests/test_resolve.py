"""Resolution: local cache, then object storage, then the upstream hub — and a
hub fetch happens exactly once in an artifact's life (FR-1.2, FR-1.3, G2).
"""

from __future__ import annotations

import pytest

from nawat.cache import Cache
from nawat.errors import InsufficientSpace, NotFound, Offline
from nawat.hub import OfflineHub
from nawat.keys import Key

MODEL = "models/unsloth/Qwen2.5-VL-7B-Instruct"


def test_a_cached_artifact_resolves_without_touching_the_store(cache, cached, hub, store, monkeypatch):
    cached(MODEL, 300)
    monkeypatch.setattr(store, "download", lambda *a, **k: pytest.fail("should not have downloaded"))

    path = cache.resolve(MODEL, lease=False)

    assert path == cache.local_path(MODEL)
    assert hub.fetches == []


def test_an_artifact_in_object_storage_is_staged_from_there(cache, store, make_artifact, hub):
    store.publish(make_artifact(400, files=2), Key.parse(MODEL))
    assert not cache.is_present(MODEL)

    path = cache.resolve(MODEL, lease=False)

    assert path.exists()
    assert cache.get(MODEL).files == 2
    assert hub.fetches == []


def test_a_first_fetch_writes_through_to_object_storage(cache, hub, store):
    hub.offer(MODEL, {"config.json": b"{}", "weights.bin": b"w" * 400})

    cache.resolve(MODEL, lease=False)

    assert hub.fetches == [MODEL]
    assert set(store.list_prefix(Key.parse(MODEL))) == {"config.json", "weights.bin"}
    assert cache.get(MODEL).replicated


def test_an_artifact_is_downloaded_from_the_internet_exactly_once_ever(cache, hub):
    """G2, stated as an assertion: fetch, evict, resolve again — still one fetch."""
    hub.offer(MODEL, {"weights.bin": b"w" * 400})

    cache.resolve(MODEL, lease=False)
    cache.evict(MODEL)
    cache.resolve(MODEL, lease=False)
    cache.evict(MODEL)
    cache.resolve(MODEL, lease=False)

    assert hub.fetches == [MODEL]


def test_resolving_something_nowhere_says_where_it_looked(cache):
    with pytest.raises(NotFound):
        cache.resolve("models/does-not-exist", lease=False)


def test_an_offline_host_refuses_to_reach_the_internet(config, store):
    cache = Cache(config, store=store, hub=OfflineHub("this host is configured offline (NAWAT_OFFLINE)"))
    with pytest.raises(Offline) as caught:
        cache.resolve(MODEL, lease=False)
    assert "NAWAT_OFFLINE" in caught.value.remedy


def test_no_hub_flag_fails_rather_than_reaching_upstream(cache, hub):
    hub.offer(MODEL, {"weights.bin": b"w" * 100})
    with pytest.raises(NotFound):
        cache.resolve(MODEL, lease=False, allow_hub=False)
    assert hub.fetches == []


# -- staging is atomic -------------------------------------------------------


def test_a_failed_fetch_leaves_nothing_that_looks_complete(cache, hub, monkeypatch):
    """An interrupted transfer must never present as a finished artifact."""
    hub.offer(MODEL, {"weights.bin": b"w" * 400})
    real_fetch = hub.fetch

    def fetch_then_die(key, dest):
        real_fetch(key, dest)
        raise OSError("connection reset midway")

    monkeypatch.setattr(hub, "fetch", fetch_then_die)

    with pytest.raises(OSError):
        cache.resolve(MODEL, lease=False)

    assert not cache.local_path(MODEL).exists()
    assert cache.get(MODEL) is None
    assert list(cache.config.staging_root.iterdir()) == [], "staging must be cleaned up"


def test_a_fetch_that_cannot_fit_is_refused_before_anything_is_written(cache, cached, hub):
    cached("models/serving", 900)
    cache.leases.acquire([Key.parse("models/serving")], holder="inference session")
    hub.offer(MODEL, {"weights.bin": b"w" * 500})

    with pytest.raises(InsufficientSpace):
        cache.resolve(MODEL, lease=False)

    assert hub.fetches == []
    assert not cache.local_path(MODEL).exists()
    assert cache.is_present("models/serving")


# -- leases ------------------------------------------------------------------


def test_resolve_holds_the_artifact_for_this_process_by_default(cache, cached):
    cached(MODEL, 300)
    cache.resolve(MODEL)
    assert cache.get(MODEL).leased


def test_resolve_can_decline_to_hold(cache, cached):
    cached(MODEL, 300)
    cache.resolve(MODEL, lease=False)
    assert not cache.get(MODEL).leased


def test_holding_stages_several_inputs_and_releases_them_at_the_end(cache, cached):
    cached(MODEL, 200)
    cached("datasets/ocr-arabic-v3", 200)

    with cache.holding(MODEL, "datasets/ocr-arabic-v3", holder="trainer") as staged:
        assert set(staged) == {MODEL, "datasets/ocr-arabic-v3"}
        assert all(path.exists() for path in staged.values())
        assert cache.get(MODEL).leased

    assert not cache.get(MODEL).leased
    assert not cache.get("datasets/ocr-arabic-v3").leased


def test_holding_releases_even_when_the_body_raises(cache, cached):
    cached(MODEL, 200)
    with pytest.raises(RuntimeError):
        with cache.holding(MODEL, holder="trainer"):
            raise RuntimeError("the trainer crashed")
    assert not cache.get(MODEL).leased


# -- recovery ----------------------------------------------------------------


def test_the_cache_describes_itself_on_disk_and_survives_losing_the_database(config, store, hub, cached, cache):
    cached(MODEL, 300)
    cache.db.close()
    config.database_path.unlink()

    recovered = Cache(config, store=store, hub=hub)
    recovered.reconcile()

    status = recovered.get(MODEL)
    assert status is not None
    assert status.bytes == 300


def test_reconcile_forgets_artifacts_deleted_behind_its_back(cache, cached):
    import shutil

    cached(MODEL, 300)
    shutil.rmtree(cache.local_path(MODEL))

    cache.reconcile()

    assert cache.get(MODEL) is None
    assert cache.status().used == 0
