"""Eviction is the path that can destroy work, so it is the path with the most
tests. Per PRD §12 these are a release gate: a change that breaks any of them
should not ship.

The invariant under test throughout: nothing leaves local disk until it has
been verified present in object storage, at the moment of deletion.
"""

from __future__ import annotations

import pytest

from nawat.errors import InsufficientSpace, Protected, VerificationFailed
from nawat.keys import Key

from .conftest import UnavailableStore


def age(cache, key: str, seconds_ago: float) -> None:
    """Backdate an artifact's last use, so LRU order is deterministic."""
    import time

    with cache.db.tx() as conn:
        conn.execute("UPDATE artifacts SET last_used = ? WHERE key = ?", (time.time() - seconds_ago, key))


# -- the ordering ------------------------------------------------------------


def test_least_recently_used_goes_first(cache, cached):
    cached("models/oldest", 300)
    cached("models/middle", 300)
    cached("models/newest", 300)
    age(cache, "models/oldest", 300)
    age(cache, "models/middle", 200)
    age(cache, "models/newest", 100)

    result = cache.collect(need=200)

    assert [str(k) for k in result.evicted] == ["models/oldest"]
    assert not cache.is_present("models/oldest")
    assert cache.is_present("models/newest")


def test_collect_stops_as_soon_as_there_is_room(cache, cached):
    for name in ("a", "b", "c"):
        cached(f"models/{name}", 300)
    age(cache, "models/a", 300)
    age(cache, "models/b", 200)
    age(cache, "models/c", 100)

    result = cache.collect(need=400)  # deficit is 900 + 400 - 1000 = 300

    assert len(result.evicted) == 1
    assert result.freed == 300


def test_collect_with_no_request_returns_to_the_ceiling(cache, cached):
    for name in ("a", "b", "c", "d"):
        cached(f"models/{name}", 300)  # 1200 against a 1000 ceiling
    assert cache.status().used == 1200

    result = cache.collect()

    assert result.freed == 300
    assert cache.status().used <= cache.config.cache_ceiling


def test_collect_does_nothing_when_already_under_the_ceiling(cache, cached):
    cached("models/a", 300)
    result = cache.collect()
    assert result.evicted == []
    assert result.target == 0
    assert cache.is_present("models/a")


# -- the refusals ------------------------------------------------------------


def test_an_artifact_not_in_object_storage_is_never_evicted(cache, cached):
    """The failure this product exists to prevent: losing the only copy."""
    cached("runs/2026-07-28-a91f/adapter", 600, replicated=False)
    cached("models/base", 600)
    age(cache, "runs/2026-07-28-a91f/adapter", 999)  # oldest, so first in line

    result = cache.collect(need=400)

    assert cache.is_present("runs/2026-07-28-a91f/adapter")
    assert [str(k) for k in result.evicted] == ["models/base"]
    reasons = {str(k): reason for k, reason in result.skipped}
    assert "not verified" in reasons["runs/2026-07-28-a91f/adapter"]


def test_an_artifact_kept_on_disk_is_never_evicted(cache, cached):
    cached("models/pinned", 600)
    cached("models/other", 600)
    age(cache, "models/pinned", 999)
    cache.pin("models/pinned")

    result = cache.collect(need=400)

    assert cache.is_present("models/pinned")
    assert ("kept on disk") in dict((str(k), r) for k, r in result.skipped)["models/pinned"]


def test_an_artifact_in_use_is_never_evicted(cache, cached):
    cached("models/serving", 600)
    cached("models/other", 600)
    age(cache, "models/serving", 999)
    cache.leases.acquire([Key.parse("models/serving")], holder="inference session")

    result = cache.collect(need=400)

    assert cache.is_present("models/serving")
    assert "in use by inference session" in dict((str(k), r) for k, r in result.skipped)["models/serving"]


def test_nothing_is_evicted_while_object_storage_is_unreachable(cache, cached, config):
    """Verification cannot be completed, so the platform keeps the data and
    reports a full disk instead (PRD principle 2)."""
    cached("models/a", 600)
    cached("models/b", 600)
    cache._store = UnavailableStore(config.local_store_root, workers=2)

    result = cache.collect(need=400)

    assert result.evicted == []
    assert cache.is_present("models/a") and cache.is_present("models/b")
    assert all("not verified" in reason for _, reason in result.skipped)


def test_a_replica_that_has_drifted_blocks_eviction(cache, cached, store):
    """Present-and-listed is not enough; sizes must agree file for file."""
    cached("models/drifted", 600)
    cached("models/clean", 300)
    age(cache, "models/drifted", 999)
    remote = store._dir(Key.parse("models/drifted"))
    truncated = sorted(p for p in remote.iterdir() if p.is_file())[0]
    truncated.write_bytes(b"short")

    result = cache.collect(need=200)

    assert cache.is_present("models/drifted")
    assert "differ in size" in dict((str(k), r) for k, r in result.skipped)["models/drifted"]


def test_a_dry_run_deletes_nothing(cache, cached):
    for name in ("a", "b", "c"):
        cached(f"models/{name}", 300)
    age(cache, "models/a", 300)

    result = cache.collect(need=400, dry_run=True)

    assert [str(k) for k in result.evicted] == ["models/a"]
    assert cache.is_present("models/a")
    assert cache.status().used == 900


# -- refusing rather than guessing -------------------------------------------


def test_ensure_space_refuses_with_the_cause_and_the_remedy(cache, cached):
    cached("models/serving", 900)
    cache.leases.acquire([Key.parse("models/serving")], holder="inference session")

    with pytest.raises(InsufficientSpace) as caught:
        cache.ensure_space(500)

    error = caught.value
    assert "Not enough space for 500 B" in error.cause
    assert "in use by inference session" in error.remedy
    assert "raise the ceiling" in error.remedy
    assert any("models/serving" in line for line in error.held)
    assert cache.is_present("models/serving"), "a refusal deletes nothing"


def test_ensure_space_succeeds_quietly_when_there_is_room(cache, cached):
    cached("models/a", 300)
    result = cache.ensure_space(200)
    assert result.evicted == []
    assert cache.is_present("models/a")


def test_ensure_space_ignores_a_request_for_nothing(cache):
    assert cache.ensure_space(0).target == 0


# -- explicit removal --------------------------------------------------------


def test_removing_an_unreplicated_artifact_by_hand_is_refused(cache, cached):
    cached("runs/x/adapter", 300, replicated=False)

    with pytest.raises(VerificationFailed) as caught:
        cache.evict("runs/x/adapter")

    assert "nawat publish" in caught.value.remedy
    assert cache.is_present("runs/x/adapter")


def test_force_removes_an_unreplicated_artifact(cache, cached):
    cached("runs/x/adapter", 300, replicated=False)
    assert cache.evict("runs/x/adapter", force=True) == 300
    assert not cache.is_present("runs/x/adapter")


def test_removing_something_kept_or_in_use_is_refused(cache, cached):
    cached("models/kept", 300)
    cache.pin("models/kept")
    with pytest.raises(Protected, match="kept on disk"):
        cache.evict("models/kept")

    cached("models/busy", 300)
    cache.leases.acquire([Key.parse("models/busy")], holder="trainer")
    with pytest.raises(Protected, match="in use by trainer"):
        cache.evict("models/busy")


def test_releasing_a_kept_artifact_makes_it_evictable_again(cache, cached):
    cached("models/a", 600)
    cached("models/b", 600)
    cache.pin("models/a")
    age(cache, "models/a", 999)

    assert cache.collect(need=400).evicted == [Key.parse("models/b")]

    cached("models/c", 600)
    cache.unpin("models/a")
    assert Key.parse("models/a") in cache.collect(need=400).evicted


# -- eviction is safe because the artifact comes back ------------------------


def test_an_evicted_artifact_returns_from_object_storage_not_the_internet(cache, cached, hub):
    cached("models/base", 900)
    cache.evict("models/base")
    assert not cache.is_present("models/base")

    path = cache.resolve("models/base", lease=False)

    assert path.exists()
    assert sum(f.stat().st_size for f in path.rglob("*") if f.is_file() and f.name != ".nawat-artifact.json") == 900
    assert hub.fetches == [], "a cached-then-evicted artifact must never be re-downloaded from the hub"


def test_eviction_leaves_no_empty_directories_behind(cache, cached):
    cached("runs/2026-07-28-a91f/adapter", 300)
    cache.evict("runs/2026-07-28-a91f/adapter")
    assert not (cache.config.cache_root / "runs" / "2026-07-28-a91f").exists()
    assert not (cache.config.cache_root / "runs").exists()
