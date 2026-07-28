"""Bring-up checks. The operator reads these when something is wrong, so each
failure must name the cause and the fix.
"""

from __future__ import annotations

from dataclasses import replace

from nawat.cache import Cache
from nawat.health import PROBE, run_checks

from .conftest import UnavailableStore


def _by_name(checks) -> dict[str, object]:
    return {check.name: check for check in checks}


def test_a_healthy_host_passes_every_check(cache):
    checks = run_checks(cache)
    assert all(check.ok for check in checks), [c for c in checks if not c.ok]
    assert "object storage round trip" in _by_name(checks)


def test_the_probe_object_is_cleaned_up(cache, store):
    run_checks(cache)
    assert not store.exists(PROBE)


def test_an_unreachable_store_reports_the_cause_and_stops(cache, config):
    cache._store = UnavailableStore(config.local_store_root, workers=2)
    checks = run_checks(cache)
    failing = checks[-1]
    assert failing.name == "object storage reachable"
    assert not failing.ok
    assert "Check the endpoint." == failing.remedy
    assert "object storage round trip" not in _by_name(checks), "no point round-tripping an unreachable store"


def test_a_ceiling_larger_than_the_disk_is_reported(config, store, hub):
    cache = Cache(replace(config, cache_ceiling=900 * 10**12), store=store, hub=hub)
    check = _by_name(run_checks(cache))["cache ceiling"]
    assert not check.ok
    assert "NAWAT_CACHE_CEILING" in check.remedy


def test_a_read_only_cache_root_is_reported(config, store, hub, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.mkdir(mode=0o500)
    try:
        cache = Cache(replace(config, cache_root=blocked / "cache"), store=store, hub=hub)
    except OSError:
        return  # ensure_dirs already refused, which is the same signal
    check = _by_name(run_checks(cache))["cache root"]
    assert not check.ok
    assert "NAWAT_CACHE_ROOT" in check.remedy
