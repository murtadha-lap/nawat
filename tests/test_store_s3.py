"""The S3 backend, driven over real HTTP.

The local backend covers the safety logic; this covers the parts only a wire
protocol exercises — pagination past the 1000-key list limit, parallel multipart
transfer, and the error wrapping that turns a botocore exception into something
a researcher can act on.
"""

from __future__ import annotations

import pytest

pytest.importorskip("moto", reason="install the dev extras to exercise the S3 backend")

from moto.server import ThreadedMotoServer  # noqa: E402

from nawat.cache import Cache  # noqa: E402
from nawat.config import Config  # noqa: E402
from nawat.errors import StoreUnavailable  # noqa: E402
from nawat.keys import Key  # noqa: E402
from nawat.store import S3ObjectStore, scan_local  # noqa: E402

BUCKET = "nawat-test"
KEY = Key.parse("runs/2026-07-28-a91f/adapter")


@pytest.fixture(scope="module")
def endpoint():
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def s3_config(tmp_path, endpoint) -> Config:
    return Config(
        cache_root=tmp_path / "cache",
        workspace_root=tmp_path / "workspace",
        state_dir=tmp_path / "state",
        cache_ceiling=100 * 10**6,
        min_free=0,
        store_backend="s3",
        bucket=BUCKET,
        endpoint=endpoint,
        region="us-east-1",
        access_key="test-access-key",
        secret_key="test-secret-key",
        transfer_workers=4,
        multipart_threshold=5 * 2**20,
        multipart_chunk=5 * 2**20,
    )


@pytest.fixture
def s3_store(s3_config: Config) -> S3ObjectStore:
    store = S3ObjectStore(s3_config)
    try:
        store.client.create_bucket(Bucket=BUCKET)
    except Exception:  # already created by an earlier test in the module
        pass
    for prefix in ("models", "datasets", "runs", "exports"):
        listed = store.client.list_objects_v2(Bucket=BUCKET, Prefix=prefix).get("Contents", [])
        if listed:
            store.client.delete_objects(Bucket=BUCKET, Delete={"Objects": [{"Key": o["Key"]} for o in listed]})
    return store


def test_round_trip_over_http(s3_store, make_artifact, tmp_path):
    source = make_artifact(64_000, files=5)
    result = s3_store.publish(source, KEY)
    assert result.ok and result.local_files == 5

    dest = tmp_path / "back"
    s3_store.download(KEY, dest)
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == {p.name: p.read_bytes() for p in source.iterdir()}


def test_multipart_transfer_of_a_file_above_the_threshold(s3_store, make_artifact, tmp_path):
    source = make_artifact(12 * 2**20, files=1)  # over the 5 MiB threshold
    s3_store.publish(source, KEY)

    dest = tmp_path / "back"
    s3_store.download(KEY, dest)
    assert scan_local(dest)["part-0.bin"].size == 12 * 2**20


def test_listing_pages_past_the_thousand_key_limit(s3_store, tmp_path):
    source = tmp_path / "many"
    source.mkdir()
    for index in range(1100):
        (source / f"shard-{index:05d}.bin").write_bytes(b"x" * 8)

    s3_store.upload(source, KEY)

    manifest = s3_store.list_prefix(KEY)
    assert len(manifest) == 1100, "a truncated listing would make verification pass on a partial replica"
    assert s3_store.verify(source, KEY).ok


def test_a_prefix_never_matches_a_sibling_with_the_same_stem(s3_store, make_artifact):
    s3_store.publish(make_artifact(1024), Key.parse("models/base"))
    s3_store.publish(make_artifact(2048), Key.parse("models/base-v2"))

    assert s3_store.total_size(Key.parse("models/base")) == 1024
    assert s3_store.total_size(Key.parse("models/base-v2")) == 2048


def test_an_unreachable_endpoint_says_what_to_check(s3_config, tmp_path):
    from dataclasses import replace

    store = S3ObjectStore(replace(s3_config, endpoint="http://127.0.0.1:1"))
    with pytest.raises(StoreUnavailable) as caught:
        store.list_prefix(KEY)
    assert "127.0.0.1:1" in caught.value.remedy
    assert "reachable" in caught.value.remedy


def test_a_missing_bucket_says_how_to_fix_it(s3_config):
    from dataclasses import replace

    store = S3ObjectStore(replace(s3_config, bucket="no-such-bucket"))
    with pytest.raises(StoreUnavailable) as caught:
        store.list_prefix(KEY)
    assert "NAWAT_S3_BUCKET" in caught.value.remedy


def test_the_whole_cache_cycle_against_a_real_endpoint(s3_config, s3_store, make_artifact):
    """Publish, reclaim, resolve again — the loop the researcher actually runs."""
    cache = Cache(s3_config, store=s3_store)
    source = make_artifact(2 * 2**20, files=3)
    original = {path.name: path.read_bytes() for path in source.iterdir()}

    result = cache.publish(source, KEY)
    assert result.local_removed and result.verification.ok
    assert cache.status().used == 0

    staged = cache.resolve(KEY, lease=False)
    assert {p.name: p.read_bytes() for p in staged.iterdir() if p.name != ".nawat-artifact.json"} == original

    assert cache.evict(KEY) == 2 * 2**20
    assert not cache.is_present(KEY)
    assert s3_store.exists(KEY), "eviction must not touch the replica"


def test_eviction_is_refused_when_the_endpoint_is_down(s3_config, s3_store, make_artifact):
    from dataclasses import replace

    cache = Cache(s3_config, store=s3_store)
    cache.publish(make_artifact(1024, files=2), KEY, keep_local=True)
    assert cache.is_present(KEY)

    cache._store = S3ObjectStore(replace(s3_config, endpoint="http://127.0.0.1:1"))
    result = cache.collect(need=s3_config.cache_ceiling)

    assert result.evicted == []
    assert cache.is_present(KEY), "an unverifiable artifact stays on disk"
    assert all("not verified" in reason for _, reason in result.skipped)
