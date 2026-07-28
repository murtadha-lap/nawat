from __future__ import annotations

import pytest

from nawat.errors import NotFound, VerificationFailed
from nawat.keys import Key
from nawat.store import PART_SUFFIX, scan_local, verify_manifests

KEY = Key.parse("runs/2026-07-28-a91f/adapter")


def test_publish_then_download_round_trips(store, make_artifact, tmp_path):
    source = make_artifact(4096, files=3)
    result = store.publish(source, KEY)
    assert result.ok
    assert result.local_files == 3
    assert result.local_bytes == 4096

    dest = tmp_path / "back"
    store.download(KEY, dest)
    assert {rel: stat.size for rel, stat in scan_local(dest).items()} == {
        rel: stat.size for rel, stat in scan_local(source).items()
    }
    assert {p.name: p.read_bytes() for p in dest.iterdir()} == {p.name: p.read_bytes() for p in source.iterdir()}


def test_downloads_leave_no_partial_files_behind(store, make_artifact, tmp_path):
    source = make_artifact(2048, files=4)
    store.publish(source, KEY)
    dest = tmp_path / "back"
    store.download(KEY, dest)
    assert not list(dest.rglob(f"*{PART_SUFFIX}"))


def test_platform_metadata_is_never_uploaded_or_compared(store, make_artifact, tmp_path):
    source = make_artifact(1024)
    (source / ".nawat-artifact.json").write_text('{"key": "runs/x/adapter"}')
    store.publish(source, KEY)
    assert ".nawat-artifact.json" not in store.list_prefix(KEY)
    assert store.verify(source, KEY).ok


def test_verification_names_a_missing_file(store, make_artifact):
    source = make_artifact(1024, files=2)
    store.publish(source, KEY)
    (source / "added-after-publish.bin").write_bytes(b"y" * 16)
    result = store.verify(source, KEY)
    assert not result.ok
    assert result.missing == ("added-after-publish.bin",)
    assert "absent from object storage" in result.reason()


def test_verification_names_a_size_mismatch(store, make_artifact):
    source = make_artifact(1024, files=2)
    store.publish(source, KEY)
    victim = sorted(source.iterdir())[0]
    victim.write_bytes(b"z" * 99)
    result = store.verify(source, KEY)
    assert not result.ok
    assert result.size_mismatch == (victim.name,)
    assert "differ in size" in result.reason()


def test_publish_raises_and_preserves_the_local_copy_when_verification_fails(store, make_artifact, monkeypatch):
    source = make_artifact(1024, files=2)

    def truncated_upload(*args, **kwargs):
        return {}  # pretend the transfer silently wrote nothing

    monkeypatch.setattr(store, "upload", truncated_upload)
    with pytest.raises(VerificationFailed) as caught:
        store.publish(source, KEY)
    assert "has been kept" in caught.value.remedy
    assert scan_local(source), "the local copy must survive a failed publish"


def test_downloading_something_absent_says_so(store, tmp_path):
    with pytest.raises(NotFound):
        store.download(Key.parse("models/never-seeded"), tmp_path / "dest")


def test_publishing_an_empty_directory_is_refused(store, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(NotFound):
        store.publish(empty, KEY)


def test_verify_manifests_reports_remote_only_files_without_failing():
    local = scan_local
    result = verify_manifests(KEY, {}, {"stale.bin": type("S", (), {"size": 10})()})
    assert result.ok
    assert result.remote_only == ("stale.bin",)
    assert local is scan_local


def test_delete_prefix_removes_every_object(store, make_artifact):
    source = make_artifact(1024, files=3)
    store.publish(source, KEY)
    assert store.delete_prefix(KEY) == 3
    assert not store.exists(KEY)
