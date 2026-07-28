"""The CLI is the Phase 1 deliverable, and must work with no API or UI running
(NFR-3.5). These drive it exactly as a shell would.
"""

from __future__ import annotations

import json
import sys

import pytest

from nawat.cli import main


@pytest.fixture
def shell(tmp_path, monkeypatch):
    """A configured `nawat` on an empty host, wired to a local object store."""
    env = {
        "NAWAT_CACHE_ROOT": str(tmp_path / "cache"),
        "NAWAT_STATE_DIR": str(tmp_path / "state"),
        "NAWAT_WORKSPACE": str(tmp_path / "workspace"),
        "NAWAT_STORE_BACKEND": "local",
        "NAWAT_LOCAL_STORE_ROOT": str(tmp_path / "store"),
        "NAWAT_CACHE_CEILING": "10000",
        "NAWAT_MIN_FREE": "0",
        "NAWAT_OFFLINE": "1",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "NAWAT_S3_ENDPOINT"):
        monkeypatch.delenv(key, raising=False)
    return tmp_path


@pytest.fixture
def outputs(tmp_path):
    directory = tmp_path / "outputs"
    directory.mkdir()
    (directory / "adapter_model.safetensors").write_bytes(b"a" * 500)
    (directory / "adapter_config.json").write_text('{"r": 16}')
    return directory


def test_status_on_an_empty_host(shell, capsys):
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "cache" in out and "0 B / 10.0 KB" in out
    assert "artifacts  0" in out


def test_ls_on_an_empty_host_says_what_to_do_first(shell, capsys):
    assert main(["ls"]) == 0
    assert "nawat resolve" in capsys.readouterr().out


def test_publish_then_ls_then_free(shell, outputs, capsys):
    assert main(["publish", str(outputs), "runs/2026-07-28-a91f/adapter"]) == 0
    published = capsys.readouterr().out
    assert "verified" in published and "Local copy removed" in published
    assert not outputs.exists()

    assert main(["registry", "--json"]) == 0
    registry = json.loads(capsys.readouterr().out)
    assert [entry["key"] for entry in registry] == ["runs/2026-07-28-a91f/adapter"]

    assert main(["resolve", "runs/2026-07-28-a91f/adapter", "--no-lease"]) == 0
    path = capsys.readouterr().out.strip()
    assert path.endswith("runs/2026-07-28-a91f/adapter")

    assert main(["ls", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["key"] == "runs/2026-07-28-a91f/adapter"
    assert listed[0]["replicated"] is True


def test_keeping_an_artifact_exempts_it_from_freeing(shell, outputs, capsys):
    main(["publish", str(outputs), "models/base", "--keep"])
    capsys.readouterr()

    assert main(["keep", "models/base"]) == 0
    assert "Keeping models/base on disk" in capsys.readouterr().out

    assert main(["--ceiling", "1", "free"]) == 6  # cannot reach the ceiling
    report = capsys.readouterr()
    assert "kept     models/base — kept on disk" in report.out
    assert "raise" in report.err

    assert main(["release", "models/base"]) == 0
    capsys.readouterr()
    assert main(["--ceiling", "1", "free"]) == 0
    assert "removed  models/base" in capsys.readouterr().out


def test_removing_an_unreplicated_artifact_is_refused_with_a_remedy(shell, outputs, capsys):
    assert main(["add", str(outputs), "runs/x/adapter", "--no-publish"]) == 0
    capsys.readouterr()

    assert main(["rm", "runs/x/adapter"]) == 5
    assert "nawat publish" in capsys.readouterr().err

    assert main(["rm", "runs/x/adapter", "--force"]) == 0
    assert "Removed runs/x/adapter" in capsys.readouterr().out


def test_an_invalid_key_is_rejected_before_anything_happens(shell, capsys):
    assert main(["resolve", "models/../../etc/passwd"]) == 2
    assert "relative path segment" in capsys.readouterr().err


def test_an_offline_host_refuses_to_reach_the_internet(shell, capsys):
    assert main(["resolve", "models/unsloth/Qwen2.5-VL-7B-Instruct"]) == 7
    assert "NAWAT_OFFLINE" in capsys.readouterr().err


def test_config_never_prints_credentials(shell, monkeypatch, capsys):
    monkeypatch.setenv("NAWAT_S3_SECRET_KEY", "hunter2-do-not-log-me")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    assert main(["config"]) == 0
    printed = capsys.readouterr().out
    assert "hunter2-do-not-log-me" not in printed
    assert "hf_secret_token" not in printed
    assert json.loads(printed)["secret_key"] == "set"


def test_hold_stages_inputs_runs_a_command_and_publishes_its_outputs(shell, outputs, capsys):
    main(["add", str(outputs), "datasets/ocr-arabic-v3"])
    capsys.readouterr()

    script = (
        "import os, pathlib, json;"
        "out = pathlib.Path(os.environ['NAWAT_OUT_DIR']);"
        "data = pathlib.Path(os.environ['NAWAT_DATASET_DIR']);"
        "assert data.is_dir(), data;"
        "assert json.loads(os.environ['NAWAT_INPUTS']);"
        "(out / 'adapter_model.safetensors').write_bytes(b'w' * 128)"
    )
    code = main(
        [
            "hold",
            "--dataset", "datasets/ocr-arabic-v3",
            "--out", "runs/2026-07-28-a91f/adapter",
            "--run-id", "2026-07-28-a91f",
            "--", sys.executable, "-c", script,
        ]
    )
    report = capsys.readouterr()
    assert code == 0, report.err
    assert "Published runs/2026-07-28-a91f/adapter" in report.out

    assert main(["registry", "--json"]) == 0
    keys = {entry["key"] for entry in json.loads(capsys.readouterr().out)}
    assert "runs/2026-07-28-a91f/adapter" in keys


def test_hold_does_not_publish_the_outputs_of_a_failed_run(shell, outputs, capsys):
    main(["add", str(outputs), "datasets/ocr-arabic-v3"])
    capsys.readouterr()

    code = main(
        [
            "hold",
            "--dataset", "datasets/ocr-arabic-v3",
            "--out", "runs/failed/adapter",
            "--", sys.executable, "-c", "import sys; sys.exit(3)",
        ]
    )
    report = capsys.readouterr()
    assert code == 3
    assert "were not published" in report.err
    assert main(["registry", "--json"]) == 0
    keys = {entry["key"] for entry in json.loads(capsys.readouterr().out)}
    assert "runs/failed/adapter" not in keys


def test_hold_releases_its_leases_when_the_command_ends(shell, outputs, capsys):
    main(["add", str(outputs), "datasets/ocr-arabic-v3"])
    main(["hold", "--dataset", "datasets/ocr-arabic-v3", "--", sys.executable, "-c", "pass"])
    capsys.readouterr()

    assert main(["leases"]) == 0
    assert "Nothing is in use." in capsys.readouterr().out
