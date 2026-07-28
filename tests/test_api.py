"""The control plane over HTTP.

Driven with a real ASGI client against the same Platform the server runs. The
exit criterion for Phase 2 is that a full experiment cycle — submit, train,
publish, serve, chat — is driveable via API alone; the last test does exactly
that.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace as dataclass_replace

import pytest

pytest.importorskip("fastapi", reason="install the api extras to test the control plane")

from fastapi.testclient import TestClient  # noqa: E402

from nawat.api import create_app  # noqa: E402
from nawat.runs import RunState  # noqa: E402

from .conftest import FakeBackend  # noqa: E402

CACHE_CEILING = 50 * 10**6

TRAINER = """
import os, pathlib
import nawat.metrics as m
out = pathlib.Path(os.environ["NAWAT_OUT_DIR"])
(out / "adapter").mkdir(parents=True, exist_ok=True)
(out / "adapter" / "adapter_model.safetensors").write_bytes(b"w" * 1024)
m.log(step=1, loss=2.31)
m.log(step=2, loss=1.87)
print("step 1 loss 2.31", flush=True)
print("step 2 loss 1.87", flush=True)
"""


@pytest.fixture
def client(cache, free_port, script):
    cache.config = dataclass_replace(cache.config, serve_port=free_port, serve_startup_timeout=20.0)
    app = create_app(cache=cache, backend=FakeBackend(), start_queue=True, sweep_idle=False)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def secured(cache, free_port):
    cache.config = dataclass_replace(
        cache.config, serve_port=free_port, api_token="a-long-random-token"
    )
    app = create_app(cache=cache, backend=FakeBackend(), start_queue=False, sweep_idle=False)
    with TestClient(app) as test_client:
        yield test_client


def wait_for(client, run_id, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/runs/{run_id}").json()
        if record["state"] in ("succeeded", "failed", "cancelled"):
            return record
        time.sleep(0.1)
    pytest.fail(f"run {run_id} did not finish: {record}")


# -- authentication ----------------------------------------------------------


def test_a_configured_token_is_required(secured):
    assert secured.get("/cache").status_code == 401
    assert secured.get("/cache", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert secured.get("/cache", headers={"Authorization": "Bearer a-long-random-token"}).status_code == 200
    assert secured.get("/cache", headers={"X-Nawat-Token": "a-long-random-token"}).status_code == 200


def test_health_needs_no_token_so_monitoring_works(secured):
    assert secured.get("/health").status_code == 200


def test_the_token_never_appears_in_config(secured):
    config = secured.get("/config", headers={"Authorization": "Bearer a-long-random-token"}).json()
    assert config["api_token"] == "set"
    assert "a-long-random-token" not in json.dumps(config)


# -- storage over http -------------------------------------------------------


def test_cache_and_registry_views(client, cached):
    cached("models/base", 4096)
    status = client.get("/cache").json()
    assert status["artifacts"] == 1
    assert status["used"] == 4096

    listed = client.get("/cache/artifacts").json()
    assert listed[0]["key"] == "models/base"
    assert listed[0]["replicated"] is True

    registry = client.get("/registry").json()
    assert {entry["key"] for entry in registry} == {"models/base"}
    assert all(entry["cached"] for entry in registry)


def test_keep_free_and_evict_over_http(client, cached):
    cached("models/a", 4096)
    cached("models/b", 4096)

    assert client.post("/cache/models/a/keep").json()["pinned"] is True

    result = client.post("/cache/free", json={"need": CACHE_CEILING, "dry_run": False}).json()
    assert "models/b" in result["evicted"]
    assert any(k["key"] == "models/a" for k in result["kept"])

    assert client.delete("/cache/models/a/keep").json()["pinned"] is False
    freed = client.delete("/cache/models/a").json()
    assert freed["freed"] == 4096


def test_errors_carry_cause_and_remedy_across_the_wire(client, cached):
    response = client.delete("/cache/models/never-cached")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "NotFound"
    assert body["cause"]
    assert "remedy" in body

    cached("runs/x/adapter", 512, replicated=False)
    response = client.delete("/cache/runs/x/adapter")
    assert response.status_code == 409
    assert "only copy" in response.json()["cause"]


def test_verify_over_http(client, cached):
    cached("models/base", 4096)
    result = client.get("/cache/models/base/verify").json()
    assert result["ok"] is True
    assert result["files"] == 1


# -- runs over http ----------------------------------------------------------


def test_the_whole_experiment_cycle_is_driveable_via_api_alone(client, cached, script):
    """Phase 2's exit criterion, as one test: submit → train → publish → serve → chat."""
    cached("datasets/ocr-arabic-v3", 4096)
    script(TRAINER)

    # submit
    submitted = client.post(
        "/runs",
        json={
            "script": "train.py",
            "datasets": ["datasets/ocr-arabic-v3"],
            "params": {"learning_rate": "2e-4"},
            "notes": "baseline OCR run",
        },
    )
    assert submitted.status_code == 201
    run_id = submitted.json()["id"]

    # train (the queue picks it up) and publish
    record = wait_for(client, run_id)
    assert record["state"] == "succeeded", record
    adapter_key = f"runs/{run_id}/adapter"
    assert adapter_key in record["artifacts"]

    # the log is readable after the fact
    log = client.get(f"/runs/{run_id}/log").text
    assert "step 2 loss 1.87" in log

    # serve the base, hot-load the trained adapter
    cached("models/served-base", 4096)
    session = client.post("/sessions", json={"model": "models/served-base"}).json()
    assert session["state"] == "ready"

    loaded = client.post("/sessions/adapters", json={"key": adapter_key, "name": "ocr-v3"})
    assert loaded.status_code == 200
    assert loaded.json()["adapters"] == {"ocr-v3": adapter_key}

    # chat with the adapter through the stable /v1 address
    completion = client.post(
        "/v1/chat/completions",
        json={"model": "ocr-v3", "messages": [{"role": "user", "content": "read this"}]},
    )
    assert completion.status_code == 200
    assert completion.json()["choices"][0]["message"]["content"] == "served by ocr-v3"

    # tear down; the GPU and the disk come back
    stopped = client.delete("/sessions").json()
    assert stopped["stopped"]["model"] == "models/served-base"
    assert client.get("/sessions/current").json() is None


def test_submission_is_validated_before_it_is_queued(client):
    response = client.post("/runs", json={"script": "no-such-script.py"})
    assert response.status_code == 404
    assert "does not exist" in response.json()["cause"]


def test_a_run_log_streams_as_server_sent_events(client, script):
    script("print('streamed line', flush=True)")
    run_id = client.post("/runs", json={"script": "train.py"}).json()["id"]
    wait_for(client, run_id)

    with client.stream("GET", f"/runs/{run_id}/log/stream") as response:
        body = "".join(chunk for chunk in response.iter_text())
    assert "data: streamed line" in body
    assert "event: state" in body


def test_cancelling_a_queued_run_over_http(client, cached, script):
    # Queue depth: put a slow run in front so the second stays queued.
    script("import time; time.sleep(30)", name="slow.py")
    script("print('never runs')", name="second.py")
    first = client.post("/runs", json={"script": "slow.py"}).json()["id"]
    second = client.post("/runs", json={"script": "second.py"}).json()["id"]

    cancelled = client.post(f"/runs/{second}/cancel").json()
    assert cancelled["state"] == "cancelled"

    client.post(f"/runs/{first}/cancel")
    record = wait_for(client, first)
    assert record["state"] == "cancelled"


def test_metrics_are_served_and_streamed_over_http(client, script):
    script(TRAINER)
    run_id = client.post("/runs", json={"script": "train.py"}).json()["id"]
    wait_for(client, run_id)

    served = client.get(f"/runs/{run_id}/metrics").json()
    assert served["points"] == 2
    assert [entry["value"] for entry in served["series"]["loss"]] == [2.31, 1.87]

    with client.stream("GET", f"/runs/{run_id}/metrics/stream") as response:
        body = "".join(chunk for chunk in response.iter_text())
    assert '"loss": 2.31' in body
    assert "event: state" in body


def test_metrics_compare_across_runs_on_one_axis(client, script):
    script(TRAINER)
    first = client.post("/runs", json={"script": "train.py", "run_id": "run-a"}).json()["id"]
    wait_for(client, first)
    second = client.post("/runs", json={"script": "train.py", "run_id": "run-b"}).json()["id"]
    wait_for(client, second)

    compared = client.get("/metrics/compare", params=[("run", "run-a"), ("run", "run-b"), ("name", "loss")]).json()

    assert set(compared) == {"run-a", "run-b"}
    assert [entry["step"] for entry in compared["run-a"]] == [1, 2]
    strip = lambda entries: [(e["step"], e["value"]) for e in entries]
    assert strip(compared["run-a"]) == strip(compared["run-b"]) == [(1, 2.31), (2, 1.87)]


def test_metrics_for_an_unknown_run_is_a_404_not_an_empty_series(client):
    assert client.get("/runs/never/metrics").status_code == 404


def test_scripts_are_listed_from_the_workspace(client, script):
    script("print('hello')", name="trainers/ocr.py")
    listed = client.get("/scripts").json()
    assert {entry["path"] for entry in listed} == {"trainers/ocr.py"}
    assert listed[0]["kind"] == "script"


# -- sessions over http ------------------------------------------------------


def test_evaluation_over_http_returns_the_number_and_stores_it(client, cached, script, tmp_path):
    cached("models/base", 4096)
    trainer = TRAINER.replace('m.log(step=2, loss=1.87)', 'm.log(step=2, loss=1.87)')
    script(trainer)
    run_id = client.post("/runs", json={"script": "train.py", "model": "models/base"}).json()["id"]
    wait_for(client, run_id)

    data = tmp_path / "eval.jsonl"
    data.write_text(json.dumps({"reference": "abcd", "prompt": "ECHO abcd"}) + "\n")

    result = client.post(f"/runs/{run_id}/evaluate", json={"data": str(data)}).json()
    assert result["cer"] == 0.0 and result["samples"] == 1
    assert "per_sample" not in result

    listed = client.get(f"/runs/{run_id}/evaluations").json()
    assert listed[0]["label"] == "eval"
    client.delete("/sessions")


def test_the_openai_route_says_when_nothing_is_serving(client):
    response = client.post("/v1/chat/completions", json={"model": "x", "messages": []})
    assert response.status_code == 404
    assert "nawat serve" in response.json()["remedy"]


def test_the_session_survives_a_control_plane_restart(cache, cached, free_port, script):
    """The stable /v1 address outlives the API process, not just the session."""
    cache.config = dataclass_replace(cache.config, serve_port=free_port, serve_startup_timeout=20.0)
    cached("models/base", 4096)

    first = create_app(cache=cache, backend=FakeBackend(), start_queue=False, sweep_idle=False)
    with TestClient(first) as client_one:
        started = client_one.post("/sessions", json={"model": "models/base"}).json()
        assert started["state"] == "ready"

    # a new control plane, same host: the session is still there and usable
    second = create_app(cache=cache, backend=FakeBackend(), start_queue=False, sweep_idle=False)
    with TestClient(second) as client_two:
        current = client_two.get("/sessions/current").json()
        assert current is not None and current["pid"] == started["pid"]
        completion = client_two.post(
            "/v1/chat/completions", json={"model": "models/base", "messages": []}
        )
        assert completion.status_code == 200
        client_two.delete("/sessions")
