"""Inference sessions.

Driven against tests/fake_server.py running as a real subprocess, exactly the
way vLLM is supervised — so process liveness, health polling, adapter hot-load,
lease-to-pid and idle teardown are all exercised for real without a GPU.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import pytest

from nawat.errors import NawatError, NotFound
from nawat.keys import Key
from nawat.sessions import SessionManager, SessionState, VLLMBackend

from .conftest import FakeBackend

CACHE_CEILING = 50 * 10**6

MODEL = "models/unsloth/Qwen2.5-VL-7B-Instruct"
ADAPTER = "runs/2026-07-28-a91f/adapter"


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


# -- lifecycle ---------------------------------------------------------------


def test_starting_a_session_serves_the_model(sessions, cached):
    cached(MODEL, 4096)

    session = sessions.start(MODEL)

    assert session.state is SessionState.READY
    assert session.model == Key.parse(MODEL)
    served = get_json(f"{session.base_url}/v1/models")
    assert [entry["id"] for entry in served["data"]] == [MODEL]


def test_the_weights_are_held_for_the_session_and_cannot_be_evicted(sessions, cached, cache):
    cached(MODEL, 4096)
    session = sessions.start(MODEL)

    status = cache.get(MODEL)
    assert status.leased
    # The lease belongs to the inference server, not to whoever started it.
    assert [record.pid for record in status.leases] == [session.pid]

    result = cache.collect(need=cache.config.cache_ceiling)
    assert result.evicted == []
    assert any("in use by serving" in reason for _, reason in result.skipped)


def test_stopping_releases_the_gpu_and_the_lease(sessions, cached, cache):
    cached(MODEL, 4096)
    session = sessions.start(MODEL)
    pid = session.pid

    sessions.stop()

    assert sessions.current() is None
    assert not cache.get(MODEL).leased
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(0.05)
    else:
        pytest.fail("the inference server was not stopped")


def test_stopping_when_nothing_runs_is_not_an_error(sessions):
    assert sessions.stop() is None


def test_only_one_session_runs_at_a_time(sessions, cached):
    cached(MODEL, 4096)
    cached("models/other", 4096)

    first = sessions.start(MODEL)
    second = sessions.start("models/other")

    assert second.model == Key.parse("models/other")
    assert second.pid != first.pid
    assert sessions.current().pid == second.pid
    with pytest.raises(OSError):
        os.kill(first.pid, 0)


def test_starting_the_model_already_served_reuses_the_session(sessions, cached):
    cached(MODEL, 4096)
    first = sessions.start(MODEL)
    again = sessions.start(MODEL)

    assert again.pid == first.pid
    assert again.last_used >= first.last_used


def test_a_server_that_dies_during_startup_points_at_its_log(cache, cached, free_port):
    from dataclasses import replace as dataclass_replace

    cached(MODEL, 4096)
    cache.config = dataclass_replace(cache.config, serve_port=free_port, serve_startup_timeout=10.0)
    manager = SessionManager(cache, backend=FakeBackend(exit_after=0.2))

    with pytest.raises(NawatError) as caught:
        manager.start(MODEL)

    assert "exited during startup" in caught.value.cause
    assert "log" in caught.value.remedy
    assert manager.current() is None
    assert not cache.get(MODEL).leased, "a failed start must not leave the weights held"


def test_a_slow_server_is_waited_for(cache, cached, free_port):
    from dataclasses import replace as dataclass_replace

    cached(MODEL, 4096)
    cache.config = dataclass_replace(cache.config, serve_port=free_port, serve_startup_timeout=20.0)
    manager = SessionManager(cache, backend=FakeBackend(ready_after=1.0))
    try:
        session = manager.start(MODEL)
        assert session.state is SessionState.READY
        assert time.time() - session.started_at >= 1.0
    finally:
        manager.stop()


# -- idle teardown -----------------------------------------------------------


def test_an_idle_session_is_torn_down(sessions, cached, cache):
    cached(MODEL, 4096)
    sessions.start(MODEL, idle_timeout=0.3)

    assert sessions.reap_if_idle() is None, "not idle yet"
    time.sleep(0.4)
    stopped = sessions.reap_if_idle()

    assert stopped is not None
    assert sessions.current() is None
    assert not cache.get(MODEL).leased


def test_using_a_session_keeps_it_alive(sessions, cached):
    cached(MODEL, 4096)
    sessions.start(MODEL, idle_timeout=0.5)

    time.sleep(0.3)
    sessions.touch()
    time.sleep(0.3)

    assert sessions.reap_if_idle() is None
    assert sessions.current() is not None


def test_an_idle_timeout_of_zero_never_tears_down(sessions, cached):
    cached(MODEL, 4096)
    sessions.start(MODEL, idle_timeout=0)
    time.sleep(0.2)
    assert sessions.reap_if_idle() is None
    assert sessions.current() is not None


# -- adapters ----------------------------------------------------------------


def test_an_adapter_loads_onto_the_running_base_without_merging(sessions, cached):
    cached(MODEL, 4096)
    cached(ADAPTER, 512)
    session = sessions.start(MODEL)
    base_pid = session.pid

    updated = sessions.load_adapter(ADAPTER)

    assert updated.pid == base_pid, "loading an adapter must not restart the server"
    assert updated.adapters == {"2026-07-28-a91f-adapter": ADAPTER}
    served = {entry["id"] for entry in get_json(f"{session.base_url}/v1/models")["data"]}
    assert served == {MODEL, "2026-07-28-a91f-adapter"}


def test_an_adapter_can_be_given_a_name(sessions, cached):
    cached(MODEL, 4096)
    cached(ADAPTER, 512)
    sessions.start(MODEL)

    updated = sessions.load_adapter(ADAPTER, name="ocr-v3")

    assert updated.adapters == {"ocr-v3": ADAPTER}


def test_a_loaded_adapter_is_held_on_disk_too(sessions, cached, cache):
    cached(MODEL, 4096)
    cached(ADAPTER, 512)
    sessions.start(MODEL)
    sessions.load_adapter(ADAPTER)

    assert cache.get(ADAPTER).leased


def test_unloading_an_adapter_releases_the_name(sessions, cached):
    cached(MODEL, 4096)
    cached(ADAPTER, 512)
    session = sessions.start(MODEL)
    sessions.load_adapter(ADAPTER, name="ocr-v3")

    updated = sessions.unload_adapter("ocr-v3")

    assert updated.adapters == {}
    served = {entry["id"] for entry in get_json(f"{session.base_url}/v1/models")["data"]}
    assert served == {MODEL}


def test_unloading_something_not_loaded_says_so(sessions, cached):
    cached(MODEL, 4096)
    sessions.start(MODEL)
    with pytest.raises(NotFound, match="No adapter named"):
        sessions.unload_adapter("never-loaded")


def test_loading_an_adapter_without_a_session_says_how_to_start_one(sessions, cached):
    cached(ADAPTER, 512)
    with pytest.raises(NotFound, match="nawat serve"):
        sessions.load_adapter(ADAPTER)


# -- state shared across processes -------------------------------------------


def test_another_process_sees_the_running_session(sessions, cached, cache):
    cached(MODEL, 4096)
    started = sessions.start(MODEL)

    observer = SessionManager(cache, backend=FakeBackend())
    seen = observer.current()

    assert seen is not None
    assert seen.id == started.id and seen.pid == started.pid


def test_a_session_whose_server_died_is_forgotten(sessions, cached, cache):
    cached(MODEL, 4096)
    session = sessions.start(MODEL)

    os.killpg(os.getpgid(session.pid), 9)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and session.alive():
        time.sleep(0.05)

    assert sessions.current() is None
    assert not cache.get(MODEL).leased, "a dead server must not keep holding the weights"


def test_a_session_from_before_a_reboot_is_not_believed(sessions, cached):
    cached(MODEL, 4096)
    session = sessions.start(MODEL)
    session.boot = "a-boot-id-from-before-the-reboot"
    assert not session.alive()


# -- the real backend --------------------------------------------------------


def test_the_vllm_command_enables_runtime_adapter_loading(tmp_path):
    backend = VLLMBackend()
    command = backend.command(model_path=tmp_path / "weights", model_name="models/base", port=8001, extra=["--dtype", "bfloat16"])

    assert command[:2] == ["vllm", "serve"]
    assert "--enable-lora" in command
    assert command[command.index("--served-model-name") + 1] == "models/base"
    assert command[command.index("--port") + 1] == "8001"
    assert command[-2:] == ["--dtype", "bfloat16"]
    # Without this vLLM refuses the runtime endpoints, and testing an adapter
    # would mean restarting the server.
    assert backend.environment()["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"


def test_a_missing_backend_binary_says_how_to_install_it(cache, cached, free_port):
    from dataclasses import replace as dataclass_replace

    class Missing(FakeBackend):
        def command(self, **kwargs):
            return ["definitely-not-installed-xyz", "serve"]

    cached(MODEL, 4096)
    cache.config = dataclass_replace(cache.config, serve_port=free_port)
    manager = SessionManager(cache, backend=Missing())

    with pytest.raises(NotFound, match="is not installed"):
        manager.start(MODEL)
