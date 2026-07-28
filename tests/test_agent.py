"""Agent-assisted authoring: the agent proposes; the researcher commits.

The safety property under test throughout: no agent output reaches the
workspace, or becomes a run, without passing through a diff and an approval —
and what leaves the host is only what the context builders compose.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from dataclasses import replace as dataclass_replace
from pathlib import Path

import pytest

from nawat.agent import (
    Agent,
    AgentBackend,
    ClaudeAgentSDKBackend,
    OpenAICompatBackend,
    Proposal,
    build_backend,
    failed_run_context,
    parse_reply,
    prior_runs_summary,
)
from nawat.errors import NawatError, NotFound
from nawat.keys import Key
from nawat.runs import RunSpec, RunState

CACHE_CEILING = 50 * 10**6

GOOD_SCRIPT = "import os\nprint('training with', os.environ.get('NAWAT_PARAM_LEARNING_RATE'))\n"


class ScriptedBackend(AgentBackend):
    """Replies with whatever the test scripted, recording what it was sent."""

    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.seen: list[tuple[str, str]] = []

    def complete(self, system: str, prompt: str) -> str:
        self.seen.append((system, prompt))
        return self.reply


def proposal_reply(code: str, summary: str = "Lower the learning rate.") -> str:
    return f"{summary}\n\n```python\n{code}```\n"


@pytest.fixture
def agent(cache, runs, workspace):
    return Agent(cache, runs, ScriptedBackend(proposal_reply(GOOD_SCRIPT)))


# -- parsing -----------------------------------------------------------------


def test_a_reply_parses_into_summary_and_code():
    summary, code = parse_reply("One-line summary.\n\n```python\nprint('hi')\n```")
    assert summary == "One-line summary."
    assert code == "print('hi')\n"


def test_a_reply_without_a_fence_is_refused_with_the_shape():
    with pytest.raises(NawatError, match="fenced code block"):
        parse_reply("I think you should lower the learning rate. Good luck!")


def test_a_bare_fence_is_accepted():
    _, code = parse_reply("```\nx = 1\n```")
    assert code == "x = 1\n"


# -- proposing ---------------------------------------------------------------


def test_a_proposal_is_a_reviewable_diff_not_an_edit(agent, workspace):
    (workspace / "train.py").write_text("old = True\n")

    proposal = agent.propose("modernise it", script="train.py")

    assert "-old = True" in proposal.diff
    assert "+import os" in proposal.diff
    assert (workspace / "train.py").read_text() == "old = True\n", "proposing must not touch the workspace"


def test_code_that_does_not_parse_is_never_offered(cache, runs, workspace):
    agent = Agent(cache, runs, ScriptedBackend(proposal_reply("def broken(:\n")))
    with pytest.raises(NawatError, match="does not parse"):
        agent.propose("write a trainer", script="train.py")


def test_the_gate_estimates_an_unrunnable_proposal(cache, runs, workspace, cached, monkeypatch):
    """FR-6.6: obviously unrunnable proposals are flagged before review."""
    cached("models/base", 16 * 10**6)  # reads as a "big" model against a tiny fake GPU
    record = runs.create(RunSpec(script="train.py", model=Key.parse("models/base")))
    monkeypatch.setattr("nawat.api.gpu_info", lambda: {"name": "tiny", "memory_total": 10**9, "memory_used": 0, "utilization": 0})
    huge = "from trl import SFTConfig\nc = SFTConfig(per_device_train_batch_size=512, max_length=32768)\n"
    agent = Agent(cache, runs, ScriptedBackend(proposal_reply(huge)))

    proposal = agent.propose("go bigger", run_id=record.id)

    assert proposal.warnings, "an over-VRAM proposal must carry a warning"
    assert "will not fit" in proposal.warnings[0]


def test_without_a_script_or_run_the_error_names_both_flags(agent):
    with pytest.raises(NotFound, match="--script"):
        agent.propose("do something")


# -- context -----------------------------------------------------------------


def test_the_context_carries_contract_hardware_registry_and_history(cache, runs, workspace, cached, script):
    cached("models/base", 4096)
    backend = ScriptedBackend(proposal_reply(GOOD_SCRIPT))
    agent = Agent(cache, runs, backend)
    runs.create(RunSpec(script="old.py", params={"lr": "1e-4"}), "prior-run")

    agent.propose("write a trainer", script="train.py")

    system, prompt = backend.seen[0]
    assert "NAWAT_MODEL_DIR" in system, "the trainer contract travels with every request"
    assert "GPU" in system
    assert "models/base" in system, "the registry keys travel"
    assert "prior-run" in system and "lr=1e-4" in system, "prior runs travel"
    assert "write a trainer" in prompt


def test_diagnosing_a_failed_run_includes_its_log_and_metrics(cache, runs, executor, script):
    body = (
        "import nawat.metrics as m\n"
        "m.log(step=1, loss=9.9)\n"
        "print('CUDA out of memory. Tried to allocate 20.00 GiB')\n"
        "import sys; sys.exit(1)\n"
    )
    record = runs.create(RunSpec(script=script(body, name="oom.py")))
    executor.execute(record.id)

    context = failed_run_context(runs, record.id)

    assert "CUDA out of memory" in context
    assert "loss: 9.9" in context
    assert "state failed" in context


def test_prior_runs_summarise_to_ids_params_and_final_readings(runs, cache):
    record = runs.create(RunSpec(script="t.py", params={"lr": "2e-4"}), "seen-run")
    runs.update(record.id, state=RunState.FAILED, error="exited 1")
    with runs.metrics_path(record.id).open("w") as fh:
        fh.write(json.dumps({"step": 5, "loss": 0.42}) + "\n")

    summary = prior_runs_summary(runs)

    assert "seen-run" in summary and "lr=2e-4" in summary
    assert "final loss 0.42" in summary and "exited 1" in summary


def test_no_credentials_ever_enter_the_context(cache, runs, workspace, monkeypatch):
    """FR-6.8: the backend receives code and metrics, never secrets."""
    monkeypatch.setenv("NAWAT_S3_SECRET_KEY", "super-secret-value")
    backend = ScriptedBackend(proposal_reply(GOOD_SCRIPT))
    agent = Agent(cache, runs, backend)

    agent.propose("write a trainer", script="train.py")

    system, prompt = backend.seen[0]
    assert "super-secret-value" not in system + prompt


# -- accepting ---------------------------------------------------------------


def test_applying_writes_the_file_and_commits_with_provenance(agent, workspace):
    proposal = agent.propose("write a trainer", script="train.py")

    commit = agent.apply(proposal)

    assert (workspace / "train.py").read_text() == GOOD_SCRIPT
    log = subprocess.run(["git", "log", "-1", "--format=%B"], cwd=workspace, capture_output=True, text=True)
    assert "agent: Lower the learning rate." in log.stdout
    assert "Instruction: write a trainer" in log.stdout
    assert "Backend: scripted" in log.stdout
    assert commit, "the commit hash comes back for the record"


def test_each_accepted_edit_is_its_own_commit(agent, workspace):
    first = agent.propose("v1", script="train.py")
    agent.apply(first)
    second = dataclass_replace(first, new=GOOD_SCRIPT + "# revised\n", instruction="v2")
    agent.apply(second)

    count = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=workspace, capture_output=True, text=True)
    assert count.stdout.strip() == "2"


# -- descriptions ------------------------------------------------------------


def test_describe_stores_a_plain_language_account_in_the_record(cache, runs, workspace):
    record = runs.create(RunSpec(script="t.py", model=Key.parse("models/base")), "descr-run")
    agent = Agent(cache, runs, ScriptedBackend("A LoRA fine-tune of the base model on the OCR set; it failed at step 3."))

    description = agent.describe_run(record.id)

    assert "LoRA fine-tune" in description
    assert runs.get(record.id).description == description


# -- backends ----------------------------------------------------------------


def test_no_backend_configured_is_a_clear_refusal_not_a_break(tmp_path):
    with pytest.raises(NotFound, match="fully usable without one"):
        build_backend({}, workspace=tmp_path)


def test_the_local_backend_needs_a_url(tmp_path):
    with pytest.raises(NotFound, match="NAWAT_AGENT_URL"):
        build_backend({"NAWAT_AGENT_BACKEND": "local"}, workspace=tmp_path)


def test_claude_selects_the_agent_sdk(tmp_path):
    backend = build_backend({"NAWAT_AGENT_BACKEND": "claude", "NAWAT_AGENT_MODEL": "claude-opus-5"}, workspace=tmp_path)
    assert isinstance(backend, ClaudeAgentSDKBackend)
    assert backend.model == "claude-opus-5"


def test_the_openai_backend_speaks_to_a_real_endpoint(cache, cached, free_port):
    """Transport over the fake server: ECHO in, text out."""
    from nawat.sessions import SessionManager

    from .conftest import FakeBackend

    cache.config = dataclass_replace(cache.config, serve_port=free_port, serve_startup_timeout=20.0)
    cached("models/base", 4096)
    manager = SessionManager(cache, backend=FakeBackend())
    try:
        session = manager.start("models/base")
        backend = OpenAICompatBackend(session.base_url, "models/base")
        assert backend.complete("system context", "ECHO the reply") == "the reply"
    finally:
        manager.stop()


def test_an_unreachable_local_endpoint_names_the_variables():
    backend = OpenAICompatBackend("http://127.0.0.1:1", "m")
    with pytest.raises(NawatError, match="NAWAT_AGENT_URL"):
        backend.complete("s", "p")


# -- the Claude Agent SDK backend --------------------------------------------


@pytest.fixture
def fake_sdk(monkeypatch):
    """A stand-in claude_agent_sdk capturing the options the backend passes."""
    module = types.ModuleType("claude_agent_sdk")
    captured: dict = {}

    class TextBlock:
        def __init__(self, text: str) -> None:
            self.text = text

    class AssistantMessage:
        def __init__(self, content) -> None:
            self.content = content

    class ResultMessage:
        def __init__(self, result: str) -> None:
            self.result = result

    class ClaudeAgentOptions:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    async def fake_query(*, prompt, options):
        captured["prompt"] = prompt
        yield AssistantMessage([TextBlock("thinking aloud")])
        yield ResultMessage("Summary.\n\n```python\nprint('from the sdk')\n```")

    module.TextBlock = TextBlock
    module.AssistantMessage = AssistantMessage
    module.ResultMessage = ResultMessage
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.query = fake_query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return captured


def test_the_sdk_backend_returns_the_result_text(tmp_path, fake_sdk):
    backend = ClaudeAgentSDKBackend(tmp_path)
    reply = backend.complete("the system context", "fix the trainer")
    assert "from the sdk" in reply
    assert fake_sdk["prompt"] == "fix the trainer"
    assert fake_sdk["system_prompt"] == "the system context"


def test_the_sdk_is_confined_to_read_only_workspace_access(tmp_path, fake_sdk):
    """FR-6.8, as options: read-only tools, no bash, no web, headless denial."""
    ClaudeAgentSDKBackend(tmp_path, model="claude-opus-5").complete("s", "p")

    assert fake_sdk["cwd"] == str(tmp_path)
    assert fake_sdk["allowed_tools"] == ["Read", "Glob", "Grep"]
    for tool in ("Bash", "Write", "Edit", "WebSearch", "WebFetch"):
        assert tool in fake_sdk["disallowed_tools"]
    assert fake_sdk["permission_mode"] == "dontAsk"
    assert fake_sdk["setting_sources"] == [], "no host-user Claude Code settings leak in"
    assert fake_sdk["model"] == "claude-opus-5"


def test_without_the_sdk_installed_the_error_names_the_alternative(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(NotFound, match="NAWAT_AGENT_BACKEND=local"):
        ClaudeAgentSDKBackend(tmp_path).complete("s", "p")
