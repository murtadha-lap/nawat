"""Agent-assisted authoring — proposes; the researcher commits.

The agent's job is narrow (PRD §6.6): write and repair training scripts that
satisfy the trainer contract, informed by the hardware envelope and the history
of prior runs. Every output is a reviewable diff; nothing reaches the workspace
without approval, and nothing is submitted as a run without a human (FR-6.4,
FR-6.9). Accepted edits are committed to git with the prompt and backend that
produced them (FR-6.5).

What leaves the host is exactly what this module composes: training code, run
parameters and metric numbers. Never datasets, never weights, never credentials
(FR-6.8) — and pointing NAWAT_AGENT_BACKEND at a local OpenAI-compatible
endpoint keeps even that on-premises. With no backend configured, everything
else in the platform works untouched (FR-6.13).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import metrics as metrics_module
from .cache import Cache
from .errors import NawatError, NotFound
from .estimator import estimate_script, params_from_bytes
from .keys import Key
from .runs import RunRecord, RunStore
from .units import human_bytes

MAX_LOG_TAIL = 4000
MAX_PRIOR_RUNS = 8


# -- backends -----------------------------------------------------------------


class AgentBackend:
    """One completion: system context in, proposal text out."""

    name = "backend"

    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatBackend(AgentBackend):
    """Any OpenAI-compatible endpoint — a local vLLM or llama.cpp keeps the
    whole loop on-premises."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 600.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = f"openai:{model}"

    def complete(self, system: str, prompt: str) -> str:
        import urllib.request

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
            }).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except Exception as exc:
            raise NawatError(
                f"The agent backend at {self.base_url} did not answer: {type(exc).__name__}.",
                "Check NAWAT_AGENT_URL, or switch backends with NAWAT_AGENT_BACKEND.",
            ) from exc
        return str(body["choices"][0]["message"]["content"])


class ClaudeAgentSDKBackend(AgentBackend):
    """The Claude Agent SDK, run headless over the workspace.

    The SDK supplies the Claude Code harness; this backend confines it to the
    agent's actual job: read-only inspection of the workspace (Read, Glob,
    Grep), no Bash, no web tools, and no file writes — a proposal arrives as
    text and still passes the diff gate, so nothing reaches the workspace
    without approval (FR-6.4, FR-6.8). Settings sources are disabled so no
    hooks or permissions from the host user's own Claude Code setup leak in.
    """

    name = "claude-agent-sdk"

    READ_ONLY_TOOLS = ["Read", "Glob", "Grep"]
    DENIED_TOOLS = ["Bash", "Write", "Edit", "NotebookEdit", "WebSearch", "WebFetch", "Task", "Agent"]

    def __init__(self, workspace: Path, model: str | None = None, max_turns: int = 12) -> None:
        self.workspace = workspace
        self.model = model
        self.max_turns = max_turns
        if model:
            self.name = f"claude-agent-sdk:{model}"

    def complete(self, system: str, prompt: str) -> str:
        try:
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                ResultMessage,
                TextBlock,
                query,
            )
        except ImportError as exc:
            raise NotFound(
                "claude-agent-sdk is not installed, so the Claude backend cannot run.",
                "Install it with: pip install claude-agent-sdk — or set NAWAT_AGENT_BACKEND=local.",
            ) from exc

        import asyncio

        self.workspace.mkdir(parents=True, exist_ok=True)
        options = ClaudeAgentOptions(
            system_prompt=system,
            cwd=str(self.workspace),
            allowed_tools=list(self.READ_ONLY_TOOLS),
            disallowed_tools=list(self.DENIED_TOOLS),
            permission_mode="dontAsk",  # headless: anything not allowed is denied, never prompted
            max_turns=self.max_turns,
            setting_sources=[],
            model=self.model,
        )

        async def run() -> str:
            final = ""
            texts: list[str] = []
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            texts.append(block.text)
                elif isinstance(message, ResultMessage):
                    final = getattr(message, "result", None) or ""
            return final or "\n".join(texts)

        try:
            return asyncio.run(run())
        except NawatError:
            raise
        except Exception as exc:
            raise NawatError(
                f"The Claude Agent SDK did not complete: {type(exc).__name__}: {str(exc)[:200]}",
                "Check that the claude CLI is installed and logged in, or set NAWAT_AGENT_BACKEND=local.",
            ) from exc


class CommandBackend(AgentBackend):
    """A CLI agent in one-shot mode — ``codex exec`` and the like.

    The prompt travels on stdin; the reply is stdout. The subprocess runs in the
    workspace with no extra environment, so it is handed no credentials by us.
    """

    def __init__(self, argv: list[str], name: str, cwd: Path | None = None) -> None:
        self.argv = argv
        self.name = name
        self.cwd = cwd

    def complete(self, system: str, prompt: str) -> str:
        try:
            completed = subprocess.run(
                self.argv,
                input=f"{system}\n\n---\n\n{prompt}",
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(self.cwd) if self.cwd else None,
            )
        except FileNotFoundError as exc:
            raise NotFound(
                f"{self.argv[0]} is not installed, so the {self.name} backend cannot run.",
                "Install it, or set NAWAT_AGENT_BACKEND=local with NAWAT_AGENT_URL.",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise NawatError(f"The {self.name} backend did not answer within 10 minutes.", "Try again.") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[:300]
            raise NawatError(f"The {self.name} backend exited {completed.returncode}: {detail}", "Check its own login/config.")
        return completed.stdout


def build_backend(env: dict[str, str] | None = None, workspace: Path | None = None) -> AgentBackend:
    """The backend the environment selects. Absent configuration is a clear
    refusal, not a broken platform (FR-6.13)."""
    env = dict(os.environ) if env is None else env
    kind = (env.get("NAWAT_AGENT_BACKEND") or "").strip().lower()
    if kind in ("local", "openai"):
        url = env.get("NAWAT_AGENT_URL", "")
        if not url:
            raise NotFound("NAWAT_AGENT_BACKEND is set but NAWAT_AGENT_URL is not.",
                           "Point it at an OpenAI-compatible endpoint, e.g. http://127.0.0.1:8001.")
        return OpenAICompatBackend(url, env.get("NAWAT_AGENT_MODEL", "default"), env.get("NAWAT_AGENT_API_KEY"))
    if kind == "claude":
        if workspace is None:
            raise NotFound("The claude backend needs a workspace to read.", "Set NAWAT_WORKSPACE.")
        return ClaudeAgentSDKBackend(workspace, model=env.get("NAWAT_AGENT_MODEL") or None)
    if kind == "codex":
        return CommandBackend(["codex", "exec", "--quiet"], "codex", cwd=workspace)
    raise NotFound(
        "No agent backend is configured; the platform is fully usable without one.",
        "Set NAWAT_AGENT_BACKEND to claude (Claude Agent SDK), local (with NAWAT_AGENT_URL), or codex.",
    )


# -- context ------------------------------------------------------------------

TRAINER_CONTRACT = """\
You write training scripts for Nawāt, a storage-orchestration platform. The contract:
- The script is ordinary Python that must also run standalone.
- Inputs arrive via environment variables: NAWAT_MODEL_DIR (local path of the staged
  base model — pass it to from_pretrained, never a hub name), NAWAT_DATASET_DIR
  (staged dataset path — pass to load_dataset), NAWAT_OUT_DIR (write artifacts here).
- Hyperparameters arrive as NAWAT_PARAM_<NAME> strings; read with os.environ.get and a default.
- The run executes offline: any attempt to download from a hub fails it.
- Save the adapter under NAWAT_OUT_DIR/adapter; each subdirectory of NAWAT_OUT_DIR
  publishes as its own artifact. Keep trainer checkpoints OUT of NAWAT_OUT_DIR
  (use tempfile.mkdtemp()).
- Log metrics with: import nawat.metrics; callbacks=[nawat.metrics.trainer_callback()]
  or nawat.metrics.log(step=..., loss=...).
Reply with a one-line summary, then the COMPLETE file in a single ```python fence.
"""


def hardware_envelope(cache: Cache) -> str:
    from .api import gpu_info

    lines = []
    gpu = None
    try:
        gpu = gpu_info()
    except Exception:
        pass
    if gpu:
        lines.append(f"GPU: {gpu['name']}, {human_bytes(gpu['memory_total'])} VRAM"
                     f" ({human_bytes(gpu['memory_total'] - gpu['memory_used'])} free now)")
    else:
        lines.append("GPU: no reading available")
    status = cache.status()
    lines.append(f"Local cache: {human_bytes(status.ceiling - status.used)} headroom under a "
                 f"{human_bytes(status.ceiling)} ceiling; disk {human_bytes(status.disk_free)} free")
    return "\n".join(lines)


def registry_summary(cache: Cache) -> str:
    try:
        keys = cache.registry(("models", "datasets"))
    except NawatError:
        return "Registry unavailable."
    if not keys:
        return "The registry is empty."
    return "\n".join(str(key) for key in sorted(keys, key=str)[:40])


def prior_runs_summary(runs: RunStore) -> str:
    lines = []
    for record in runs.list(limit=MAX_PRIOR_RUNS):
        points = metrics_module.read_points(runs.metrics_path(record.id))
        grouped = metrics_module.series(points)
        readings = []
        for name in ("loss", "cer", "wer"):
            if grouped.get(name):
                readings.append(f"final {name} {grouped[name][-1]['value']:.4g}")
        params = " ".join(f"{k}={v}" for k, v in record.spec.params.items())
        lines.append(f"- {record.id}: {record.spec.script} {params} → {record.state.value}"
                     + (f" ({', '.join(readings)})" if readings else "")
                     + (f" — {record.error}" if record.error else ""))
    return "\n".join(lines) or "No prior runs."


def failed_run_context(runs: RunStore, run_id: str) -> str:
    record = runs.get(run_id)
    log_tail = runs.read_log(run_id)[-MAX_LOG_TAIL:]
    points = metrics_module.read_points(runs.metrics_path(run_id))
    grouped = metrics_module.series(points)
    series_lines = [
        f"{name}: " + ", ".join(f"{entry['value']:.4g}" for entry in entries[-12:])
        for name, entries in grouped.items()
    ]
    return (
        f"Run {run_id} — state {record.state.value}"
        + (f", error: {record.error}" if record.error else "")
        + f"\nSpec: script={record.spec.script} model={record.spec.model}"
        + f" params={dict(record.spec.params)}"
        + ("\nMetric series (last points):\n" + "\n".join(series_lines) if series_lines else "")
        + ("\nLog tail:\n" + log_tail if log_tail else "")
    )


# -- proposals ----------------------------------------------------------------


@dataclass
class Proposal:
    path: str            # workspace-relative
    old: str
    new: str
    summary: str
    backend: str
    instruction: str
    warnings: list[str] = field(default_factory=list)

    @property
    def diff(self) -> str:
        return "".join(difflib.unified_diff(
            self.old.splitlines(keepends=True), self.new.splitlines(keepends=True),
            fromfile=f"a/{self.path}", tofile=f"b/{self.path}",
        ))

    def to_json(self) -> dict[str, Any]:
        return {"path": self.path, "summary": self.summary, "diff": self.diff,
                "new_content": self.new, "backend": self.backend, "warnings": self.warnings}


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def parse_reply(reply: str) -> tuple[str, str]:
    """(summary, code) out of a backend reply. No fence, no proposal."""
    match = _FENCE.search(reply)
    if not match:
        raise NawatError(
            "The agent replied without a fenced code block, so there is nothing to review.",
            "Ask again; the reply must contain the complete script in one ```python fence.",
        )
    code = match.group(1)
    summary = reply[: match.start()].strip().splitlines()
    return (summary[0].strip("# ").strip() if summary else "Proposed change."), code


class Agent:
    """Compose context, obtain a proposal, gate it, and record acceptance."""

    def __init__(self, cache: Cache, runs: RunStore, backend: AgentBackend | None = None) -> None:
        self.cache = cache
        self.runs = runs
        self.workspace = cache.config.workspace_root
        self.backend = backend or build_backend(workspace=self.workspace)

    # -- proposing ---------------------------------------------------------

    def propose(self, instruction: str, *, script: str | None = None, run_id: str | None = None) -> Proposal:
        record = self.runs.get(run_id) if run_id else None
        script = script or (record.spec.script if record else None)
        if not script:
            raise NotFound("No script to work on.", "Name one with --script, or a run with --run.")

        path = self.workspace / script
        old = path.read_text() if path.is_file() else ""

        system = "\n\n".join(filter(None, [
            TRAINER_CONTRACT,
            "HARDWARE\n" + hardware_envelope(self.cache),
            "AVAILABLE MODELS AND DATASETS\n" + registry_summary(self.cache),
            "PRIOR RUNS\n" + prior_runs_summary(self.runs),
        ]))
        prompt = "\n\n".join(filter(None, [
            f"TASK\n{instruction}",
            failed_run_context(self.runs, run_id) if run_id else None,
            f"CURRENT {script}\n```python\n{old}\n```" if old else f"There is no {script} yet; write it.",
        ]))

        summary, code = parse_reply(self.backend.complete(system, prompt))
        proposal = Proposal(path=script, old=old, new=code, summary=summary,
                            backend=self.backend.name, instruction=instruction)
        self._gate(proposal, record)
        return proposal

    def _gate(self, proposal: Proposal, record: RunRecord | None) -> None:
        """Checks before the proposal is even offered (FR-6.6)."""
        try:
            compile(proposal.new, proposal.path, "exec")
        except SyntaxError as exc:
            raise NawatError(
                f"The agent proposed code that does not parse: line {exc.lineno}: {exc.msg}.",
                "Nothing was offered for review. Ask again.",
            ) from exc
        model_key = record.spec.model if record else None
        if model_key is not None:
            status = self.cache.get(model_key)
            if status is not None:
                quantized = "4bit" in str(model_key).lower() or "bnb" in str(model_key).lower()
                params = params_from_bytes(status.bytes, quantized=quantized)
                from .api import gpu_info

                gpu = None
                try:
                    gpu = gpu_info()
                except Exception:
                    pass
                verdict = estimate_script(
                    proposal.new, params=params,
                    vram_available=gpu["memory_total"] if gpu else None,
                )
                if verdict.fits is False:
                    proposal.warnings.append(verdict.verdict())

    # -- accepting ---------------------------------------------------------

    def apply(self, proposal: Proposal) -> str:
        """Write the accepted edit and commit it with its provenance (FR-6.5)."""
        path = self.workspace / proposal.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(proposal.new)
        return self._commit(
            proposal.path,
            f"agent: {proposal.summary}\n\nInstruction: {proposal.instruction}\nBackend: {proposal.backend}",
        )

    def _git(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-c", "user.name=nawat", "-c", "user.email=nawat@localhost", *argv],
            cwd=str(self.workspace), capture_output=True, text=True,
        )

    def _commit(self, rel_path: str, message: str) -> str:
        self.workspace.mkdir(parents=True, exist_ok=True)
        if not (self.workspace / ".git").exists():
            self._git("init", "-q")
        self._git("add", rel_path)
        committed = self._git("commit", "-q", "-m", message)
        if committed.returncode != 0 and "nothing to commit" not in committed.stdout + committed.stderr:
            raise NawatError(
                f"git commit failed in the workspace: {(committed.stderr or committed.stdout).strip()[:200]}",
                "The file was written; commit it by hand once git is happy.",
            )
        head = self._git("rev-parse", "--short", "HEAD")
        return head.stdout.strip()

    # -- descriptions ------------------------------------------------------

    def describe_run(self, run_id: str) -> str:
        """A plain-language account of what a run was and did (FR-6.11)."""
        record = self.runs.get(run_id)
        prompt = (
            "Describe this training run in 2-3 plain sentences for a lab notebook: base model, "
            "adaptation method if inferable, data, budget, and outcome. No markdown, no headers.\n\n"
            + failed_run_context(self.runs, run_id)
        )
        description = self.backend.complete(
            "You summarise machine-learning training runs precisely and briefly.", prompt
        ).strip()
        if description:
            self.runs.update(record.id, description=description)
        return description
