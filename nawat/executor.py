"""Running a training job.

Stage the declared inputs under lease, run the script in a subprocess with the
upstream hub switched off, then publish what it produced and give the disk back.
The script is ordinary Python that runs standalone: everything it needs arrives
through the environment, so nothing about it is specific to this platform
(PRD principle 4, FR-2.1 through FR-2.12).

Artifact classes come out as separate keys named for the subdirectory the script
wrote them to, so a run that writes ``adapter/`` and ``gguf/`` publishes
``runs/<id>/adapter`` and ``runs/<id>/gguf`` without declaring anything.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .cache import Cache
from .errors import InvalidKey, NawatError, NotFound, Protected
from .keys import Key
from .leases import LeaseRecord
from .process import process_alive, terminate_group
from .runs import RunRecord, RunSpec, RunState, RunStore

#: Written into the run directory to distinguish a cancelled run from a crashed
#: one. A file rather than a flag in memory, so the CLI can cancel a run the API
#: process started.
CANCEL_SENTINEL = "CANCELLED"

#: Grace between asking a trainer to stop and insisting.
TERMINATE_GRACE = 10.0

#: Loose files in the output directory publish under this name.
LOOSE_OUTPUT = "output"

#: The run's own log and record.
RECORD_CLASS = "record"

_ENV_NAME = re.compile(r"[^A-Z0-9]+")


def _param_env_name(name: str) -> str:
    return "NAWAT_PARAM_" + _ENV_NAME.sub("_", name.upper()).strip("_")


@dataclass(frozen=True)
class Plan:
    """What the executor will do, resolved before anything is started."""

    script: Path
    command: list[str]
    out_dir: Path
    inputs: dict[str, Path]


class Executor:
    """Runs one job at a time, start to finish."""

    def __init__(self, cache: Cache, runs: RunStore) -> None:
        self.cache = cache
        self.runs = runs
        self.config = cache.config

    # -- validation --------------------------------------------------------

    def resolve_script(self, script: str) -> Path:
        """Locate a training script inside the workspace, refusing anything outside it."""
        workspace = self.config.workspace_root
        candidate = Path(script).expanduser()
        path = (candidate if candidate.is_absolute() else workspace / candidate).resolve()
        if workspace.resolve() not in path.parents and path != workspace.resolve():
            raise InvalidKey(
                f"{script} is outside the workspace.",
                f"Training scripts must live under {workspace}.",
            )
        if not path.is_file():
            raise NotFound(f"{path} does not exist.", "Check the script name, or list them with: nawat scripts")
        if path.suffix not in (".py", ".ipynb"):
            raise InvalidKey(
                f"{path.name} is neither a Python script nor a notebook.",
                "Submit a .py or .ipynb entrypoint.",
            )
        return path

    def validate(self, spec: RunSpec) -> None:
        """Everything checkable before GPU time is spent (FR-5.4)."""
        self.resolve_script(spec.script)
        for key in spec.staged_keys():
            if self.cache.get(key) is None and not self.cache.store.exists(key):
                if key.hub_repo_id is None or self.config.offline:
                    raise NotFound(
                        f"{key} is not cached and not in object storage.",
                        "Publish or seed it first, then resubmit.",
                    )

    # -- planning ----------------------------------------------------------

    def plan(self, record: RunRecord, inputs: dict[str, Path]) -> Plan:
        script = self.resolve_script(record.spec.script)
        out_dir = self.output_dir(record.id)
        out_dir.mkdir(parents=True, exist_ok=True)
        if record.spec.is_notebook:
            executed = out_dir / RECORD_CLASS / f"{script.stem}.executed.ipynb"
            executed.parent.mkdir(parents=True, exist_ok=True)
            # The executed notebook *is* the run record for a notebook run (FR-2.5).
            command = [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--allow-errors",
                "--ExecutePreprocessor.timeout=-1",
                "--output",
                str(executed),
                str(script),
            ]
        else:
            command = [sys.executable, "-u", str(script)]
        return Plan(script=script, command=command, out_dir=out_dir, inputs=inputs)

    def output_dir(self, run_id: str) -> Path:
        """The run's own directory in the cache, where artifact classes are written."""
        return self.config.cache_root / "runs" / run_id

    def environment(self, record: RunRecord, plan: Plan) -> dict[str, str]:
        """Configuration for the training script, injected not embedded (FR-2.4)."""
        env = dict(os.environ)
        spec = record.spec
        env.update(
            {
                "NAWAT_RUN_ID": record.id,
                "NAWAT_OUT_DIR": str(plan.out_dir),
                "NAWAT_WORKSPACE": str(self.config.workspace_root),
                "NAWAT_INPUTS": json.dumps(plan.inputs),
                "NAWAT_PARAMS": json.dumps(dict(spec.params)),
                "NAWAT_METRICS_PATH": str(self.runs.metrics_path(record.id)),
                "PYTHONUNBUFFERED": "1",
                # The run is offline by construction: an input that was not
                # declared and staged fails here rather than being downloaded
                # silently, which would break both reproducibility and the disk
                # budget (FR-2.3).
                "NAWAT_OFFLINE": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
            }
        )
        if spec.model:
            env["NAWAT_MODEL_DIR"] = plan.inputs[str(spec.model)]
        if spec.datasets:
            env["NAWAT_DATASET_DIR"] = plan.inputs[str(spec.datasets[0])]
            env["NAWAT_DATASET_DIRS"] = json.dumps([plan.inputs[str(k)] for k in spec.datasets])
        for name, value in spec.params.items():
            env[_param_env_name(name)] = str(value)
        return env

    # -- execution ---------------------------------------------------------

    def execute(self, run_id: str) -> RunRecord:
        """Stage, run, publish, reclaim. Returns the finished record."""
        record = self.runs.get(run_id)
        if record.state.terminal:
            raise Protected(f"Run {run_id} has already finished.", "Submit a new run instead.")
        leases: list[LeaseRecord] = []
        self._clear_cancel(run_id)
        try:
            record = self.runs.update(run_id, state=RunState.STAGING, started_at=time.time())
            inputs: dict[str, Path] = {}
            for key in record.spec.staged_keys():
                path = self.cache.resolve(key, lease=False)
                leases.extend(self.cache.leases.acquire([key], holder=f"run {run_id}"))
                inputs[str(key)] = str(path)

            plan = self.plan(record, inputs)
            record = self._run_process(record, plan)

            if record.state is RunState.CANCELLED:
                return record
            if record.exit_code == 0:
                record = self.runs.update(run_id, state=RunState.PUBLISHING)
                artifacts = self._publish_outputs(record, plan)
                record = self.runs.update(
                    run_id, state=RunState.SUCCEEDED, finished_at=time.time(), artifacts=artifacts
                )
            else:
                record = self.runs.update(
                    run_id,
                    state=RunState.FAILED,
                    finished_at=time.time(),
                    error=f"The training script exited {record.exit_code}.",
                )
        except NawatError as exc:
            record = self._fail(run_id, f"{exc.cause} {exc.remedy or ''}".strip())
        except Exception as exc:  # noqa: BLE001 - the record must survive anything
            record = self._fail(run_id, f"{type(exc).__name__}: {exc}")
        finally:
            self.cache.leases.release(lease.id for lease in leases)
        # The record and its log are published whatever happened (FR-2.11).
        self._publish_record(record)
        return self.runs.get(run_id)

    def _fail(self, run_id: str, error: str) -> RunRecord:
        record = self.runs.find(run_id)
        finished = time.time()
        if record is not None and record.state is RunState.CANCELLED:
            return record
        return self.runs.update(run_id, state=RunState.FAILED, finished_at=finished, error=error)

    def _run_process(self, record: RunRecord, plan: Plan) -> RunRecord:
        log_path = self.runs.log_path(record.id)
        env = self.environment(record, plan)
        with log_path.open("ab", buffering=0) as log:
            log.write(self._log_header(record, plan).encode())
            process = subprocess.Popen(
                plan.command,
                cwd=str(self.config.workspace_root),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                # Its own process group, so cancelling reaches the dataloader
                # workers a trainer spawns, not just the trainer.
                start_new_session=True,
            )
            record = self.runs.update(record.id, state=RunState.RUNNING, pid=process.pid)
            exit_code = process.wait()
            log.write(f"\n[nawat] exited {exit_code} after {self._elapsed(record)}\n".encode())

        if self._cancel_requested(record.id):
            return self.runs.update(
                record.id,
                state=RunState.CANCELLED,
                finished_at=time.time(),
                exit_code=exit_code,
                error="Cancelled.",
            )
        return self.runs.update(record.id, exit_code=exit_code)

    def _elapsed(self, record: RunRecord) -> str:
        from .units import human_age

        return human_age(record.duration or 0.0)

    def _log_header(self, record: RunRecord, plan: Plan) -> str:
        lines = [
            f"[nawat] run {record.id}",
            f"[nawat] script  {plan.script}",
            f"[nawat] outputs {plan.out_dir}",
        ]
        for key, path in plan.inputs.items():
            lines.append(f"[nawat] input   {key} -> {path}")
        for name, value in record.spec.params.items():
            lines.append(f"[nawat] param   {name}={value}")
        lines.append(f"[nawat] command {' '.join(plan.command)}")
        return "\n".join(lines) + "\n\n"

    # -- publication -------------------------------------------------------

    def _publish_outputs(self, record: RunRecord, plan: Plan) -> list[str]:
        """Publish each artifact class the script produced, then reclaim the disk."""
        published: list[str] = []
        out_dir = plan.out_dir
        if not out_dir.exists():
            return published

        loose = [entry for entry in out_dir.iterdir() if entry.is_file() and not entry.name.startswith(".nawat")]
        if loose:
            staging = out_dir / LOOSE_OUTPUT
            staging.mkdir(exist_ok=True)
            for entry in loose:
                entry.rename(staging / entry.name)

        for entry in sorted(out_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".nawat"):
                continue
            if not any(entry.rglob("*")):
                continue
            key = Key.parse(f"runs/{record.id}/{entry.name}")
            self.cache.publish(entry, key)
            published.append(str(key))
        self._prune(out_dir)
        return published

    def _publish_record(self, record: RunRecord) -> None:
        """Copy the log and the record into object storage.

        A copy, because the local originals stay put: they live in the state
        directory rather than the cache, so nothing reclaims them and a failed
        run stays readable (FR-2.11).
        """
        directory = self.runs.directory(record.id)
        sources = [path for path in directory.iterdir() if path.is_file() and path.name != CANCEL_SENTINEL]
        if not sources:
            return
        with tempfile.TemporaryDirectory(dir=str(self.config.staging_root)) as scratch:
            for path in sources:
                shutil.copy2(path, Path(scratch) / path.name)
            try:
                self.cache.store.publish(Path(scratch), Key.parse(f"runs/{record.id}/{RECORD_CLASS}"))
            except NawatError:
                # Losing the replica of a log must not turn a successful run
                # into a failed one; the local copy is still there.
                pass

    def _prune(self, directory: Path) -> None:
        try:
            if directory.exists() and not any(directory.iterdir()):
                directory.rmdir()
        except OSError:
            pass

    # -- cancellation ------------------------------------------------------

    def cancel(self, run_id: str) -> RunRecord:
        """Stop a run and release what it holds (FR-2.10).

        Works from any process, because it acts on the recorded pid rather than
        on a handle only the executor holds.
        """
        record = self.runs.get(run_id)
        if record.state.terminal:
            raise Protected(f"Run {run_id} has already finished.", "There is nothing to cancel.")
        self._request_cancel(run_id)
        if record.state is RunState.QUEUED:
            return self.runs.update(
                run_id, state=RunState.CANCELLED, finished_at=time.time(), error="Cancelled before it started."
            )
        if record.pid is not None:
            self._terminate(record.pid)
        return self.runs.get(run_id)

    def _terminate(self, pid: int) -> None:
        terminate_group(pid, grace=TERMINATE_GRACE)

    def _cancel_path(self, run_id: str) -> Path:
        return self.runs.directory(run_id) / CANCEL_SENTINEL

    def _request_cancel(self, run_id: str) -> None:
        self._cancel_path(run_id).write_text(str(time.time()))

    def _cancel_requested(self, run_id: str) -> bool:
        return self._cancel_path(run_id).exists()

    def _clear_cancel(self, run_id: str) -> None:
        self._cancel_path(run_id).unlink(missing_ok=True)


class RunQueue:
    """Serial execution. The host has one GPU, so runs wait their turn (FR-2.12)."""

    def __init__(self, executor: Executor, *, poll: float = 0.2) -> None:
        self.executor = executor
        self.runs = executor.runs
        self.poll = poll
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()

    def submit(self, spec: RunSpec, run_id: str | None = None) -> RunRecord:
        self.executor.validate(spec)
        record = self.runs.create(spec, run_id)
        self._idle.clear()
        return record

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._serve, name="nawat-run-queue", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued or running. For tests and shutdown."""
        return self._idle.wait(timeout)

    def _serve(self) -> None:
        while not self._stop.is_set():
            record = self.runs.next_queued()
            if record is None:
                self._idle.set()
                self._stop.wait(self.poll)
                continue
            self._idle.clear()
            try:
                self.executor.execute(record.id)
            except Exception:  # noqa: BLE001 - one bad run must not stop the queue
                pass
