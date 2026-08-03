"""Running a training job.

Stage the declared inputs under lease, run the script in a subprocess with the
upstream hub switched off, then publish what it produced and give the disk back.
The script is ordinary Python that runs standalone: everything it needs arrives
through the environment, so nothing about it is specific to this platform
(PRD principle 4, FR-2.1 through FR-2.12).

Artifact classes come out as separate keys named for the subdirectory the script
wrote them to, so a run that writes ``adapter/`` and ``gguf/`` publishes
``runs/<id>/adapter`` and ``runs/<id>/gguf`` without declaring anything.

Outputs are published only when the script exits 0, which leaves the question of
what a run that took two days and then died has to show for itself. The answer
is its checkpoints: they are written outside the output directory, into a
durable lineage that outlives the run (see :mod:`nawat.checkpoints`), kept
whenever the run does not succeed, and offered back to the next run of the same
shape through ``NAWAT_RESUME_FROM``.
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
from typing import Callable

from . import checkpoints as checkpoint_store
from .cache import Cache
from .errors import InvalidKey, NawatError, NotFound, Protected
from .keys import Key
from .leases import LeaseRecord
from .process import process_alive, terminate_group
from .runs import RunRecord, RunSpec, RunState, RunStore
from .units import human_bytes

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

#: Where a checkpoint goes if a failed run asked for one to be replicated.
CHECKPOINT_CLASS = "checkpoint"

_ENV_NAME = re.compile(r"[^A-Z0-9]+")

#: Asked for a run's name at the end of it. Returns None to keep the run id.
NamePrompt = Callable[["RunRecord"], "str | None"]


def param_env_name(name: str) -> str:
    return "NAWAT_PARAM_" + _ENV_NAME.sub("_", name.upper()).strip("_")


@dataclass(frozen=True)
class Plan:
    """What the executor will do, resolved before anything is started."""

    script: Path
    command: list[str]
    out_dir: Path
    inputs: dict[str, Path]
    checkpoint_dir: Path
    resume_from: Path | None = None


# -- publication --------------------------------------------------------------
#
# Free functions rather than methods, because a run driven from a notebook
# kernel (see :mod:`nawat.notebook`) has no subprocess and therefore no
# Executor, and must still publish byte-for-byte identically: same artifact
# classes, same keys, same verified-then-reclaimed order.


def run_output_dir(config, run_id: str) -> Path:
    """The run's own directory in the cache, where artifact classes are written."""
    return config.cache_root / "runs" / run_id


def publish_outputs(cache: Cache, prefix: str, out_dir: Path) -> list[str]:
    """Publish each artifact class the run produced, then reclaim the disk.

    ``prefix`` is the run's folder in object storage — its name when the
    researcher gave it one, its id otherwise.
    """
    published: list[str] = []
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
        key = Key.parse(f"runs/{prefix}/{entry.name}")
        cache.publish(entry, key)
        published.append(str(key))
    prune_empty(out_dir)
    return published


def publish_record(cache: Cache, runs: RunStore, run_id: str, prefix: str | None = None) -> None:
    """Copy the log, metrics and record into object storage.

    A copy, because the local originals stay put: they live in the state
    directory rather than the cache, so nothing reclaims them and a failed run
    stays readable (FR-2.11).
    """
    directory = runs.directory(run_id)
    sources = [path for path in directory.iterdir() if path.is_file() and path.name != CANCEL_SENTINEL]
    if not sources:
        return
    with tempfile.TemporaryDirectory(dir=str(cache.config.staging_root)) as scratch:
        for path in sources:
            shutil.copy2(path, Path(scratch) / path.name)
        try:
            cache.store.publish(Path(scratch), Key.parse(f"runs/{prefix or run_id}/{RECORD_CLASS}"))
        except NawatError:
            # Losing the replica of a log must not turn a successful run into a
            # failed one; the local copy is still there.
            pass


def prune_empty(directory: Path) -> None:
    try:
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    except OSError:
        pass


class Executor:
    """Runs one job at a time, start to finish."""

    def __init__(self, cache: Cache, runs: RunStore, ask_name: "NamePrompt | None" = None) -> None:
        self.cache = cache
        self.runs = runs
        self.config = cache.config
        #: Asked for a name once the training process is over and before
        #: anything is uploaded. The CLI supplies one when a terminal is
        #: attached; the API and the queue leave it None and keep the run id.
        self.ask_name = ask_name

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
        spec.checkpoint_lineage  # rejects an unusable --checkpoint-lineage here, not at hour 60
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
        checkpoint_dir, resume_from = self.prepare_checkpoints(record)
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
        return Plan(
            script=script,
            command=command,
            out_dir=out_dir,
            inputs=inputs,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume_from,
        )

    def output_dir(self, run_id: str) -> Path:
        return run_output_dir(self.config, run_id)

    def checkpoint_dir(self, record: RunRecord) -> Path:
        return checkpoint_store.lineage_dir(self.config, record.spec.checkpoint_lineage)

    def prepare_checkpoints(self, record: RunRecord) -> tuple[Path, Path | None]:
        """Open the run's checkpoint lineage and decide what it resumes from.

        A lineage is shared by every run of the same script over the same
        inputs, which is what makes a resubmission continue rather than start
        over. ``--fresh`` (``resume=False``) is the way to say that is not
        wanted, and is the only path here that deletes anything.
        """
        directory = self.checkpoint_dir(record)
        policy = record.spec.checkpoints
        if not policy.resume:
            checkpoint_store.discard(directory)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_store.note_owner(directory, record.id, record.spec.script)
        newest = checkpoint_store.latest(directory) if policy.resume else None
        return directory, newest.path if newest else None

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
                # Where the trainer should write checkpoints, and what it can
                # pick up: a script that passes these to the trainer survives a
                # crash, and one that ignores them behaves exactly as before.
                "NAWAT_CHECKPOINT_DIR": str(plan.checkpoint_dir),
                "NAWAT_CHECKPOINT_LINEAGE": spec.checkpoint_lineage,
                "NAWAT_RESUME_FROM": str(plan.resume_from) if plan.resume_from else "",
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
            env[param_env_name(name)] = str(value)
        return env

    # -- execution ---------------------------------------------------------

    def execute(self, run_id: str) -> RunRecord:
        """Stage, run, publish, reclaim. Returns the finished record."""
        record = self.runs.get(run_id)
        if record.state.terminal:
            raise Protected(f"Run {run_id} has already finished.", "Submit a new run instead.")
        leases: list[LeaseRecord] = []
        plan: Plan | None = None
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
                return self._settle_checkpoints(record, plan)

            # Named before anything is uploaded, because the name *is* the
            # folder it is uploaded into.
            record = self._name_run(record)
            record = self.runs.update(run_id, state=RunState.PUBLISHING)
            if record.exit_code == 0:
                artifacts = self._publish_outputs(record, plan)
                record = self.runs.update(
                    run_id, state=RunState.SUCCEEDED, finished_at=time.time(), artifacts=artifacts
                )
            else:
                # A failed run's outputs go to object storage too. Whatever it
                # managed to write — an emergency adapter, the reports, the
                # partial exports — is all it has to show for the GPU time, and
                # a local disk is not where that should be left.
                artifacts = self._publish_partial(record, plan)
                record = self.runs.update(
                    run_id,
                    state=RunState.FAILED,
                    finished_at=time.time(),
                    error=f"The training script exited {record.exit_code}.",
                    artifacts=artifacts,
                )
            record = self._settle_checkpoints(record, plan)
        except NawatError as exc:
            record = self._fail(run_id, f"{exc.cause} {exc.remedy or ''}".strip())
            record = self._settle_checkpoints(record, plan)
        except Exception as exc:  # noqa: BLE001 - the record must survive anything
            record = self._fail(run_id, f"{type(exc).__name__}: {exc}")
            record = self._settle_checkpoints(record, plan)
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
        lines.append(f"[nawat] ckpts   {plan.checkpoint_dir}")
        if plan.resume_from is not None:
            lines.append(f"[nawat] resume  {plan.resume_from.name} — training continues from there")
        elif not record.spec.checkpoints.resume:
            lines.append("[nawat] resume  no (--fresh); any previous checkpoints of this lineage were removed")
        else:
            lines.append("[nawat] resume  nothing to resume from; this run starts at step 0")
        lines.append(f"[nawat] command {' '.join(plan.command)}")
        return "\n".join(lines) + "\n\n"

    # -- publication -------------------------------------------------------

    def _publish_outputs(self, record: RunRecord, plan: Plan) -> list[str]:
        return publish_outputs(self.cache, record.publish_prefix, plan.out_dir)

    def _publish_partial(self, record: RunRecord, plan: Plan) -> list[str]:
        """Publish what a run that did not succeed managed to produce.

        Guarded, because a store that is unreachable must not turn "the script
        failed" into a different and less useful error — the outputs simply
        stay on local disk, and the log says why.
        """
        try:
            return self._publish_outputs(record, plan)
        except Exception as exc:  # noqa: BLE001 - the original failure is the one that matters
            self._log(record.id, f"could not publish the outputs of the failed run: {type(exc).__name__}: {exc}")
            self._log(record.id, f"they are still on disk at {plan.out_dir}")
            return [str(k) for k in record.artifacts]

    def _publish_record(self, record: RunRecord) -> None:
        publish_record(self.cache, self.runs, record.id, record.publish_prefix)

    # -- naming ------------------------------------------------------------

    def _name_run(self, record: RunRecord) -> RunRecord:
        """Let the researcher name the run before its artifacts are uploaded.

        Asked at the end rather than the beginning, because that is when there
        is something to say about it. It is only ever a prompt on an interactive
        submit: a queued run, an API run, or a shell with no terminal keeps the
        generated id rather than blocking on an answer nobody is there to give.
        """
        if record.name or self.ask_name is None:
            return record
        try:
            chosen = self.ask_name(record)
        except Exception:  # noqa: BLE001 - naming is a convenience, never a failure
            return record
        if not chosen:
            return record
        try:
            named = self.runs.update(record.id, name=chosen)
        except NawatError as exc:
            self._log(record.id, f"could not use that name: {exc.cause}")
            return record
        self._log(record.id, f"named {named.name} — artifacts publish under runs/{named.publish_prefix}")
        return named

    # -- checkpoints -------------------------------------------------------

    def _settle_checkpoints(self, record: RunRecord, plan: Plan | None) -> RunRecord:
        """Replicate what the run reached, then apply the keep-on-failure rule.

        Called on every terminal path, including the one where publishing itself
        threw — a run that trained for two days and then failed to upload is the
        case where the checkpoint matters most.

        The replication happens first and regardless of outcome, because object
        storage is the only durable copy this host has: a successful run's
        checkpoint is then safe to delete locally, and a failed one has a second
        copy of the only state that cannot be recomputed.
        """
        if plan is None:
            return record
        succeeded = record.state is RunState.SUCCEEDED
        artifacts = list(record.artifacts)
        reached = checkpoint_store.latest(plan.checkpoint_dir)
        replicated = False
        if reached is not None and record.spec.checkpoints.publish:
            key = self._replicate_checkpoint(record, reached)
            replicated = key is not None
            if key is not None and key not in artifacts:
                artifacts.append(key)

        newest = checkpoint_store.settle(
            plan.checkpoint_dir,
            succeeded=succeeded,
            keep=record.spec.checkpoints.keep,
            replicated=replicated,
            log=lambda message: self._log(record.id, message),
        )
        if newest is not None and not succeeded:
            self._log(record.id, f"resume it with: nawat resume {record.id}")
        return self.runs.update(
            record.id,
            checkpoint=str(newest.path) if newest else None,
            artifacts=[str(k) for k in artifacts],
        )

    def _replicate_checkpoint(self, record: RunRecord, checkpoint) -> Key | None:
        """Copy the run's last checkpoint into object storage.

        The local copy is untouched, so resuming a failed run still reads from
        the lineage on disk rather than pulling gigabytes back down. Turn it off
        with ``--no-publish-checkpoints`` when the upload costs more than the
        insurance is worth.
        """
        key = Key.parse(f"runs/{record.publish_prefix}/{CHECKPOINT_CLASS}")
        self._log(record.id, f"replicating {checkpoint.name} ({human_bytes(checkpoint.bytes)}) to {key}")
        try:
            self.cache.store.publish(checkpoint.path, key)
        except Exception as exc:  # noqa: BLE001 - a failed upload must not mask the failure it followed
            self._log(record.id, f"could not replicate the checkpoint: {type(exc).__name__}: {exc}")
            return None
        self._log(record.id, f"replicated {key}")
        return key

    def _log(self, run_id: str, message: str) -> None:
        """Append to the run log after the script's own output has ended."""
        try:
            with self.runs.log_path(run_id).open("a") as handle:
                handle.write(f"[nawat] {message}\n")
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

    def submit(self, spec: RunSpec, run_id: str | None = None, name: str | None = None) -> RunRecord:
        self.executor.validate(spec)
        record = self.runs.create(spec, run_id, name=name)
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
