"""Runs driven from a live kernel.

``nawat submit`` runs a script in a subprocess: the executor stages the inputs,
injects ``NAWAT_*`` into the environment, and publishes whatever the script left
in ``NAWAT_OUT_DIR``. A notebook has no subprocess — the kernel *is* the run —
so this module does the same four things in-process:

    import nawat

    run = nawat.begin_run(
        model   = "models/unsloth/Qwen3.5-0.8B",
        dataset = "datasets/unsloth/LaTeX_OCR",
        params  = {"max_steps": 30, "learning_rate": 2e-4},
    )

    model, tokenizer = FastVisionModel.from_pretrained(run.model_dir)
    ...
    model.save_pretrained(run.artifact_dir("adapter"))
    run.finish()

Between those two calls the inputs are leased to the kernel's pid, so nothing
can evict weights out from under a cell; metrics written by the trainer land in
the run's durable series; and the hub is switched off, so a stray
``from_pretrained("some/other-model")`` fails loudly instead of quietly filling
the disk.

The point of the accessors — :func:`model_dir`, :func:`dataset_dir`,
:func:`out_dir`, :func:`param` — is that they read the environment when one is
set and the live run otherwise. Cells written with them run unchanged as a
submitted script, which is how a notebook graduates into ``nawat submit``
without an edit (PRD principle 4).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import checkpoints as checkpoint_store
from . import metrics
from .cache import Cache
from .checkpoints import CheckpointPolicy
from .errors import InvalidKey, NawatError, NotFound, Protected
from .executor import param_env_name, publish_outputs, publish_record, run_output_dir
from .hub import OFFLINE_VARS
from .keys import Key
from .leases import LeaseRecord
from .runs import RunRecord, RunSpec, RunState, RunStore

#: What a run started from a kernel calls itself when the notebook's own name
#: cannot be discovered. It goes in the run record's ``script`` field, which is
#: "what was submitted" — for a kernel run, the honest answer is the kernel.
UNKNOWN_SCRIPT = "<notebook>"

_active: "Run | None" = None
_active_lock = threading.Lock()


# -- finding the notebook -----------------------------------------------------


def notebook_path() -> Path | None:
    """The notebook driving this kernel, if the host makes it discoverable.

    Jupyter ≥ 7 exports ``JPY_SESSION_NAME``; VS Code injects
    ``__vsc_ipynb_file__`` into the user namespace. Colab exposes neither, so
    None is a normal answer, not a failure.
    """
    for name in ("JPY_SESSION_NAME", "NAWAT_NOTEBOOK"):
        value = os.environ.get(name)
        if value and Path(value).is_file():
            return Path(value).resolve()
    try:  # the user namespace, where VS Code leaves the path
        import IPython  # type: ignore

        shell = IPython.get_ipython()
        candidate = (shell.user_ns.get("__vsc_ipynb_file__") if shell else None) or ""
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    except Exception:  # noqa: BLE001 - discovery is best-effort by construction
        pass
    return None


# -- the run ------------------------------------------------------------------


class Run:
    """One recorded experiment, driven a cell at a time.

    Constructed by :func:`begin_run`, which stages and leases the inputs before
    returning: by the time you hold one of these, every path on it exists.
    """

    def __init__(
        self,
        cache: Cache,
        runs: RunStore,
        record: RunRecord,
        inputs: dict[str, Path],
        leases: list[LeaseRecord],
        *,
        offline: bool,
    ) -> None:
        self.cache = cache
        self.runs = runs
        self.record = record
        self.inputs = inputs
        self._leases = leases
        self._offline = offline
        self._saved_env: dict[str, str | None] = {}
        self._scratch: list[Path] = []
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- identity ----------------------------------------------------------

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def spec(self) -> RunSpec:
        return self.record.spec

    @property
    def state(self) -> RunState:
        return self.runs.get(self.id).state

    @property
    def finished(self) -> bool:
        return self.state.terminal

    def __repr__(self) -> str:
        return f"<nawat run {self.id} {self.state.value}>"

    # -- where things are --------------------------------------------------

    @property
    def model_dir(self) -> str:
        """The staged model directory, as a string ready for ``from_pretrained``."""
        if self.spec.model is None:
            raise NotFound(
                f"Run {self.id} was started without a model.",
                'Pass one: nawat.begin_run(model="models/unsloth/Qwen3.5-0.8B", ...)',
            )
        return str(self.inputs[str(self.spec.model)])

    @property
    def dataset_dir(self) -> str:
        """The first staged dataset, as a string ready for ``load_dataset``."""
        return self.dataset_dirs[0]

    @property
    def dataset_dirs(self) -> list[str]:
        if not self.spec.datasets:
            raise NotFound(
                f"Run {self.id} was started without a dataset.",
                'Pass one: nawat.begin_run(dataset="datasets/unsloth/LaTeX_OCR", ...)',
            )
        return [str(self.inputs[str(key)]) for key in self.spec.datasets]

    @property
    def out_dir(self) -> Path:
        """Where artifacts go. Each subdirectory becomes its own published key."""
        return run_output_dir(self.cache.config, self.id)

    @property
    def metrics_path(self) -> Path:
        return self.runs.metrics_path(self.id)

    @property
    def log_path(self) -> Path:
        return self.runs.log_path(self.id)

    def dir_for(self, key: str | Key) -> str:
        """The staged path of any declared input, by key."""
        parsed = str(Key.parse(key))
        if parsed not in self.inputs:
            raise NotFound(
                f"{parsed} is not an input of run {self.id}.",
                f"Declared inputs: {', '.join(self.inputs) or 'none'}.",
            )
        return str(self.inputs[parsed])

    def artifact_dir(self, name: str = "adapter") -> Path:
        """A directory that will be published as ``runs/<id>/<name>``.

        ``adapter``, ``merged`` and ``gguf`` are the conventional names; any
        other is equally valid and publishes under whatever you call it.
        """
        if not name or "/" in name or name.startswith("."):
            raise InvalidKey(
                f"{name!r} is not usable as an artifact class.",
                "Use a plain directory name such as adapter, merged or gguf.",
            )
        path = self.out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def scratch_dir(self, name: str = "trainer") -> Path:
        """Scratch space that will *not* be published, and is deleted at the end.

        For working files a later run has no use for: a tokenised copy of the
        corpus, an intermediate export, a merge staging area. Deleted when the
        run ends, however it ends.

        Trainer checkpoints do **not** belong here — put them in
        :attr:`checkpoint_dir`, which survives the run, so a failure at hour 60
        costs an hour instead of the whole thing.
        """
        path = self.cache.config.staging_root / f"scratch-{self.id}-{name}"
        path.mkdir(parents=True, exist_ok=True)
        if path not in self._scratch:
            self._scratch.append(path)
        return path

    # -- checkpoints -------------------------------------------------------

    @property
    def lineage(self) -> str:
        """The checkpoint lineage this run trains into. Shared with every run of
        the same script over the same inputs, which is what lets one resume
        another."""
        return self.spec.checkpoint_lineage

    @property
    def checkpoint_dir(self) -> Path:
        """Where the trainer should write checkpoints — durable across runs.

            SFTConfig(**nawat.checkpoint_args(save_steps=250), ...)

        Unlike :meth:`scratch_dir` and :attr:`out_dir`, nothing removes this
        when the run ends badly. It is the only copy of a partly trained model,
        and unlike every artifact in the cache it cannot be fetched again.
        """
        path = checkpoint_store.lineage_dir(self.cache.config, self.lineage)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def resume_from(self) -> str | None:
        """The checkpoint to continue from, or None to start at step 0.

            trainer.train(resume_from_checkpoint=run.resume_from)
        """
        if not self.spec.checkpoints.resume:
            return None
        newest = checkpoint_store.latest(self.checkpoint_dir)
        return str(newest.path) if newest else None

    @property
    def checkpoints(self) -> list[checkpoint_store.Checkpoint]:
        """Every checkpoint in this run's lineage, oldest step first."""
        return checkpoint_store.read(self.checkpoint_dir)

    def _prepare_checkpoints(self) -> None:
        directory = checkpoint_store.lineage_dir(self.cache.config, self.lineage)
        if not self.spec.checkpoints.resume:
            checkpoint_store.discard(directory)
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint_store.note_owner(directory, self.id, self.spec.script)

    # -- parameters --------------------------------------------------------

    def param(self, name: str, default: Any = None, cast: Callable[[str], Any] | None = None) -> Any:
        """A hyperparameter, with the notebook's value as the default.

        Values arrive as strings from ``nawat submit --param``; the type of
        ``default`` decides what they are converted to, so ``param("max_steps",
        30)`` is an int in both worlds.
        """
        raw = self.spec.params.get(name)
        if raw is None:
            return default
        return _coerce(raw, default, cast)

    @property
    def params(self) -> dict[str, str]:
        return dict(self.spec.params)

    # -- metrics and the log ----------------------------------------------

    def callback(self):
        """A ``transformers`` callback that streams this run's metric series.

        ``SFTTrainer(..., callbacks=[run.callback()])`` — loss, learning rate,
        grad norm and steps/second land in ``nawat metrics <id>``, live.
        """
        return metrics.trainer_callback()

    def log_metric(self, step: int | None = None, event: str | None = None, **fields: Any) -> dict[str, Any]:
        """Append one point to the series, for training loops without a Trainer."""
        return metrics.log(step=step, event=event, **fields)

    def log(self, message: str) -> None:
        """Append a line to the run log, so ``nawat logs <id>`` shows it."""
        with self.log_path.open("a") as handle:
            handle.write(f"[nawat] {message}\n")

    # -- finishing ---------------------------------------------------------

    def finish(self, *, description: str | None = None, name: str | None = None) -> RunRecord:
        """Publish everything under :attr:`out_dir`, then release the inputs.

        Each subdirectory is uploaded, verified file by file, and reclaimed from
        local disk — the same path a submitted script takes on exit 0.

        ``name`` files the artifacts under ``runs/<name>/`` in object storage
        instead of ``runs/<id>/``, which is worth doing at the end of a run that
        turned out to be worth keeping.
        """
        self._require_active()
        if name:
            self.record = self.runs.update(self.id, name=name)
        self.record = self.runs.update(self.id, state=RunState.PUBLISHING)
        try:
            artifacts = publish_outputs(self.cache, self.record.publish_prefix, self.out_dir)
        except BaseException as exc:  # noqa: BLE001 - a failed publish is a failed run
            self.fail(f"Publishing failed: {exc}")
            raise
        for key in artifacts:
            self.log(f"published {key}")
        self.record = self.runs.update(
            self.id,
            state=RunState.SUCCEEDED,
            finished_at=time.time(),
            exit_code=0,
            artifacts=artifacts,
            **({"description": description} if description is not None else {}),
        )
        self._settle(succeeded=True)
        return self.record

    def fail(self, error: str | BaseException) -> RunRecord:
        """Record the run as failed. Outputs and checkpoints stay on local disk.

        The checkpoints are the point: the cell can be re-run, and
        :attr:`resume_from` picks up where the failure interrupted it.
        """
        self._require_active()
        text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        self.log(f"failed: {text}")
        self.record = self.runs.update(
            self.id, state=RunState.FAILED, finished_at=time.time(), error=text
        )
        self._settle(succeeded=False)
        return self.record

    def cancel(self) -> RunRecord:
        """Abandon the run without publishing and without marking it a failure."""
        self._require_active()
        self.log("cancelled")
        self.record = self.runs.update(
            self.id, state=RunState.CANCELLED, finished_at=time.time(), error="Cancelled."
        )
        self._settle(succeeded=False)
        return self.record

    def close(self) -> None:
        """Let go of everything this run holds, leaving the record as it stands.

        Idempotent, and safe on a run whose record went terminal elsewhere — a
        crash reconciliation, or ``nawat cancel`` from another shell. Without
        it, a kernel could go on holding leases against a run the rest of the
        platform considers over.
        """
        global _active
        self.cache.leases.release(lease.id for lease in self._leases)
        self._leases = []
        for path in self._scratch:
            shutil.rmtree(path, ignore_errors=True)
        self._scratch = []
        self._restore_env()
        with _active_lock:
            if _active is self:
                _active = None

    def _require_active(self) -> None:
        state = self.state
        if state.terminal:
            raise Protected(
                f"Run {self.id} has already finished ({state.value}).",
                "Start another with nawat.begin_run(...), or let go of this one with run.close().",
            )

    def _settle(self, succeeded: bool) -> None:
        """Clean up, then archive the record — after a transition this run made."""
        self._record_checkpoints(succeeded)
        self.close()
        self._archive_source()
        publish_record(self.cache, self.runs, self.id, self.record.publish_prefix)

    def _record_checkpoints(self, succeeded: bool) -> None:
        """Keep what a bad ending leaves behind, and note it in the record."""
        newest = checkpoint_store.settle(
            checkpoint_store.lineage_dir(self.cache.config, self.lineage),
            succeeded=succeeded,
            keep=self.spec.checkpoints.keep,
            log=self.log,
        )
        self.record = self.runs.update(self.id, checkpoint=str(newest.path) if newest else None)

    def _archive_source(self) -> None:
        """Keep a copy of the notebook beside the log, when we can find it.

        The executor archives the executed notebook as the run record; a kernel
        run can only archive the file on disk, which may be a cell or two behind
        what actually ran. Better than nothing, and clearly labelled.
        """
        source = self.spec.script
        if source == UNKNOWN_SCRIPT:
            return
        path = Path(source)
        if not path.is_file():
            return
        try:
            shutil.copy2(path, self.runs.directory(self.id) / f"source-{path.name}")
        except OSError:
            pass

    # -- the environment ---------------------------------------------------

    def _apply_env(self) -> None:
        """Set what a submitted script would have been given.

        Cells that read ``os.environ["NAWAT_MODEL_DIR"]`` — copied from a
        training script, or destined to become one — work unchanged, and
        anything the kernel spawns inherits the same view.
        """
        env: dict[str, str] = {
            "NAWAT_RUN_ID": self.id,
            "NAWAT_OUT_DIR": str(self.out_dir),
            "NAWAT_WORKSPACE": str(self.cache.config.workspace_root),
            "NAWAT_INPUTS": json.dumps({k: str(v) for k, v in self.inputs.items()}),
            "NAWAT_PARAMS": json.dumps(self.params),
            "NAWAT_CHECKPOINT_DIR": str(self.checkpoint_dir),
            "NAWAT_CHECKPOINT_LINEAGE": self.lineage,
            "NAWAT_RESUME_FROM": self.resume_from or "",
            metrics.ENV_VAR: str(self.metrics_path),
        }
        if self.spec.model is not None:
            env["NAWAT_MODEL_DIR"] = self.model_dir
        if self.spec.datasets:
            env["NAWAT_DATASET_DIR"] = self.dataset_dir
            env["NAWAT_DATASET_DIRS"] = json.dumps(self.dataset_dirs)
        for name, value in self.params.items():
            env[param_env_name(name)] = str(value)
        if self._offline:
            # Staging is done, so nothing legitimate needs the hub now. A
            # download from here would be an undeclared input: not in the run
            # record, and not accounted for against the cache ceiling.
            env.update({name: "1" for name in OFFLINE_VARS})
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        for name, value in env.items():
            self._saved_env.setdefault(name, os.environ.get(name))
            os.environ[name] = value

    def _restore_env(self) -> None:
        for name, previous in self._saved_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous
        self._saved_env = {}

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.finished:
            self.close()  # ended elsewhere; still let go of what we hold
        elif exc is None:
            self.finish()
        else:
            self.fail(exc)
        return False


# -- starting one -------------------------------------------------------------


def begin_run(
    *,
    model: str | Key | None = None,
    dataset: str | Key | None = None,
    datasets: Sequence[str | Key] = (),
    inputs: Sequence[str | Key] = (),
    params: Mapping[str, Any] | None = None,
    notes: str = "",
    run_id: str | None = None,
    script: str | None = None,
    offline: bool = True,
    cache: Cache | None = None,
    checkpoint_lineage: str = "",
    resume: bool = True,
    keep_checkpoints: bool = False,
) -> Run:
    """Stage the inputs, open a run record, and hold everything for the kernel.

    Returns once every input is on local disk and leased to this process. The
    lease dies with the kernel — including a crash — so a forgotten notebook
    cannot wedge the cache, and a live one cannot have its weights evicted.

    Set ``offline=False`` only when the kernel itself must reach the internet;
    artifacts always resolve through the cache regardless.

    Checkpoints go to a lineage shared by every run with the same script and
    inputs, so re-running a notebook after a crash continues from the last saved
    step. ``resume=False`` starts that lineage over; ``keep_checkpoints=True``
    holds on to them even after the run succeeds.
    """
    from . import default_cache

    global _active
    with _active_lock:
        if _active is not None and not _active.finished:
            raise Protected(
                f"Run {_active.id} is still open in this kernel.",
                "Finish it with run.finish(), or abandon it with run.cancel().",
            )

    cache = cache or default_cache()
    runs = RunStore(cache.db, cache.config.state_dir / "runs")

    if script is None:
        found = notebook_path()
        script = str(found) if found else UNKNOWN_SCRIPT

    every_dataset = ((dataset,) if dataset is not None else ()) + tuple(datasets)
    spec = RunSpec(
        script=script,
        model=Key.parse(model) if model is not None else None,
        datasets=tuple(Key.parse(k) for k in every_dataset),
        inputs=tuple(Key.parse(k) for k in inputs),
        params={name: str(value) for name, value in (params or {}).items()},
        notes=notes,
        checkpoints=CheckpointPolicy(lineage=checkpoint_lineage, resume=resume, keep=keep_checkpoints),
    )

    record = runs.create(spec, run_id)
    staged: dict[str, Path] = {}
    leases: list[LeaseRecord] = []
    try:
        record = runs.update(record.id, state=RunState.STAGING, started_at=time.time())
        for key in spec.staged_keys():
            staged[str(key)] = cache.resolve(key, lease=False)
            leases.extend(cache.leases.acquire([key], holder=f"notebook {record.id}"))
        record = runs.update(record.id, state=RunState.RUNNING, pid=os.getpid())
    except BaseException as exc:  # noqa: BLE001 - the record must survive anything
        cache.leases.release(lease.id for lease in leases)
        message = f"{exc.cause} {exc.remedy or ''}".strip() if isinstance(exc, NawatError) else f"{type(exc).__name__}: {exc}"
        runs.update(record.id, state=RunState.FAILED, finished_at=time.time(), error=message)
        raise

    run = Run(cache, runs, record, staged, leases, offline=offline)
    run._prepare_checkpoints()
    run._apply_env()
    run.log(f"run {run.id} opened in kernel pid {os.getpid()}")
    run.log(f"script  {spec.script}")
    run.log(f"outputs {run.out_dir}")
    run.log(f"ckpts   {run.checkpoint_dir}")
    if run.resume_from:
        run.log(f"resume  {Path(run.resume_from).name} — training continues from there")
    for key, path in staged.items():
        run.log(f"input   {key} -> {path}")
    for name, value in run.params.items():
        run.log(f"param   {name}={value}")
    with _active_lock:
        _active = run
    return run


def current_run() -> Run | None:
    """The run open in this kernel, if there is one."""
    with _active_lock:
        return _active


def require_run() -> Run:
    run = current_run()
    if run is None:
        raise NotFound(
            "No run is open in this kernel.",
            "Start one with nawat.begin_run(model=..., dataset=...).",
        )
    return run


# -- accessors that work in both worlds ---------------------------------------
#
# Under `nawat submit` the executor puts these in the environment; in a kernel
# the open run answers instead. Code written against them moves between a
# notebook cell and a submitted script without an edit.


def model_dir() -> str:
    """The staged model directory."""
    value = os.environ.get("NAWAT_MODEL_DIR")
    if value:
        return value
    return require_run().model_dir


def dataset_dir(index: int = 0) -> str:
    """A staged dataset directory; the first one by default."""
    if index == 0 and os.environ.get("NAWAT_DATASET_DIR"):
        return os.environ["NAWAT_DATASET_DIR"]
    listed = os.environ.get("NAWAT_DATASET_DIRS")
    if listed:
        return json.loads(listed)[index]
    return require_run().dataset_dirs[index]


def out_dir() -> Path:
    """Where artifacts are written to be published."""
    value = os.environ.get("NAWAT_OUT_DIR")
    return Path(value) if value else require_run().out_dir


def artifact_dir(name: str = "adapter") -> Path:
    """A directory that publishes as ``runs/<id>/<name>``."""
    value = os.environ.get("NAWAT_OUT_DIR")
    if value:
        path = Path(value) / name
        path.mkdir(parents=True, exist_ok=True)
        return path
    return require_run().artifact_dir(name)


def checkpoint_dir() -> Path:
    """Where to write trainer checkpoints so that a failed run is not a lost one.

    Durable and shared with the next run of the same script and inputs, which is
    what makes resuming possible at all. Answers outside the platform too — a
    script run by hand still gets a real directory — so the same line works
    under ``nawat submit``, in a notebook, and under plain ``python``.
    """
    value = os.environ.get("NAWAT_CHECKPOINT_DIR")
    if value:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path
    run = current_run()
    if run is not None:
        return run.checkpoint_dir
    from . import default_cache

    config = default_cache().config
    lineage = os.environ.get("NAWAT_CHECKPOINT_LINEAGE") or checkpoint_store.lineage_name(sys.argv[0] or "adhoc")
    path = checkpoint_store.lineage_dir(config, lineage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resume_from() -> str | None:
    """The checkpoint to continue from, or None to start at step 0.

        trainer.train(resume_from_checkpoint=nawat.resume_from())

    An open run answers before the environment does, so re-running a cell after
    a failure picks up the checkpoints written since the run was opened, rather
    than the state of the world when it started.
    """
    run = current_run()
    if run is not None:
        return run.resume_from
    value = os.environ.get("NAWAT_RESUME_FROM")
    if value:
        return value
    if "NAWAT_RESUME_FROM" in os.environ:
        return None  # the executor looked and found nothing; do not look again
    newest = checkpoint_store.latest(checkpoint_dir())
    return str(newest.path) if newest else None


def checkpoint_args(save_steps: int = 250, save_total_limit: int = 3, **overrides: Any) -> dict[str, Any]:
    """Trainer arguments that make a long run survivable.

        args = SFTConfig(**nawat.checkpoint_args(save_steps=250), ...)
        trainer.train(resume_from_checkpoint=nawat.resume_from())

    Saving every N *steps* rather than once an epoch is the whole difference:
    an epoch over a large corpus can be a day wide, and a crash lands between
    two of them. ``save_total_limit`` bounds what that costs in disk.
    """
    args: dict[str, Any] = {
        "output_dir": str(checkpoint_dir()),
        "save_strategy": "steps",
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
    }
    args.update(overrides)
    return args


def param(name: str, default: Any = None, cast: Callable[[str], Any] | None = None) -> Any:
    """A hyperparameter from ``--param``, falling back to the value written here."""
    raw = os.environ.get(param_env_name(name))
    if raw is not None:
        return _coerce(raw, default, cast)
    run = current_run()
    return run.param(name, default, cast) if run is not None else default


def run_id() -> str | None:
    """The id of the run this code is part of, if any."""
    value = os.environ.get("NAWAT_RUN_ID")
    if value:
        return value
    run = current_run()
    return run.id if run is not None else None


# -- reading the history from a kernel ----------------------------------------


def _store(cache: Cache | None) -> RunStore:
    from . import default_cache

    cache = cache or default_cache()
    return RunStore(cache.db, cache.config.state_dir / "runs")


def history(limit: int = 20, cache: Cache | None = None) -> list[RunRecord]:
    """Recent runs, newest first — the notebook equivalent of ``nawat runs``."""
    return _store(cache).list(limit=limit)


def run_record(identifier: str | None = None, cache: Cache | None = None) -> RunRecord:
    """One run's record: what was submitted, what happened, what came out."""
    return _store(cache).get(identifier or _named_or_open(identifier))


def trace(identifier: str | None = None, cache: Cache | None = None) -> dict[str, list[dict[str, Any]]]:
    """A run's metric series, grouped by name, ready to plot.

    ``matplotlib.pyplot.plot([p["step"] for p in loss], [p["value"] for p in loss])``
    — the series is read from the durable file, so it plots identically long
    after the weights themselves have been reclaimed.
    """
    store = _store(cache)
    return metrics.series(metrics.read_points(store.metrics_path(_named_or_open(identifier))))


def _named_or_open(identifier: str | None) -> str:
    resolved = identifier or run_id()
    if resolved is None:
        raise NotFound("No run was named and none is open in this kernel.", "Pass a run id.")
    return resolved


# -- helpers ------------------------------------------------------------------


def _coerce(raw: str, default: Any, cast: Callable[[str], Any] | None) -> Any:
    if cast is not None:
        return cast(raw)
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    for kind in (int, float):
        if isinstance(default, kind):
            try:
                return kind(raw)
            except ValueError as exc:
                raise InvalidKey(
                    f"Parameter {raw!r} is not a {kind.__name__}.",
                    f"Pass a {kind.__name__} value.",
                ) from exc
    return raw


__all__ = [
    "Run",
    "UNKNOWN_SCRIPT",
    "artifact_dir",
    "begin_run",
    "checkpoint_args",
    "checkpoint_dir",
    "current_run",
    "dataset_dir",
    "history",
    "resume_from",
    "model_dir",
    "notebook_path",
    "out_dir",
    "param",
    "require_run",
    "run_id",
    "run_record",
    "trace",
]
