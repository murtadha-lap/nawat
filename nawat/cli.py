"""The cache CLI.

Usable standalone, with no API running (NFR-3.5), and sharing one
implementation with everything else that touches the cache (FR-1.11).

Commands are named for what the researcher controls rather than for how the
system works (PRD 9.8): ``keep`` and ``release`` rather than pin and unpin,
``free`` rather than garbage-collect. The mechanical names are kept as hidden
aliases so muscle memory from the design docs still works.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

from . import checkpoints
from .cache import Cache, open_cache
from .checkpoints import CheckpointPolicy
from .config import UNSET, Config
from .errors import NawatError
from .keys import KINDS, Key
from .runs import RunSpec, RunState, RunStore, validate_run_name
from .units import bar, human_age, human_bytes, parse_size

PROGRAM = "nawat"


# -- output -------------------------------------------------------------------


def _err(message: str) -> None:
    sys.stdout.flush()  # keep the two streams in order when either is a pipe
    print(message, file=sys.stderr)


def _table(headers: list[str], rows: list[list[str]], right: set[int] = frozenset()) -> None:
    if not rows:
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    line = "  ".join(
        (h.upper().rjust(widths[i]) if i in right else h.upper().ljust(widths[i])) for i, h in enumerate(headers)
    )
    print(line.rstrip())
    for row in rows:
        line = "  ".join(
            (cell.rjust(widths[i]) if i in right else cell.ljust(widths[i])) for i, cell in enumerate(row)
        )
        print(line.rstrip())


def _progress_reporter():
    if not sys.stderr.isatty():
        return None
    state = {"last": 0.0}

    def report(name: str, done: int, total: int) -> None:
        now = time.monotonic()
        finished = bool(total) and done >= total
        if not finished and now - state["last"] < 0.1:
            return
        state["last"] = now
        fraction = (done / total) if total else 0.0
        sys.stderr.write(f"\r{name}  {bar(fraction, 20)}  {human_bytes(done)} / {human_bytes(total)}   ")
        if finished:
            sys.stderr.write("\n")
        sys.stderr.flush()

    return report


# -- commands -----------------------------------------------------------------


def cmd_status(cache: Cache, args) -> int:
    cache.reconcile()
    status = cache.status()
    if args.json:
        print(json.dumps(status.__dict__, indent=2, default=str))
        return 0
    print(f"cache      {bar(status.fraction)}  {human_bytes(status.used)} / {human_bytes(status.ceiling)}"
          f"  ({status.fraction * 100:.0f}%)")
    print(f"disk       {human_bytes(status.disk_free)} free of {human_bytes(status.disk_total)}")
    print(
        f"held       {human_bytes(status.pinned_bytes)} kept"
        f" · {human_bytes(status.leased_bytes)} in use"
        f" · {human_bytes(status.unreplicated_bytes)} not yet in object storage"
    )
    print(f"artifacts  {status.artifacts}")
    if status.checkpoint_bytes:
        print(f"ckpts      {human_bytes(status.checkpoint_bytes)} kept for resuming — {PROGRAM} checkpoints")
    if status.headroom < 0:
        _err(f"\nOver the ceiling by {human_bytes(-status.headroom)}. Run: {PROGRAM} free")
    if status.unreplicated_bytes:
        _err(f"\n{human_bytes(status.unreplicated_bytes)} exists only on this disk. Publish it before it is lost.")
    return 0


def cmd_ls(cache: Cache, args) -> int:
    cache.reconcile()
    artifacts = cache.list()
    if args.kind:
        artifacts = [a for a in artifacts if a.key.kind == args.kind]
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "key": str(a.key),
                        "bytes": a.bytes,
                        "files": a.files,
                        "last_used": a.last_used,
                        "pinned": a.pinned,
                        "leased": a.leased,
                        "replicated": a.replicated,
                        "holders": [r.describe() for r in a.leases],
                    }
                    for a in artifacts
                ],
                indent=2,
            )
        )
        return 0
    if not artifacts:
        print("Nothing cached locally.")
        print(f"Stage something with: {PROGRAM} resolve models/<repo>")
        return 0
    now = time.time()
    rows = [
        [
            str(a.key),
            human_bytes(a.bytes),
            str(a.files),
            human_age(now - a.last_used),
            a.flags(),
            ", ".join(r.describe() for r in a.leases),
        ]
        for a in artifacts
    ]
    _table(["key", "size", "files", "last use", "flags", "used by"], rows, right={1, 2, 3})
    print("\nflags: K kept on disk · L in use · R in object storage")
    return 0


def cmd_resolve(cache: Cache, args) -> int:
    path = cache.resolve(
        args.key,
        lease=not args.no_lease,
        holder=args.holder or f"{PROGRAM} resolve",
        allow_hub=not args.no_hub,
        progress=_progress_reporter(),
    )
    print(path)
    if not args.no_lease:
        _err(f"Held for this process only; it is released as soon as {PROGRAM} exits.")
    return 0


def cmd_path(cache: Cache, args) -> int:
    key = Key.parse(args.key)
    status = cache.get(key)
    if status is None:
        _err(f"{key} is not on local disk. Stage it with: {PROGRAM} resolve {key}")
        return 3
    print(cache.local_path(key))
    return 0


def cmd_keep(cache: Cache, args) -> int:
    status = cache.pin(args.key)
    print(f"Keeping {status.key} on disk ({human_bytes(status.bytes)}).")
    return 0


def cmd_release(cache: Cache, args) -> int:
    status = cache.unpin(args.key)
    print(f"{status.key} is no longer kept on disk; it may be freed when space is needed.")
    return 0


def cmd_verify(cache: Cache, args) -> int:
    result = cache.verify(args.key)
    if result.ok:
        print(f"{result.key} matches object storage: {result.local_files} files, {human_bytes(result.local_bytes)}.")
        return 0
    _err(f"{result.key} does not match object storage: {result.reason()}.")
    for rel in list(result.missing)[:10]:
        _err(f"  absent remotely  {rel}")
    for rel in list(result.size_mismatch)[:10]:
        _err(f"  size differs     {rel}")
    _err(f"The local copy has been kept. Republish it with: {PROGRAM} publish {cache.local_path(args.key)} {args.key}")
    return 5


def cmd_free(cache: Cache, args) -> int:
    need = parse_size(args.need) if args.need else 0
    result = cache.collect(need, dry_run=args.dry_run)
    verb = "Would free" if args.dry_run else "Freed"
    if result.target <= 0:
        print("Nothing to free — the cache is already under the ceiling.")
        return 0
    print(f"{verb} {human_bytes(result.freed)} of {human_bytes(result.target)} needed.")
    for key in result.evicted:
        print(f"  removed  {key}")
    for key, reason in result.skipped:
        print(f"  kept     {key} — {reason}")
    if not result.satisfied:
        _err(f"\nStill {human_bytes(result.target - result.freed)} short. Release something above, or raise "
             f"NAWAT_CACHE_CEILING (currently {human_bytes(cache.config.cache_ceiling)}).")
        return 6
    return 0


def cmd_rm(cache: Cache, args) -> int:
    if args.force:
        _err("--force skips the object-storage check. If this artifact is not replicated, it is gone.")
    freed = cache.evict(args.key, force=args.force)
    print(f"Removed {args.key} from local disk, freeing {human_bytes(freed)}.")
    return 0


def cmd_publish(cache: Cache, args) -> int:
    result = cache.publish(
        args.directory,
        args.key,
        keep_local=args.keep,
        progress=_progress_reporter(),
    )
    print(f"Published {result.key}: {result.verification.local_files} files, {human_bytes(result.bytes)}, verified.")
    if result.local_removed:
        print(f"Local copy removed, freeing {human_bytes(result.bytes)}.")
    else:
        print(f"Local copy retained — {result.retained_reason}.")
    return 0


def cmd_add(cache: Cache, args) -> int:
    status = cache.add(args.directory, args.key, publish=not args.no_publish)
    where = "cached and published" if not args.no_publish else "cached only, not yet in object storage"
    print(f"{status.key}: {status.files} files, {human_bytes(status.bytes)} — {where}.")
    return 0


def cmd_registry(cache: Cache, args) -> int:
    keys = cache.registry(args.kind or None)
    cached = {str(a.key) for a in cache.list()}
    if args.json:
        print(json.dumps([{"key": str(k), "cached": str(k) in cached} for k in keys], indent=2))
        return 0
    if not keys:
        print("Object storage holds no artifacts yet.")
        return 0
    rows = [[str(key), "cached" if str(key) in cached else ""] for key in sorted(keys, key=str)]
    _table(["key", "local"], rows)
    return 0


def cmd_leases(cache: Cache, args) -> int:
    if args.reap:
        cleared = cache.leases.reap()
        print(f"Cleared {cleared} lease(s) whose holder is gone.")
        return 0
    records = cache.leases.live()
    if not records:
        print("Nothing is in use.")
        return 0
    now = time.time()
    rows = [[str(r.key), r.describe(), human_age(now - r.acquired_at)] for r in records]
    _table(["key", "held by", "for"], rows, right={2})
    return 0


def cmd_hold(cache: Cache, args) -> int:
    """Stage inputs, run a command against them, publish its outputs.

    This is the Phase 1 shape of a run: everything the executor will later do
    from the API, driven from the shell with no manual file management.
    """
    if not args.command:
        _err(f"Nothing to run. Put the command after --, e.g. {PROGRAM} hold --model models/x -- python train.py")
        return 2

    run_id = args.run_id or f"{time.strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:4]}"
    inputs: list[Key] = [Key.parse(k) for k in (*args.model, *args.dataset, *args.input)]
    holder = f"{PROGRAM} hold {run_id}"

    env = dict(os.environ)
    env["NAWAT_RUN_ID"] = run_id
    progress = _progress_reporter()

    with cache.holding(*inputs, holder=holder, progress=progress) as staged:
        env["NAWAT_INPUTS"] = json.dumps({key: str(path) for key, path in staged.items()})
        if args.model:
            env["NAWAT_MODEL_DIR"] = str(staged[str(Key.parse(args.model[0]))])
        if args.dataset:
            env["NAWAT_DATASET_DIR"] = str(staged[str(Key.parse(args.dataset[0]))])
            env["NAWAT_DATASET_DIRS"] = json.dumps(
                [str(staged[str(Key.parse(k))]) for k in args.dataset]
            )

        out_key = Key.parse(args.out) if args.out else None
        if out_key is not None:
            out_dir = cache.local_path(out_key)
        elif os.environ.get("NAWAT_OUT_DIR"):
            out_dir = Path(os.environ["NAWAT_OUT_DIR"])
        else:
            out_dir = None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            env["NAWAT_OUT_DIR"] = str(out_dir)

        _err(f"run {run_id}  staged {len(staged)} input(s)")
        for key, path in staged.items():
            _err(f"  {key} -> {path}")
        if out_dir is not None:
            _err(f"  outputs -> {out_dir}")
        _err(f"$ {' '.join(shlex.quote(part) for part in args.command)}")

        started = time.monotonic()
        completed = subprocess.run(args.command, env=env)
        elapsed = time.monotonic() - started
        _err(f"run {run_id} exited {completed.returncode} after {human_age(elapsed)}")

        if completed.returncode != 0:
            _err("Outputs were not published because the command failed. They are still on disk"
                 + (f" at {out_dir}." if out_dir else "."))
            return completed.returncode

    if out_key is not None and out_dir is not None and any(out_dir.iterdir()):
        result = cache.publish(out_dir, out_key, keep_local=args.keep_outputs, progress=progress)
        print(f"Published {result.key}: {human_bytes(result.bytes)}, verified.")
    elif out_key is not None:
        _err(f"The command wrote nothing to {out_dir}; nothing was published.")
    return 0


# -- runs ---------------------------------------------------------------------


def _run_store(cache: Cache) -> RunStore:
    return RunStore(cache.db, cache.config.state_dir / "runs")


def _executor(cache: Cache, ask_name=None):
    from .executor import Executor

    return Executor(cache, _run_store(cache), ask_name=ask_name)


#: How long the end-of-run name prompt waits before publishing under the run id.
#: A run that finishes at four in the morning must not sit there holding its
#: artifacts hostage until someone comes back to the terminal.
NAME_TIMEOUT = 300.0


def _ask_name(record) -> str | None:
    """Ask what to call the run, once it is over and before anything is uploaded.

    Only on a terminal, and only for as long as someone is plausibly watching.
    """
    if not (sys.stdin.isatty() and sys.stderr.isatty()):
        return None
    outcome = "finished" if record.exit_code == 0 else f"exited {record.exit_code}"
    _err(f"\nrun {record.id} {outcome}.")
    _err("Name it for object storage — artifacts publish under runs/<name>/.")
    # One deadline for the whole exchange, retries included: the artifacts of a
    # finished run are not held up for longer than someone might be away from
    # the terminal, however many times the name comes back unusable.
    deadline = time.monotonic() + NAME_TIMEOUT
    while True:
        _err(f"name [{record.id}]: ")
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([sys.stdin], [], [], remaining)[0]:
            _err(f"\nNo answer in {human_age(NAME_TIMEOUT)}; publishing under runs/{record.id}/.")
            return None
        answer = sys.stdin.readline().strip()
        if not answer:
            return None
        try:
            return validate_run_name(answer)
        except NawatError as exc:
            _err(f"{exc.cause} {exc.remedy or ''}".strip())


def _parse_params(pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise NawatError(f"{pair!r} is not a parameter.", "Write parameters as name=value.")
        name, value = pair.split("=", 1)
        params[name.strip()] = value.strip()
    return params


def cmd_submit(cache: Cache, args) -> int:
    executor = _executor(cache, ask_name=None if args.name or args.no_name else _ask_name)
    spec = RunSpec(
        script=args.script,
        model=Key.parse(args.model) if args.model else None,
        datasets=tuple(Key.parse(k) for k in args.dataset),
        inputs=tuple(Key.parse(k) for k in args.input),
        params=_parse_params(args.param),
        notes=args.notes or "",
        checkpoints=CheckpointPolicy(
            lineage=args.checkpoint_lineage or "",
            resume=not args.fresh,
            keep=args.keep_checkpoints,
            publish=not args.no_publish_checkpoints,
        ),
    )
    executor.validate(spec)
    record = executor.runs.create(spec, args.run_id, name=args.name)
    return _launch(executor, record, queue=args.queue)


def cmd_resume(cache: Cache, args) -> int:
    """Submit the same run again, so it continues from its last checkpoint."""
    executor = _executor(cache, ask_name=None if args.name else _ask_name)
    previous = executor.runs.get(args.run_id)
    lineage = previous.spec.checkpoint_lineage
    directory = checkpoints.lineage_dir(cache.config, lineage)
    newest = checkpoints.latest(directory)

    if newest is None:
        _err(f"Lineage {lineage} holds no checkpoint, so there is nothing to continue from.")
        _err(f"Inspect it with: {PROGRAM} checkpoints {lineage}")
        if not args.force:
            return 3
        _err("Continuing anyway (--force); this starts from step 0.")
    else:
        _err(f"run {previous.id} reached {newest.describe()}")

    # The lineage is pinned rather than re-derived, so the new run continues
    # this one even if it was submitted with an explicit lineage of its own.
    spec = replace(
        previous.spec,
        params={**previous.spec.params, **_parse_params(args.param)},
        notes=args.notes if args.notes is not None else f"resumes {previous.id}",
        checkpoints=replace(previous.spec.checkpoints, lineage=lineage, resume=True),
    )
    executor.validate(spec)
    # The name carries over by default: a resumed run is the same experiment,
    # so its artifacts belong in the folder the first attempt was filed under.
    record = executor.runs.create(spec, args.run_id_new, name=args.name or previous.name)
    _err(f"resuming as run {record.id}")
    return _launch(executor, record, queue=args.queue)


def _launch(executor, record, *, queue: bool) -> int:
    """Run it here and stream the log, or hand it to the control plane."""
    if queue:
        print(f"Queued run {record.id}. It starts when the control plane reaches it.")
        return 0

    _err(f"run {record.id}  {record.spec.script}")
    finished: list = []
    worker = threading.Thread(target=lambda: finished.append(executor.execute(record.id)), daemon=True)
    worker.start()
    for chunk in executor.runs.follow_log(record.id):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    worker.join()

    final = executor.runs.get(record.id)
    named = f" ({final.name})" if final.name else ""
    print(f"\nrun {final.id}{named} {final.state.value} in {human_age(final.duration or 0)}")
    for key in final.artifacts:
        print(f"  published  {key}")
    if final.checkpoint:
        print(f"  checkpoint {final.checkpoint}")
    if final.error:
        _err(final.error)
    if final.state is not RunState.SUCCEEDED and final.checkpoint:
        _err(f"Nothing is lost up to that checkpoint. Continue from it with: {PROGRAM} resume {final.id}")
    return 0 if final.state is RunState.SUCCEEDED else 1


def cmd_runs(cache: Cache, args) -> int:
    runs = _run_store(cache)
    records = runs.list(state=RunState(args.state) if args.state else None, limit=args.limit)
    if args.json:
        print(json.dumps([r.to_json() for r in records], indent=2))
        return 0
    if not records:
        print("No runs yet.")
        print(f"Submit one with: {PROGRAM} submit train.py --model models/<repo> --dataset datasets/<name>")
        return 0
    now = time.time()
    rows = [
        [
            r.id,
            r.name or "",
            r.state.value,
            r.spec.script,
            human_age(now - r.submitted_at),
            human_age(r.duration) if r.duration else "",
            str(len(r.artifacts)),
        ]
        for r in records
    ]
    _table(["run", "name", "state", "script", "age", "took", "artifacts"], rows, right={4, 5, 6})
    return 0


def cmd_run(cache: Cache, args) -> int:
    record = _run_store(cache).get(args.run_id)
    if args.json:
        print(json.dumps(record.to_json(), indent=2))
        return 0
    print(f"run        {record.id}")
    if record.name:
        print(f"name       {record.name}")
    print(f"state      {record.state.value}" + (f" (exit {record.exit_code})" if record.exit_code is not None else ""))
    print(f"script     {record.spec.script}")
    if record.spec.model:
        print(f"model      {record.spec.model}")
    for key in record.spec.datasets:
        print(f"dataset    {key}")
    for name, value in record.spec.params.items():
        print(f"param      {name}={value}")
    if record.duration:
        print(f"took       {human_age(record.duration)}")
    for key in record.artifacts:
        print(f"artifact   {key}")
    print(f"lineage    {record.spec.checkpoint_lineage}")
    if record.checkpoint:
        print(f"checkpoint {record.checkpoint}")
    if record.error:
        print(f"error      {record.error}")
    if record.checkpoint and record.state is not RunState.SUCCEEDED:
        print(f"\nContinue from that checkpoint with: {PROGRAM} resume {record.id}")
    return 0


def cmd_checkpoints(cache: Cache, args) -> int:
    """What can be resumed, and what it costs in disk."""
    root = cache.config.checkpoint_root
    runs = _run_store(cache)
    busy = {r.spec.checkpoint_lineage: r.id for r in runs.active()}
    found = checkpoints.lineages(root)
    if args.lineage:
        found = [l for l in found if l.name == args.lineage]
        if not found:
            _err(f"No checkpoint lineage {args.lineage} under {root}.")
            return 3

    if args.rm or args.prune:
        return _remove_checkpoints(found, busy, keep=args.keep, remove_all=args.rm, named=bool(args.lineage))

    if args.json:
        print(json.dumps([l.to_json() for l in found], indent=2))
        return 0
    if not found:
        print(f"No checkpoints under {root}.")
        print("They appear once a run writes one; a failed run's are kept so it can be resumed.")
        return 0

    if args.lineage:
        lineage = found[0]
        print(f"lineage    {lineage.name}")
        print(f"path       {lineage.path}")
        if lineage.run_id:
            print(f"last run   {lineage.run_id}")
        for checkpoint in lineage.checkpoints:
            print(f"  {checkpoint.describe()}")
        if lineage.latest:
            print(f"\nA run of this lineage continues from {lineage.latest.name}.")
        return 0

    now = time.time()
    rows = [
        [
            l.name,
            str(l.latest.step) if l.latest else "-",
            str(len(l.checkpoints)),
            human_bytes(l.bytes),
            human_age(now - (l.updated_at or now)),
            l.run_id or "",
        ]
        for l in found
    ]
    _table(["lineage", "step", "saved", "size", "age", "last run"], rows, right={1, 2, 3, 4})
    total = sum(l.bytes for l in found)
    print(f"\n{human_bytes(total)} kept so failed runs can be resumed. Free it with: {PROGRAM} checkpoints --prune")
    return 0


def _remove_checkpoints(found, busy: dict[str, str], *, keep: int, remove_all: bool, named: bool) -> int:
    """Delete checkpoints, refusing any lineage a live run is still writing to.

    Pruning always leaves one resumable checkpoint behind; removing a lineage
    outright leaves nothing, so it has to be named.
    """
    freed = 0
    for lineage in found:
        if lineage.name in busy:
            _err(f"  kept     {lineage.name} — run {busy[lineage.name]} is still writing to it")
            continue
        if remove_all:
            if not named:
                _err(f"  kept     {lineage.name} — name a lineage to remove it outright")
                continue
            freed += checkpoints.discard(lineage.path)
            print(f"  removed  {lineage.name}")
            continue
        removed = checkpoints.prune(lineage.path, keep=max(1, keep))
        freed += sum(c.bytes for c in removed)
        survivor = checkpoints.latest(lineage.path)
        print(
            f"  pruned   {lineage.name} — removed {len(removed)},"
            f" kept {survivor.name if survivor else 'nothing'}"
        )
    print(f"Freed {human_bytes(freed)}.")
    return 0


def cmd_logs(cache: Cache, args) -> int:
    runs = _run_store(cache)
    runs.get(args.run_id)
    if args.follow:
        for chunk in runs.follow_log(args.run_id):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        return 0
    sys.stdout.write(runs.read_log(args.run_id, tail=args.tail))
    return 0


def cmd_cancel(cache: Cache, args) -> int:
    record = _executor(cache).cancel(args.run_id)
    print(f"run {record.id} {record.state.value}.")
    return 0


def cmd_metrics(cache: Cache, args) -> int:
    from . import metrics
    from .units import spark

    runs = _run_store(cache)
    runs.get(args.run_id)
    path = runs.metrics_path(args.run_id)

    if args.follow:
        def running() -> bool:
            record = runs.find(args.run_id)
            return record is not None and not record.state.terminal

        for point in metrics.follow(path, running):
            print(json.dumps(point))
        return 0

    points = metrics.read_points(path)
    if args.json:
        print(json.dumps({"series": metrics.series(points), "events": metrics.events(points)}, indent=2))
        return 0
    if not points:
        print(f"Run {args.run_id} recorded no metrics.")
        print("Log them from the script with nawat.metrics.log(step=..., loss=...),")
        print("or pass callbacks=[nawat.metrics.trainer_callback()] to the trainer.")
        return 0

    print(f"run {args.run_id} · {len(points)} points\n")
    all_series = metrics.series(points)
    name_width = max(len(name) for name in all_series)
    for name in sorted(all_series):
        entries = all_series[name]
        values = [entry["value"] for entry in entries]
        lowest = min(entries, key=lambda entry: entry["value"])
        line = spark(values)
        print(
            f"{name.ljust(name_width)}  {line}  {values[0]:.4g} → {values[-1]:.4g}"
            f"   min {lowest['value']:.4g} @ {lowest['step']}"
        )
    marks = metrics.events(points)
    if marks:
        print()
        for mark in marks:
            print(f"event  step {mark['step']}  {mark['event']}")
    return 0


def cmd_scripts(cache: Cache, args) -> int:
    root = cache.config.workspace_root
    if not root.is_dir():
        print(f"The workspace {root} does not exist yet.")
        print("Create it and put a training script there, or set NAWAT_WORKSPACE.")
        return 0
    found = sorted(
        path
        for path in root.rglob("*")
        if path.suffix in (".py", ".ipynb")
        and not any(part.startswith((".", "__")) for part in path.relative_to(root).parts)
    )
    if not found:
        print(f"No scripts or notebooks under {root}.")
        return 0
    rows = [[str(p.relative_to(root)), "notebook" if p.suffix == ".ipynb" else "script"] for p in found]
    _table(["script", "kind"], rows)
    return 0


# -- serving ------------------------------------------------------------------


def _sessions(cache: Cache):
    from .sessions import SessionManager

    return SessionManager(cache, runs=_run_store(cache))


def cmd_serve(cache: Cache, args) -> int:
    manager = _sessions(cache)
    _err(f"Starting an inference server for {args.key} — this takes a minute for a large base.")
    session = manager.start(args.key, idle_timeout=args.idle_timeout, wait=not args.no_wait)
    print(f"Serving {session.model} at {session.base_url}/v1 ({session.state.value}).")
    print(f"Idle timeout {human_age(session.idle_timeout)}; the GPU is released after that.")
    return 0


def cmd_session(cache: Cache, args) -> int:
    manager = _sessions(cache)
    if args.stop:
        stopped = manager.stop()
        print(f"Stopped {stopped.model}; GPU released." if stopped else "Nothing was running.")
        return 0
    if args.log:
        sys.stdout.write(manager.log(tail=args.tail))
        return 0
    session = manager.current()
    if session is None:
        print("No inference server is running.")
        print(f"Start one with: {PROGRAM} serve models/<repo>")
        return 0
    if args.json:
        print(json.dumps(session.to_json(), indent=2))
        return 0
    print(f"model      {session.model}")
    print(f"state      {session.state.value}")
    print(f"url        {session.base_url}/v1")
    print(f"pid        {session.pid}")
    print(f"idle for   {human_age(session.idle_for)} of {human_age(session.idle_timeout)}")
    for name, key in session.adapters.items():
        print(f"adapter    {name} -> {key}")
    return 0


def cmd_adapter(cache: Cache, args) -> int:
    manager = _sessions(cache)
    if args.unload:
        session = manager.unload_adapter(args.unload)
        print(f"Unloaded {args.unload}.")
        return 0
    if not args.key:
        _err(f"Name an adapter to load, or --unload one. Try: {PROGRAM} adapter runs/<id>/adapter")
        return 2
    session = manager.load_adapter(args.key, args.name)
    loaded = args.name or Key.parse(args.key).path.replace("/", "-")
    print(f"Loaded {args.key} as {loaded}; no merge, no restart.")
    print(f"Use it by setting model={loaded} against {session.base_url}/v1.")
    return 0


def cmd_estimate(cache: Cache, args) -> int:
    from .api import gpu_info
    from .estimator import estimate, params_from_bytes, parse_params

    if args.params:
        params = parse_params(args.params)
    elif args.model:
        status = cache.get(args.model)
        if status is None:
            _err(f"{args.model} is not cached, so its size cannot be read. Stage it, or pass --params 7B.")
            return 3
        params = params_from_bytes(status.bytes, quantized="4bit" in args.model.lower())
    else:
        _err("Name the model with --model KEY, or its size with --params 7B.")
        return 2
    gpu = gpu_info()
    result = estimate(
        params=params, method=args.method, bits=args.bits, batch=args.batch,
        seq_len=args.seq, grad_checkpointing=not args.no_checkpointing,
        vram_available=gpu["memory_total"] if gpu else None,
    )
    if args.json:
        print(json.dumps(result.to_json(), indent=2))
        return 0
    breakdown = result.to_json()["breakdown"]
    for name, value in breakdown.items():
        print(f"{name:<12} {human_bytes(value):>10}")
    print(f"{'total':<12} {human_bytes(result.vram_total):>10}")
    print()
    print(result.verdict())
    return 0 if result.fits is not False else 1


def cmd_eval(cache: Cache, args) -> int:
    """Evaluate a run's adapter against a held-out set, into its record."""
    from .evaluate import Evaluator

    evaluator = Evaluator(cache, _run_store(cache), _sessions(cache))
    _err(f"Evaluating run {args.run_id} — the base model is served if it is not already.")

    def report(done: int, total: int) -> None:
        if sys.stderr.isatty():
            _err(f"\r  sample {done}/{total}   ") if done < total else _err(f"\r  sample {done}/{total}")

    result = evaluator.evaluate_run(
        args.run_id,
        args.data,
        base=args.base,
        file_name=args.file,
        limit=args.limit,
        label=args.label,
        progress=report,
    )
    print(f"run {args.run_id} · {result.samples} samples against {Path(args.data).name}")
    print(f"CER  {result.cer * 100:.2f}%")
    print(f"WER  {result.wer * 100:.2f}%")
    print(f"Written into the run record; compare with: {PROGRAM} metrics {args.run_id}")
    return 0


def cmd_lab(cache: Cache, args) -> int:
    """JupyterLab in the workspace, sharing the cache through the environment."""
    config = cache.config
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "jupyterlab", "--notebook-dir", str(config.workspace_root)]
    if args.port:
        command += ["--port", str(args.port)]
    try:
        completed = subprocess.run(command)
    except FileNotFoundError:
        _err("JupyterLab is not installed. Install it with: pip install jupyterlab")
        return 1
    return completed.returncode


def cmd_api(cache: Cache, args) -> int:
    try:
        from .api import serve as serve_api
    except ImportError as exc:
        _err(f"The control plane needs its extra: pip install -e '.[api]'  ({exc})")
        return 1
    config = cache.config
    _err(f"Nawāt control plane on http://{config.api_host}:{config.api_port}")
    if not config.api_token:
        _err("No NAWAT_API_TOKEN is set, so the API is unauthenticated. Bind it to the loopback only.")
    serve_api(config)
    return 0


def cmd_shard(cache: Cache, args) -> int:
    """Pack a small-file corpus into tar shards and publish it as a dataset."""
    import tempfile

    from . import shards

    source = Path(args.directory).expanduser()
    shard_size = parse_size(args.shard_size)

    def report(done: int, total: int) -> None:
        if sys.stderr.isatty() and (done % 500 == 0 or done == total):
            _err(f"\r  packed {done}/{total} files   ")

    with tempfile.TemporaryDirectory(dir=str(cache.config.staging_root)) as scratch:
        dest = Path(scratch) / "sharded"
        index = shards.pack(source, dest, shard_size=shard_size, progress=report)
        shards.verify(dest)
        print(f"{index.total_files} files → {len(index.shards)} shard(s), {human_bytes(index.total_bytes)} of content.")
        if args.no_publish:
            final = source.parent / (source.name + "-sharded")
            dest.rename(final)
            print(f"Written to {final}; publish later with: {PROGRAM} add {final} datasets/<name>")
            return 0
        status = cache.add(dest, args.key)
        print(f"Published as {status.key} ({human_bytes(status.bytes)}), verified in object storage.")
    print("Stream it in training with webdataset or datasets over the shard-*.tar files.")
    return 0


def cmd_config(cache: Cache, args) -> int:
    print(json.dumps(cache.config.redacted(), indent=2))
    return 0


def cmd_check(cache: Cache, args) -> int:
    """Bring-up check: can this host actually store, verify and reclaim?"""
    from .health import run_checks

    checks = run_checks(cache, create_bucket=args.create_bucket)
    width = max(len(check.name) for check in checks)
    source = cache.config.env_file or "the environment alone — no .env was found"
    print(f"{'ok  '}  {'configuration'.ljust(width)}  {source}")
    failed = [check for check in checks if not check.ok]
    for check in checks:
        print(f"{'ok  ' if check.ok else 'FAIL'}  {check.name.ljust(width)}  {check.detail}")
        if not check.ok and check.remedy:
            print(f"        {check.remedy}")
    if failed:
        _err(f"\n{len(failed)} of {len(checks)} checks failed. Nawāt will not run correctly until they pass.")
        return 4
    print("\nReady. Object storage is reachable and this host can publish, verify and reclaim.")
    return 0


# -- argument parsing ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Local cache over object storage. Object storage is truth; local disk is disposable.",
    )
    parser.add_argument("--cache-root", help="override NAWAT_CACHE_ROOT")
    parser.add_argument("--ceiling", help="override NAWAT_CACHE_CEILING, e.g. 80GB")
    parser.add_argument("--env-file", metavar="PATH", help="read configuration from this file instead of the nearest .env")
    parser.add_argument("--no-env-file", action="store_true", help="use the environment alone, ignoring any .env")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="cache occupancy against the ceiling")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_status)

    p = sub.add_parser("ls", help="what is on local disk")
    p.add_argument("--kind", choices=KINDS)
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_ls)

    p = sub.add_parser("resolve", help="make an artifact present locally and print its path")
    p.add_argument("key")
    p.add_argument("--no-lease", action="store_true", help="do not hold it for this process")
    p.add_argument("--no-hub", action="store_true", help="fail rather than fetching from the upstream hub")
    p.add_argument("--holder", help="label recorded against the lease")
    p.set_defaults(run=cmd_resolve)

    p = sub.add_parser("path", help="print the local path of an already-cached artifact")
    p.add_argument("key")
    p.set_defaults(run=cmd_path)

    for name, aliases, help_text, fn in (
        ("keep", ["pin"], "keep on disk until released", cmd_keep),
        ("release", ["unpin"], "stop keeping on disk", cmd_release),
    ):
        p = sub.add_parser(name, aliases=aliases, help=help_text)
        p.add_argument("key")
        p.set_defaults(run=fn)

    p = sub.add_parser("verify", help="compare the local copy against object storage")
    p.add_argument("key")
    p.set_defaults(run=cmd_verify)

    p = sub.add_parser("free", aliases=["collect"], help="reclaim local disk, least recently used first")
    p.add_argument("--need", help="free at least this much, e.g. 16GB")
    p.add_argument("--dry-run", action="store_true", help="report what would go, delete nothing")
    p.set_defaults(run=cmd_free)

    p = sub.add_parser("rm", help="remove one artifact from local disk")
    p.add_argument("key")
    p.add_argument("--force", action="store_true", help="skip the object-storage check (can lose data)")
    p.set_defaults(run=cmd_rm)

    p = sub.add_parser("publish", help="upload a directory, verify it, then reclaim the local copy")
    p.add_argument("directory")
    p.add_argument("key")
    p.add_argument("--keep", action="store_true", help="keep the local copy in the cache after publishing")
    p.set_defaults(run=cmd_publish)

    p = sub.add_parser("add", help="adopt a directory already on disk as a cached artifact")
    p.add_argument("directory")
    p.add_argument("key")
    p.add_argument("--no-publish", action="store_true", help="cache it without uploading (it will not be evictable)")
    p.set_defaults(run=cmd_add)

    p = sub.add_parser("registry", help="what object storage holds, cached or not")
    p.add_argument("kind", nargs="*", choices=KINDS, help="limit to these kinds")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_registry)

    p = sub.add_parser("leases", help="what is in use, and by whom")
    p.add_argument("--reap", action="store_true", help="clear leases whose holder is gone")
    p.set_defaults(run=cmd_leases)

    p = sub.add_parser("hold", help="stage inputs, run a command against them, publish its outputs")
    p.add_argument("--model", action="append", default=[], metavar="KEY")
    p.add_argument("--dataset", action="append", default=[], metavar="KEY")
    p.add_argument("--input", action="append", default=[], metavar="KEY", help="any other artifact to stage")
    p.add_argument("--out", metavar="KEY", help="publish the command's output directory under this key")
    p.add_argument("--run-id", help="identify this run; generated if omitted")
    p.add_argument("--keep-outputs", action="store_true", help="keep outputs on disk after publishing")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- followed by the command to run")
    p.set_defaults(run=cmd_hold)

    p = sub.add_parser("submit", help="run a training script: stage, execute, publish")
    p.add_argument("script", help="a .py or .ipynb inside the workspace")
    p.add_argument("--model", metavar="KEY")
    p.add_argument("--dataset", action="append", default=[], metavar="KEY")
    p.add_argument("--input", action="append", default=[], metavar="KEY")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--notes", help="why this run exists")
    p.add_argument("--name", metavar="NAME",
                   help="what to call this run in object storage; asked for at the end if omitted")
    p.add_argument("--no-name", action="store_true", help="do not ask for a name; publish under the run id")
    p.add_argument("--run-id", help="identify this run; generated if omitted")
    p.add_argument("--queue", action="store_true", help="enqueue for the control plane instead of running here")
    p.add_argument("--checkpoint-lineage", metavar="NAME",
                   help="checkpoint directory to train into; derived from the script and inputs if omitted")
    p.add_argument("--fresh", action="store_true",
                   help="ignore and delete this lineage's checkpoints, starting from step 0")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="keep the checkpoints on local disk even after the run succeeds")
    p.add_argument("--no-publish-checkpoints", action="store_true",
                   help="do not replicate the last checkpoint to object storage")
    p.set_defaults(run=cmd_submit)

    p = sub.add_parser("resume", help="run a failed run again, continuing from its last checkpoint")
    p.add_argument("run_id", help="the run to continue")
    p.add_argument("--param", action="append", default=[], metavar="NAME=VALUE", help="change a parameter")
    p.add_argument("--notes")
    p.add_argument("--name", metavar="NAME", help="publish under this name; the previous run's by default")
    p.add_argument("--run-id", dest="run_id_new", help="identify the new run; generated if omitted")
    p.add_argument("--queue", action="store_true", help="enqueue for the control plane instead of running here")
    p.add_argument("--force", action="store_true", help="submit even if there is no checkpoint to continue from")
    p.set_defaults(run=cmd_resume)

    p = sub.add_parser("checkpoints", help="what can be resumed, and what it costs in disk")
    p.add_argument("lineage", nargs="?", help="one lineage, in detail")
    p.add_argument("--prune", action="store_true", help="keep only the newest checkpoints of each lineage")
    p.add_argument("--keep", type=int, default=1, metavar="N", help="how many to keep when pruning (default 1)")
    p.add_argument("--rm", action="store_true", help="remove the named lineage outright")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_checkpoints)

    p = sub.add_parser("runs", help="run history")
    p.add_argument("--state", choices=[s.value for s in RunState])
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_runs)

    p = sub.add_parser("run", help="everything recorded about one run")
    p.add_argument("run_id")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_run)

    p = sub.add_parser("logs", help="a run's log")
    p.add_argument("run_id")
    p.add_argument("-f", "--follow", action="store_true", help="stream until the run ends")
    p.add_argument("--tail", type=int, help="only the last N lines")
    p.set_defaults(run=cmd_logs)

    p = sub.add_parser("cancel", help="stop a run and release what it holds")
    p.add_argument("run_id")
    p.set_defaults(run=cmd_cancel)

    p = sub.add_parser("metrics", help="a run's metric series — the trace, in the terminal")
    p.add_argument("run_id")
    p.add_argument("-f", "--follow", action="store_true", help="stream points as they are written")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_metrics)

    p = sub.add_parser("scripts", help="training scripts and notebooks in the workspace")
    p.set_defaults(run=cmd_scripts)

    p = sub.add_parser("serve", help="start an inference server for a model")
    p.add_argument("key")
    p.add_argument("--idle-timeout", type=float, help="seconds of inactivity before the GPU is released")
    p.add_argument("--no-wait", action="store_true", help="return before the server is ready")
    p.set_defaults(run=cmd_serve)

    p = sub.add_parser("session", help="the running inference server")
    p.add_argument("--stop", action="store_true", help="tear it down and release the GPU")
    p.add_argument("--log", action="store_true")
    p.add_argument("--tail", type=int, default=200)
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_session)

    p = sub.add_parser("adapter", help="load a trained adapter onto the running base, without merging")
    p.add_argument("key", nargs="?")
    p.add_argument("--name", help="what to call it when requesting completions")
    p.add_argument("--unload", metavar="NAME")
    p.set_defaults(run=cmd_adapter)

    p = sub.add_parser("estimate", help="will it fit? VRAM estimate for a training configuration")
    p.add_argument("--model", metavar="KEY")
    p.add_argument("--params", help="model size when it is not cached, e.g. 7B")
    p.add_argument("--method", choices=["lora", "full"], default="lora")
    p.add_argument("--bits", type=int, default=16, choices=[4, 8, 16])
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--no-checkpointing", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_estimate)

    p = sub.add_parser("eval", help="score a run's adapter against a held-out set (CER/WER)")
    p.add_argument("run_id")
    p.add_argument("--data", required=True, help="a dataset key or a local path holding the eval JSONL")
    p.add_argument("--file", help="file name inside the dataset (default: eval/test/val.jsonl)")
    p.add_argument("--base", metavar="KEY", help="base model to serve; defaults to the run's model")
    p.add_argument("--limit", type=int, help="score only the first N samples")
    p.add_argument("--label", help="name for this evaluation in the record")
    p.set_defaults(run=cmd_eval)

    p = sub.add_parser("api", help="run the control plane over HTTP")
    p.set_defaults(run=cmd_api)

    p = sub.add_parser("lab", help="JupyterLab in the workspace, sharing the cache")
    p.add_argument("--port", type=int)
    p.set_defaults(run=cmd_lab)

    p = sub.add_parser("shard", help="pack a small-file corpus into tar shards and publish it")
    p.add_argument("directory", help="the corpus to shard")
    p.add_argument("key", help="dataset key to publish under, e.g. datasets/ocr-arabic-v3-sharded")
    p.add_argument("--shard-size", default="512MB")
    p.add_argument("--no-publish", action="store_true", help="write the shards beside the source instead")
    p.set_defaults(run=cmd_shard)

    p = sub.add_parser("config", help="the configuration in force, with credentials redacted")
    p.set_defaults(run=cmd_config)

    p = sub.add_parser("check", help="verify this host can store, verify and reclaim")
    p.add_argument("--create-bucket", action="store_true", help="create the bucket if it is missing")
    p.set_defaults(run=cmd_check)

    return parser


def _config_from(args) -> Config:
    overrides = {}
    if args.cache_root:
        overrides["NAWAT_CACHE_ROOT"] = args.cache_root
    if args.ceiling:
        overrides["NAWAT_CACHE_CEILING"] = args.ceiling
    if args.no_env_file:
        env_file = None
    elif args.env_file:
        env_file = args.env_file
    else:
        env_file = UNSET
    return Config.from_env({**os.environ, **overrides}, env_file=env_file)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "run", None) is cmd_hold and args.command and args.command[0] == "--":
        args.command = args.command[1:]  # argparse.REMAINDER keeps the leading --
    try:
        cache = open_cache(_config_from(args))
        return args.run(cache, args)
    except NawatError as exc:
        _err(f"{PROGRAM}: {exc}")
        for line in getattr(exc, "held", []):
            _err(f"  {line}")
        return exc.exit_code
    except KeyboardInterrupt:
        _err("\nInterrupted. Nothing was deleted.")
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
