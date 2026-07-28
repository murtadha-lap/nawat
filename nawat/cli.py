"""The cache CLI.

Usable standalone, with no API or UI running (NFR-3.5), and sharing one
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
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .cache import Cache, open_cache
from .config import UNSET, Config
from .errors import NawatError
from .keys import KINDS, Key
from .runs import RunSpec, RunState, RunStore
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


def _executor(cache: Cache):
    from .executor import Executor

    return Executor(cache, _run_store(cache))


def _parse_params(pairs: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise NawatError(f"{pair!r} is not a parameter.", "Write parameters as name=value.")
        name, value = pair.split("=", 1)
        params[name.strip()] = value.strip()
    return params


def cmd_submit(cache: Cache, args) -> int:
    executor = _executor(cache)
    spec = RunSpec(
        script=args.script,
        model=Key.parse(args.model) if args.model else None,
        datasets=tuple(Key.parse(k) for k in args.dataset),
        inputs=tuple(Key.parse(k) for k in args.input),
        params=_parse_params(args.param),
        notes=args.notes or "",
    )
    executor.validate(spec)
    record = executor.runs.create(spec, args.run_id)

    if args.queue:
        print(f"Queued run {record.id}. It starts when the control plane reaches it.")
        return 0

    _err(f"run {record.id}  {spec.script}")
    finished: list = []
    worker = threading.Thread(target=lambda: finished.append(executor.execute(record.id)), daemon=True)
    worker.start()
    for chunk in executor.runs.follow_log(record.id):
        sys.stdout.write(chunk)
        sys.stdout.flush()
    worker.join()

    final = executor.runs.get(record.id)
    print(f"\nrun {final.id} {final.state.value} in {human_age(final.duration or 0)}")
    for key in final.artifacts:
        print(f"  published  {key}")
    if final.error:
        _err(final.error)
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
            r.state.value,
            r.spec.script,
            str(r.spec.model or ""),
            human_age(now - r.submitted_at),
            human_age(r.duration) if r.duration else "",
            str(len(r.artifacts)),
        ]
        for r in records
    ]
    _table(["run", "state", "script", "model", "age", "took", "artifacts"], rows, right={4, 5, 6})
    return 0


def cmd_run(cache: Cache, args) -> int:
    record = _run_store(cache).get(args.run_id)
    if args.json:
        print(json.dumps(record.to_json(), indent=2))
        return 0
    print(f"run        {record.id}")
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
    if record.error:
        print(f"error      {record.error}")
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
    p.add_argument("--run-id", help="identify this run; generated if omitted")
    p.add_argument("--queue", action="store_true", help="enqueue for the control plane instead of running here")
    p.set_defaults(run=cmd_submit)

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

    p = sub.add_parser("eval", help="score a run's adapter against a held-out set (CER/WER)")
    p.add_argument("run_id")
    p.add_argument("--data", required=True, help="a dataset key or a local path holding the eval JSONL")
    p.add_argument("--file", help="file name inside the dataset (default: eval/test/val.jsonl)")
    p.add_argument("--base", metavar="KEY", help="base model to serve; defaults to the run's model")
    p.add_argument("--limit", type=int, help="score only the first N samples")
    p.add_argument("--label", help="name for this evaluation in the record")
    p.set_defaults(run=cmd_eval)

    p = sub.add_parser("api", help="run the control plane and web interface")
    p.set_defaults(run=cmd_api)

    p = sub.add_parser("lab", help="JupyterLab in the workspace, sharing the cache")
    p.add_argument("--port", type=int)
    p.set_defaults(run=cmd_lab)

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
