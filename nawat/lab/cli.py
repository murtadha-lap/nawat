"""Commands for the lab: presets, blocks, experiments, sweeps, object storage, the sandbox."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ..errors import NawatError, NotFound, Protected
from ..units import human_age, human_bytes
from . import experiment as exp_mod
from . import report as rep
from . import sandbox as sbx
from . import storage
from . import sweep as sweep_mod
from .blocks import KINDS, Library
from .experiment import SANDBOX_ENV, Experiment, find, list_experiments, plan_experiment
from .params import format_value
from .presets import Presets, apply_sets, apply_uses


def _main():
    from .. import cli

    return cli


def _err(message: str) -> None:
    _main()._err(message)


def _table(headers, rows, right=frozenset()):
    _main()._table(headers, rows, right=right)


def _pairs(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise NawatError(f"{item!r} is not a {what}.", f"Write it as name=value, e.g. {what == 'setting' and 'train.learning_rate=1e-4' or 'dataset=my-data'}.")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _confirm(prompt: str, yes: bool) -> bool:
    if yes:
        return True
    if not _interactive():
        raise Protected("This deletes data and needs confirmation.", "Run it at a terminal, or pass --yes.")
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _live_keys(cache) -> set[str]:
    keys = {str(r.key) for r in cache.leases.live()}
    for record in _main()._run_store(cache).list(limit=200):
        if record.state.active:
            keys.add(f"runs/{record.publish_prefix}")
    return keys


# -- presets --------------------------------------------------------------------


def cmd_preset_ls(cache, args) -> int:
    rows = [[p.name, p.backbone, p.source, (p.model or "").replace("models/", ""), p.description[:70]]
            for p in Presets(cache.config).list()]
    _table(["preset", "backbone", "from", "model", "description"], rows)
    return 0


def cmd_preset_show(cache, args) -> int:
    preset = Presets(cache.config).get(args.name)
    if args.toml:
        print(preset.path.read_text() if preset.path else preset.to_toml())
        return 0
    backbone, backbone_params, plans = exp_mod.resolve_plan(Library(cache.config), preset)
    print(f"preset    {preset.name} ({preset.source})" + (f" ← {preset.parent}" if preset.parent else ""))
    if preset.description:
        print(f"about     {preset.description}")
    print(f"backbone  {backbone.name} — {backbone.summary}")
    print(f"model     {preset.model or '— (give --model)'}")
    print(f"datasets  {', '.join(preset.datasets) or '— (give --dataset)'}")
    for plan in plans:
        print(f"{plan.slot:9} {plan.block.name} — {plan.block.summary}")
    print()
    rows = []
    for label, block, resolved in [("train", backbone, backbone_params)] + [(p.key, p.block, p.params) for p in plans]:
        for name, value in resolved.values.items():
            changed = resolved.sources.get(name) != "default"
            spec = block.params[name]
            rows.append([f"{label}.{name}", rep.truncate(format_value(value), 44) + (" ◀" if changed else ""),
                         rep.truncate(format_value(spec.default), 30), spec.help[:70]])
    _table(["parameter", "value", "default", "meaning"], rows)
    print(f"\nChange values with --set <parameter>=<value> on `nawat exp plan`, or save them: "
          f"nawat preset fork {preset.name} --as <new> --set …")
    return 0


def cmd_preset_fork(cache, args) -> int:
    preset = Presets(cache.config).fork(
        args.source, args.new_name, sets=_pairs(args.set, "setting"), uses=_pairs(args.use, "block choice"),
        model=args.model, datasets=args.dataset or None, description=args.description,
    )
    exp_mod.resolve_plan(Library(cache.config), preset)  # fail now, not at plan time
    print(f"Saved preset {preset.name} ← {args.source}: {preset.path}")
    return 0


# -- blocks ---------------------------------------------------------------------


def cmd_block_ls(cache, args) -> int:
    library = Library(cache.config)
    rows = []
    for block in library.list(args.kind):
        flags = ("locked" if library.is_locked(block) else "") if block.source == "workspace" else ""
        fits = ",".join(block.meta.get("backbones", ())) or "any"
        rows.append([block.ref, block.source, fits, flags, block.summary[:72]])
    _table(["block", "from", "backbones", "", "what it does"], rows)
    return 0


def cmd_block_show(cache, args) -> int:
    library = Library(cache.config)
    block = library.get(args.ref)
    print(f"{block.ref}  ({block.source}{', locked' if library.is_locked(block) else ''})")
    print(f"file      {block.path}")
    print(f"sha256    {block.sha256}")
    for key in ("backbones", "models", "model_type", "config_class", "uses_training_data"):
        if key in block.meta:
            print(f"{key:9} {block.meta[key]}")
    if block.doc.strip():
        print()
        print(block.doc.strip())
    if block.params:
        print()
        _table(["parameter", "default", "meaning"],
               [[n, format_value(s.default), s.help + (f"  [{' | '.join(map(str, s.choices))}]" if s.choices else "")]
                for n, s in block.params.items()])
    return 0


def cmd_block_fork(cache, args) -> int:
    block = Library(cache.config).fork(args.ref, args.new_name)
    print(f"Created {block.ref}: {block.path}")
    print("Edit it, then use it: nawat preset fork <preset> --as <new> --use " f"{block.kind}={block.name}")
    return 0


def cmd_block_new(cache, args) -> int:
    block = Library(cache.config).new(args.kind, args.name)
    print(f"Created {block.ref} from the {args.kind} template: {block.path}")
    return 0


def cmd_block_save_as(cache, args) -> int:
    restored, created = Library(cache.config).save_as(args.ref, args.new_name)
    print(f"Your edit is now {created.ref} ({created.path}); {restored.ref} is back to its original.")
    return 0


# -- experiments ------------------------------------------------------------------


def cmd_exp_plan(cache, args) -> int:
    sets, uses = _pairs(args.set, "setting"), _pairs(args.use, "block choice")
    if args.save_preset:
        # Saved first, so a preset name that is already taken fails before an experiment exists.
        presets = Presets(cache.config)
        saved = presets.get(args.preset)
        saved.name, saved.parent = args.save_preset, args.preset
        apply_uses(saved, uses)
        apply_sets(saved, sets)
        if args.model:
            saved.model = args.model
        if args.dataset:
            saved.datasets = list(args.dataset)
        exp_mod.resolve_plan(Library(cache.config), saved)  # a bad choice fails here, before anything is written
        presets.save_new(saved)
        print(f"saved preset {args.save_preset}")
    experiment = plan_experiment(
        cache.config, args.preset, name=args.name, group=args.group, sets=sets, uses=uses, model=args.model,
        datasets=args.dataset or None, notes=args.notes or "", store=cache.store,
    )
    _print_plan(experiment)
    print(f"\nPlanned {experiment.name} in {experiment.dir}")
    print(f"Approve it at a terminal on the host: nawat exp approve {experiment.name}")
    return 0


def _print_plan(experiment: Experiment, *, only_changed: bool = False) -> None:
    d = experiment.data
    print(f"experiment  {d['name']}")
    print(f"backbone    {d['backbone']['name']}@{d['backbone']['sha256'][:12]}")
    print(f"model       {d['inputs']['model']}")
    print(f"datasets    {', '.join(d['inputs']['datasets'])}")
    if d["inputs"]["inputs"]:
        print(f"inputs      {', '.join(d['inputs']['inputs'])}")
    for slot, entries in d["blocks"].items():
        print(f"{slot:11} {', '.join(e['name'] + '@' + e['sha256'][:8] for e in entries)}")
    print()
    rows = [[f"{slot}.{name}", rep.truncate(value, 44) + ("" if source == "default" else " ◀"), rep.truncate(default, 30), help_text[:60]]
            for slot, name, value, default, source, help_text in exp_mod.param_rows(experiment)
            if not only_changed or source != "default"]
    _table(["parameter", "value", "default", "meaning"], rows)
    print(f"\n◀ = set by the preset or --set.  Library defaults also in effect: {experiment.path('plan.md')}")
    print(f"Approval code: {experiment.code}")


def cmd_exp_approve(cache, args) -> int:
    config = cache.config
    if os.environ.get(SANDBOX_ENV):
        raise Protected("Approval is not possible inside the sandbox.", "Approve on the host.")
    if not _interactive():
        raise Protected("Approval needs a person at a terminal.",
                        "Run `nawat exp approve` yourself in an interactive shell; agents and scripts cannot approve.")
    if sweep_mod.is_sweep(config, args.name):
        sweep = sweep_mod.load(config, args.name)
        experiments = sweep.experiments(config)
        print(f"Sweep {sweep.id}: {len(experiments)} experiments from preset {sweep.data['preset']}")
        result = rep.compare(experiments, show_all=False, config=config)
        print(rep.render_compare(result).split("## Results")[0])
        _print_plan(experiments[0])
        print("\nEvery experiment above shares the parameters shown for the first, except the varied ones.")
        typed = input(f"Type the sweep's approval code ({sweep.code[:2]}…) to approve all {len(experiments)}: ")
        sweep_mod.approve(config, sweep, typed)
        print(f"Approved sweep {sweep.id}. Run it: nawat sweep run {sweep.id}")
        return 0
    experiment = find(config, args.name)
    _print_plan(experiment)
    typed = input("\nRead the table above. Type the approval code to approve: ")
    exp_mod.approve(experiment, typed, config)
    print(f"Approved {experiment.name}. Run it: nawat exp run {experiment.name}")
    return 0


def experiment_spec(experiment: Experiment, fresh: bool = False):
    from ..checkpoints import CheckpointPolicy
    from ..keys import Key
    from ..runs import RunSpec

    d = experiment.data
    flat = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in rep.flat_params(experiment).items()}
    return RunSpec(
        script=str(Path("lab") / "experiments" / experiment.name / "run.py"),
        model=Key.parse(d["inputs"]["model"]) if d["inputs"]["model"] else None,
        datasets=tuple(Key.parse(k) for k in d["inputs"]["datasets"]),
        inputs=tuple(Key.parse(k) for k in d["inputs"]["inputs"]),
        params=flat,
        notes=d.get("notes", "") or f"experiment {experiment.name}",
        checkpoints=CheckpointPolicy(resume=not fresh, publish=True),
    )


def run_experiment(cache, experiment: Experiment, *, queue: bool = False, fresh: bool = False, here: bool = False) -> int:
    experiment.require_approved(cache.config)
    runtime = (experiment.data["blocks"].get("runtime") or [{"name": "host"}])[0]["name"]
    if runtime == "sandbox" and not os.environ.get(SANDBOX_ENV) and not here:
        container = experiment.data["params"].get("runtime", {}).get("container", sbx.CONTAINER)
        if not sbx.running(container):
            raise NawatError(f"Experiment {experiment.name} runs in the sandbox, which is not running.", "Start it: nawat sandbox up")
        print(f"running {experiment.name} inside sandbox {container}")
        # The approval was verified above with the host's key. The sandbox has none, so it is told what was verified.
        return sbx.call(sbx.exec_args(container, ["nawat", "exp", "run", experiment.name] + (["--fresh"] if fresh else []),
                                      interactive=False, env={exp_mod.HOST_VERIFIED_ENV: experiment.plan_hash}))
    cli = _main()
    executor = cli._executor(cache)
    spec = experiment_spec(experiment, fresh)
    executor.validate(spec)
    record = executor.runs.create(spec, None, name=experiment.name)
    experiment.add_attempt(record.id)
    return cli._launch(executor, record, queue=queue)


def cmd_exp_run(cache, args) -> int:
    return run_experiment(cache, find(cache.config, args.name), queue=args.queue, fresh=args.fresh, here=args.here)


def _latest_run(experiment: Experiment) -> str:
    attempts = experiment.attempts()
    if not attempts:
        raise NotFound(f"Experiment {experiment.name} has not run yet.", f"Run it: nawat exp run {experiment.name}")
    return attempts[-1]


def _in_sandbox(cache, experiment: Experiment, command: list[str]) -> int | None:
    """Run a nawat command inside the sandbox when that is where this experiment's runs live; None otherwise."""
    if os.environ.get(SANDBOX_ENV):
        return None
    runtime = (experiment.data["blocks"].get("runtime") or [{"name": "host"}])[0]["name"]
    if runtime != "sandbox":
        return None
    container = experiment.data["params"].get("runtime", {}).get("container", sbx.CONTAINER)
    if not sbx.running(container):
        return None
    return sbx.call(sbx.exec_args(container, ["nawat", *command], interactive=False))


def _not_here(experiment: Experiment, run_id: str, what: str) -> NotFound:
    return NotFound(f"Run {run_id} of {experiment.name} did not happen on this machine, and {what} is not here.",
                    "If it ran in the sandbox, start the sandbox (nawat sandbox up) to reach it there.")


def cmd_exp_logs(cache, args) -> int:
    """The latest attempt's log: from this machine's run store, live from the sandbox, or the copy in attempts/."""
    experiment = find(cache.config, args.name)
    run_id = _latest_run(experiment)
    main = _main()
    if main._run_store(cache).find(run_id) is not None:
        return main.cmd_logs(cache, argparse.Namespace(run_id=run_id, follow=args.follow, tail=args.tail))
    delegated = _in_sandbox(cache, experiment, ["logs", run_id] + (["-f"] if args.follow else [])
                            + (["--tail", str(args.tail)] if args.tail else []))
    if delegated is not None:
        return delegated
    copy = experiment.path("attempts", run_id, "run.log")
    if not copy.is_file():
        raise _not_here(experiment, run_id, "its log")
    text = copy.read_text(errors="replace")
    sys.stdout.write("".join(text.splitlines(keepends=True)[-args.tail:]) if args.tail else text)
    return 0


def cmd_exp_metrics(cache, args) -> int:
    experiment = find(cache.config, args.name)
    run_id = _latest_run(experiment)
    main = _main()
    if main._run_store(cache).find(run_id) is not None:
        return main.cmd_metrics(cache, argparse.Namespace(run_id=run_id, follow=args.follow, json=args.json))
    delegated = _in_sandbox(cache, experiment, ["metrics", run_id] + (["-f"] if args.follow else [])
                            + (["--json"] if args.json else []))
    if delegated is not None:
        return delegated
    copy = experiment.path("attempts", run_id, "metrics.jsonl")
    if not copy.is_file():
        raise _not_here(experiment, run_id, "its metric series")
    return main.print_metrics(copy, run_id, as_json=args.json)


def cmd_exp_cancel(cache, args) -> int:
    experiment = find(cache.config, args.name)
    run_id = _latest_run(experiment)
    main = _main()
    if main._run_store(cache).find(run_id) is None:
        delegated = _in_sandbox(cache, experiment, ["cancel", run_id])
        if delegated is not None:
            return delegated
        raise _not_here(experiment, run_id, "a process to stop")
    return main.cmd_cancel(cache, argparse.Namespace(run_id=run_id))


def cmd_exp_ls(cache, args) -> int:
    experiments = [e for e in list_experiments(cache.config) if not args.group or e.data.get("group") == args.group]
    if args.json:
        print(json.dumps([{"name": e.name, "status": rep.status(e, cache.config), "preset": e.data["preset"]["name"],
                           "created_at": e.data.get("created_at"), "sweep": e.data.get("sweep")} for e in experiments], indent=2))
        return 0
    if not experiments:
        print("No experiments yet. Start with: nawat preset ls, then nawat exp plan <preset>")
        return 0
    now = time.time()
    rows = [[e.name, rep.status(e, cache.config), e.data["backbone"]["name"], (e.data["inputs"]["model"] or "").replace("models/", ""),
             human_age(now - e.data.get("created_at", now)), e.data.get("sweep") or ""] for e in experiments]
    _table(["experiment", "status", "backbone", "model", "age", "sweep"], rows, right={4})
    return 0


def cmd_exp_show(cache, args) -> int:
    print(find(cache.config, args.name).path("plan.md").read_text())
    return 0


def cmd_exp_report(cache, args) -> int:
    experiments = [find(cache.config, n) for n in args.names]
    if args.compare or len(experiments) > 1:
        result = rep.compare(experiments, show_all=args.all, force=args.force, config=cache.config)
        print(json.dumps(result, indent=2, default=str) if args.json else rep.render_compare(result, for_agent=True))
        return 0
    report = rep.build(experiments[0], cache.config)
    print(json.dumps(report, indent=2, default=str) if args.json else rep.render(report))
    return 0


def cmd_exp_compare(cache, args) -> int:
    experiments = [find(cache.config, n) for n in args.names]
    if len(experiments) < 2:
        raise NawatError("Compare needs at least two experiments.")
    result = rep.compare(experiments, show_all=args.all, force=args.force, config=cache.config)
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0
    for warning in result["warnings"]:
        _err(f"⚠ {warning}")
    names = [n.split("/", 1)[-1] for n in result["experiments"]]
    if result["differences"]:
        print("What differs")
        _table([""] + names, [[label] + ["—" if v is None else rep._fmt(v) for v in values] for label, values in result["differences"]])
    else:
        print("Nothing in the setup differs.")
    print("\nResults")
    _table([""] + names, [[label] + ["—" if v is None else rep._fmt(v) for v in values] for label, values in result["results"]])
    print(f"\n{len(result['shared'])} identical parameters hidden (--all shows them).")
    return 0


def cmd_exp_traces(cache, args) -> int:
    experiments = [find(cache.config, n) for n in args.names]
    if args.clear:
        return _clear_traces(cache, experiments, args.yes)
    kind = args.kind
    if len(experiments) == 1 and args.step is None and not args.prompt:
        experiment = experiments[0]
        if kind == "rollouts":
            steps = rep.rollout_steps(experiment)
            if not steps:
                print("No rollout traces. (Probe samples, if any: --kind samples)")
                return 0
            if args.json:
                print(json.dumps(steps, indent=2))
                return 0
            reward_names = [k for k in steps[0] if k not in ("step", "completions", "total_mean", "total_max")]
            _table(["step", "n", "total mean", "total max"] + reward_names,
                   [[str(s["step"]), str(s["completions"]), rep._fmt(s["total_mean"]), rep._fmt(s["total_max"])]
                    + [rep._fmt(s.get(r)) for r in reward_names] for s in steps[-args.limit:]], right=set(range(4 + len(reward_names))))
            print(f"\nOne step in full: nawat exp traces {experiment.name} --step {steps[-1]['step']}")
            return 0
    groups = rep.side_by_side(experiments, kind=kind, step=args.step, prompt_filter=args.prompt, limit=args.limit)
    if args.json:
        print(json.dumps(groups, indent=2, default=str, ensure_ascii=False))
        return 0
    if not groups:
        print("No matching traces" + (" common to all of them" if len(experiments) > 1 else "") + ".")
        return 0
    width = args.width
    for group in groups:
        print("=" * min(width, 100))
        print(f"PROMPT  {rep.truncate(group['prompt'], width)}")
        for name, records in group["by_experiment"].items():
            print(f"--- {name}")
            for r in records[: args.per_prompt]:
                if kind == "rollouts":
                    rewards = " ".join(f"{k}={rep._fmt(v)}" for k, v in (r.get("rewards") or {}).items())
                    print(f"  step {r['step']} total {rep._fmt(r.get('total'))} adv {rep._fmt(r.get('advantage'))}  {rewards}")
                    print(f"    {rep.truncate(r.get('completion'), width)}")
                else:
                    print(f"  step {r['step']} ({r.get('event')}): {rep.truncate(r.get('output'), width)}")
    return 0


def _clear_traces(cache, experiments, yes) -> int:
    for experiment in experiments:
        directory = experiment.path("traces")
        size = sum(f.stat().st_size for f in directory.glob("*")) if directory.is_dir() else 0
        if not _confirm(f"Delete the traces of {experiment.name} ({human_bytes(size)} here, and in object storage)?", yes):
            continue
        # Object storage first: if that is refused (a live run, the sandbox), the local copy is still here.
        count, freed = storage.delete(cache.store, f"runs/{experiment.name}/traces", live_keys=_live_keys(cache))
        if directory.is_dir():
            for f in directory.glob("*"):
                f.unlink()
        print(f"cleared traces of {experiment.name}: {human_bytes(size)} local, {count} objects / {human_bytes(freed)} remote")
    return 0


PARTS = ("adapter", "merged", "gguf", "checkpoint", "traces", "record", "experiment")


def cmd_exp_du(cache, args) -> int:
    experiment = find(cache.config, args.name)
    rows = []
    for folder, count, size in storage.usage(cache.store, f"runs/{experiment.name}", depth=4):
        rows.append([folder.rsplit("/", 1)[-1], str(count), human_bytes(size)])
    local = sum(f.stat().st_size for f in experiment.dir.rglob("*") if f.is_file())
    _table(["part (object storage)", "objects", "size"], rows, right={1, 2})
    print(f"\nexperiment folder on this machine: {human_bytes(local)}  ({experiment.dir})")
    return 0


def cmd_exp_rm(cache, args) -> int:
    experiment = find(cache.config, args.name)
    parts = PARTS if args.parts == ["all"] else args.parts
    unknown = set(parts) - set(PARTS)
    if unknown:
        raise NawatError(f"Unknown part(s): {', '.join(sorted(unknown))}.", f"Parts are: {', '.join(PARTS)}, or all.")
    where = [w for w, on in (("object storage", args.remote), ("this machine", args.local)) if on] or ["object storage", "this machine"]
    if not _confirm(f"Delete {', '.join(parts)} of {experiment.name} from {' and '.join(where)}?", args.yes):
        print("Nothing deleted.")
        return 0
    live = _live_keys(cache)
    if "object storage" in where:
        for part in parts:
            count, size = storage.delete(cache.store, f"runs/{experiment.name}/{part}", live_keys=live)
            if count:
                print(f"remote  runs/{experiment.name}/{part}: {count} objects, {human_bytes(size)}")
                try:
                    status = cache.get(f"runs/{experiment.name}/{part}")
                    if status is not None:
                        cache.mark_unreplicated(status.key)
                except NawatError:
                    pass
    if "this machine" in where:
        import shutil

        for part in [p for p in PARTS if p in parts]:  # the experiment folder goes last: the others read it
            if part == "traces":
                for f in experiment.path("traces").glob("*"):
                    f.unlink()
            elif part == "experiment":
                shutil.rmtree(experiment.dir, ignore_errors=True)
                print(f"local   {experiment.dir} removed")
            elif part in ("adapter", "merged", "gguf", "record"):
                key = f"runs/{experiment.name}/{part}"
                if cache.get(key) is not None:
                    freed = cache.evict(key, force=True)
                    print(f"local   {key}: {human_bytes(freed)}")
            elif part == "checkpoint":
                from .. import checkpoints

                lineage = _main()._run_store(cache)
                for run_id in experiment.attempts():
                    try:
                        record = lineage.get(run_id)
                    except NawatError:
                        continue
                    directory = checkpoints.lineage_dir(cache.config, record.spec.checkpoint_lineage)
                    if directory.exists():
                        shutil.rmtree(directory, ignore_errors=True)
                        print(f"local   checkpoints {directory.name} removed")
                        break
    return 0


def cmd_exp_pull(cache, args) -> int:
    name = args.name
    target = exp_mod.experiments_root(cache.config) / name
    count, size = storage.download(cache.store, f"runs/{name}/experiment", target)
    if not count:
        raise NotFound(f"Object storage has no runs/{name}/experiment.", "List experiments there with: nawat store ls runs")
    print(f"pulled {count} files, {human_bytes(size)} into {target}")
    if args.traces:
        c2, s2 = storage.download(cache.store, f"runs/{name}/traces", target / "traces")
        print(f"pulled {c2} trace files, {human_bytes(s2)}")
    return 0


def cmd_exp_publish(cache, args) -> int:
    from .finalize import publish, write_report

    experiment = find(cache.config, args.name)
    write_report(experiment)
    publish(cache, experiment)
    return 0


# -- sweeps -----------------------------------------------------------------------


def cmd_sweep_plan(cache, args) -> int:
    models = [m for item in (args.model or []) for m in sweep_mod.split_values(item)]
    sweep = sweep_mod.plan_sweep(cache.config, args.preset, vary=_pairs(args.vary, "setting"), models=models,
                                 sets=_pairs(args.set, "setting"), uses=_pairs(args.use, "block choice"),
                                 datasets=args.dataset or None, notes=args.notes or "", group=args.group, store=cache.store)
    experiments = sweep.experiments(cache.config)
    result = rep.compare(experiments, config=cache.config)
    names = [n.split("/", 1)[-1] for n in result["experiments"]]
    print(f"Sweep {sweep.id}: {len(experiments)} experiments")
    _table([""] + names, [[label] + [rep._fmt(v) for v in values] for label, values in result["differences"]])
    print(f"\nApprove all of them at a terminal on the host: nawat exp approve {sweep.id}")
    return 0


def cmd_sweep_ls(cache, args) -> int:
    rows = []
    for sweep in sweep_mod.list_sweeps(cache.config):
        statuses = [rep.status(e, cache.config) for e in sweep.experiments(cache.config)]
        summary = ", ".join(f"{statuses.count(s)} {s}" for s in sorted(set(statuses)))
        rows.append([sweep.id, sweep.data["preset"], str(len(statuses)), summary])
    _table(["sweep", "preset", "experiments", "status"], rows)
    return 0


def cmd_sweep_run(cache, args) -> int:
    sweep = sweep_mod.load(cache.config, args.id)
    code = 0
    for experiment in sweep.experiments(cache.config):
        state = rep.status(experiment, cache.config)
        if state == "succeeded" and not args.again:
            print(f"skip {experiment.name} (already succeeded)")
            continue
        print(f"\n### {experiment.name}  {experiment.data.get('sweep_values')}")
        result = run_experiment(cache, experiment, queue=args.queue)
        code = code or result
    return code


# -- object storage ---------------------------------------------------------------


def cmd_store_ls(cache, args) -> int:
    prefix = (args.prefix or "").strip("/")
    depth = args.depth or (prefix.count("/") + 2 if prefix else 1)
    rows = [[folder, str(count), human_bytes(size)] for folder, count, size in storage.usage(cache.store, prefix, depth)]
    if not rows:
        print("Nothing there." if prefix else "Object storage is empty.")
        return 0
    _table(["prefix", "objects", "size"], rows, right={1, 2})
    total = sum(int(r[1]) for r in rows)
    print(f"\n{total} objects. Deeper: nawat store ls {prefix or 'runs'} --depth {depth + 1}")
    return 0


def cmd_store_du(cache, args) -> int:
    prefix = (args.prefix or "").strip("/")
    count = size = 0
    for obj in storage.walk(cache.store, prefix):
        count += 1
        size += obj.size
    print(f"{prefix or '(bucket)'}: {count} objects, {human_bytes(size)}")
    return 0


def cmd_store_rm(cache, args) -> int:
    prefix = args.prefix.strip("/")
    objects = list(storage.walk(cache.store, prefix))
    if not objects:
        print(f"Nothing under {prefix}.")
        return 0
    size = sum(o.size for o in objects)
    if not _confirm(f"Permanently delete {prefix} from object storage ({len(objects)} objects, {human_bytes(size)})?", args.yes):
        print("Nothing deleted.")
        return 0
    count, freed = storage.delete(cache.store, prefix, live_keys=_live_keys(cache))
    print(f"Deleted {count} objects, {human_bytes(freed)}, under {prefix}.")
    for status in cache.list():
        if str(status.key) == prefix or str(status.key).startswith(prefix + "/"):
            cache.mark_unreplicated(status.key)
            _err(f"note: {status.key} is still cached here and is now its only copy.")
    return 0


def cmd_store_get(cache, args) -> int:
    count, size = storage.download(cache.store, args.prefix, Path(args.dest).expanduser())
    print(f"Downloaded {count} files, {human_bytes(size)}, into {args.dest}.")
    return 0


# -- sandbox ----------------------------------------------------------------------


def cmd_sandbox(cache, args) -> int:
    action = args.action
    config = cache.config
    if action == "agents":
        print(sbx.AGENTS_MD)
        return 0
    if action == "dockerfile":
        print(sbx.DOCKERFILE)
        print("# training-stack.sh\n" + sbx.TRAINING_STACK)
        return 0
    if action == "build":
        return sbx.build(args.image)
    if action == "up":
        if sbx.running(args.name):
            print(f"Sandbox {args.name} is already running.")
            return 0
        argv, env = sbx.run_args(config, image=args.image, name=args.name, network=args.network,
                                 gpus="" if args.gpus == "none" else args.gpus)
        written = sbx.write_agents_md(config)
        if written:
            print(f"wrote agent rules: {written}")
        code = sbx.call(argv, env)
        if code == 0:
            print(f"Sandbox {args.name} is up. Open OpenCode in it: nawat sandbox opencode")
        return code
    if action == "shell":
        return sbx.call(sbx.exec_args(args.name, ["bash", "-l"], interactive=True))
    if action == "opencode":
        return sbx.call(sbx.exec_args(args.name, ["opencode", *args.rest], interactive=True))
    if action == "exec":
        rest = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
        return sbx.call(sbx.exec_args(args.name, rest, interactive=_interactive()))
    if action == "status":
        print(f"sandbox {args.name}: {'running' if sbx.running(args.name) else 'not running'}")
        return 0
    if action == "down":
        code = sbx.call([sbx.docker(), "rm", "-f", args.name])
        if args.volumes:
            sbx.call([sbx.docker(), "volume", "rm", sbx.CACHE_VOLUME, sbx.HOME_VOLUME])
        return code
    raise NawatError(f"Unknown sandbox action {action!r}.")


# -- registration -----------------------------------------------------------------


def register(sub) -> None:
    def setting_args(p):
        p.add_argument("--set", action="append", default=[], metavar="SLOT.PARAM=VALUE",
                       help="change a parameter, e.g. train.learning_rate=1e-4 or model.r=64")
        p.add_argument("--use", action="append", default=[], metavar="SLOT=BLOCK",
                       help="swap a block, e.g. dataset=my-data or rewards=format-exact,answer-match")
        p.add_argument("--model", metavar="KEY", help="model key, e.g. models/unsloth/Qwen3-4B-Base")
        p.add_argument("--dataset", action="append", default=[], metavar="KEY", help="dataset key (repeat for several)")

    preset = sub.add_parser("preset", help="presets: a backbone, its blocks and parameter values").add_subparsers(dest="preset_cmd", required=True)
    p = preset.add_parser("ls", help="every preset")
    p.set_defaults(run=cmd_preset_ls)
    p = preset.add_parser("show", help="a preset's blocks and every parameter")
    p.add_argument("name")
    p.add_argument("--toml", action="store_true", help="print the file itself")
    p.set_defaults(run=cmd_preset_show)
    p = preset.add_parser("fork", help="save a changed copy under a new name (presets are never edited)")
    p.add_argument("source")
    p.add_argument("--as", dest="new_name", required=True)
    p.add_argument("--description")
    setting_args(p)
    p.set_defaults(run=cmd_preset_fork)

    block = sub.add_parser("block", help="blocks: backbones and the swappable parts").add_subparsers(dest="block_cmd", required=True)
    p = block.add_parser("ls", help="every block, or every block of a kind")
    p.add_argument("kind", nargs="?", choices=KINDS)
    p.set_defaults(run=cmd_block_ls)
    p = block.add_parser("show", help="what a block does and its parameters")
    p.add_argument("ref", help="<kind>/<name>, e.g. dataset/chat-messages")
    p.set_defaults(run=cmd_block_show)
    p = block.add_parser("fork", help="copy a block under a new name, to edit")
    p.add_argument("ref")
    p.add_argument("--as", dest="new_name", required=True)
    p.set_defaults(run=cmd_block_fork)
    p = block.add_parser("new", help="start a new block from the template for its kind")
    p.add_argument("kind", choices=[k for k in KINDS if k not in ("backbone", "runtime")])
    p.add_argument("name")
    p.set_defaults(run=cmd_block_new)
    p = block.add_parser("save-as", help="move an in-place edit of a used block to a new block, restoring the original")
    p.add_argument("ref")
    p.add_argument("new_name")
    p.set_defaults(run=cmd_block_save_as)

    exp = sub.add_parser("exp", help="experiments: plan, approve, run, report, compare").add_subparsers(dest="exp_cmd", required=True)
    p = exp.add_parser("plan", help="freeze an experiment from a preset and print every parameter")
    p.add_argument("preset")
    p.add_argument("--name", help="experiment name (filed under its group)")
    p.add_argument("--group", help="folder to file it under; the preset name by default")
    p.add_argument("--notes", help="why this experiment exists")
    p.add_argument("--save-preset", metavar="NAME", help="also save these choices as a new preset")
    setting_args(p)
    p.set_defaults(run=cmd_exp_plan)
    p = exp.add_parser("approve", help="read the plan and approve it (a person, at a terminal, on the host)")
    p.add_argument("name", help="experiment or sweep id")
    p.set_defaults(run=cmd_exp_approve)
    p = exp.add_parser("run", help="run an approved experiment")
    p.add_argument("name")
    p.add_argument("--queue", action="store_true", help="hand it to the control plane instead of running here")
    p.add_argument("--fresh", action="store_true", help="ignore its checkpoints and start from step 0")
    p.add_argument("--here", action="store_true", help="run on this machine even if its runtime is the sandbox")
    p.set_defaults(run=cmd_exp_run)
    p = exp.add_parser("logs", help="the log of an experiment's latest run")
    p.add_argument("name")
    p.add_argument("-f", "--follow", action="store_true", help="stream until it ends")
    p.add_argument("--tail", type=int, help="only the last N lines")
    p.set_defaults(run=cmd_exp_logs)
    p = exp.add_parser("metrics", help="the metric series of an experiment's latest run")
    p.add_argument("name")
    p.add_argument("-f", "--follow", action="store_true", help="stream points as they are written")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_exp_metrics)
    p = exp.add_parser("cancel", help="stop an experiment's running attempt")
    p.add_argument("name")
    p.set_defaults(run=cmd_exp_cancel)
    p = exp.add_parser("ls", help="every experiment")
    p.add_argument("--group")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_exp_ls)
    p = exp.add_parser("show", help="an experiment's plan (plan.md)")
    p.add_argument("name")
    p.set_defaults(run=cmd_exp_show)
    p = exp.add_parser("report", help="what an experiment did, as text for a person or an agent")
    p.add_argument("names", nargs="+")
    p.add_argument("--compare", action="store_true", help="one comparison text for several experiments")
    p.add_argument("--all", action="store_true", help="with --compare: include identical parameters")
    p.add_argument("--force", action="store_true", help="show eval numbers even when measured differently")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_exp_report)
    p = exp.add_parser("compare", help="only what differs between experiments, beside their results")
    p.add_argument("names", nargs="+")
    p.add_argument("--all", action="store_true", help="show identical parameters too")
    p.add_argument("--force", action="store_true", help="show eval numbers even when measured differently (marked ⚠)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(run=cmd_exp_compare)
    p = exp.add_parser("traces", help="what the model generated while training; several experiments side by side")
    p.add_argument("names", nargs="+")
    p.add_argument("--kind", choices=("rollouts", "samples"), default="rollouts")
    p.add_argument("--step", type=int, help="only this training step")
    p.add_argument("--prompt", help="only prompts containing this text")
    p.add_argument("--limit", type=int, default=10, help="prompts (or steps) to show")
    p.add_argument("--per-prompt", type=int, default=4, help="completions per prompt and experiment")
    p.add_argument("--width", type=int, default=160)
    p.add_argument("--json", action="store_true")
    p.add_argument("--clear", action="store_true", help="delete the traces, here and in object storage")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(run=cmd_exp_traces)
    p = exp.add_parser("du", help="how much each part of an experiment takes in object storage")
    p.add_argument("name")
    p.set_defaults(run=cmd_exp_du)
    p = exp.add_parser("rm", help="delete parts of an experiment, here and/or in object storage")
    p.add_argument("name")
    p.add_argument("--parts", nargs="+", default=["traces"], metavar="PART",
                   help=f"any of {', '.join(PARTS)}, or all (default: traces)")
    p.add_argument("--remote", action="store_true", help="only from object storage")
    p.add_argument("--local", action="store_true", help="only from this machine")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(run=cmd_exp_rm)
    p = exp.add_parser("pull", help="download an experiment folder from object storage (e.g. on another machine)")
    p.add_argument("name", help="<group>/<name>")
    p.add_argument("--traces", action="store_true", help="the traces too")
    p.set_defaults(run=cmd_exp_pull)
    p = exp.add_parser("publish", help="rewrite the report and upload the experiment folder again")
    p.add_argument("name")
    p.set_defaults(run=cmd_exp_publish)

    sweep = sub.add_parser("sweep", help="a grid of experiments from one preset, approved once").add_subparsers(dest="sweep_cmd", required=True)
    p = sweep.add_parser("plan", help="plan every combination")
    p.add_argument("preset")
    p.add_argument("--model", action="append", default=[], metavar="KEY[,KEY…]", help="models to try")
    p.add_argument("--vary", action="append", default=[], metavar="SLOT.PARAM=V1,V2", help="a parameter and its values")
    p.add_argument("--set", action="append", default=[], metavar="SLOT.PARAM=VALUE", help="a fixed change for all")
    p.add_argument("--use", action="append", default=[], metavar="SLOT=BLOCK")
    p.add_argument("--dataset", action="append", default=[], metavar="KEY")
    p.add_argument("--group")
    p.add_argument("--notes")
    p.set_defaults(run=cmd_sweep_plan)
    p = sweep.add_parser("ls", help="every sweep")
    p.set_defaults(run=cmd_sweep_ls)
    p = sweep.add_parser("run", help="run every experiment of an approved sweep, one after another")
    p.add_argument("id")
    p.add_argument("--queue", action="store_true")
    p.add_argument("--again", action="store_true", help="also re-run those that already succeeded")
    p.set_defaults(run=cmd_sweep_run)
    p = sweep.add_parser("approve", help="same as: nawat exp approve <sweep id>")
    p.add_argument("name")
    p.set_defaults(run=cmd_exp_approve)

    store = sub.add_parser("store", help="what object storage holds: list, size, download, delete").add_subparsers(dest="store_cmd", required=True)
    p = store.add_parser("ls", help="folders and their sizes")
    p.add_argument("prefix", nargs="?", default="")
    p.add_argument("--depth", type=int, help="path segments to group by")
    p.set_defaults(run=cmd_store_ls)
    p = store.add_parser("du", help="total size under a prefix")
    p.add_argument("prefix", nargs="?", default="")
    p.set_defaults(run=cmd_store_du)
    p = store.add_parser("rm", help="permanently delete everything under a prefix")
    p.add_argument("prefix")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(run=cmd_store_rm)
    p = store.add_parser("get", help="download everything under a prefix")
    p.add_argument("prefix")
    p.add_argument("dest")
    p.set_defaults(run=cmd_store_get)

    p = sub.add_parser("sandbox", help="the isolated OpenCode container: build, up, opencode, shell, down")
    p.add_argument("action", choices=("dockerfile", "build", "up", "opencode", "shell", "exec", "status", "down", "agents"))
    p.add_argument("rest", nargs=argparse.REMAINDER, help="for opencode / exec: arguments to pass")
    p.add_argument("--image", default=sbx.IMAGE)
    p.add_argument("--name", default=sbx.CONTAINER)
    p.add_argument("--network", default="auto", choices=("auto", "host", "bridge"),
                   help="auto: the host's network when object storage is on localhost")
    p.add_argument("--gpus", default="all", help="'all', a device list, or 'none'")
    p.add_argument("--volumes", action="store_true", help="with down: also delete the sandbox's cache and home")
    p.set_defaults(run=cmd_sandbox)
