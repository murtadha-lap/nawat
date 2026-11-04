"""Reports — what an experiment did, and what differs between experiments, as text.

Written for two readers at once: a person at a terminal, and a local agent
drafting a paper or comparing against an article. So everything is plain
Markdown or JSON, every number says where it came from, and nothing that varies
between experiments is left out of a comparison.

**Fair comparison.** Measurements only compare when they were taken the same way.
Each eval block carries a fingerprint (its code, its test data, its metric, its
generation settings — and for held-out evals the training data split). ``compare``
puts eval numbers side by side only when the fingerprints match; otherwise it
says why it will not, unless told to show them anyway, marked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..metrics import read_points
from ..process import process_alive
from .experiment import Experiment
from .params import format_value
from .trace import read as read_traces

#: Training fields worth a line in a report when present.
KEY_FIELDS = ("loss", "eval_loss", "grad_norm", "learning_rate", "reward", "reward_std", "kl", "entropy",
              "completions/mean_length", "completions/clipped_ratio", "frac_reward_zero_std",
              "rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins", "eval_wer",
              "reward_mean", "reward_max")


# -- one experiment -----------------------------------------------------------


def attempt_records(experiment: Experiment, config=None) -> list[dict[str, Any]]:
    out = []
    live_config = None
    for run_id in experiment.attempts():
        path = experiment.path("attempts", run_id, "record.json")
        if path.is_file():
            try:
                out.append(json.loads(path.read_text()))
                continue
            except ValueError:
                pass
        # No record yet: the run is live, or it stopped before finishing. Ask the run store which.
        live_config = live_config or _config(config)
        out.append(_live_record(run_id, live_config) or {"id": run_id, "state": "unknown"})
    return out


def _live_record(run_id: str, config) -> dict[str, Any] | None:
    """The run store's record of an attempt that has not been finalised yet, if this host knows the run.

    ``record.json`` is only written once a run ends, but the run store keeps
    ``<state_dir>/runs/<id>/run.json`` current through every state change. A run
    that says it is running but whose trainer process is gone was stopped without
    the platform noticing, and is reported as interrupted.
    """
    if config is None:
        return None
    path = Path(config.state_dir) / "runs" / run_id / "run.json"  # RunStore's record_path
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("id") != run_id:
        return None
    pid = record.get("pid")
    if record.get("state") == "running" and record.get("exit_code") is None and pid and not process_alive(pid):
        record["state"] = "interrupted"
    return record


def latest_attempt(experiment: Experiment, config=None) -> dict[str, Any] | None:
    records = attempt_records(experiment, config)
    return records[-1] if records else None


def status(experiment: Experiment, config=None) -> str:
    last = latest_attempt(experiment, config)
    if last is not None:
        state = str(last.get("state", "unknown"))
        return "interrupted" if state == "unknown" else state  # stopped before its record was written
    return "approved" if experiment.approved else "planned"


def metric_points(experiment: Experiment) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for run_id in experiment.attempts():
        path = experiment.path("attempts", run_id, "metrics.jsonl")
        if path.is_file():
            points.extend(read_points(path))
    return points


def summarise_series(points: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """first / last / min / max / mean of the last tenth, for every numeric training field."""
    fields: dict[str, list[tuple[int, float]]] = {}
    for point in points:
        if point.get("event") and point["event"] != "rollout":
            continue
        step = point.get("step", 0)
        for key, value in point.items():
            if key in ("t", "step", "event", "epoch") or not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            fields.setdefault(key, []).append((step, float(value)))
    out = {}
    for key, series in fields.items():
        values = [v for _, v in series]
        tail = values[-max(1, len(values) // 10):]
        low = min(series, key=lambda sv: sv[1])
        out[key] = {"points": len(values), "first": values[0], "last": values[-1], "min": low[1], "min_step": low[0],
                    "max": max(values), "last_tenth_mean": sum(tail) / len(tail), "last_step": series[-1][0]}
    return out


def eval_results(experiment: Experiment) -> dict[str, dict[str, Any]]:
    """The newest result of each eval block, with the fingerprint it was measured under."""
    out: dict[str, dict[str, Any]] = {}
    fingerprints = experiment.data.get("eval_fingerprints", {})
    for run_id in experiment.attempts():
        for path in sorted(experiment.dir.glob(f"eval/*-{run_id}.json")):
            try:
                record = json.loads(path.read_text())
            except ValueError:
                continue
            name = record.get("block") or path.name.rsplit("-", 1)[0]
            out[name] = {"fingerprint": record.get("fingerprint") or fingerprints.get(name),
                         "metrics": record.get("metrics", {}), "error": record.get("error"), "run": run_id}
    for name, fp in fingerprints.items():
        out.setdefault(name, {"fingerprint": fp, "metrics": {}, "error": "not measured yet"})
    return out


def trace_stats(experiment: Experiment) -> dict[str, Any]:
    directory = experiment.path("traces")
    files = sorted(directory.glob("*.jsonl")) if directory.is_dir() else []
    stats: dict[str, Any] = {"files": len(files), "bytes": sum(f.stat().st_size for f in files)}
    for kind in ("rollouts", "samples"):
        stats[kind] = sum(1 for f in files if f.name.startswith(kind) for _ in f.open(encoding="utf-8"))
    return stats


def changed_params(experiment: Experiment) -> dict[str, Any]:
    out = {}
    sources = experiment.data.get("param_sources", {})
    for slot, values in experiment.data["params"].items():
        label = "train" if slot == "backbone" else slot
        for name, value in values.items():
            if sources.get(slot, {}).get(name, "default") != "default":
                out[f"{label}.{name}"] = value
    return out


def flat_params(experiment: Experiment) -> dict[str, Any]:
    return {f"{'train' if slot == 'backbone' else slot}.{name}": value
            for slot, values in experiment.data["params"].items() for name, value in values.items()}


def environment(experiment: Experiment) -> dict[str, Any]:
    for run_id in reversed(experiment.attempts()):
        path = experiment.path(f"env-{run_id}.json")
        if path.is_file():
            try:
                env = json.loads(path.read_text())
            except ValueError:
                continue
            env.pop("installed", None)
            env.pop("pip_freeze", None)  # older records
            return env
    return {}


def build(experiment: Experiment, config=None) -> dict[str, Any]:
    d = experiment.data
    points = metric_points(experiment)
    series = summarise_series(points)
    config = _config(config)
    attempts = attempt_records(experiment, config)
    last = attempts[-1] if attempts else {}
    result = {}
    if last.get("id"):
        path = experiment.path(f"result-{last['id']}.json")
        if path.is_file():
            try:
                result = json.loads(path.read_text())
                result.pop("traceback", None)
            except ValueError:
                pass
    return {
        "name": d["name"],
        "group": d.get("group"),
        "sweep": d.get("sweep"),
        "status": status(experiment, config),
        "notes": d.get("notes", ""),
        "plan_code": experiment.code,
        "plan_hash": d["plan_hash"],
        "approval": experiment.approval,
        "approval_state": _approval_state(experiment, config),
        "preset": d["preset"],
        "backbone": {"name": d["backbone"]["name"], "sha256": d["backbone"]["sha256"],
                     "config_class": d["backbone"].get("config_class")},
        "blocks": {slot: [f"{b['name']}@{b['sha256'][:12]}" for b in entries] for slot, entries in d["blocks"].items()},
        "inputs": d["inputs"],
        "params_changed": changed_params(experiment),
        "params": flat_params(experiment),
        "environment": environment(experiment),
        "attempts": [{k: a.get(k) for k in ("id", "state", "started_at", "finished_at", "exit_code", "error", "artifacts")}
                     for a in attempts],
        "artifacts": sorted({key for a in attempts for key in (a.get("artifacts") or [])}),
        "training": {k: v for k, v in series.items()},
        "result": result,
        "eval": eval_results(experiment),
        "traces": trace_stats(experiment),
    }


def _config(config):
    """The given config, or the one from the environment; None when there is none to be had."""
    if config is not None:
        return config
    try:
        from ..config import Config

        return Config.from_env()
    except Exception:  # noqa: BLE001
        return None


def _approval_state(experiment: Experiment, config) -> str:
    from .experiment import approval_state

    config = _config(config)
    if config is None:
        return "unknown"
    return approval_state(config, experiment.approval)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return format_value(value)


def render(report: Mapping[str, Any]) -> str:
    r = report
    lines = [f"# Experiment {r['name']}", ""]
    lines.append(f"- **Status:** {r['status']}" + (f" ({len(r['attempts'])} attempt(s))" if r["attempts"] else ""))
    if r.get("notes"):
        lines.append(f"- **Purpose:** {r['notes']}")
    lines.append(f"- **Preset:** {r['preset']['name']}" + (f" (from {r['preset']['parent']})" if r["preset"].get("parent") else "")
                 + (f" — {r['preset']['description']}" if r["preset"].get("description") else ""))
    lines.append(f"- **Backbone:** {r['backbone']['name']}@{r['backbone']['sha256'][:12]}"
                 + (f" ({r['backbone']['config_class']})" if r["backbone"].get("config_class") else ""))
    lines.append(f"- **Model:** {r['inputs']['model'] or '—'}")
    lines.append(f"- **Datasets:** {', '.join(r['inputs']['datasets']) or '—'}")
    if r["inputs"]["inputs"]:
        lines.append(f"- **Other inputs:** {', '.join(r['inputs']['inputs'])}")
    approval = r.get("approval")
    lines.append(f"- **Plan:** code {r['plan_code']}, " + (f"approved by {approval['approved_by']} ({r.get('approval_state')})" if approval else "not approved"))
    if r.get("sweep"):
        lines.append(f"- **Sweep:** {r['sweep']}")
    lines += ["", "## Blocks", ""]
    lines += [f"- {slot}: {', '.join(names)}" for slot, names in r["blocks"].items()]
    lines += ["", "## Parameters changed from their defaults", ""]
    lines += [f"- `{k}` = `{_fmt(v)}`" for k, v in r["params_changed"].items()] or ["- none (every parameter at its default)"]
    lines.append("")
    lines.append("All parameters, with defaults and meanings: `plan.md`.")
    env = r.get("environment") or {}
    if env:
        packages = env.get("packages", {})
        keep = ", ".join(f"{k} {v}" for k, v in packages.items()
                         if k in ("torch", "transformers", "trl", "peft", "unsloth", "datasets", "vllm", "sentence-transformers"))
        lines += ["", "## Environment", "", f"- GPU: {', '.join(env.get('gpus', [])) or env.get('device', 'unknown')}",
                  f"- Python {env.get('python')}, CUDA {env.get('torch_cuda')}", f"- {keep}"]
    if r["attempts"]:
        lines += ["", "## Attempts", ""]
        for a in r["attempts"]:
            took = ""
            if a.get("started_at") and a.get("finished_at"):
                took = f", {(a['finished_at'] - a['started_at']) / 60:.1f} min"
            lines.append(f"- {a['id']}: {a.get('state')}{took}" + (f" — {a['error']}" if a.get("error") else ""))
    training = r.get("training") or {}
    if training:
        lines += ["", "## Training", "", "| metric | first | last | min (step) | mean of last 10 % |", "| --- | --- | --- | --- | --- |"]
        shown = [k for k in KEY_FIELDS if k in training] + sorted(k for k in training if k.startswith(("rewards/", "reward_")) and k not in KEY_FIELDS)
        for key in shown:
            s = training[key]
            lines.append(f"| {key} | {_fmt(s['first'])} | {_fmt(s['last'])} | {_fmt(s['min'])} ({s['min_step']}) | {_fmt(s['last_tenth_mean'])} |")
        steps = max((s.get("last_step", 0) for s in training.values()), default=0)
        lines.append("")
        lines.append(f"Steps logged: {steps}. Every point: `attempts/<run>/metrics.jsonl`.")
    result = r.get("result") or {}
    if result.get("train"):
        t = result["train"]
        bits = [f"{k}={_fmt(t[k])}" for k in ("train_runtime", "train_loss", "peak_reserved_gb", "training_reserved_gb", "rollout_rounds") if k in t]
        if bits:
            lines += ["", "Trainer summary: " + ", ".join(bits)]
    if result.get("status") == "failed":
        lines += ["", f"**Failed:** {result.get('error')}"]
    if r["eval"]:
        lines += ["", "## Evaluation", "", "| eval | metric | value | fingerprint |", "| --- | --- | --- | --- |"]
        for name, e in r["eval"].items():
            fp = (e.get("fingerprint") or "")[:10]
            if e.get("error"):
                lines.append(f"| {name} | — | {e['error']} | {fp} |")
                continue
            for metric, value in e["metrics"].items():
                lines.append(f"| {name} | {metric} | {_fmt(value)} | {fp} |")
        lines.append("")
        lines.append("Per-item predictions: `eval/<block>-<run>.predictions.jsonl`.")
    t = r.get("traces") or {}
    if t.get("files"):
        lines += ["", "## Traces", "", f"- {t.get('rollouts', 0)} rollout completions, {t.get('samples', 0)} probe samples, "
                  f"{t['files']} file(s), {t['bytes'] / 1e6:.1f} MB in `traces/`",
                  f"- Browse: `nawat exp traces {r['name']}`"]
    if r["artifacts"]:
        lines += ["", "## Artifacts in object storage", ""] + [f"- `{k}`" for k in r["artifacts"]]
    lines += ["", "## Where everything is", "",
              "- `plan.md` — every parameter; `code/` — the exact code that ran; `preset.toml` — reusable preset",
              "- `attempts/<run>/` — log, metrics, run record; `env-<run>.json` — library versions and GPU",
              "- `effective_config-<run>.json` — every trainer field in force; `eval/`, `traces/`", ""]
    return "\n".join(lines)


# -- comparing ------------------------------------------------------------------


def _differs(values: Sequence[Any]) -> bool:
    first = json.dumps(values[0], sort_keys=True, default=str)
    return any(json.dumps(v, sort_keys=True, default=str) != first for v in values[1:])


def compare(experiments: Sequence[Experiment], *, show_all: bool = False, force: bool = False, config=None) -> dict[str, Any]:
    """What differs between experiments, and their results side by side where that is fair."""
    config = _config(config)
    reports = [build(e, config) for e in experiments]
    names = [r["name"] for r in reports]
    setup: list[tuple[str, list[Any]]] = []

    def add(label: str, values: list[Any]) -> None:
        if show_all or _differs(values):
            setup.append((label, values))

    add("preset", [r["preset"]["name"] for r in reports])
    add("backbone", [f"{r['backbone']['name']}@{r['backbone']['sha256'][:8]}" for r in reports])
    add("model", [r["inputs"]["model"] for r in reports])
    add("datasets", [", ".join(r["inputs"]["datasets"]) for r in reports])
    slots = sorted({slot for r in reports for slot in r["blocks"]})
    for slot in slots:
        add(f"block {slot}", [", ".join(r["blocks"].get(slot, [])) or "—" for r in reports])
    keys = sorted({k for r in reports for k in r["params"]})
    shared_params = {}
    for key in keys:
        values = [r["params"].get(key, "—") for r in reports]
        if show_all or _differs(values):
            setup.append((key, values))
        else:
            shared_params[key] = values[0]

    results: list[tuple[str, list[Any]]] = [("status", [r["status"] for r in reports])]
    for field in ("loss", "eval_loss", "reward", "reward_mean", "kl"):
        if any(field in r["training"] for r in reports):
            results.append((f"{field} (last 10 % mean)", [r["training"].get(field, {}).get("last_tenth_mean") for r in reports]))
    for r_field in sorted({k for r in reports for k in r["training"] if k.startswith("rewards/") and k.endswith("/mean")}):
        results.append((f"{r_field} (last 10 %)", [r["training"].get(r_field, {}).get("last_tenth_mean") for r in reports]))

    warnings: list[str] = []
    eval_names = sorted({n for r in reports for n in r["eval"]})
    for name in eval_names:
        prints = [r["eval"].get(name, {}).get("fingerprint") for r in reports]
        present = [p for p in prints if p]
        comparable = len(present) == len(prints) and len(set(present)) == 1
        metrics = sorted({m for r in reports for m in (r["eval"].get(name, {}).get("metrics") or {})})
        if not comparable and len(set(present)) > 1:
            groups: dict[str, list[str]] = {}
            for exp_name, fp in zip(names, prints):
                groups.setdefault(fp or "missing", []).append(exp_name)
            warnings.append(f"eval {name}: measured differently ({len(groups)} versions: "
                            + "; ".join(f"{fp[:10]} ← {', '.join(ns)}" for fp, ns in groups.items())
                            + "). Its numbers are " + ("shown marked ⚠" if force else "withheld; pass --force to see them anyway") + ".")
            if not force:
                results.append((f"{name}", ["≠ fingerprint"] * len(reports)))
                continue
        for metric in metrics:
            values = [(r["eval"].get(name, {}).get("metrics") or {}).get(metric) for r in reports]
            label = f"{name} · {metric}" + (" ⚠" if not comparable and len(set(present)) > 1 else "")
            results.append((label, values))
    return {"experiments": names, "differences": setup, "shared": shared_params, "results": results, "warnings": warnings}


def render_table(rows: Sequence[tuple[str, Sequence[Any]]], names: Sequence[str]) -> list[str]:
    header = "| | " + " | ".join(names) + " |"
    sep = "| --- |" + " --- |" * len(names)
    out = [header, sep]
    for label, values in rows:
        out.append(f"| {label} | " + " | ".join("—" if v is None else _fmt(v) for v in values) + " |")
    return out


def render_compare(result: Mapping[str, Any], *, for_agent: bool = False) -> str:
    names = result["experiments"]
    lines = [f"# Comparison of {len(names)} experiments", ""]
    if result["warnings"]:
        lines += ["**Not comparable as measured:**", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    lines += ["## What differs", ""]
    lines += render_table(result["differences"], names) if result["differences"] else ["Nothing in the setup differs."]
    lines += ["", "## Results", ""] + render_table(result["results"], names)
    if for_agent:
        lines += ["", "## Shared setup (identical in every experiment)", ""]
        lines += [f"- `{k}` = `{_fmt(v)}`" for k, v in result["shared"].items()]
        lines += ["", "Eval numbers appear side by side only where every experiment was measured with the same eval "
                  "fingerprint (same eval code, test data, metric and generation settings).", ""]
    else:
        lines += ["", f"{len(result['shared'])} parameters are identical in all of them and hidden (--all shows them)."]
    return "\n".join(lines)


# -- traces -------------------------------------------------------------------


def prompt_key(prompt: Any) -> str:
    """The user's words, for matching the same prompt across experiments."""
    if isinstance(prompt, list):
        users = [m.get("content") for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
        content = users[-1] if users else prompt
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        return str(content).strip()
    return str(prompt).strip()


def rollout_steps(experiment: Experiment) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for record in read_traces(experiment.path("traces"), "rollouts"):
        s = by_step.setdefault(record["step"], {"step": record["step"], "n": 0, "totals": [], "rewards": {}})
        s["n"] += 1
        if record.get("total") is not None:
            s["totals"].append(record["total"])
        for name, value in (record.get("rewards") or {}).items():
            if value is not None:
                s["rewards"].setdefault(name, []).append(value)
    out = []
    for step in sorted(by_step):
        s = by_step[step]
        totals = s["totals"]
        out.append({"step": step, "completions": s["n"],
                    "total_mean": sum(totals) / len(totals) if totals else None,
                    "total_max": max(totals) if totals else None,
                    **{f"{k}": sum(v) / len(v) for k, v in s["rewards"].items()}})
    return out


def records_at(experiment: Experiment, kind: str, step: int | None, prompt_filter: str | None) -> list[dict[str, Any]]:
    out = []
    for record in read_traces(experiment.path("traces"), kind):
        if step is not None and record.get("step") != step:
            continue
        text = prompt_key(record.get("prompt") if kind == "rollouts" else record.get("input"))
        if prompt_filter and prompt_filter.lower() not in text.lower():
            continue
        out.append(record)
    return out


def side_by_side(experiments: Sequence[Experiment], *, kind: str, step: int | None, prompt_filter: str | None,
                 limit: int) -> list[dict[str, Any]]:
    """For each prompt seen by every experiment (at the step, if given): what each one produced."""
    per_exp = []
    for experiment in experiments:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records_at(experiment, kind, step, prompt_filter):
            key = prompt_key(record.get("prompt") if kind == "rollouts" else record.get("input"))
            grouped.setdefault(key, []).append(record)
        per_exp.append(grouped)
    common = [k for k in per_exp[0] if all(k in g for g in per_exp[1:])] if per_exp else []
    out = []
    for key in common[:limit]:
        entry = {"prompt": key, "by_experiment": {}}
        for experiment, grouped in zip(experiments, per_exp):
            records = grouped[key]
            if kind == "rollouts":
                records = sorted(records, key=lambda r: (r.get("total") is None, -(r.get("total") or 0)))
            entry["by_experiment"][experiment.name] = records
        out.append(entry)
    return out


def truncate(text: Any, width: int) -> str:
    s = str(text).replace("\n", "⏎ ")
    return s if len(s) <= width else s[: width - 1] + "…"
