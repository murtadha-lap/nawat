"""Decision rows {state, questions, gold} → decision items, with a held-out share.

``questions`` maps a name to a typed question — ``choice`` with named
``criteria``, ``noul`` (yes / no), or ``score`` with ordered levels — and
``gold`` holds the right answer for each. Built and split with
``FastDecisionModel.build_dataset`` and ``split_holdout``, as in the notebook.
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Typed decision rows (state / questions / gold) → decision items",
         "backbones": ("decision",), "models": ("decision",)}

PARAMS = {
    "split": P("train", "Dataset split to read"),
    "subset": P("all", "Dataset configuration (typed-decisions: all)"),
    "max_samples": P(0, "Use at most N rows (0 = all)"),
    "holdout_fraction": P(0.1, "Share of decisions held out for calibration and eval"),
    "seed": P(3407, "Seed for the held-out split"),
}


def load(ctx, p, model, tokenizer):
    from unsloth import FastDecisionModel

    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    ds = support.limit(ds, p.max_samples, p.seed)
    items, report = FastDecisionModel.build_dataset(ds, tokenizer, model)
    ctx.write_json("data/decision-build-report.json", report)
    ctx.log(f"decisions: skipped {report['skipped']} of {report['total']}" + (f" — {report['reason']}" if report.get("reason") else ""))
    train, held = FastDecisionModel.split_holdout(items, seed=p.seed, fraction=p.holdout_fraction)
    return {"train": train, "eval": held, "heldout": [], "probes": []}
