"""<What this dataset block reads, and what it turns each row into.>

Return a dict:
    train     the training rows, in the shape the backbone needs
              (sft: "messages" or "text"; dpo: prompt/chosen/rejected; grpo: "prompt" + reward columns)
    eval      held-out rows in the same shape, or None
    heldout   [{"input": <messages or text>, "reference": <answer>}] for generation evals
    probes    a few items like heldout, re-asked during training for the trace
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "<one line>", "backbones": ("sft",)}

PARAMS = {
    "split": P("train", "Dataset split to read"),
    "max_samples": P(0, "Use at most N rows (0 = all)"),
    "eval_size": P(0.0, "Held out: a fraction (< 1) or a row count"),
    "seed": P(3407, "Seed for sampling and the held-out split"),
}


def load(ctx, p, model, tokenizer):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split)
    ds = support.limit(ds, p.max_samples, p.seed)
    # ds = ds.map(...)   your processing here — every number it uses belongs in PARAMS
    train, held = support.split_off(ds, p.eval_size, p.seed)
    return {"train": train, "eval": held, "heldout": [], "probes": []}
