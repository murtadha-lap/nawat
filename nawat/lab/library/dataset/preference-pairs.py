"""Preference pairs for DPO: a prompt, a chosen answer and a rejected answer.

Accepts both shapes in common use. Conversational pairs where ``chosen`` and
``rejected`` are whole conversations (UltraFeedback-binarized) are split, as the
DPO notebook does, into the shared prompt turns and the final assistant turn of
each. Plain-text pairs pass through.
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Prompt / chosen / rejected pairs for DPO", "backbones": ("dpo",)}

PARAMS = {
    "split": P("train_prefs", "Dataset split (UltraFeedback-binarized: train_prefs)"),
    "subset": P("", "Dataset configuration (subset) name"),
    "prompt_column": P("prompt", "Column with the prompt"),
    "chosen_column": P("chosen", "Column with the preferred answer"),
    "rejected_column": P("rejected", "Column with the rejected answer"),
    "max_samples": P(0, "Use at most N pairs (0 = all)"),
    "eval_size": P(0.0, "Held out for evaluation: a fraction (< 1) or a row count"),
    "probe_count": P(4, "Held-out prompts the trace answers during training"),
    "seed": P(3407, "Seed for sampling and the held-out split"),
}


def load(ctx, p, model, tokenizer):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    ds = support.limit(ds, p.max_samples, p.seed)
    pc, cc, rc = p.prompt_column, p.chosen_column, p.rejected_column

    def convert(row):
        chosen, rejected = row[cc], row[rc]
        if isinstance(chosen, list):
            prompt = chosen[:-1] if len(chosen) > 1 else [{"role": "user", "content": row.get(pc, "")}]
            return {"prompt": prompt, "chosen": [chosen[-1]], "rejected": [rejected[-1]]}
        return {"prompt": row[pc], "chosen": chosen, "rejected": rejected}

    ds = ds.map(convert, remove_columns=ds.column_names)
    train, held = support.split_off(ds, p.eval_size, p.seed)

    def items(dataset, count=None):
        out = []
        for r in support.rows(dataset, count):
            ref = r["chosen"][-1]["content"] if isinstance(r["chosen"], list) else r["chosen"]
            out.append({"input": r["prompt"], "reference": ref})
        return out

    return {"train": train, "eval": held, "probes": items(held if held is not None else train, p.probe_count),
            "heldout": items(held)}
