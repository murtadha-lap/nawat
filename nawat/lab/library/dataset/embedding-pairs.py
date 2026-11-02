"""Anchor / positive (/ negative) pairs for embedding training.

Columns are used in order — first the anchor, then the positive, then any hard
negatives — which is how sentence-transformers' contrastive losses read them.
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Anchor / positive (/ negative) text pairs for embeddings",
         "backbones": ("embedding",), "models": ("embedding",)}

PARAMS = {
    "split": P("train", "Dataset split to read"),
    "subset": P("", "Dataset configuration name"),
    "columns": P([], "Columns to use, in order (anchor, positive, negatives…); empty = all, as stored", type=list),
    "max_samples": P(0, "Use at most N pairs (0 = all)"),
    "eval_size": P(0.0, "Pairs held out for retrieval eval: a fraction (< 1) or a count"),
    "seed": P(3407, "Seed for sampling and the held-out split"),
}


def load(ctx, p, model, tokenizer):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    if p.columns:
        ds = ds.select_columns(list(p.columns))
    ds = support.limit(ds, p.max_samples, p.seed)
    train, held = support.split_off(ds, p.eval_size, p.seed)
    first, second = (held.column_names[:2] if held is not None else (None, None))
    pairs = [{"query": r[first], "positive": r[second]} for r in support.rows(held)] if held is not None else []
    return {"train": train, "eval": None, "heldout": pairs, "probes": []}
