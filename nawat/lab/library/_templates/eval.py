"""<What this eval measures, on which data, and how it is scored.>

``inputs(p)`` names the dataset keys it needs; Nawāt stages them (fetching from
the hub once and keeping them in object storage). Keep every setting that
changes the score — data, metric, prompt, generation settings — in PARAMS:
they make up the eval's fingerprint, and `nawat exp compare` only compares
experiments whose fingerprints match.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "<one line>"}

PARAMS = {
    "dataset": P("datasets/<owner>/<name>", "Dataset key with the test data"),
    "split": P("test", "Split to score"),
    "max_items": P(0, "Score the first N items (0 = all)"),
    "max_new_tokens": P(512, "Generation length"),
    "temperature": P(0.0, "Sampling temperature (0 = greedy)"),
    "batch_size": P(8, "Generation batch size"),
}


def inputs(p):
    return [p.dataset]


def evaluate(ctx, p, model, tokenizer, data):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.input_dir(p.dataset), split=p.split)
    items = [{"input": [{"role": "user", "content": r["question"]}], "reference": r["answer"]}
             for r in support.rows(ds, p.max_items)]
    return support.generation_eval(ctx, model, tokenizer, "my-eval", items, metric="exact", extract="none",
                                   max_new_tokens=p.max_new_tokens, temperature=p.temperature, batch_size=p.batch_size)
