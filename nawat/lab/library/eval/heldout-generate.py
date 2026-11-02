"""Held-out generation: the model answers the held-out part of its own training data.

Compares the extracted answer with the reference — exact match, numeric match,
CER or WER. Depends on the training dataset and its split, so it only compares
across experiments trained on the same data split the same way (the fingerprint
says so). For a measurement that stays fixed across datasets, use a benchmark
block or `evalset`.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "Generate on the held-out split; exact / number / CER / WER",
         "uses_training_data": True}

PARAMS = {
    "metric": P("exact", "How an answer is scored", choices=("exact", "number", "contains", "math", "cer", "wer")),
    "extract": P("none", "How the answer is read from the generation",
                 choices=("none", "auto", "solution-tags", "boxed", "last-number", "gsm8k")),
    "max_items": P(200, "Score at most N held-out items (0 = all)"),
    "max_new_tokens": P(512, "Generation length"),
    "temperature": P(0.0, "Sampling temperature (0 = greedy, the comparable setting)"),
    "batch_size": P(8, "Generation batch size (does not change results under greedy decoding)"),
}


def evaluate(ctx, p, model, tokenizer, data):
    from nawat.lab import support

    items = list(data.get("heldout") or [])
    if p.max_items > 0:
        items = items[: p.max_items]
    if not items:
        return {"metrics": {"n": 0}, "note": "the dataset block held nothing out; set dataset.eval_size"}
    return support.generation_eval(ctx, model, tokenizer, "heldout-generate", items, metric=p.metric, extract=p.extract,
                                   max_new_tokens=p.max_new_tokens, temperature=p.temperature, batch_size=p.batch_size)
