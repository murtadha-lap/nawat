"""Your own evaluation set — a fixed test file you own, scored the same way every time.

Point `dataset` at a dataset key (`datasets/my-evals/ocr-v1`); it is staged
like any input, so it is fetched once and kept in object storage. Rows need an
input column and a reference column, and optionally an image column. Because
the set, the metric and the generation settings are all parameters, every
experiment that uses the same `evalset` settings is measured identically, and
`nawat exp compare` will compare them.

Write a JSONL file (`{"prompt": ..., "reference": ...}` per line), publish it
with `nawat publish ./my-evals datasets/my-evals/ocr-v1`, and it is ready.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "Your own fixed test set (any dataset key), scored with a chosen metric"}

PARAMS = {
    "dataset": P("", "Dataset key holding the eval set, e.g. datasets/my-evals/ocr-v1"),
    "split": P("test", "Split to read (a file named test.jsonl is the test split)"),
    "subset": P("", "Dataset configuration name"),
    "input_column": P("prompt", "Column with the input (text, or a list of chat messages)"),
    "reference_column": P("reference", "Column with the expected answer"),
    "image_column": P("", "Column with an image, for vision models"),
    "system_prompt": P("", "System message added before each input"),
    "metric": P("exact", "How an answer is scored", choices=("exact", "number", "contains", "math", "cer", "wer")),
    "extract": P("none", "How the answer is read from the generation",
                 choices=("none", "auto", "solution-tags", "boxed", "last-number", "gsm8k")),
    "max_items": P(0, "Score at most N items (0 = all)"),
    "max_new_tokens": P(512, "Generation length"),
    "temperature": P(0.0, "Sampling temperature (0 = greedy, the comparable setting)"),
    "batch_size": P(8, "Generation batch size (does not change results under greedy decoding)"),
}


def inputs(p):
    return [p.dataset] if p.dataset else []


def evaluate(ctx, p, model, tokenizer, data):
    from nawat.lab import support

    if not p.dataset:
        raise ValueError("eval.evalset needs a dataset: --set eval.evalset.dataset=datasets/<your-set>")
    ds = support.load_dataset_dir(ctx.input_dir(p.dataset), split=p.split, subset=p.subset or None)
    ds = ds if p.max_items <= 0 else ds.select(range(min(p.max_items, len(ds))))
    items = []
    for row in ds:
        value = row[p.input_column]
        if isinstance(value, str):
            content = value if not p.image_column else [{"type": "image"}, {"type": "text", "text": value}]
            value = [{"role": "user", "content": content}]
        if p.system_prompt:
            value = [{"role": "system", "content": p.system_prompt}] + list(value)
        item = {"input": value, "reference": str(row[p.reference_column])}
        if p.image_column:
            item["image"] = row[p.image_column]
        items.append(item)
    return support.generation_eval(ctx, model, tokenizer, "evalset", items, metric=p.metric, extract=p.extract,
                                   max_new_tokens=p.max_new_tokens, temperature=p.temperature, batch_size=p.batch_size)
