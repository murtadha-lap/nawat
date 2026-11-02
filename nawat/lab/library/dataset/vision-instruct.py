"""Image + text pairs for vision SFT or vision GRPO.

For SFT each row becomes the conversation the vision notebook builds — the
instruction and the image from the user, the target text from the assistant.
For GRPO it becomes a ``prompt`` with an image placeholder, the ``image`` and
the reference ``answer``, as in the vision GRPO notebook (which also resizes to
512×512, converts to RGB and keeps only numeric answers — all parameters here).
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Image + text rows for vision SFT (conversations) or vision GRPO (prompts)",
         "backbones": ("sft", "grpo"), "models": ("vlm",)}

PARAMS = {
    "split": P("train", "Dataset split (MathVista: testmini)"),
    "subset": P("", "Dataset configuration name"),
    "image_column": P("image", "Column with the image (MathVista: decoded_image)"),
    "text_column": P("text", "Column with the target text / reference answer"),
    "question_column": P("", "Column with a per-row question; empty = use the fixed instruction"),
    "instruction": P("Write the LaTeX representation for this image.", "Instruction used when there is no question column"),
    "resize": P(0, "Resize images to N×N pixels; 0 = keep"),
    "convert_rgb": P(False, "Convert images to RGB"),
    "numeric_only": P(False, "Keep only rows whose answer is a number"),
    "max_samples": P(0, "Use at most N rows (0 = all)"),
    "eval_size": P(0.0, "Held out for evaluation: a fraction (< 1) or a row count"),
    "probe_count": P(4, "Held-out rows the trace re-asks during training"),
    "seed": P(3407, "Seed for sampling and the held-out split"),
}


def _number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load(ctx, p, model, tokenizer):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    ic, tc, qc, instruction = p.image_column, p.text_column, p.question_column, p.instruction
    if p.numeric_only:
        ds = ds.filter(lambda x: _number(x[tc]))
    size, rgb = p.resize, p.convert_rgb
    if size or rgb:
        def fix(x):
            image = x[ic]
            if size:
                image = image.resize((size, size))
            if rgb and image.mode != "RGB":
                image = image.convert("RGB")
            return {ic: image}
        ds = ds.map(fix)
    ds = support.limit(ds, p.max_samples, p.seed)
    train, held = support.split_off(ds, p.eval_size, p.seed)

    def question(row):
        return row[qc] if qc else instruction

    def items(dataset, count=None):
        return [{"input": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question(r)}]}],
                 "image": r[ic], "reference": str(r[tc])} for r in support.rows(dataset, count)]

    if ctx.backbone.family == "grpo":
        def to_prompt(x):
            return {"prompt": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question(x)}]}],
                    "image": x[ic], "answer": str(x[tc])}
        keep = [c for c in train.column_names if c != ic]
        train = train.map(to_prompt, remove_columns=keep)
        return {"train": train, "eval": None, "format": "prompts", "heldout": items(held),
                "probes": items(held if held is not None else train, p.probe_count) if held is not None else []}

    def conversation(r):
        return {"messages": [
            {"role": "user", "content": [{"type": "text", "text": question(r)}, {"type": "image", "image": r[ic]}]},
            {"role": "assistant", "content": [{"type": "text", "text": str(r[tc])}]},
        ]}

    return {"train": [conversation(r) for r in train], "eval": [conversation(r) for r in held] if held is not None else None,
            "format": "vision-messages", "heldout": items(held),
            "probes": items(held if held is not None else train, p.probe_count)}
