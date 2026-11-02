"""Raw text for continued pretraining: each row formatted by a template and ended with EOS.

The template is Python ``str.format`` over the row's columns, e.g. the CPT
notebook's ``"Wikipedia Article\n{title}\n{text}"``. EOS is appended because
without it the model never learns where a document ends (the notebook: "Must
add EOS_TOKEN").
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Raw text documents (templated, EOS-terminated) for CPT or plain SFT",
         "backbones": ("cpt", "sft"), "models": ("llm",)}

PARAMS = {
    "split": P("train", "Dataset split to read"),
    "subset": P("", "Dataset configuration (subset) name"),
    "text_template": P("{text}", "How a row becomes text, formatted with its columns"),
    "add_eos": P(True, "Append the tokenizer's EOS token to every document"),
    "max_samples": P(0, "Use at most N rows (0 = all; the notebook takes 1% of Wikipedia)"),
    "eval_size": P(0.0, "Held out for evaluation: a fraction (< 1) or a row count"),
    "probe_count": P(4, "Held-out documents whose openings the trace continues"),
    "probe_chars": P(200, "Characters of a document given as the probe's opening"),
    "seed": P(3407, "Seed for sampling and the held-out split"),
}


def load(ctx, p, model, tokenizer):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    ds = support.limit(ds, p.max_samples, p.seed)
    template, eos = p.text_template, (getattr(tokenizer, "eos_token", "") or "") if p.add_eos else ""
    ds = ds.map(lambda row: {"text": template.format(**row) + eos}, remove_columns=ds.column_names)
    train, held = support.split_off(ds, p.eval_size, p.seed)
    cut = p.probe_chars
    probes = [{"input": r["text"][:cut], "reference": r["text"][cut:cut + 400]}
              for r in support.rows(held if held is not None else train, p.probe_count)]
    return {"train": train, "eval": held, "text_field": "text", "probes": probes, "heldout": []}
