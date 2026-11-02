"""Prompts with reference answers for GRPO / DAPO, plus an optional priming set.

Each row becomes ``prompt`` (a user message; the template adds the system
prompt) and ``answer`` (passed to every reward function as ``answer=``). The
priming set — the second dataset of the experiment, as in the Qwen3 GRPO
notebook — holds worked solutions the template turns into format examples.
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Question + reference answer prompts for GRPO, with optional priming set",
         "backbones": ("grpo",), "models": ("llm",)}

PARAMS = {
    "split": P("train", "Dataset split to read"),
    "subset": P("", "Dataset configuration (DAPO-Math-17k-Processed: en)"),
    "question_column": P("prompt", "Column with the question"),
    "answer_column": P("solution", "Column with the reference answer"),
    "answer_extractor": P("none", "How the reference answer is read from its column",
                          choices=("none", "gsm8k", "boxed", "last-number")),
    "numeric_only": P(False, "Keep only rows whose answer is a number"),
    "max_samples": P(0, "Use at most N prompts (0 = all)"),
    "eval_size": P(0.0, "Prompts held out for evaluation: a fraction (< 1) or a count"),
    "seed": P(3407, "Seed for sampling and the held-out split"),
    "prime": P(False, "Read a priming set from the experiment's second dataset"),
    "prime_split": P("cot", "Priming dataset split"),
    "prime_subset": P("", "Priming dataset configuration"),
    "prime_problem_column": P("problem", "Priming: the problem"),
    "prime_solution_column": P("generated_solution", "Priming: the worked solution"),
    "prime_answer_column": P("expected_answer", "Priming: the final answer"),
    "prime_numeric_only": P(True, "Priming: keep only numeric answers (as the notebook does)"),
    "prime_max_samples": P(0, "Priming: at most N examples (0 = all)"),
}


def _extract(kind, text):
    from nawat.lab import support

    text = "" if text is None else str(text)
    if kind == "gsm8k":
        return text.split("####")[-1].strip().replace(",", "") if "####" in text else None
    if kind == "boxed":
        return support.boxed(text)
    if kind == "last-number":
        return support.last_number(text)
    return text


def _is_number(value):
    try:
        float(str(value).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def load(ctx, p, model, tokenizer):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    qc, ac, kind = p.question_column, p.answer_column, p.answer_extractor
    ds = ds.map(lambda x: {"prompt": [{"role": "user", "content": x[qc]}], "answer": _extract(kind, x[ac])},
                remove_columns=ds.column_names)
    numeric = p.numeric_only
    ds = ds.filter(lambda x: x["answer"] is not None and (not numeric or _is_number(x["answer"])))
    ds = support.limit(ds, p.max_samples, p.seed)
    train, held = support.split_off(ds, p.eval_size, p.seed)
    data = {"train": train, "eval": None, "format": "prompts",
            "heldout": [{"input": r["prompt"], "reference": r["answer"]} for r in support.rows(held)],
            "probes": [{"input": r["prompt"], "reference": r["answer"]} for r in support.rows(held if held is not None else train, 8)]}
    if p.prime:
        dirs = ctx.dataset_dirs
        if len(dirs) < 2:
            raise ValueError("dataset.prime is on, but the experiment has no second dataset to prime from.")
        prime = support.load_dataset_dir(dirs[1], split=p.prime_split, subset=p.prime_subset or None)
        pc, sc, xc = p.prime_problem_column, p.prime_solution_column, p.prime_answer_column
        prime = prime.map(lambda x: {"problem": x[pc], "thoughts": x[sc], "answer": str(x[xc])},
                          remove_columns=prime.column_names)
        if p.prime_numeric_only:
            prime = prime.filter(lambda x: _is_number(x["answer"]))
        data["prime_raw"] = support.limit(prime, p.prime_max_samples, p.seed)
    return data
