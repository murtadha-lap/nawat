"""MATH-500 — competition maths, answer in \\boxed{} (HuggingFaceH4/MATH-500).

Zero-shot, greedy. Answers are compared after LaTeX normalisation (spacing,
\\left/\\right, dfrac, \\text{}) or numerically.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "MATH-500 competition maths (HuggingFaceH4/MATH-500), accuracy", "models": ("llm",)}

PARAMS = {
    "dataset": P("datasets/HuggingFaceH4/MATH-500", "Where MATH-500 comes from"),
    "split": P("test", "Split"),
    "max_items": P(0, "Score the first N problems (0 = all 500)"),
    "prompt": P("{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}.", "User turn"),
    "extract": P("auto", "How the answer is read", choices=("auto", "boxed", "solution-tags", "last-number")),
    "max_new_tokens": P(1024, "Generation length"),
    "temperature": P(0.0, "Sampling temperature (0 = greedy, the comparable setting)"),
    "batch_size": P(8, "Generation batch size"),
}


def inputs(p):
    return [p.dataset]


def evaluate(ctx, p, model, tokenizer, data):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.input_dir(p.dataset), split=p.split)
    rows = support.rows(ds, p.max_items)
    items = [{"input": [{"role": "user", "content": p.prompt.format(problem=r["problem"])}], "reference": r["answer"]}
             for r in rows]
    return support.generation_eval(ctx, model, tokenizer, "math-500", items, metric="math", extract=p.extract,
                                   max_new_tokens=p.max_new_tokens, temperature=p.temperature, batch_size=p.batch_size)
