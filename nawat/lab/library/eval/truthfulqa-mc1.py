"""TruthfulQA MC1 — does the model prefer the true answer over common misconceptions? (truthfulqa/truthful_qa)

Zero-shot "Q: …\nA:" and each candidate answer; exactly one is true. Accuracy
is how often the true one has the highest log-likelihood.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "TruthfulQA MC1 (truthfulqa/truthful_qa), log-likelihood accuracy",
         "models": ("llm",)}

PARAMS = {
    "dataset": P("datasets/truthfulqa/truthful_qa", "Where TruthfulQA comes from"),
    "subset": P("multiple_choice", "Configuration"),
    "split": P("validation", "Split (817 questions)"),
    "max_items": P(0, "Score the first N questions (0 = all)"),
}


def inputs(p):
    return [p.dataset]


def evaluate(ctx, p, model, tokenizer, data):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.input_dir(p.dataset), split=p.split, subset=p.subset or None)
    questions = []
    for r in support.rows(ds, p.max_items):
        targets = r["mc1_targets"]
        questions.append({"context": f"Q: {r['question']}\nA:", "choices": [" " + c for c in targets["choices"]],
                          "gold": list(targets["labels"]).index(1)})
    return support.choice_eval(ctx, model, tokenizer, "truthfulqa-mc1", questions)
