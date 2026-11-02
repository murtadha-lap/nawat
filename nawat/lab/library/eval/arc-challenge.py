"""ARC-Challenge — grade-school science questions, multiple choice by log-likelihood (allenai/ai2_arc).

The lm-eval formulation: "Question: …\nAnswer:" followed by each answer's text;
reports accuracy and length-normalised accuracy.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "ARC-Challenge science questions (allenai/ai2_arc), acc / acc_norm",
         "models": ("llm",)}

PARAMS = {
    "dataset": P("datasets/allenai/ai2_arc", "Where ARC comes from"),
    "subset": P("ARC-Challenge", "ARC-Challenge or ARC-Easy"),
    "split": P("test", "Split (1,172 questions)"),
    "max_items": P(0, "Score the first N questions (0 = all)"),
}


def inputs(p):
    return [p.dataset]


def evaluate(ctx, p, model, tokenizer, data):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.input_dir(p.dataset), split=p.split, subset=p.subset or None)
    questions = []
    for r in support.rows(ds, p.max_items):
        labels = list(r["choices"]["label"])
        if r["answerKey"] not in labels:
            continue
        questions.append({"context": f"Question: {r['question']}\nAnswer:",
                          "choices": [" " + t for t in r["choices"]["text"]], "gold": labels.index(r["answerKey"])})
    return support.choice_eval(ctx, model, tokenizer, "arc-challenge", questions)
