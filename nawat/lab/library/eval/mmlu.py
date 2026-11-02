"""MMLU — 57 subjects of four-way multiple choice, scored by log-likelihood (cais/mmlu).

Zero-shot, the standard prompt: "The following are multiple choice questions
(with answers) about <subject>." then the question, the lettered options and
"Answer:". The letter with the highest probability is the answer — nothing is
generated, so the score does not depend on sampling.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "MMLU 57-subject multiple choice (cais/mmlu), log-likelihood accuracy",
         "models": ("llm",)}

PARAMS = {
    "dataset": P("datasets/cais/mmlu", "Where MMLU comes from"),
    "subset": P("all", "Configuration: all, or one subject"),
    "split": P("test", "Split (14,042 questions in all)"),
    "max_items": P(1000, "Score the first N questions (0 = all)"),
}


def inputs(p):
    return [p.dataset]


def evaluate(ctx, p, model, tokenizer, data):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.input_dir(p.dataset), split=p.split, subset=p.subset or None)
    questions = []
    for r in support.rows(ds, p.max_items):
        subject = str(r.get("subject", "")).replace("_", " ")
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", r["choices"]))
        context = (f"The following are multiple choice questions (with answers) about {subject}.\n\n"
                   f"{r['question'].strip()}\n{options}\nAnswer:")
        questions.append({"context": context, "choices": [" A", " B", " C", " D"], "gold": int(r["answer"]),
                          "subject": subject})
    return support.choice_eval(ctx, model, tokenizer, "mmlu", questions)
