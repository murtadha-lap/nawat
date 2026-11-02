"""GSM8K — grade-school maths word problems, exact numeric answer (openai/gsm8k).

Zero-shot, chat-formatted, greedy. The reference is the number after `####`;
the prediction is read with `extract` (`auto`: solution tags, then \\boxed,
then the last number — so tag-trained GRPO models and plain chat models are
both scored on their final answer). Fetched through Nawāt on first use and kept
in object storage.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "GSM8K maths word problems (openai/gsm8k), accuracy", "models": ("llm",)}

PARAMS = {
    "dataset": P("datasets/openai/gsm8k", "Where GSM8K comes from"),
    "subset": P("main", "Configuration"),
    "split": P("test", "Split (1,319 problems)"),
    "max_items": P(250, "Score the first N problems (0 = all)"),
    "prompt": P("{question}\nPlease reason step by step, and put your final answer after '####'.", "User turn"),
    "extract": P("auto", "How the answer is read", choices=("auto", "gsm8k", "last-number", "solution-tags", "boxed")),
    "max_new_tokens": P(512, "Generation length"),
    "temperature": P(0.0, "Sampling temperature (0 = greedy, the comparable setting)"),
    "batch_size": P(8, "Generation batch size (does not change results under greedy decoding)"),
}


def inputs(p):
    return [p.dataset]


def evaluate(ctx, p, model, tokenizer, data):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.input_dir(p.dataset), split=p.split, subset=p.subset or None)
    rows = support.rows(ds, p.max_items)
    items = [{"input": [{"role": "user", "content": p.prompt.format(question=r["question"])}],
              "reference": r["answer"].split("####")[-1].strip().replace(",", "")} for r in rows]
    return support.generation_eval(ctx, model, tokenizer, "gsm8k", items, metric="number", extract=p.extract,
                                   max_new_tokens=p.max_new_tokens, temperature=p.temperature, batch_size=p.batch_size)
