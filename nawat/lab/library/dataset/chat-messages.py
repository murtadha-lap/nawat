"""Conversations for SFT: a messages / ShareGPT column, or a prompt and a response column.

ShareGPT turns (``from``/``value``, ``human``/``gpt``) are normalised to
``role``/``content``, as Unsloth's ``standardize_sharegpt`` does, so the chat
template block sees one format whatever the source used.
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Conversations (messages / ShareGPT / prompt+response columns) for SFT",
         "backbones": ("sft",), "models": ("llm",)}

PARAMS = {
    "split": P("train", "Dataset split to read"),
    "subset": P("", "Dataset configuration (subset) name, if it has several"),
    "messages_column": P("", "Column holding the conversation; empty = find messages / conversations"),
    "prompt_column": P("", "Instead of a conversation: column with the user turn"),
    "response_column": P("", "…and the column with the assistant turn"),
    "system_prompt": P("", "System message prepended to every conversation"),
    "max_samples": P(0, "Use at most N rows (0 = all)"),
    "eval_size": P(0.0, "Held out for evaluation: a fraction (< 1) or a row count"),
    "probe_count": P(8, "Held-out conversations the trace re-asks during training"),
    "seed": P(3407, "Seed for sampling and the held-out split"),
}

ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "model": "assistant",
         "bot": "assistant", "system": "system", "tool": "tool"}


def _normalise(turns):
    out = []
    for turn in turns:
        role = turn.get("role", turn.get("from"))
        content = turn.get("content", turn.get("value"))
        out.append({"role": ROLES.get(str(role).lower(), str(role)), "content": content})
    return out


def load(ctx, p, model, tokenizer):
    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    ds = support.limit(ds, p.max_samples, p.seed)
    column = p.messages_column or next((c for c in ("messages", "conversations", "conversation", "chat")
                                        if c in ds.column_names), "")
    prompt_col, response_col, system = p.prompt_column, p.response_column, p.system_prompt
    if not column and not (prompt_col and response_col):
        raise ValueError(f"No conversation column among {ds.column_names}; set dataset.messages_column, "
                         "or dataset.prompt_column and dataset.response_column.")

    def convert(row):
        if column:
            messages = _normalise(row[column])
        else:
            messages = [{"role": "user", "content": row[prompt_col]}, {"role": "assistant", "content": row[response_col]}]
        if system and (not messages or messages[0]["role"] != "system"):
            messages = [{"role": "system", "content": system}] + messages
        return {"messages": messages}

    ds = ds.map(convert, remove_columns=ds.column_names)
    train, held = support.split_off(ds, p.eval_size, p.seed)

    def as_items(dataset, count=None):
        items = []
        for row in support.rows(dataset, count):
            m = row["messages"]
            if m and m[-1]["role"] == "assistant":
                items.append({"input": m[:-1], "reference": m[-1]["content"]})
        return items

    return {"train": train, "eval": held, "format": "messages",
            "probes": as_items(held if held is not None else train, p.probe_count),
            "heldout": as_items(held)}
