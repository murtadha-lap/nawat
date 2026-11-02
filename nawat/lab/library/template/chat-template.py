"""Chat template: pick it, render conversations with it, and train only on the answers.

``chat_template`` names one of Unsloth's templates (``qwen3``, ``llama-3.1``,
``gemma-3``, ``chatml``, …); empty keeps the template the model ships with.
Conversations become a ``text`` column. With ``train_on_responses_only`` the
user and system turns are masked out of the loss, Unsloth detecting the turn
markers from the template unless they are given.
"""

from nawat.lab import P

BLOCK = {"kind": "template", "summary": "Chat template + response-only loss for conversations",
         "backbones": ("sft", "dpo"), "models": ("llm",)}

PARAMS = {
    "chat_template": P("", "Unsloth template name (qwen3, llama-3.1, gemma-3, chatml…); empty = the model's own"),
    "train_on_responses_only": P(True, "Mask user/system turns out of the loss (SFT)"),
    "instruction_part": P("", "Marker that starts a user turn; empty = detect from the template"),
    "response_part": P("", "Marker that starts an assistant turn; empty = detect from the template"),
    "fallback_template": P("", "Jinja template to use if the tokenizer has none (DPO notebook: Zephyr's)"),
}


def apply(ctx, p, tokenizer, data):
    if p.chat_template:
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(tokenizer, chat_template=p.chat_template)
    elif not getattr(tokenizer, "chat_template", None) and p.fallback_template:
        tokenizer.chat_template = p.fallback_template
    if data.get("format") != "messages":
        return tokenizer, data  # DPO: TRL renders prompt / chosen / rejected with the template itself

    def render(batch):
        return {"text": [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]}

    for split in ("train", "eval"):
        if data.get(split) is not None:
            data[split] = data[split].map(render, batched=True, remove_columns=data[split].column_names)
    data["text_field"] = "text"
    return tokenizer, data


def wrap_trainer(ctx, p, trainer, tokenizer):
    if not p.train_on_responses_only:
        return trainer
    from unsloth.chat_templates import train_on_responses_only

    return train_on_responses_only(trainer, instruction_part=p.instruction_part or None,
                                   response_part=p.response_part or None)
