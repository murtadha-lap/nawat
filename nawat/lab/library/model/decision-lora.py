"""A typed decision model via FastDecisionModel, with LoRA.

A plain language model (Qwen3.5, …) gets a fresh Clef decision head; an
existing decision checkpoint (Laya, Clef) loads as it is. Values from the
decision notebook: 4-bit, rank 16.
"""

from nawat.lab import P

BLOCK = {"kind": "model", "model_type": "decision", "summary": "Typed decision model via FastDecisionModel + LoRA"}

PARAMS = {
    "max_seq_length": P(2048, "Longest input; long inputs keep the question and options, the end of the state is cut"),
    "load_in_4bit": P(True, "4-bit quantisation (Laya checkpoints train in 16-bit only)"),
    "full_finetuning": P(False, "Train every weight instead of LoRA adapters"),
    "r": P(16, "LoRA rank"),
    "lora_alpha": P(16, "LoRA alpha"),
    "lora_dropout": P(0.0, "LoRA dropout"),
    "use_gradient_checkpointing": P("unsloth", "true, false or \"unsloth\"", type=str),
    "random_state": P(3407, "Seed for adapter and head initialisation"),
}


def load(ctx, p):
    from unsloth import FastDecisionModel

    return FastDecisionModel.from_pretrained(model_name=ctx.model_dir, max_seq_length=p.max_seq_length,
                                             load_in_4bit=p.load_in_4bit, full_finetuning=p.full_finetuning,
                                             random_state=p.random_state)


def add_adapters(ctx, p, model):
    from unsloth import FastDecisionModel

    if p.full_finetuning:
        return model
    return FastDecisionModel.get_peft_model(
        model, r=p.r, lora_alpha=p.lora_alpha, lora_dropout=p.lora_dropout, random_state=p.random_state,
        use_gradient_checkpointing={"true": True, "false": False}.get(str(p.use_gradient_checkpointing).lower(), p.use_gradient_checkpointing),
    )


def save(ctx, p, model, tokenizer, directory):
    model.save_pretrained(str(directory))


def save_merged(ctx, p, model, tokenizer, directory):
    model.save_pretrained_merged(str(directory), tokenizer, save_method="merged_16bit")
