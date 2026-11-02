"""A speech LLM (Orpheus-style: text in, SNAC codec tokens out) via FastLanguageModel.

Values from the Orpheus notebook: 16-bit LoRA, rank 64 on every projection.
"""

from nawat.lab import P

BLOCK = {"kind": "model", "model_type": "tts", "summary": "Orpheus-style TTS LLM via FastLanguageModel + LoRA"}

PARAMS = {
    "max_seq_length": P(2048, "Context length (text tokens + audio codes)"),
    "load_in_4bit": P(False, "4-bit QLoRA"),
    "full_finetuning": P(False, "Train every weight instead of LoRA adapters"),
    "r": P(64, "LoRA rank"),
    "lora_alpha": P(64, "LoRA alpha"),
    "lora_dropout": P(0.0, "LoRA dropout"),
    "target_modules": P(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "Adapted modules", type=list),
    "bias": P("none", "LoRA bias", choices=("none", "all", "lora_only")),
    "use_gradient_checkpointing": P("unsloth", "true, false or \"unsloth\"", type=str),
    "random_state": P(3407, "Seed for adapter initialisation"),
    "use_rslora": P(False, "Rank-stabilised LoRA"),
}


def load(ctx, p):
    from unsloth import FastLanguageModel

    return FastLanguageModel.from_pretrained(model_name=ctx.model_dir, max_seq_length=p.max_seq_length,
                                             dtype=None, load_in_4bit=p.load_in_4bit, full_finetuning=p.full_finetuning)


def add_adapters(ctx, p, model):
    from unsloth import FastLanguageModel

    if p.full_finetuning:
        return model
    return FastLanguageModel.get_peft_model(
        model, r=p.r, target_modules=list(p.target_modules), lora_alpha=p.lora_alpha, lora_dropout=p.lora_dropout,
        bias=p.bias, random_state=p.random_state, use_rslora=p.use_rslora, loftq_config=None,
        use_gradient_checkpointing={"true": True, "false": False}.get(str(p.use_gradient_checkpointing).lower(), p.use_gradient_checkpointing),
    )


def save(ctx, p, model, tokenizer, directory):
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
