"""<Which models this loads and how they are adapted.>"""

from nawat.lab import P

BLOCK = {"kind": "model", "model_type": "llm", "summary": "<one line>"}

PARAMS = {
    "max_seq_length": P(2048, "Context length"),
    "r": P(16, "LoRA rank"),
    "lora_alpha": P(16, "LoRA alpha"),
}


def load(ctx, p):
    from unsloth import FastLanguageModel

    return FastLanguageModel.from_pretrained(model_name=ctx.model_dir, max_seq_length=p.max_seq_length)


def add_adapters(ctx, p, model):
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(model, r=p.r, lora_alpha=p.lora_alpha)
