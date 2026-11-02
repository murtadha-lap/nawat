"""A sentence-embedding model via FastSentenceTransformer, with LoRA.

Values from the Qwen3-Embedding notebook. The tokenizer is the model's own
(``model.tokenizer``); gradient checkpointing is off because it conflicts with
torch.compile here.
"""

from nawat.lab import P

BLOCK = {"kind": "model", "model_type": "embedding", "summary": "Embedding model via FastSentenceTransformer + LoRA"}

PARAMS = {
    "max_seq_length": P(512, "Longest input, in tokens"),
    "full_finetuning": P(False, "Train every weight instead of LoRA adapters"),
    "r": P(32, "LoRA rank"),
    "lora_alpha": P(32, "LoRA alpha"),
    "lora_dropout": P(0.0, "LoRA dropout"),
    "target_modules": P(["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "Adapted modules", type=list),
    "bias": P("none", "LoRA bias", choices=("none", "all", "lora_only")),
    "use_gradient_checkpointing": P(False, "Gradient checkpointing (conflicts with torch.compile)", type=bool),
    "random_state": P(3407, "Seed for adapter initialisation"),
    "use_rslora": P(False, "Rank-stabilised LoRA"),
}


def load(ctx, p):
    from unsloth import FastSentenceTransformer

    model = FastSentenceTransformer.from_pretrained(model_name=ctx.model_dir, max_seq_length=p.max_seq_length,
                                                    full_finetuning=p.full_finetuning)
    return model, model.tokenizer


def add_adapters(ctx, p, model):
    from unsloth import FastSentenceTransformer

    if p.full_finetuning:
        return model
    return FastSentenceTransformer.get_peft_model(
        model, r=p.r, target_modules=list(p.target_modules), lora_alpha=p.lora_alpha, lora_dropout=p.lora_dropout,
        bias=p.bias, use_gradient_checkpointing=p.use_gradient_checkpointing, random_state=p.random_state,
        use_rslora=p.use_rslora, loftq_config=None, task_type="FEATURE_EXTRACTION",
    )


def encode(ctx, p, model, texts, batch_size=64):
    import torch

    with torch.inference_mode():
        return model.encode(list(texts), batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)


def save(ctx, p, model, tokenizer, directory):
    model.save_pretrained(str(directory))
    model.tokenizer.save_pretrained(str(directory))


def save_merged(ctx, p, model, tokenizer, directory):
    model.save_pretrained_merged(str(directory), tokenizer=model.tokenizer, save_method="merged_16bit")
