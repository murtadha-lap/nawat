"""Whisper speech recognition via FastModel, with LoRA on q/v projections.

Values from the Whisper notebook. ``task_type`` must stay None for Whisper.
"""

from nawat.lab import P

BLOCK = {"kind": "model", "model_type": "stt", "summary": "Whisper STT via FastModel + LoRA"}

PARAMS = {
    "whisper_language": P("English", "Language the tokenizer is set up for"),
    "whisper_task": P("transcribe", "transcribe or translate", choices=("transcribe", "translate")),
    "generation_language": P("<|en|>", "Language token forced at generation"),
    "load_in_4bit": P(False, "4-bit loading"),
    "full_finetuning": P(False, "Train every weight instead of LoRA adapters"),
    "r": P(64, "LoRA rank"),
    "lora_alpha": P(64, "LoRA alpha"),
    "lora_dropout": P(0.0, "LoRA dropout"),
    "target_modules": P(["q_proj", "v_proj"], "Adapted modules", type=list),
    "bias": P("none", "LoRA bias", choices=("none", "all", "lora_only")),
    "use_gradient_checkpointing": P("unsloth", "true, false or \"unsloth\"", type=str),
    "random_state": P(3407, "Seed for adapter initialisation"),
    "use_rslora": P(False, "Rank-stabilised LoRA"),
}


def load(ctx, p):
    from transformers import WhisperForConditionalGeneration
    from unsloth import FastModel

    return FastModel.from_pretrained(model_name=ctx.model_dir, dtype=None, load_in_4bit=p.load_in_4bit,
                                     full_finetuning=p.full_finetuning, auto_model=WhisperForConditionalGeneration,
                                     whisper_language=p.whisper_language, whisper_task=p.whisper_task)


def add_adapters(ctx, p, model):
    from unsloth import FastModel

    if not p.full_finetuning:
        model = FastModel.get_peft_model(
            model, r=p.r, target_modules=list(p.target_modules), lora_alpha=p.lora_alpha, lora_dropout=p.lora_dropout,
            bias=p.bias, random_state=p.random_state, use_rslora=p.use_rslora, loftq_config=None, task_type=None,
            use_gradient_checkpointing={"true": True, "false": False}.get(str(p.use_gradient_checkpointing).lower(), p.use_gradient_checkpointing),
        )
    model.generation_config.language = p.generation_language
    model.generation_config.task = p.whisper_task
    model.config.suppress_tokens = []
    model.generation_config.forced_decoder_ids = None
    return model


def generate(ctx, p, model, tokenizer, items, max_new_tokens=225, temperature=0.0, top_p=1.0, batch_size=8):
    """Transcribe items carrying {"audio": {"array", "sampling_rate"}}."""
    import torch

    outputs = []
    model.eval()
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            features = tokenizer.feature_extractor([it["audio"]["array"] for it in batch],
                                                   sampling_rate=batch[0]["audio"]["sampling_rate"], return_tensors="pt")
            inputs = features.input_features.to(model.device, dtype=next(model.parameters()).dtype)
            with torch.no_grad():
                generated = model.generate(input_features=inputs, max_new_tokens=max_new_tokens)
            outputs += tokenizer.batch_decode(generated, skip_special_tokens=True)
    finally:
        model.train()
    return outputs


def save(ctx, p, model, tokenizer, directory):
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
