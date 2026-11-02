"""A vision-language model through FastVisionModel, with LoRA on chosen parts.

The values are the Qwen3.5 vision notebook's: vision and language layers,
attention and MLP all trained, rank 16.
"""

from nawat.lab import P

BLOCK = {"kind": "model", "model_type": "vlm", "summary": "Vision-language model via FastVisionModel + LoRA"}

PARAMS = {
    "max_seq_length": P(2048, "Context length (text + image tokens)"),
    "load_in_4bit": P(False, "4-bit QLoRA; false for 16-bit LoRA"),
    "full_finetuning": P(False, "Train every weight instead of LoRA adapters"),
    "fast_inference": P(False, "vLLM generation inside training (GRPO)"),
    "finetune_vision_layers": P(True, "Adapt the vision encoder (the vision GRPO notebook: false)"),
    "finetune_language_layers": P(True, "Adapt the language model"),
    "finetune_attention_modules": P(True, "Adapt attention layers"),
    "finetune_mlp_modules": P(True, "Adapt MLP layers"),
    "r": P(16, "LoRA rank"),
    "lora_alpha": P(16, "LoRA alpha (at least the rank)"),
    "lora_dropout": P(0.0, "LoRA dropout"),
    "bias": P("none", "LoRA bias", choices=("none", "all", "lora_only")),
    "use_gradient_checkpointing": P("unsloth", "true, false or \"unsloth\"", type=str),
    "random_state": P(3407, "Seed for adapter initialisation"),
    "use_rslora": P(False, "Rank-stabilised LoRA"),
}


def vllm_compat():
    """vLLM 0.11 reads ``tokenizer.all_special_tokens_extended``, which transformers 5 removed.

    Puts the same tokens back under the old name, and only when it is missing, so
    nothing changes on library versions that still have it.
    """
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))


def load(ctx, p):
    from unsloth import FastVisionModel

    kwargs = dict(model_name=ctx.model_dir, max_seq_length=p.max_seq_length, load_in_4bit=p.load_in_4bit,
                  full_finetuning=p.full_finetuning,
                  use_gradient_checkpointing={"true": True, "false": False}.get(str(p.use_gradient_checkpointing).lower(), p.use_gradient_checkpointing))
    if p.fast_inference:
        vllm_compat()
        kwargs["fast_inference"] = True
    return FastVisionModel.from_pretrained(**kwargs)


def add_adapters(ctx, p, model):
    from unsloth import FastVisionModel

    if p.full_finetuning:
        return model
    return FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=p.finetune_vision_layers,
        finetune_language_layers=p.finetune_language_layers,
        finetune_attention_modules=p.finetune_attention_modules,
        finetune_mlp_modules=p.finetune_mlp_modules,
        r=p.r,
        lora_alpha=p.lora_alpha,
        lora_dropout=p.lora_dropout,
        bias=p.bias,
        random_state=p.random_state,
        use_rslora=p.use_rslora,
        loftq_config=None,
        use_gradient_checkpointing={"true": True, "false": False}.get(str(p.use_gradient_checkpointing).lower(), p.use_gradient_checkpointing),
    )


def trainer_hooks(ctx, p, model, tokenizer, method):
    """What SFT needs for images: the vision collator and an untouched dataset (notebook: "Must use!")."""
    if method != "sft":
        return {}
    from unsloth.trainer import UnslothVisionDataCollator

    return {
        "data_collator": UnslothVisionDataCollator(model, tokenizer),
        "config": {"remove_unused_columns": False, "dataset_text_field": "",
                   "dataset_kwargs": {"skip_prepare_dataset": True}},
    }


def for_training(ctx, p, model):
    from unsloth import FastVisionModel

    FastVisionModel.for_training(model)


def generate(ctx, p, model, tokenizer, items, max_new_tokens=256, temperature=0.0, top_p=1.0, batch_size=1):
    """One item at a time, as the notebook does: image + chat-templated text through the processor."""
    import torch
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    outputs = []
    try:
        for item in items:
            messages = item["input"]
            if isinstance(messages, str):
                content = ([{"type": "image"}] if item.get("image") is not None else []) + [{"type": "text", "text": messages}]
                messages = [{"role": "user", "content": content}]
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            image = item.get("image")
            inputs = (tokenizer(image, text, add_special_tokens=False, return_tensors="pt") if image is not None
                      else tokenizer(text=text, add_special_tokens=False, return_tensors="pt")).to(model.device)
            kwargs = {"max_new_tokens": max_new_tokens, "use_cache": True}
            kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False))
            with torch.no_grad():
                generated = model.generate(**inputs, **kwargs)
            width = inputs["input_ids"].shape[1]
            outputs.append(tokenizer.decode(generated[0][width:], skip_special_tokens=True))
    finally:
        FastVisionModel.for_training(model)
    return outputs


def save(ctx, p, model, tokenizer, directory):
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
