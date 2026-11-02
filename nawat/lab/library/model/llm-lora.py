"""A text LLM through FastLanguageModel, with LoRA (or full fine-tuning).

Loads from the staged model directory — the experiment's model input — never
from the hub by name. Every argument the Unsloth notebooks pass to
``from_pretrained`` and ``get_peft_model`` is a parameter here.
"""

from nawat.lab import P

BLOCK = {"kind": "model", "model_type": "llm", "summary": "Text LLM via FastLanguageModel + LoRA"}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

PARAMS = {
    "max_seq_length": P(2048, "Context length to train with"),
    "load_in_4bit": P(False, "4-bit QLoRA (less memory, slightly lower quality)"),
    "load_in_8bit": P(False, "8-bit loading"),
    "full_finetuning": P(False, "Train every weight instead of LoRA adapters"),
    "fast_inference": P(False, "vLLM generation inside training (GRPO); needs vllm in the training environment"),
    "max_lora_rank": P(32, "With fast_inference: the largest LoRA rank vLLM must accept"),
    "gpu_memory_utilization": P(0.9, "With fast_inference: share of VRAM vLLM may take"),
    "r": P(32, "LoRA rank"),
    "lora_alpha": P(32, "LoRA alpha (rank or 2× rank)"),
    "lora_dropout": P(0.0, "LoRA dropout (0 is optimised)"),
    "target_modules": P(LORA_TARGETS, "Modules that get adapters; add embed_tokens, lm_head for continued pretraining", type=list),
    "bias": P("none", "LoRA bias", choices=("none", "all", "lora_only")),
    "use_gradient_checkpointing": P("unsloth", "true, false or \"unsloth\" (long context)", type=str),
    "random_state": P(3407, "Seed for adapter initialisation"),
    "use_rslora": P(False, "Rank-stabilised LoRA (the CPT notebook uses true)"),
}


def _checkpointing(value):
    return {"true": True, "false": False}.get(str(value).lower(), value)


def vllm_compat():
    """vLLM 0.11 reads ``tokenizer.all_special_tokens_extended``, which transformers 5 removed.

    Puts the same tokens back under the old name, and only when it is missing, so
    nothing changes on library versions that still have it.
    """
    from transformers import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(self.all_special_tokens))


def load(ctx, p):
    from unsloth import FastLanguageModel

    kwargs = dict(
        model_name=ctx.model_dir,
        max_seq_length=p.max_seq_length,
        load_in_4bit=p.load_in_4bit,
        load_in_8bit=p.load_in_8bit,
        full_finetuning=p.full_finetuning,
        use_gradient_checkpointing=_checkpointing(p.use_gradient_checkpointing),
        random_state=p.random_state,
    )
    if p.fast_inference:
        vllm_compat()
        kwargs.update(fast_inference=True, max_lora_rank=p.max_lora_rank, gpu_memory_utilization=p.gpu_memory_utilization)
    return FastLanguageModel.from_pretrained(**kwargs)


def add_adapters(ctx, p, model):
    from unsloth import FastLanguageModel

    if p.full_finetuning:
        return model
    return FastLanguageModel.get_peft_model(
        model,
        r=p.r,
        target_modules=list(p.target_modules),
        lora_alpha=p.lora_alpha,
        lora_dropout=p.lora_dropout,
        bias=p.bias,
        use_gradient_checkpointing=_checkpointing(p.use_gradient_checkpointing),
        random_state=p.random_state,
        use_rslora=p.use_rslora,
        loftq_config=None,
    )


def for_training(ctx, p, model):
    from unsloth import FastLanguageModel

    FastLanguageModel.for_training(model)


def generate(ctx, p, model, tokenizer, items, max_new_tokens=256, temperature=0.0, top_p=1.0, batch_size=8):
    from unsloth import FastLanguageModel

    from nawat.lab import support

    FastLanguageModel.for_inference(model)
    try:
        return support.causal_generate(model, tokenizer, items, max_new_tokens=max_new_tokens,
                                       temperature=temperature, top_p=top_p, batch_size=batch_size)
    finally:
        FastLanguageModel.for_training(model)


def score(ctx, p, model, tokenizer, context, choices):
    from unsloth import FastLanguageModel

    from nawat.lab import support

    FastLanguageModel.for_inference(model)
    try:
        return support.choice_logprobs(model, tokenizer, context, choices)
    finally:
        FastLanguageModel.for_training(model)


def save(ctx, p, model, tokenizer, directory):
    model.save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
