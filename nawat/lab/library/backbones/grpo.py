"""GRPO-family reinforcement learning — Unsloth's GRPO notebooks, step for step.

GRPO, DAPO, Dr. GRPO, BNPO and GSPO are this one backbone with different
parameters (``loss_type``, ``epsilon_high``, ``importance_sampling_level``,
``scale_rewards``, ``beta``), not different code — the presets ``llm-grpo``,
``llm-dapo``, ``llm-gspo`` and ``vlm-grpo`` name the combinations.

Fixed flow:

1. load model → add LoRA
2. load prompts (and an optional priming set) → reasoning template
3. *priming* (optional, as in the Qwen3 notebook): a short SFT pass so the model
   learns the answer format before it is rewarded for it
4. prompt-length cut: drop prompts above a length quantile and give the rest of
   the context to the completion (only when ``max_prompt_length`` is 0)
5. GRPOTrainer with one reward function per reward block — every completion of
   every round is written to ``traces/rollouts-*.jsonl`` with each reward, the
   weighted total and its group advantage
6. eval → export

Sources: unslothai/notebooks nb/Qwen3_(4B)-GRPO.ipynb and
nb/Qwen3_5_(4B)_Vision_GRPO.ipynb.

One deliberate difference: the Qwen3 notebook leaves ``loss_type`` at TRL's
default, which in TRL 0.22 is ``"dapo"``. Here it is always an explicit
parameter, so a run called GRPO and a run called DAPO cannot silently be the
same thing.
"""

from nawat.lab import P

BLOCK = {
    "kind": "backbone",
    "summary": "GRPO / DAPO / GSPO / Dr. GRPO with TRL GRPOTrainer, every rollout traced",
    "config_class": "trl.GRPOConfig",
    "models": ("llm", "vlm"),
    "slots": {"model": "required", "dataset": "required", "template": "optional", "rewards": "required",
              "eval": "optional", "export": "required", "runtime": "optional"},
}

PARAMS = {
    # the algorithm
    "loss_type": P("grpo", "grpo | dapo | dr_grpo | bnpo — how token losses are normalised",
                   choices=("grpo", "dapo", "dr_grpo", "bnpo")),
    "beta": P(0.0, "KL coefficient against the reference model (0 = no reference model loaded)"),
    "epsilon": P(0.2, "Lower clipping bound"),
    "epsilon_high": P(None, "Upper clipping bound; DAPO's clip-higher uses 0.28. None = same as epsilon", type=float),
    "importance_sampling_level": P("token", "token = GRPO, sequence = GSPO", choices=("token", "sequence")),
    "scale_rewards": P("group", "Advantage scaling: group std, batch std, or none (Dr. GRPO)",
                       choices=("group", "batch", "none")),
    "mask_truncated_completions": P(False, "Exclude cut-off completions from the loss (Unsloth recommends false: KL issues)"),
    "num_iterations": P(1, "Optimisation passes per generation batch (μ)"),
    # generation
    "num_generations": P(4, "Completions per prompt (group size G)"),
    "temperature": P(1.0, "Sampling temperature for rollouts"),
    "min_p": P(0.1, "Rollout sampling min_p"),
    "top_p": P(1.0, "Rollout sampling top_p"),
    "top_k": P(-1, "Rollout sampling top_k; -1 = off"),
    "max_prompt_length": P(0, "Longest prompt in tokens; 0 = measure the data and cut at prompt_length_quantile"),
    "max_completion_length": P(0, "Longest completion in tokens; 0 = the model's max_seq_length minus the prompt"),
    "prompt_length_quantile": P(0.9, "With automatic lengths: drop prompts longer than this quantile"),
    # optimisation
    "learning_rate": P(5e-6, "Peak learning rate"),
    "adam_beta1": P(0.9, "AdamW β1"),
    "adam_beta2": P(0.999, "AdamW β2 (the vision notebook uses 0.99)"),
    "weight_decay": P(0.001, "Weight decay"),
    "warmup_ratio": P(0.1, "Fraction of training spent warming up"),
    "lr_scheduler_type": P("linear", "Learning-rate schedule"),
    "optim": P("adamw_8bit", "Optimizer"),
    "max_grad_norm": P(1.0, "Gradient clipping norm (the vision notebook uses 0.1)"),
    "per_device_train_batch_size": P(1, "Prompts per GPU per step"),
    "gradient_accumulation_steps": P(1, "Steps accumulated per optimizer update (notebook: 4 for smoother training)"),
    "max_steps": P(100, "Optimizer steps"),
    "logging_steps": P(1, "Log metrics every N steps"),
    "seed": P(3407, "Trainer and sampling seed"),
    "save_steps": P(100, "Write a checkpoint every N steps"),
    "save_total_limit": P(3, "Checkpoints kept on local disk"),
    # priming (Qwen3 GRPO notebook, step 3)
    "prime_epochs": P(0.0, "Format-priming SFT epochs before GRPO; 0 = skip (the Qwen3 notebook uses 2)"),
    "prime_learning_rate": P(2e-4, "Priming learning rate"),
    "prime_batch_size": P(1, "Priming batch size"),
    "prime_gradient_accumulation_steps": P(1, "Priming gradient accumulation"),
    "prime_warmup_steps": P(5, "Priming warm-up steps"),
    "prime_weight_decay": P(0.001, "Priming weight decay"),
    "prime_logging_steps": P(5, "Priming: log every N steps"),
    # tracing
    "trace_max_mb": P(256.0, "Rotate rollout trace files at this size"),
    "extra": P({}, "Any other GRPOConfig field, as a table", type=dict),
}


def run(ctx, p):
    from nawat.lab import support

    model_block = ctx.block("model")
    model, tokenizer = model_block.call(ctx, "load")
    model = model_block.call(ctx, "add_adapters", model)
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer)
    if ctx.has("template"):
        tokenizer, data = ctx.block("template").call(ctx, "apply", tokenizer, data)

    if p.prime_epochs > 0 and data.get("prime") is not None:
        _prime(ctx, p, model, tokenizer, data["prime"])

    train = data["train"]
    max_seq_length = model_block.p.get("max_seq_length", 2048)
    max_prompt_length, max_completion_length = p.max_prompt_length, p.max_completion_length
    if max_prompt_length <= 0:
        train, max_prompt_length = _cut_long_prompts(ctx, tokenizer, train, p.prompt_length_quantile)
    if max_completion_length <= 0:
        max_completion_length = max_seq_length - max_prompt_length
    ctx.log(f"prompts: {len(train)} · max_prompt_length {max_prompt_length} · max_completion_length {max_completion_length}")

    reward_blocks = ctx.blocks("rewards")
    funcs = []
    for block in reward_blocks:
        fn = block.call(ctx, "reward")
        fn.__name__ = block.name.replace("-", "_")
        funcs.append(fn)
    weights = [float(block.p.get("weight", 1.0)) for block in reward_blocks]
    tracer = ctx.rollout_tracer(num_generations=p.num_generations, weights=weights,
                                scale_rewards=p.scale_rewards, max_mb=p.trace_max_mb)
    traced = tracer.wrap(funcs)

    from trl import GRPOConfig, GRPOTrainer

    values = dict(
        loss_type=p.loss_type,
        beta=p.beta,
        epsilon=p.epsilon,
        epsilon_high=p.epsilon_high,
        importance_sampling_level=p.importance_sampling_level,
        scale_rewards=p.scale_rewards,
        mask_truncated_completions=p.mask_truncated_completions,
        num_iterations=p.num_iterations,
        num_generations=p.num_generations,
        temperature=p.temperature,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        learning_rate=p.learning_rate,
        adam_beta1=p.adam_beta1,
        adam_beta2=p.adam_beta2,
        weight_decay=p.weight_decay,
        warmup_ratio=p.warmup_ratio,
        lr_scheduler_type=p.lr_scheduler_type,
        optim=p.optim,
        max_grad_norm=p.max_grad_norm,
        per_device_train_batch_size=p.per_device_train_batch_size,
        gradient_accumulation_steps=p.gradient_accumulation_steps,
        max_steps=p.max_steps,
        logging_steps=p.logging_steps,
        seed=p.seed,
        reward_weights=weights,
        log_completions=False,  # traced in full instead
        report_to="none",
        **ctx.checkpoint_args(p.save_steps, p.save_total_limit),
    )
    if model_block.p.get("fast_inference"):
        from vllm import SamplingParams

        values["vllm_sampling_params"] = SamplingParams(
            min_p=p.min_p, top_p=p.top_p, top_k=p.top_k, seed=p.seed,
            stop=[tokenizer.eos_token], include_stop_str_in_output=True,
        )
    else:
        values.update(min_p=p.min_p, top_p=p.top_p, top_k=None if p.top_k < 0 else p.top_k)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=traced,
        train_dataset=train,
        eval_dataset=data.get("eval"),
        callbacks=[ctx.metrics_callback()],
        args=support.build_config(GRPOConfig, values, p.extra),
    )
    stats = support.train(ctx, trainer)
    stats["rollout_rounds"] = tracer.round
    return {"train": stats, **support.finish(ctx, model, tokenizer, data)}


def _prime(ctx, p, model, tokenizer, dataset):
    """Step 3 of the Qwen3 GRPO notebook: teach the format before rewarding it."""
    import gc

    import torch
    from trl import SFTConfig, SFTTrainer

    ctx.log(f"priming on {len(dataset)} formatted examples for {p.prime_epochs} epoch(s)")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        callbacks=[ctx.metrics_callback()],
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=p.prime_batch_size,
            gradient_accumulation_steps=p.prime_gradient_accumulation_steps,
            warmup_steps=p.prime_warmup_steps,
            num_train_epochs=p.prime_epochs,
            learning_rate=p.prime_learning_rate,
            logging_steps=p.prime_logging_steps,
            optim=p.optim,
            weight_decay=p.prime_weight_decay,
            lr_scheduler_type=p.lr_scheduler_type,
            seed=p.seed,
            report_to="none",
            output_dir=str(ctx.path("prime", "trainer")),
            save_strategy="no",
        ),
    )
    trainer.train()
    ctx.log_metric(event="priming-done")
    del trainer
    torch.cuda.empty_cache()
    gc.collect()


def _cut_long_prompts(ctx, tokenizer, dataset, quantile):
    """Drop the longest prompts and size the prompt window to the rest (notebook step 4)."""
    import numpy as np

    def length(prompt):
        if isinstance(prompt, list):
            return len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True))
        return len(getattr(tokenizer, "tokenizer", tokenizer)(prompt)["input_ids"])

    lengths = np.array([length(x) for x in dataset["prompt"]])
    maximum = int(np.quantile(lengths, quantile))
    keep = np.where(lengths <= maximum)[0]
    ctx.log(f"prompt lengths: kept {len(keep)} of {len(lengths)} at ≤ {maximum} tokens (quantile {quantile})")
    return dataset.select(keep), maximum + 1
