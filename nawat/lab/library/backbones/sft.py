"""Supervised fine-tuning — Unsloth's conversational and vision notebooks, step for step.

Fixed flow: load model → add LoRA → load data → chat template → SFTTrainer →
train on responses only (when the template says so) → train → eval → export.

Sources: unslothai/notebooks nb/Qwen3_(14B)-Reasoning-Conversational.ipynb and
nb/Qwen3_5_(0_8B)_Vision.ipynb. Every value those notebooks write inline is a
parameter below, with the notebook's value as its default.
"""

from nawat.lab import P

BLOCK = {
    "kind": "backbone",
    "summary": "SFT with TRL SFTTrainer, as in Unsloth's conversational and vision notebooks",
    "config_class": "trl.SFTConfig",
    "models": ("llm", "vlm"),
    "slots": {"model": "required", "dataset": "required", "template": "optional", "eval": "optional",
              "export": "required", "runtime": "optional"},
}

PARAMS = {
    "per_device_train_batch_size": P(2, "Samples per GPU per step"),
    "gradient_accumulation_steps": P(4, "Steps accumulated per optimizer update (effective batch = batch × this)"),
    "warmup_steps": P(5, "Linear warm-up steps"),
    "max_steps": P(30, "Optimizer steps to train; -1 to train by epochs"),
    "num_train_epochs": P(1.0, "Epochs, used only when max_steps is -1"),
    "learning_rate": P(2e-4, "Peak learning rate (the notebook suggests 2e-5 for long runs)"),
    "lr_scheduler_type": P("linear", "Learning-rate schedule"),
    "optim": P("adamw_8bit", "Optimizer"),
    "weight_decay": P(0.001, "Weight decay"),
    "max_grad_norm": P(1.0, "Gradient clipping norm"),
    "logging_steps": P(1, "Log metrics every N steps"),
    "seed": P(3407, "Trainer seed"),
    "max_length": P(2048, "Longest training sequence, in tokens"),
    "packing": P(False, "Pack short sequences into one"),
    "padding_free": P(False, "Padding-free batches (the notebook: true only above 17 GB VRAM)"),
    "save_steps": P(250, "Write a checkpoint every N steps (kept if the run fails, copied to object storage)"),
    "save_total_limit": P(3, "Checkpoints kept on local disk"),
    "eval_steps": P(0, "Loss on the held-out split every N steps; 0 = not during training"),
    "sample_every": P(25, "Trace: answer the probe inputs every N steps; 0 = off"),
    "sample_probes": P(4, "Trace: how many probe inputs"),
    "sample_max_new_tokens": P(256, "Trace: generation length for probes"),
    "sample_temperature": P(0.0, "Trace: temperature for probes (0 = greedy)"),
    "extra": P({}, "Any other SFTConfig field, as a table", type=dict),
}


def run(ctx, p):
    from nawat.lab import support

    model_block = ctx.block("model")
    model, tokenizer = model_block.call(ctx, "load")
    model = model_block.call(ctx, "add_adapters", model)
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer)
    if ctx.has("template"):
        tokenizer, data = ctx.block("template").call(ctx, "apply", tokenizer, data)
    hooks = (model_block.call(ctx, "trainer_hooks", model, tokenizer, "sft") if model_block.has("trainer_hooks") else None) or {}
    if model_block.has("for_training"):
        model_block.call(ctx, "for_training", model)

    from trl import SFTConfig, SFTTrainer

    values = dict(
        per_device_train_batch_size=p.per_device_train_batch_size,
        gradient_accumulation_steps=p.gradient_accumulation_steps,
        warmup_steps=p.warmup_steps,
        max_steps=p.max_steps,
        num_train_epochs=p.num_train_epochs,
        learning_rate=p.learning_rate,
        lr_scheduler_type=p.lr_scheduler_type,
        optim=p.optim,
        weight_decay=p.weight_decay,
        max_grad_norm=p.max_grad_norm,
        logging_steps=p.logging_steps,
        seed=p.seed,
        max_length=p.max_length,
        packing=p.packing,
        padding_free=p.padding_free,
        dataset_text_field=data.get("text_field", "text"),
        report_to="none",
        **ctx.checkpoint_args(p.save_steps, p.save_total_limit),
    )
    if data.get("eval") is not None and p.eval_steps > 0:
        values.update(eval_strategy="steps", eval_steps=p.eval_steps)
    values.update(hooks.get("config", {}))
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data.get("eval"),
        data_collator=hooks.get("data_collator"),
        callbacks=support.callbacks(ctx, model, tokenizer, data, p),
        args=support.build_config(SFTConfig, values, p.extra),
    )
    if ctx.has("template") and ctx.block("template").has("wrap_trainer"):
        trainer = ctx.block("template").call(ctx, "wrap_trainer", trainer, tokenizer)
    stats = support.train(ctx, trainer)
    return {"train": stats, **support.finish(ctx, model, tokenizer, data)}
