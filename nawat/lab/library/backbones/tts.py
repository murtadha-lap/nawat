"""Text-to-speech fine-tuning — Unsloth's Orpheus notebook, step for step.

Fixed flow: load the speech LLM → add LoRA → audio encoded into codec tokens by
the dataset block (SNAC for Orpheus) → transformers Trainer on the token
sequences → train → eval → export.

Source: unslothai/notebooks nb/Orpheus_(3B)-TTS.ipynb.
"""

from nawat.lab import P

BLOCK = {
    "kind": "backbone",
    "summary": "TTS: a speech LLM trained on codec tokens with transformers Trainer (Orpheus notebook)",
    "config_class": "transformers.TrainingArguments",
    "models": ("tts",),
    "slots": {"model": "required", "dataset": "required", "eval": "optional", "export": "required", "runtime": "optional"},
}

PARAMS = {
    "per_device_train_batch_size": P(1, "Utterances per GPU per step"),
    "gradient_accumulation_steps": P(4, "Steps accumulated per optimizer update"),
    "warmup_steps": P(5, "Linear warm-up steps"),
    "max_steps": P(60, "Optimizer steps; -1 to train by epochs"),
    "num_train_epochs": P(1.0, "Epochs, used only when max_steps is -1"),
    "learning_rate": P(2e-4, "Peak learning rate"),
    "lr_scheduler_type": P("linear", "Learning-rate schedule"),
    "optim": P("adamw_8bit", "Optimizer"),
    "weight_decay": P(0.001, "Weight decay"),
    "max_grad_norm": P(1.0, "Gradient clipping norm"),
    "logging_steps": P(1, "Log metrics every N steps"),
    "seed": P(3407, "Trainer seed"),
    "pad_batches": P(False, "Pad variable-length batches (needed only when batch size > 1; the notebook uses 1)"),
    "save_steps": P(250, "Write a checkpoint every N steps"),
    "save_total_limit": P(3, "Checkpoints kept on local disk"),
    "eval_steps": P(0, "Loss on the held-out utterances every N steps; 0 = not during training"),
    "extra": P({}, "Any other TrainingArguments field, as a table", type=dict),
}


def run(ctx, p):
    from nawat.lab import support

    model_block = ctx.block("model")
    model, tokenizer = model_block.call(ctx, "load")
    model = model_block.call(ctx, "add_adapters", model)
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer)

    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

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
        report_to="none",
        **ctx.checkpoint_args(p.save_steps, p.save_total_limit),
    )
    if data.get("eval") is not None and p.eval_steps > 0:
        values.update(eval_strategy="steps", eval_steps=p.eval_steps)
    trainer = Trainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data.get("eval"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True) if p.pad_batches else None,
        callbacks=[ctx.metrics_callback()],
        args=support.build_config(TrainingArguments, values, p.extra),
    )
    stats = support.train(ctx, trainer)
    return {"train": stats, **support.finish(ctx, model, tokenizer, data)}
