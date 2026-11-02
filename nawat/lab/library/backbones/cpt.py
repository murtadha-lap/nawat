"""Continued pretraining — Unsloth's CPT notebook, step for step.

Fixed flow: load model → add LoRA (with embed_tokens and lm_head) → raw text
corpus → UnslothTrainer with a separate, smaller embedding learning rate →
train → eval → export.

Source: unslothai/notebooks nb/Mistral_v0.3_(7B)-CPT.ipynb.
"""

from nawat.lab import P

BLOCK = {
    "kind": "backbone",
    "summary": "Continued pretraining with UnslothTrainer and a separate embedding learning rate",
    "config_class": "unsloth.UnslothTrainingArguments",
    "models": ("llm",),
    "slots": {"model": "required", "dataset": "required", "eval": "optional", "export": "required", "runtime": "optional"},
}

PARAMS = {
    "per_device_train_batch_size": P(2, "Samples per GPU per step"),
    "gradient_accumulation_steps": P(8, "Steps accumulated per optimizer update"),
    "warmup_steps": P(10, "Linear warm-up steps"),
    "max_steps": P(120, "Optimizer steps; -1 to train by epochs"),
    "num_train_epochs": P(1.0, "Epochs, used only when max_steps is -1"),
    "learning_rate": P(5e-5, "Learning rate for the LoRA weights"),
    "embedding_learning_rate": P(1e-5, "Learning rate for embed_tokens / lm_head (2–10× smaller)"),
    "lr_scheduler_type": P("linear", "Learning-rate schedule"),
    "optim": P("adamw_8bit", "Optimizer"),
    "weight_decay": P(0.001, "Weight decay"),
    "max_grad_norm": P(1.0, "Gradient clipping norm"),
    "logging_steps": P(1, "Log metrics every N steps"),
    "seed": P(3407, "Trainer seed"),
    "max_length": P(2048, "Longest training sequence, in tokens"),
    "dataset_num_proc": P(4, "Processes used to tokenize the corpus"),
    "packing": P(False, "Pack short documents into one sequence"),
    "save_steps": P(250, "Write a checkpoint every N steps"),
    "save_total_limit": P(3, "Checkpoints kept on local disk"),
    "eval_steps": P(0, "Loss on the held-out split every N steps; 0 = not during training"),
    "sample_every": P(0, "Trace: continue the probe texts every N steps; 0 = off"),
    "sample_probes": P(4, "Trace: how many probe texts"),
    "sample_max_new_tokens": P(128, "Trace: generation length"),
    "sample_temperature": P(0.0, "Trace: temperature (0 = greedy)"),
    "extra": P({}, "Any other UnslothTrainingArguments field, as a table", type=dict),
}


def run(ctx, p):
    from nawat.lab import support

    model_block = ctx.block("model")
    model, tokenizer = model_block.call(ctx, "load")
    model = model_block.call(ctx, "add_adapters", model)
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer)

    from unsloth import UnslothTrainer, UnslothTrainingArguments

    values = dict(
        per_device_train_batch_size=p.per_device_train_batch_size,
        gradient_accumulation_steps=p.gradient_accumulation_steps,
        warmup_steps=p.warmup_steps,
        max_steps=p.max_steps,
        num_train_epochs=p.num_train_epochs,
        learning_rate=p.learning_rate,
        embedding_learning_rate=p.embedding_learning_rate,
        lr_scheduler_type=p.lr_scheduler_type,
        optim=p.optim,
        weight_decay=p.weight_decay,
        max_grad_norm=p.max_grad_norm,
        logging_steps=p.logging_steps,
        seed=p.seed,
        max_length=p.max_length,
        dataset_num_proc=p.dataset_num_proc,
        packing=p.packing,
        dataset_text_field=data.get("text_field", "text"),
        report_to="none",
        **ctx.checkpoint_args(p.save_steps, p.save_total_limit),
    )
    if data.get("eval") is not None and p.eval_steps > 0:
        values.update(eval_strategy="steps", eval_steps=p.eval_steps)
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data.get("eval"),
        callbacks=support.callbacks(ctx, model, tokenizer, data, p),
        args=support.build_config(UnslothTrainingArguments, values, p.extra),
    )
    stats = support.train(ctx, trainer)
    return {"train": stats, **support.finish(ctx, model, tokenizer, data)}
