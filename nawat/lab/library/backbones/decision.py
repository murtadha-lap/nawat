"""Typed decision models — Unsloth's decision notebook, step for step.

A decision model reads an input (the *state*) and answers declared questions
with typed answers: a ``choice`` among named options, a ``noul`` yes/no, or a
``score`` on an ordered scale — with calibrated probabilities over every
option. The open alternative to hosted decision APIs, trained on your own
decisions.

Fixed flow: load (a plain LM gets a fresh Clef decision head) → add LoRA →
rows {state, questions, gold} built into decision items, a held-out share kept
back → DecisionTrainer → temperature calibration on the held-out items → eval →
export.

Source: unslothai/notebooks nb/Qwen3_5_(4B)-Decision.ipynb.
"""

from nawat.lab import P

BLOCK = {
    "kind": "backbone",
    "summary": "Typed decisions (choice / yes-no / score) with DecisionTrainer and calibration",
    "config_class": "transformers.TrainingArguments",
    "models": ("decision",),
    "slots": {"model": "required", "dataset": "required", "eval": "optional", "export": "required", "runtime": "optional"},
}

PARAMS = {
    "per_device_train_batch_size": P(8, "Decision rows per GPU per step"),
    "gradient_accumulation_steps": P(4, "Steps accumulated per optimizer update"),
    "warmup_steps": P(10, "Linear warm-up steps"),
    "max_steps": P(60, "Optimizer steps; -1 to train by epochs"),
    "num_train_epochs": P(1.0, "Epochs, used only when max_steps is -1"),
    "learning_rate": P(2e-4, "Learning rate for the LoRA weights"),
    "head_learning_rate": P(None, "Learning rate for the decision head; None = Unsloth's default (1e-4)", type=float),
    "lr_scheduler_type": P("cosine", "Learning-rate schedule"),
    "weight_decay": P(0.01, "Weight decay"),
    "optim": P("adamw_torch", "Optimizer (TrainingArguments default; the notebook does not set one)"),
    "label_smoothing": P(None, "Label smoothing; None = the model's recipe", type=float),
    "kl_weight": P(0.0, "KL penalty to the starting model, against forgetting (Clef + LoRA only)"),
    "eval_strategy": P("epoch", "When to score the held-out decisions during training", choices=("no", "steps", "epoch")),
    "eval_steps": P(0, "With eval_strategy=steps: every N steps"),
    "logging_steps": P(10, "Log metrics every N steps"),
    "seed": P(3407, "Trainer seed"),
    "calibrate": P(True, "Fit answer temperatures on the held-out decisions after training"),
    "save_steps": P(250, "Write a checkpoint every N steps"),
    "save_total_limit": P(3, "Checkpoints kept on local disk"),
    "extra": P({}, "Any other TrainingArguments field, as a table", type=dict),
}


def run(ctx, p):
    from nawat.lab import support

    model_block = ctx.block("model")
    model, tokenizer = model_block.call(ctx, "load")
    model = model_block.call(ctx, "add_adapters", model)
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer)

    from transformers import TrainingArguments
    from unsloth import DecisionTrainer, FastDecisionModel

    bf16 = support.bf16()
    values = dict(
        per_device_train_batch_size=p.per_device_train_batch_size,
        gradient_accumulation_steps=p.gradient_accumulation_steps,
        warmup_steps=p.warmup_steps,
        max_steps=p.max_steps,
        num_train_epochs=p.num_train_epochs,
        learning_rate=p.learning_rate,
        lr_scheduler_type=p.lr_scheduler_type,
        weight_decay=p.weight_decay,
        optim=p.optim,
        bf16=bf16,
        fp16=not bf16,
        eval_strategy=p.eval_strategy if data.get("eval") else "no",
        logging_steps=p.logging_steps,
        seed=p.seed,
        report_to="none",
        **ctx.checkpoint_args(p.save_steps, p.save_total_limit),
    )
    if p.eval_strategy == "steps" and p.eval_steps > 0:
        values["eval_steps"] = p.eval_steps
    trainer_kwargs = {"kl_weight": p.kl_weight}
    if p.head_learning_rate is not None:
        trainer_kwargs["head_learning_rate"] = p.head_learning_rate
    if p.label_smoothing is not None:
        trainer_kwargs["label_smoothing"] = p.label_smoothing
    trainer = DecisionTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data.get("eval") or None,
        callbacks=[ctx.metrics_callback()],
        args=support.build_config(TrainingArguments, values, p.extra),
        **trainer_kwargs,
    )
    stats = support.train(ctx, trainer)
    if p.calibrate and data.get("eval"):
        calibration = FastDecisionModel.calibrate(model, tokenizer, data["eval"])
        ctx.write_json(f"eval/calibration-{ctx.run_id}.json", calibration)
        stats["calibrated"] = True
    return {"train": stats, **support.finish(ctx, model, tokenizer, data)}
