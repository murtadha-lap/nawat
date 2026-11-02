"""Direct preference optimisation — Unsloth's DPO notebook, step for step.

Fixed flow: patch DPOTrainer → load model → add LoRA → prompt / chosen /
rejected pairs → chat template → DPOTrainer with no separate reference model
(the adapter-free base is the reference) → train → eval → export.

Source: unslothai/notebooks nb/Zephyr_(7B)-DPO.ipynb.
"""

from nawat.lab import P

BLOCK = {
    "kind": "backbone",
    "summary": "Preference tuning with TRL DPOTrainer, as in Unsloth's DPO notebook",
    "config_class": "trl.DPOConfig",
    "models": ("llm", "vlm"),
    "slots": {"model": "required", "dataset": "required", "template": "optional", "eval": "optional",
              "export": "required", "runtime": "optional"},
}

PARAMS = {
    "beta": P(0.1, "Strength of the pull towards the reference model"),
    "loss_type": P("sigmoid", "DPO loss variant", choices=("sigmoid", "hinge", "ipo", "exo_pair", "nca_pair", "robust",
                                                        "bco_pair", "sppo_hard", "aot", "aot_pair", "apo_zero", "apo_down")),
    "max_length": P(1024, "Longest prompt + response, in tokens"),
    "max_prompt_length": P(512, "Longest prompt, in tokens"),
    "per_device_train_batch_size": P(2, "Pairs per GPU per step"),
    "gradient_accumulation_steps": P(4, "Steps accumulated per optimizer update"),
    "warmup_ratio": P(0.1, "Fraction of training spent warming up"),
    "num_train_epochs": P(3.0, "Epochs, used when max_steps is -1"),
    "max_steps": P(-1, "Optimizer steps; -1 to train by epochs"),
    "learning_rate": P(5e-6, "Peak learning rate"),
    "lr_scheduler_type": P("linear", "Learning-rate schedule"),
    "optim": P("adamw_8bit", "Optimizer"),
    "weight_decay": P(0.0, "Weight decay"),
    "max_grad_norm": P(1.0, "Gradient clipping norm"),
    "logging_steps": P(1, "Log metrics every N steps"),
    "seed": P(42, "Trainer seed (the notebook uses 42)"),
    "save_steps": P(250, "Write a checkpoint every N steps"),
    "save_total_limit": P(3, "Checkpoints kept on local disk"),
    "eval_steps": P(0, "DPO metrics on the held-out pairs every N steps; 0 = not during training"),
    "sample_every": P(25, "Trace: answer the probe prompts every N steps; 0 = off"),
    "sample_probes": P(4, "Trace: how many probe prompts"),
    "sample_max_new_tokens": P(256, "Trace: generation length"),
    "sample_temperature": P(0.0, "Trace: temperature (0 = greedy)"),
    "extra": P({}, "Any other DPOConfig field, as a table", type=dict),
}


def run(ctx, p):
    from unsloth import PatchDPOTrainer

    PatchDPOTrainer()
    from nawat.lab import support

    model_block = ctx.block("model")
    model, tokenizer = model_block.call(ctx, "load")
    model = model_block.call(ctx, "add_adapters", model)
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer)
    if ctx.has("template"):
        tokenizer, data = ctx.block("template").call(ctx, "apply", tokenizer, data)

    from trl import DPOConfig, DPOTrainer

    values = dict(
        beta=p.beta,
        loss_type=p.loss_type,
        max_length=p.max_length,
        max_prompt_length=p.max_prompt_length,
        per_device_train_batch_size=p.per_device_train_batch_size,
        gradient_accumulation_steps=p.gradient_accumulation_steps,
        warmup_ratio=p.warmup_ratio,
        num_train_epochs=p.num_train_epochs,
        max_steps=p.max_steps,
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
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=data["train"],
        eval_dataset=data.get("eval"),
        callbacks=support.callbacks(ctx, model, tokenizer, data, p),
        args=support.build_config(DPOConfig, values, p.extra),
    )
    stats = support.train(ctx, trainer)
    return {"train": stats, **support.finish(ctx, model, tokenizer, data)}
