"""Embedding-model fine-tuning — Unsloth's sentence-transformers notebook, step for step.

Fixed flow: load the embedding model → add LoRA (FEATURE_EXTRACTION) →
anchor / positive (/ negative) pairs → SentenceTransformerTrainer with a
contrastive loss and a no-duplicates batch sampler → train → eval → export.

Source: unslothai/notebooks nb/Qwen3_Embedding_(0_6B).ipynb.
"""

from nawat.lab import P

BLOCK = {
    "kind": "backbone",
    "summary": "Embeddings: SentenceTransformerTrainer with a contrastive loss (Qwen3-Embedding notebook)",
    "config_class": "sentence_transformers.SentenceTransformerTrainingArguments",
    "models": ("embedding",),
    "slots": {"model": "required", "dataset": "required", "eval": "optional", "export": "required", "runtime": "optional"},
}

PARAMS = {
    "loss": P("MultipleNegativesRankingLoss", "sentence-transformers loss",
              choices=("MultipleNegativesRankingLoss", "CachedMultipleNegativesRankingLoss",
                       "MultipleNegativesSymmetricRankingLoss", "CoSENTLoss", "AnglELoss", "TripletLoss")),
    "mini_batch_size": P(32, "Only for the Cached loss: GPU mini-batch size"),
    "batch_sampler": P("no_duplicates", "How batches are drawn", choices=("batch_sampler", "no_duplicates", "group_by_label")),
    "num_train_epochs": P(2.0, "Epochs"),
    "max_steps": P(-1, "Optimizer steps; -1 to train by epochs"),
    "per_device_train_batch_size": P(256, "Pairs per GPU per step (in-batch negatives grow with it)"),
    "gradient_accumulation_steps": P(1, "Steps accumulated per optimizer update"),
    "learning_rate": P(3e-5, "Peak learning rate"),
    "warmup_ratio": P(0.03, "Fraction of training spent warming up"),
    "lr_scheduler_type": P("constant_with_warmup", "Learning-rate schedule"),
    "optim": P("adamw_torch", "Optimizer (the notebook keeps the TrainingArguments default)"),
    "weight_decay": P(0.0, "Weight decay"),
    "logging_steps": P(50, "Log metrics every N steps"),
    "seed": P(42, "Trainer seed (TrainingArguments default; the notebook does not set one)"),
    "save_steps": P(500, "Write a checkpoint every N steps"),
    "save_total_limit": P(3, "Checkpoints kept on local disk"),
    "extra": P({}, "Any other SentenceTransformerTrainingArguments field, as a table", type=dict),
}


def run(ctx, p):
    from nawat.lab import support

    model_block = ctx.block("model")
    model, tokenizer = model_block.call(ctx, "load")
    model = model_block.call(ctx, "add_adapters", model)
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer)

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers

    loss_cls = getattr(losses, p.loss)
    loss = loss_cls(model, mini_batch_size=p.mini_batch_size) if p.loss.startswith("Cached") else loss_cls(model)
    bf16 = support.bf16()
    values = dict(
        num_train_epochs=p.num_train_epochs,
        max_steps=p.max_steps,
        per_device_train_batch_size=p.per_device_train_batch_size,
        gradient_accumulation_steps=p.gradient_accumulation_steps,
        learning_rate=p.learning_rate,
        warmup_ratio=p.warmup_ratio,
        lr_scheduler_type=p.lr_scheduler_type,
        optim=p.optim,
        weight_decay=p.weight_decay,
        logging_steps=p.logging_steps,
        seed=p.seed,
        fp16=not bf16,
        bf16=bf16,
        batch_sampler=BatchSamplers(p.batch_sampler),
        report_to="none",
        **ctx.checkpoint_args(p.save_steps, p.save_total_limit),
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data.get("eval"),
        loss=loss,
        callbacks=[ctx.metrics_callback()],
        args=support.build_config(SentenceTransformerTrainingArguments, values, p.extra),
    )
    stats = support.train(ctx, trainer)
    return {"train": stats, **support.finish(ctx, model, tokenizer, data)}
