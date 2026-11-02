"""Speech-to-text fine-tuning — Unsloth's Whisper notebook, step for step.

Fixed flow: load Whisper → add LoRA (task_type None, as Whisper requires) →
audio resampled to 16 kHz and turned into log-mel features by the dataset block
→ Seq2SeqTrainer with the notebook's speech collator, WER on the held-out split
while training → train → eval → export.

Source: unslothai/notebooks nb/Whisper.ipynb. The notebook computes WER with the
``evaluate`` package; here the same word error rate is computed directly, so
there is no extra dependency.
"""

from dataclasses import dataclass
from typing import Any

from nawat.lab import P

BLOCK = {
    "kind": "backbone",
    "summary": "STT: Whisper with Seq2SeqTrainer and live held-out WER (Whisper notebook)",
    "config_class": "transformers.Seq2SeqTrainingArguments",
    "models": ("stt",),
    "slots": {"model": "required", "dataset": "required", "eval": "optional", "export": "required", "runtime": "optional"},
}

PARAMS = {
    "per_device_train_batch_size": P(1, "Utterances per GPU per step"),
    "gradient_accumulation_steps": P(4, "Steps accumulated per optimizer update"),
    "warmup_steps": P(5, "Linear warm-up steps"),
    "max_steps": P(60, "Optimizer steps; -1 to train by epochs"),
    "num_train_epochs": P(1.0, "Epochs, used only when max_steps is -1"),
    "learning_rate": P(1e-4, "Peak learning rate"),
    "lr_scheduler_type": P("linear", "Learning-rate schedule"),
    "optim": P("adamw_8bit", "Optimizer"),
    "weight_decay": P(0.001, "Weight decay"),
    "max_grad_norm": P(1.0, "Gradient clipping norm"),
    "logging_steps": P(1, "Log metrics every N steps"),
    "eval_steps": P(5, "Teacher-forced WER on the held-out split every N steps; 0 = off"),
    "seed": P(3407, "Trainer seed"),
    "save_steps": P(250, "Write a checkpoint every N steps"),
    "save_total_limit": P(3, "Checkpoints kept on local disk"),
    "extra": P({}, "Any other Seq2SeqTrainingArguments field, as a table", type=dict),
}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """The notebook's collator, unchanged."""

    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def run(ctx, p):
    import numpy as np

    from nawat.lab import support

    model_block = ctx.block("model")
    model, tokenizer = model_block.call(ctx, "load")
    model = model_block.call(ctx, "add_adapters", model)
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer)

    def compute_metrics(pred):
        pred_logits = pred.predictions[0]
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": 100 * support.wer(pred_str, label_str)}

    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

    bf16 = support.bf16()
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
        fp16=not bf16,
        bf16=bf16,
        remove_unused_columns=False,  # the PeftModel forward does not show Whisper's signature
        label_names=["labels"],
        report_to="none",
        **ctx.checkpoint_args(p.save_steps, p.save_total_limit),
    )
    has_eval = bool(data.get("eval"))
    if has_eval and p.eval_steps > 0:
        values.update(eval_strategy="steps", eval_steps=p.eval_steps)
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data.get("eval") if has_eval else None,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=tokenizer),
        processing_class=tokenizer.feature_extractor,
        compute_metrics=compute_metrics if has_eval else None,
        callbacks=[ctx.metrics_callback()],
        args=support.build_config(Seq2SeqTrainingArguments, values, p.extra),
    )
    stats = support.train(ctx, trainer)
    return {"train": stats, **support.finish(ctx, model, tokenizer, data)}
