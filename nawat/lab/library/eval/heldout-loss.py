"""Held-out loss and perplexity, from the trainer itself.

Works for every trainer-based backbone, including the ones that do not
generate text (TTS). Depends on the training data split.
"""

BLOCK = {"kind": "eval", "summary": "Trainer loss and perplexity on the held-out split", "uses_training_data": True}

PARAMS = {}


def evaluate(ctx, p, model, tokenizer, data):
    import math

    trainer = ctx.state.get("trainer")
    if trainer is None or data.get("eval") is None:
        return {"metrics": {}, "note": "no held-out split; set dataset.eval_size"}
    measured = trainer.evaluate(eval_dataset=data["eval"])
    metrics = {k.replace("eval_", ""): v for k, v in measured.items() if isinstance(v, (int, float))}
    if "loss" in metrics:
        metrics["perplexity"] = math.exp(min(50.0, metrics["loss"]))
    return {"metrics": metrics}
