"""Decision accuracy and calibration on the held-out decisions (FastDecisionModel.evaluate)."""

BLOCK = {"kind": "eval", "summary": "Typed-decision accuracy / calibration on the held-out share",
         "uses_training_data": True, "models": ("decision",)}

PARAMS = {}


def evaluate(ctx, p, model, tokenizer, data):
    from unsloth import FastDecisionModel

    if not data.get("eval"):
        return {"metrics": {}, "note": "nothing held out"}
    measured = FastDecisionModel.evaluate(model, tokenizer, data["eval"])
    return {"metrics": {k: v for k, v in measured.items() if isinstance(v, (int, float))}, "detail": measured}
