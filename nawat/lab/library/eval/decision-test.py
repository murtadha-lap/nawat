"""Decision accuracy on a separate test split of the training dataset (the notebook's final step)."""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "Typed-decision accuracy on the dataset's test split",
         "uses_training_data": True, "models": ("decision",)}

PARAMS = {
    "split": P("test", "Split to score"),
    "subset": P("all", "Dataset configuration"),
}


def evaluate(ctx, p, model, tokenizer, data):
    from unsloth import FastDecisionModel

    from nawat.lab import support

    test = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    items, report = FastDecisionModel.build_dataset(test, tokenizer, model)
    measured = FastDecisionModel.evaluate(model, tokenizer, items)
    return {"metrics": {k: v for k, v in measured.items() if isinstance(v, (int, float))}, "detail": measured,
            "build": report}
