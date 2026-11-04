"""Evaluate a model as stored — no training — with the same eval blocks as a training run.

Fixed flow: load model → load data (if a dataset block is used) → chat template
(if a template block is used) → every eval block → every export block. No
adapter is added and no optimizer step is taken, so the numbers measure the
model input exactly as it was stored:

- ``models/<repo>`` — a base model, for a baseline;
- ``runs/<group>/<experiment>/adapter`` — the output of a finished run (after
  full fine-tuning that folder holds the full weights), to score it on an eval
  it was not planned with, or on general benchmarks to see what fine-tuning cost.

One preset per comparison, with the experiments differing only in ``--model``,
keeps every eval fingerprint identical, so ``nawat exp compare`` shows them side
by side. For Qwen3, set ``model.chat_template_kwargs = {enable_thinking = false}``
in that preset (and in the training runs you compare with) so the base model and
its fine-tuned versions answer under the same generation prompt.

Blocks written for any training method fit here: the dataset and template blocks
of the runs being compared, so a held-out eval sees the same split.
"""

BLOCK = {
    "kind": "backbone",
    "summary": "Evaluate a model as stored — no training — with the same eval blocks as a training run",
    "models": ("llm", "vlm"),
    "trains": False,
    "any_family": True,
    "slots": {"model": "required", "dataset": "optional", "template": "optional", "eval": "required",
              "export": "optional", "runtime": "optional"},
}

PARAMS = {}


def run(ctx, p):
    from nawat.lab import support

    model, tokenizer = ctx.block("model").call(ctx, "load")
    data = ctx.block("dataset").call(ctx, "load", model, tokenizer) if ctx.has("dataset") else {}
    if ctx.has("template"):
        tokenizer, data = ctx.block("template").call(ctx, "apply", tokenizer, data)
    ctx.log(f"no training: evaluating {ctx.model_dir} as stored")
    return support.finish(ctx, model, tokenizer, data)
