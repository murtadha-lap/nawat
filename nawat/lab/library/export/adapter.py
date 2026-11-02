"""Export the trained adapter (or full weights, after full fine-tuning) and its tokenizer.

Published as ``runs/<group>/<experiment>/adapter``. vLLM serves it on top of
the base model without merging — ``nawat serve`` / ``nawat adapter``.
"""

BLOCK = {"kind": "export", "summary": "LoRA adapter + tokenizer → runs/<experiment>/adapter"}

PARAMS = {}


def export(ctx, p, model, tokenizer):
    directory = ctx.out("adapter")
    model_block = ctx.block("model")
    if model_block.has("save"):
        model_block.call(ctx, "save", model, tokenizer, directory)
    else:
        model.save_pretrained(str(directory))
        tokenizer.save_pretrained(str(directory))
    ctx.log(f"adapter written to {directory}")
