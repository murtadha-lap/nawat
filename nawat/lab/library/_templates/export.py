"""<What this export writes.> Whatever goes under ctx.out(<name>) is published as runs/<experiment>/<name>."""

BLOCK = {"kind": "export", "summary": "<one line>"}

PARAMS = {}


def export(ctx, p, model, tokenizer):
    directory = ctx.out("my-export")
    model.save_pretrained(str(directory))
