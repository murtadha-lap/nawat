"""<How prompts are formatted for the model.>"""

BLOCK = {"kind": "template", "summary": "<one line>", "backbones": ("sft",)}

PARAMS = {}


def apply(ctx, p, tokenizer, data):
    return tokenizer, data
