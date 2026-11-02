"""Export merged 16-bit weights (base + adapter in one model), roughly the size of the base model.

Opt-in: vLLM serves the adapter directly, so a merged copy is for deployment
elsewhere. Published as ``runs/<group>/<experiment>/merged``.
"""

from nawat.lab import P

BLOCK = {"kind": "export", "summary": "Merged 16-bit model → runs/<experiment>/merged (large)"}

PARAMS = {
    "save_method": P("merged_16bit", "Unsloth save method", choices=("merged_16bit", "merged_4bit")),
}


def export(ctx, p, model, tokenizer):
    directory = ctx.out("merged")
    model_block = ctx.block("model")
    if model_block.has("save_merged"):
        model_block.call(ctx, "save_merged", model, tokenizer, directory)
    else:
        model.save_pretrained_merged(str(directory), tokenizer, save_method=p.save_method)
