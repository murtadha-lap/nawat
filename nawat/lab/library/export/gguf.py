"""Export GGUF for llama.cpp / Ollama.

A failed conversion — most often an architecture the converter does not know,
vision encoders especially — is logged and the run still succeeds with its
other exports, rather than a finished training being thrown away.
"""

from nawat.lab import P

BLOCK = {"kind": "export", "summary": "GGUF for llama.cpp / Ollama → runs/<experiment>/gguf"}

PARAMS = {
    "quantization_method": P("q4_k_m", "GGUF quantisation (q4_k_m, q5_k_m, q8_0, f16, …)"),
}


def export(ctx, p, model, tokenizer):
    directory = ctx.out("gguf")
    try:
        model.save_pretrained_gguf(str(directory), tokenizer, quantization_method=p.quantization_method)
    except Exception as exc:  # noqa: BLE001 - a failed export is not a failed run
        ctx.log(f"gguf conversion failed, publishing without it: {type(exc).__name__}: {exc}")
