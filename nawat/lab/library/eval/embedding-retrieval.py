"""Retrieval on held-out pairs: does each query find its own positive among all of them?

Recall@1/5/10 and MRR over cosine similarity of normalised embeddings. Depends
on the training data split.
"""

from nawat.lab import P

BLOCK = {"kind": "eval", "summary": "Recall@k and MRR on held-out query / positive pairs",
         "uses_training_data": True, "models": ("embedding",)}

PARAMS = {
    "max_items": P(1000, "Use at most N held-out pairs (0 = all)"),
    "batch_size": P(64, "Encoding batch size"),
}


def evaluate(ctx, p, model, tokenizer, data):
    import torch

    pairs = list(data.get("heldout") or [])
    if p.max_items > 0:
        pairs = pairs[: p.max_items]
    if not pairs:
        return {"metrics": {"n": 0}, "note": "no held-out pairs; set dataset.eval_size"}
    encode = ctx.block("model")
    queries = encode.call(ctx, "encode", model, [x["query"] for x in pairs], batch_size=p.batch_size)
    docs = encode.call(ctx, "encode", model, [x["positive"] for x in pairs], batch_size=p.batch_size)
    sims = queries @ docs.T
    gold = torch.arange(len(pairs), device=sims.device)
    ranks = (sims > sims[gold, gold].unsqueeze(1)).sum(dim=1) + 1
    metrics = {"n": len(pairs), "mrr": float((1.0 / ranks.float()).mean())}
    for k in (1, 5, 10):
        metrics[f"recall@{k}"] = float((ranks <= k).float().mean())
    return {"metrics": metrics}
