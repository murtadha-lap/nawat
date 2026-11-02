"""Reward: partial credit for each tag used exactly once (Qwen3 GRPO notebook, match_format_approximately)."""

from nawat.lab import P

BLOCK = {"kind": "rewards", "summary": "Approximate format: +/− for each closing tag present exactly once",
         "backbones": ("grpo",)}

PARAMS = {
    "weight": P(1.0, "Weight of this reward in the total"),
    "hit": P(0.5, "Added per tag that appears exactly once"),
    "miss": P(-1.0, "Added per tag that does not"),
}


def reward(ctx, p):
    from nawat.lab import support

    tags = ctx.state.get("tags") or {"reasoning_end": "<end_working_out>", "solution_start": "<SOLUTION>", "solution_end": "</SOLUTION>"}
    checks = (tags["reasoning_end"], tags["solution_start"], tags["solution_end"])
    hit, miss = p.hit, p.miss

    def match_format_approximately(completions, **kwargs):
        scores = []
        for completion in completions:
            response = support.completion_text(completion)
            scores.append(sum(hit if response.count(tag) == 1 else miss for tag in checks))
        return scores

    return match_format_approximately
