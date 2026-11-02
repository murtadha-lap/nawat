"""Reward: the completion follows the format exactly (Qwen3 GRPO notebook, match_format_exactly).

Full marks when the reasoning closes and the answer sits between the solution
tags at the end of the completion. Tags come from the template block.
"""

from nawat.lab import P

BLOCK = {"kind": "rewards", "summary": "Exact format: reasoning end, then <SOLUTION>…</SOLUTION> at the end",
         "backbones": ("grpo",)}

PARAMS = {
    "weight": P(1.0, "Weight of this reward in the total"),
    "score": P(3.0, "Reward when the format matches"),
}


def reward(ctx, p):
    from nawat.lab import support

    match_format, score = support.solution_pattern(ctx), p.score

    def match_format_exactly(completions, **kwargs):
        return [score if match_format.search(support.completion_text(c)) is not None else 0.0 for c in completions]

    return match_format_exactly
