"""<What this reward scores, and why.>

``reward(ctx, p)`` returns a function TRL calls with the batch:
``fn(prompts, completions, **columns)`` → one float per completion. Dataset
columns (e.g. ``answer``) arrive as keyword arguments. Every score is a
parameter, so changing it is a new preset, not an edit.
"""

from nawat.lab import P

BLOCK = {"kind": "rewards", "summary": "<one line>", "backbones": ("grpo",)}

PARAMS = {
    "weight": P(1.0, "Weight of this reward in the total"),
    "correct": P(1.0, "Reward when …"),
    "incorrect": P(0.0, "Reward otherwise"),
}


def reward(ctx, p):
    from nawat.lab import support

    correct, incorrect = p.correct, p.incorrect

    def my_reward(prompts, completions, answer, **kwargs):
        return [correct if str(a) in support.completion_text(c) else incorrect for c, a in zip(completions, answer)]

    return my_reward
