"""Reward: a penalty for completions longer than a budget.

Not from an Unsloth notebook — a common addition against runaway reasoning.
Zero up to ``target_chars``, then falling linearly to ``penalty`` at
``max_chars`` and beyond.
"""

from nawat.lab import P

BLOCK = {"kind": "rewards", "summary": "Linear penalty past a length budget (not from a notebook)", "backbones": ("grpo",)}

PARAMS = {
    "weight": P(1.0, "Weight of this reward in the total"),
    "target_chars": P(2000, "No penalty up to this many characters"),
    "max_chars": P(6000, "Full penalty from this many characters"),
    "penalty": P(-1.0, "Reward at and beyond max_chars"),
}


def reward(ctx, p):
    from nawat.lab import support

    target, maximum, penalty = p.target_chars, max(p.max_chars, p.target_chars + 1), p.penalty

    def length_penalty(completions, **kwargs):
        out = []
        for completion in completions:
            n = len(support.completion_text(completion))
            out.append(0.0 if n <= target else penalty * min(1.0, (n - target) / (maximum - target)))
        return out

    return length_penalty
