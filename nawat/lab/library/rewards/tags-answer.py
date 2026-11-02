"""Reward: the single answer block equals the reference (vision GRPO notebook, correctness_reward_func)."""

from nawat.lab import P

BLOCK = {"kind": "rewards", "summary": "The one answer block equals the reference exactly", "backbones": ("grpo",)}

PARAMS = {
    "weight": P(1.0, "Weight of this reward in the total"),
    "correct": P(2.0, "Answer equals the reference"),
    "incorrect": P(0.0, "Anything else"),
}


def reward(ctx, p):
    import re

    from nawat.lab import support

    tags = ctx.state.get("tags") or {"solution_start": "<SOLUTION>", "solution_end": "</SOLUTION>"}
    pattern = re.compile(f"{re.escape(tags['solution_start'])}(.*?){re.escape(tags['solution_end'])}", re.DOTALL)
    correct, incorrect = p.correct, p.incorrect

    def correctness_reward_func(prompts, completions, answer, **kwargs):
        responses = [pattern.findall(support.completion_text(c)) for c in completions]
        return [correct if len(r) == 1 and str(a) == r[0].replace("\n", "") else incorrect for r, a in zip(responses, answer)]

    return correctness_reward_func
