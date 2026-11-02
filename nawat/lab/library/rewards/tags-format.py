"""Reward: one reasoning block and one answer block, no degenerate repetition (vision GRPO notebook, formatting_reward_func)."""

from nawat.lab import P

BLOCK = {"kind": "rewards", "summary": "One reasoning and one answer block; penalise degenerate repetition",
         "backbones": ("grpo",)}

PARAMS = {
    "weight": P(1.0, "Weight of this reward in the total"),
    "reasoning_block": P(1.0, "Exactly one reasoning block"),
    "answer_block": P(1.0, "Exactly one answer block"),
    "spam_penalty": P(-2.0, "When a junk token makes up half the completion"),
    "spam_token": P("addCriterion", "The junk token the notebook guards against"),
}


def reward(ctx, p):
    import re

    from nawat.lab import support

    tags = ctx.state.get("tags") or {"reasoning_start": "<REASONING>", "reasoning_end": "</REASONING>",
                                     "solution_start": "<SOLUTION>", "solution_end": "</SOLUTION>"}
    thinking = re.compile(f"{re.escape(tags['reasoning_start'])}(.*?){re.escape(tags['reasoning_end'])}", re.DOTALL)
    answer = re.compile(f"{re.escape(tags['solution_start'])}(.*?){re.escape(tags['solution_end'])}", re.DOTALL)
    s = p.as_dict()

    def formatting_reward_func(completions, **kwargs):
        scores = []
        for completion in completions:
            text = support.completion_text(completion)
            score = 0.0
            if len(thinking.findall(text)) == 1:
                score += s["reasoning_block"]
            if len(answer.findall(text)) == 1:
                score += s["answer_block"]
            if text:
                removal = text.replace(s["spam_token"], "").replace("\n", "")
                if (len(text) - len(removal)) / len(text) >= 0.5:
                    score += s["spam_penalty"]
            scores.append(score)
        return scores

    return formatting_reward_func
