"""Reward: the first number after the solution tag equals the reference (Qwen3 GRPO notebook, check_numbers)."""

from nawat.lab import P

BLOCK = {"kind": "rewards", "summary": "First number after <SOLUTION> equals the numeric reference",
         "backbones": ("grpo",)}

PARAMS = {
    "weight": P(1.0, "Weight of this reward in the total"),
    "correct": P(3.5, "Number equals the reference"),
    "incorrect": P(-1.5, "Number differs"),
    "no_number": P(-2.5, "No number after the solution tag"),
    "unparseable": P(0.0, "Reference or guess not a number"),
}


def reward(ctx, p):
    import re

    from nawat.lab import support

    tags = ctx.state.get("tags") or {"solution_start": "<SOLUTION>"}
    match_numbers = re.compile(re.escape(tags["solution_start"]) + r".*?[\s]{0,}([-]?[\d\.\,]{1,})", flags=re.MULTILINE | re.DOTALL)
    s = p.as_dict()

    def check_numbers(prompts, completions, answer, **kwargs):
        scores = []
        for completion, true_answer in zip(completions, answer):
            found = match_numbers.search(support.completion_text(completion))
            if found is None:
                scores.append(s["no_number"])
                continue
            try:
                truth = float(str(true_answer).strip())
                guess = float(found.group(1).strip().replace(",", ""))
                scores.append(s["correct"] if guess == truth else s["incorrect"])
            except Exception:  # noqa: BLE001
                scores.append(s["unparseable"])
        return scores

    return check_numbers
