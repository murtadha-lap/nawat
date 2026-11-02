"""Reward: the answer between the solution tags matches the reference (Qwen3 GRPO notebook, check_answer).

Exact match scores highest, a whitespace-only difference a little less, and a
numeric answer close to the reference gets partial credit by ratio. Reads the
reference from the dataset's ``answer`` column.
"""

from nawat.lab import P

BLOCK = {"kind": "rewards", "summary": "Answer correctness inside the solution tags, with partial credit by ratio",
         "backbones": ("grpo",)}

PARAMS = {
    "weight": P(1.0, "Weight of this reward in the total"),
    "exact": P(5.0, "Answer identical to the reference"),
    "stripped": P(3.5, "Identical after stripping whitespace"),
    "within_10pct": P(2.0, "Numeric answer within 10 %"),
    "within_20pct": P(1.5, "Numeric answer within 20 %"),
    "wrong": P(-2.5, "Numeric but further off"),
    "unparseable": P(-4.5, "Not a number and not equal"),
    "no_answer": P(-2.0, "No answer found in the tags"),
}


def reward(ctx, p):
    from nawat.lab import support

    fmt = support.solution_pattern(ctx)
    s = p.as_dict()

    def check_answer(prompts, completions, answer, **kwargs):
        scores = []
        for completion, true_answer in zip(completions, answer):
            match = fmt.search(support.completion_text(completion))
            guess = match.group(1) if match else None
            if guess is None:
                scores.append(s["no_answer"])
                continue
            if guess == true_answer:
                scores.append(s["exact"])
            elif guess.strip() == str(true_answer).strip():
                scores.append(s["stripped"])
            else:
                try:
                    ratio = float(guess) / float(true_answer)
                    if 0.9 <= ratio <= 1.1:
                        scores.append(s["within_10pct"])
                    elif 0.8 <= ratio <= 1.2:
                        scores.append(s["within_20pct"])
                    else:
                        scores.append(s["wrong"])
                except Exception:  # noqa: BLE001 - the notebook's bare except
                    scores.append(s["unparseable"])
        return scores

    return check_answer
