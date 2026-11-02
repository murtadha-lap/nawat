"""Reasoning format for GRPO: tags around the working and the answer.

Two modes, one per Unsloth notebook:

``chat-template`` (Qwen3 GRPO notebook)
    A system prompt names the tags, and the notebook's chat template is
    installed, which opens the reasoning tag at generation time. The priming set
    becomes worked examples in exactly this format.

``inline`` (vision GRPO notebook)
    The model keeps its own template; the instruction naming the tags is written
    into the user turn after the question.

The tags are shared with the reward blocks through ``ctx.state["tags"]``, so a
reward that checks the format always checks this format.
"""

from nawat.lab import P

BLOCK = {"kind": "template", "summary": "Reasoning / solution tags, system prompt and GRPO chat template",
         "backbones": ("grpo",)}

PARAMS = {
    "mode": P("chat-template", "chat-template (Qwen3 notebook) or inline (vision notebook)",
              choices=("chat-template", "inline")),
    "reasoning_start": P("<start_working_out>", "Opens the reasoning"),
    "reasoning_end": P("<end_working_out>", "Closes the reasoning"),
    "solution_start": P("<SOLUTION>", "Opens the answer"),
    "solution_end": P("</SOLUTION>", "Closes the answer"),
    "system_prompt": P("You are given a problem.\nThink about the problem and provide your working out.\n"
                       "Place it between {reasoning_start} and {reasoning_end}.\n"
                       "Then, provide your solution between {solution_start}{solution_end}",
                       "chat-template mode: system prompt ({tags} are filled in)"),
    "inline_instruction": P("{question}. Also first provide your reasoning or working out on how you would go about "
                            "solving the question between {reasoning_start} and {reasoning_end} and then your final "
                            "answer between {solution_start} and (put a single float here) {solution_end}",
                            "inline mode: the user turn ({question} and {tags} are filled in)"),
    "prime_max_fraction": P(0.5, "Priming: keep worked examples up to this share of max_seq_length"),
}

CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '{system_prompt}' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
    "{% endif %}"
)


def apply(ctx, p, tokenizer, data):
    tags = {"reasoning_start": p.reasoning_start, "reasoning_end": p.reasoning_end,
            "solution_start": p.solution_start, "solution_end": p.solution_end}
    ctx.state["tags"] = tags
    ctx.state["eos_token"] = getattr(getattr(tokenizer, "tokenizer", tokenizer), "eos_token", "") or ""

    if p.mode == "inline":
        template = p.inline_instruction

        def rewrite(message_list):
            out = []
            for message in message_list:
                if message["role"] == "user" and isinstance(message["content"], list):
                    content = [dict(part, text=template.format(question=part["text"], **tags)) if part.get("type") == "text" else part
                               for part in message["content"]]
                    message = {**message, "content": content}
                elif message["role"] == "user":
                    message = {**message, "content": template.format(question=message["content"], **tags)}
                out.append(message)
            return out

        data["train"] = data["train"].map(lambda x: {"prompt": rewrite(x["prompt"])})
        for key in ("heldout", "probes"):
            data[key] = [{**item, "input": rewrite(item["input"])} for item in data.get(key) or []]
        return tokenizer, data

    system = p.system_prompt.format(**tags)
    tokenizer.chat_template = (CHAT_TEMPLATE.replace("'{system_prompt}'", repr(system))
                               .replace("'{reasoning_start}'", repr(p.reasoning_start)))

    def with_system(messages):
        return [{"role": "system", "content": system}] + [m for m in messages if m["role"] != "system"]

    data["train"] = data["train"].map(lambda x: {"prompt": with_system(x["prompt"])})
    for key in ("heldout", "probes"):
        data[key] = [{**item, "input": with_system(item["input"])} for item in data.get(key) or []]

    if data.get("prime_raw") is not None:
        limit = ctx.block("model").p.get("max_seq_length", 2048) * p.prime_max_fraction

        def worked(x):
            thoughts = x["thoughts"].replace("<think>", "").replace("</think>", "").strip()
            answer = p.reasoning_start + thoughts + p.reasoning_end + p.solution_start + x["answer"] + p.solution_end
            messages = [{"role": "system", "content": system}, {"role": "user", "content": x["problem"]},
                        {"role": "assistant", "content": answer}]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False),
                    "N": len(tokenizer.apply_chat_template(messages))}

        prime = data["prime_raw"].map(worked, remove_columns=data["prime_raw"].column_names)
        data["prime"] = prime.filter(lambda x: x["N"] <= limit).remove_columns(["N"])
        ctx.log(f"priming set: {len(data['prime'])} worked examples ≤ {int(limit)} tokens")
    return tokenizer, data
