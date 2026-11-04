"""Shared helpers for backbones and blocks.

Frozen with the experiment: planning copies this file into ``code/support.py``
and the runner imports that copy, so an upgrade of Nawāt cannot change what an
old experiment does when it is run again. Keep it to plumbing — anything that
decides a number belongs in a block's PARAMS, where the plan shows it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

# -- data ---------------------------------------------------------------------


def load_dataset_dir(path: str, *, split: str = "train", subset: str | None = None, data_files: Any = None):
    """A dataset from a staged directory: a hub snapshot, a save_to_disk folder, or plain files."""
    from datasets import load_dataset, load_from_disk

    directory = Path(path)
    if (directory / "dataset_dict.json").is_file() or (directory / "state.json").is_file():
        loaded = load_from_disk(str(directory))
        return loaded[split] if hasattr(loaded, "keys") else loaded
    kwargs: dict[str, Any] = {}
    if subset:
        kwargs["name"] = subset
    if data_files:
        kwargs["data_files"] = data_files
    return load_dataset(str(directory), split=split, **kwargs)


def limit(dataset, count: int, seed: int | None = None):
    """The first ``count`` rows (after a seeded shuffle when a seed is given); all of them for 0."""
    if not count or count <= 0 or len(dataset) <= count:
        return dataset
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)
    return dataset.select(range(count))


def split_off(dataset, size: float, seed: int):
    """(train, held_out). ``size`` < 1 is a fraction, ≥ 1 a row count, 0 holds nothing out."""
    if not size or size <= 0:
        return dataset, None
    count = int(len(dataset) * size) if size < 1 else int(size)
    count = max(1, min(count, len(dataset) - 1))
    parts = dataset.train_test_split(test_size=count, seed=seed, shuffle=True)
    return parts["train"], parts["test"]


def rows(dataset, count: int | None = None) -> list[dict[str, Any]]:
    if dataset is None:
        return []
    n = len(dataset) if count is None or count <= 0 else min(count, len(dataset))
    return [dataset[i] for i in range(n)]


# -- trainer configuration -------------------------------------------------------


def build_config(cls, values: dict[str, Any], extra: dict[str, Any] | None = None):
    """A trainer config from declared parameters plus ``train.extra``.

    ``extra`` reaches any field of the library's config that the backbone does
    not declare — still visible in the plan, still part of the approval hash —
    without editing code. A field may not be set both ways.
    """
    import inspect

    extra = dict(extra or {})
    names = {f.name for f in dataclasses.fields(cls)}
    for klass in cls.__mro__:  # Unsloth's argument classes add keyword-only fields in __init__
        init = klass.__dict__.get("__init__")
        if init is not None:
            names |= {n for n, prm in inspect.signature(init).parameters.items()
                      if n != "self" and prm.kind in (prm.POSITIONAL_OR_KEYWORD, prm.KEYWORD_ONLY)}
    unknown = sorted((set(values) | set(extra)) - names)
    if unknown:
        from nawat.errors import NawatError

        raise NawatError(f"{cls.__name__} has no field(s): {', '.join(unknown)}.",
                         "Check the spelling in train.extra against the trainer library's documentation.")
    both = sorted(set(values) & set(extra))
    if both:
        from nawat.errors import NawatError

        raise NawatError(f"train.extra sets {', '.join(both)}, which already have their own parameters.",
                         "Set them through their own parameter instead.")
    return cls(**values, **extra)


def record_config(ctx, trainer) -> None:
    try:
        ctx.write_json(f"effective_config-{ctx.run_id}.json", trainer.args.to_dict())
    except Exception as exc:  # noqa: BLE001
        ctx.log(f"could not record the effective config: {exc}")


def train(ctx, trainer) -> dict[str, Any]:
    """Train (continuing from the lineage's newest checkpoint) and record what happened."""
    record_config(ctx, trainer)
    ctx.state["trainer"] = trainer
    start_memory = _gpu_reserved()
    started = time.time()
    stats = trainer.train(resume_from_checkpoint=ctx.resume_from)
    metrics = dict(getattr(stats, "metrics", {}) or {})
    metrics["wall_seconds"] = time.time() - started
    peak = _gpu_reserved()
    if peak is not None:
        metrics["peak_reserved_gb"] = peak
        metrics["training_reserved_gb"] = round(peak - (start_memory or 0.0), 3)
    ctx.write_json(f"train-{ctx.run_id}.json", {"metrics": metrics, "log_history": trainer.state.log_history})
    return metrics


def _gpu_reserved() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    except Exception:  # noqa: BLE001
        pass
    return None


def callbacks(ctx, model, tokenizer, data, p) -> list:
    """The metric stream, plus a sample trace when the backbone asks for one."""
    out = [ctx.metrics_callback()]
    probes = (data or {}).get("probes") or []
    every = int(getattr(p, "sample_every", 0) or 0) if "sample_every" in p else 0
    if every > 0 and probes and ctx.block("model").has("generate"):
        settings = {"max_new_tokens": p.sample_max_new_tokens, "temperature": p.sample_temperature}
        out.append(ctx.sample_callback(lambda items: generate(ctx, model, tokenizer, items, **settings),
                                       probes[: max(1, p.sample_probes)], every))
    return out


def finish(ctx, model, tokenizer, data) -> dict[str, Any]:
    """Run every eval block, then every export block. Returns the measurements."""
    return {"eval": evaluate_all(ctx, model, tokenizer, data), "exports": export_all(ctx, model, tokenizer)}


def evaluate_all(ctx, model, tokenizer, data) -> dict[str, Any]:
    results: dict[str, Any] = {}
    fingerprints = ctx.experiment.data.get("eval_fingerprints", {})
    for block in ctx.blocks("eval"):
        ctx.log(f"eval {block.name} …")
        started = time.time()
        try:
            measured = block.call(ctx, "evaluate", model, tokenizer, data) or {}
        except Exception as exc:  # noqa: BLE001 - a broken eval must not lose a finished training
            ctx.log(f"eval {block.name} failed: {type(exc).__name__}: {exc}")
            measured = {"error": f"{type(exc).__name__}: {exc}"}
        metrics = measured.get("metrics", measured) if isinstance(measured, dict) else {}
        record = {"block": block.name, "fingerprint": fingerprints.get(block.name), "params": block.p.as_dict(),
                  "seconds": round(time.time() - started, 1), **(measured if isinstance(measured, dict) else {})}
        ctx.write_json(f"eval/{block.name}-{ctx.run_id}.json", record)
        numeric = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if numeric:
            ctx.log_metric(event=f"eval:{block.name}", **numeric)
        results[block.name] = metrics
        ctx.log(f"eval {block.name}: " + ", ".join(f"{k}={_fmt(v)}" for k, v in numeric.items()))
    return results


def export_all(ctx, model, tokenizer) -> list[str]:
    done = []
    for block in ctx.blocks("export"):
        ctx.log(f"export {block.name} …")
        block.call(ctx, "export", model, tokenizer)
        done.append(block.name)
    return done


def _fmt(value: Any) -> str:
    return f"{value:.4g}" if isinstance(value, float) else str(value)


# -- generation and scoring ------------------------------------------------------


def generate(ctx, model, tokenizer, items: list[dict[str, Any]], **settings) -> list[str]:
    """Answers from the model block's own generate(), so eval and traces share one path."""
    return ctx.block("model").call(ctx, "generate", model, tokenizer, items, **settings)


def render_prompt(tokenizer, value: Any, chat_template_kwargs: dict[str, Any] | None = None) -> tuple[str, bool]:
    """(text, came_from_chat_template). Messages go through the chat template.

    ``chat_template_kwargs`` are extra variables for the template — for Qwen3,
    ``{"enable_thinking": False}`` makes the generation prompt ask for a direct
    answer. They come from the model block's parameter of the same name, so the
    plan shows them and the eval fingerprints count them.
    """
    if isinstance(value, list):
        text = tokenizer.apply_chat_template(value, tokenize=False, add_generation_prompt=True,
                                             **dict(chat_template_kwargs or {}))
        return text, True
    return str(value), False


def causal_generate(model, tokenizer, items, *, max_new_tokens: int, temperature: float, top_p: float = 1.0,
                    batch_size: int = 8, chat_template_kwargs: dict[str, Any] | None = None) -> list[str]:
    """Left-padded batched generation for a causal language model."""
    import torch

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    previous_side = tok.padding_side
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    outputs: list[str] = []
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            rendered = [render_prompt(tokenizer, item["input"], chat_template_kwargs) for item in batch]
            encoded = tok([text for text, _ in rendered], return_tensors="pt", padding=True,
                          add_special_tokens=not rendered[0][1]).to(model.device)
            kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.pad_token_id, "use_cache": True}
            if temperature and temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            else:
                kwargs.update(do_sample=False)
            with torch.no_grad():
                generated = model.generate(**encoded, **kwargs)
            width = encoded["input_ids"].shape[1]
            outputs += [tok.decode(row[width:], skip_special_tokens=True) for row in generated]
    finally:
        tok.padding_side = previous_side
    return outputs


def choice_logprobs(model, tokenizer, context: str, choices: Sequence[str]) -> list[float]:
    """log P(choice | context) summed over the choice's tokens, for each choice."""
    import torch

    tok = getattr(tokenizer, "tokenizer", tokenizer)
    scores = []
    context_ids = tok(context, add_special_tokens=True)["input_ids"]
    for choice in choices:
        choice_ids = tok(choice, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([context_ids + choice_ids], device=model.device)
        with torch.no_grad():
            logits = model(input_ids=ids).logits[0, :-1].float()
        logp = torch.log_softmax(logits, dim=-1)
        start = len(context_ids) - 1
        target = ids[0, len(context_ids):]
        scores.append(float(logp[start:start + len(choice_ids)].gather(1, target.unsqueeze(1)).sum()))
    return scores


# -- metrics ------------------------------------------------------------------


def edit_distance(a: Sequence, b: Sequence) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def cer(predictions: Sequence[str], references: Sequence[str]) -> float:
    total = sum(len(r) for r in references) or 1
    return sum(edit_distance(p, r) for p, r in zip(predictions, references)) / total


def wer(predictions: Sequence[str], references: Sequence[str]) -> float:
    total = sum(len(r.split()) for r in references) or 1
    return sum(edit_distance(p.split(), r.split()) for p, r in zip(predictions, references)) / total


def normalize(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def last_number(text: str) -> str | None:
    found = _NUMBER.findall(str(text).replace("$", ""))
    if not found:
        return None
    value = found[-1].replace(",", "").rstrip(".")
    try:
        number = float(value)
    except ValueError:
        return None
    return str(int(number)) if number.is_integer() else str(number)


def boxed(text: str) -> str | None:
    """The content of the last \\boxed{...}, braces balanced."""
    start = str(text).rfind("\\boxed")
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        return None
    depth, j = 0, i
    while j < len(text):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j].strip()
        j += 1
    return None


def completion_text(completion: Any) -> str:
    """A GRPO completion as text, whether conversational ([{"content": ...}]) or plain."""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


DEFAULT_TAGS = {"reasoning_start": "<start_working_out>", "reasoning_end": "<end_working_out>",
                "solution_start": "<SOLUTION>", "solution_end": "</SOLUTION>"}


def solution_pattern(ctx):
    """The Qwen3 GRPO notebook's ``match_format``: reasoning end, then the answer in solution tags at the end.

    Group 1 is the answer. Shared by every reward and eval that has to agree on
    what "the model gave an answer" means.
    """
    tags = ctx.state.get("tags") or DEFAULT_TAGS
    eos = ctx.state.get("eos_token") or ""
    end = re.escape(tags["solution_end"]) + r"[\s]{0,}" + (f"(?:{re.escape(eos)})?" if eos else "")
    return re.compile(rf"{re.escape(tags['reasoning_end'])}.*?{re.escape(tags['solution_start'])}(.+?){end}[\s]{{0,}}$",
                      flags=re.MULTILINE | re.DOTALL)


def bf16() -> bool:
    try:
        from unsloth import is_bfloat16_supported

        return bool(is_bfloat16_supported())
    except Exception:  # noqa: BLE001
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def mean(values: Iterable[float]) -> float | None:
    values = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(values) / len(values) if values else None


def write_predictions(ctx, name: str, records: list[dict[str, Any]]) -> None:
    """Every scored item of an eval, beside its summary, for later inspection."""
    path = ctx.path("eval", f"{name}-{ctx.run_id}.predictions.jsonl")
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


# -- evaluation harness ---------------------------------------------------------


def extract_answer(ctx, kind: str, text: str) -> str | None:
    """Pull the answer out of a generation. ``auto`` tries solution tags, then \\boxed{}, then the last number."""
    text = "" if text is None else str(text)
    if kind == "none":
        return text
    if kind == "solution-tags":
        tags = ctx.state.get("tags") or DEFAULT_TAGS
        found = re.findall(re.escape(tags["solution_start"]) + r"(.*?)" + re.escape(tags["solution_end"]), text, re.DOTALL)
        return found[-1].strip() if found else None
    if kind == "boxed":
        return boxed(text)
    if kind == "last-number":
        return last_number(text)
    if kind == "gsm8k":
        return last_number(text.split("####")[-1]) if "####" in text else last_number(text)
    if kind == "auto":
        return (extract_answer(ctx, "solution-tags", text) or boxed(text) or last_number(text))
    raise ValueError(f"unknown answer extractor {kind!r}")


def normalize_math(text: Any) -> str:
    s = str(text).strip().strip("$").replace(" ", "")
    for token in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\displaystyle", "^\\circ", "^{\\circ}", "\\%", "%"):
        s = s.replace(token, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac").rstrip(".")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def same_number(a: Any, b: Any) -> bool:
    try:
        return abs(float(str(a).replace(",", "")) - float(str(b).replace(",", ""))) < 1e-6
    except (TypeError, ValueError):
        return False


def item_score(metric: str, prediction: Any, reference: Any) -> float:
    if prediction is None:
        return 0.0
    if metric == "exact":
        return float(normalize(prediction) == normalize(reference))
    if metric == "contains":
        return float(normalize(reference) in normalize(prediction))
    if metric == "number":
        return float(same_number(prediction, reference))
    if metric == "math":
        return float(normalize_math(prediction) == normalize_math(reference) or same_number(prediction, reference))
    raise ValueError(f"unknown per-item metric {metric!r}")


def generation_eval(ctx, model, tokenizer, name: str, items: list[dict[str, Any]], *, metric: str, extract: str,
                    max_new_tokens: int, temperature: float, batch_size: int) -> dict[str, Any]:
    """Generate for every item, extract, score, write every prediction. Returns the summary."""
    if not items:
        return {"metrics": {"n": 0}, "note": "no items to evaluate"}
    outputs = generate(ctx, model, tokenizer, items, max_new_tokens=max_new_tokens, temperature=temperature,
                       batch_size=batch_size)
    answers = [extract_answer(ctx, extract, o) for o in outputs]
    references = [str(it.get("reference", "")) for it in items]
    records = [{"i": i, "input": it.get("input"), "reference": ref, "output": out, "answer": ans}
               for i, (it, ref, out, ans) in enumerate(zip(items, references, outputs, answers))]
    metrics: dict[str, Any] = {"n": len(items), "answered": sum(a is not None for a in answers) / len(items)}
    if metric in ("cer", "wer"):
        preds = [a or "" for a in answers]
        metrics[metric] = cer(preds, references) if metric == "cer" else wer(preds, references)
        for r, pr, ref in zip(records, preds, references):
            r["errors"] = edit_distance(pr, ref) if metric == "cer" else edit_distance(pr.split(), ref.split())
    else:
        scores = [item_score(metric, a, ref) for a, ref in zip(answers, references)]
        metrics["accuracy"] = sum(scores) / len(scores)
        for r, sc in zip(records, scores):
            r["correct"] = bool(sc)
    write_predictions(ctx, name, records)
    return {"metrics": metrics}


def choice_eval(ctx, model, tokenizer, name: str, questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Multiple choice by log-likelihood: each question has context, choices, gold index.

    ``accuracy`` picks the choice with the highest total log-probability;
    ``accuracy_norm`` divides by the choice's length in characters (lm-eval's acc_norm).
    """
    if not questions:
        return {"metrics": {"n": 0}}
    model_block = ctx.block("model")
    if not model_block.has("score"):
        return {"error": f"model block {model_block.name} cannot score choices", "metrics": {}}
    correct = correct_norm = 0
    records = []
    for i, q in enumerate(questions):
        logps = model_block.call(ctx, "score", model, tokenizer, q["context"], q["choices"])
        pick = max(range(len(logps)), key=lambda k: logps[k])
        norm = [lp / max(1, len(c)) for lp, c in zip(logps, q["choices"])]
        pick_norm = max(range(len(norm)), key=lambda k: norm[k])
        correct += pick == q["gold"]
        correct_norm += pick_norm == q["gold"]
        records.append({"i": i, "gold": q["gold"], "pick": pick, "pick_norm": pick_norm, "logprobs": logps,
                        **({"subject": q["subject"]} if "subject" in q else {})})
    write_predictions(ctx, name, records)
    n = len(questions)
    return {"metrics": {"n": n, "accuracy": correct / n, "accuracy_norm": correct_norm / n}}
