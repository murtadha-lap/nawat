"""Batch evaluation — adapter quality as a number in the run record.

An evaluation set is a JSONL file: one sample per line, with a ``reference``
transcription, a ``prompt``, and optionally an ``image`` path relative to the
file (or a data URL) for vision models. The runner sends each sample to the
running inference session, computes character and word error rates against the
reference, and writes the result three ways (FR-4.7):

- ``eval-<label>.json`` beside the run log — the full per-sample record;
- a point in the run's metric series (``cer``, ``wer``, event ``eval``), so
  quality plots on the same axis as training progress (FR-7.5) and compares
  across runs with the same tools as loss (FR-4.8);
- republished with the run record in object storage.

CER and WER are corpus-level: total edit distance over total reference length,
so long samples weigh what they should.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .errors import NawatError, NotFound
from .keys import Key


# -- edit distance -----------------------------------------------------------


def edit_distance(reference: list | str, prediction: list | str) -> int:
    """Levenshtein distance between two sequences, two-row dynamic programme."""
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)
    previous = list(range(len(prediction) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row] + [0] * len(prediction)
        for col, pred_item in enumerate(prediction, start=1):
            current[col] = min(
                previous[col] + 1,          # deletion
                current[col - 1] + 1,       # insertion
                previous[col - 1] + (ref_item != pred_item),  # substitution
            )
        previous = current
    return previous[-1]


def cer(reference: str, prediction: str) -> float:
    """Character error rate for one sample. 0 is perfect."""
    reference = reference.strip()
    if not reference:
        return 0.0 if not prediction.strip() else 1.0
    return edit_distance(list(reference), list(prediction.strip())) / len(reference)


def wer(reference: str, prediction: str) -> float:
    """Word error rate for one sample, on whitespace tokens."""
    ref_words = reference.split()
    if not ref_words:
        return 0.0 if not prediction.split() else 1.0
    return edit_distance(ref_words, prediction.split()) / len(ref_words)


@dataclass
class Corpus:
    """Corpus-level accumulation: edits over reference length, not a mean of rates."""

    char_edits: int = 0
    char_total: int = 0
    word_edits: int = 0
    word_total: int = 0
    samples: int = 0

    def add(self, reference: str, prediction: str) -> tuple[float, float]:
        reference = reference.strip()
        prediction = prediction.strip()
        sample_cer = cer(reference, prediction)
        sample_wer = wer(reference, prediction)
        self.char_edits += edit_distance(list(reference), list(prediction))
        self.char_total += max(len(reference), 1)
        self.word_edits += edit_distance(reference.split(), prediction.split())
        self.word_total += max(len(reference.split()), 1)
        self.samples += 1
        return sample_cer, sample_wer

    @property
    def cer(self) -> float:
        return self.char_edits / self.char_total if self.char_total else 0.0

    @property
    def wer(self) -> float:
        return self.word_edits / self.word_total if self.word_total else 0.0


# -- samples -----------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    reference: str
    prompt: str
    image: str | None = None  # a data URL, ready to send


DEFAULT_PROMPT = "Transcribe the text in this image."


def load_samples(path: Path, *, limit: int | None = None) -> list[Sample]:
    """Read an evaluation set, inlining referenced images as data URLs."""
    if not path.is_file():
        raise NotFound(
            f"{path} does not exist, so there is nothing to evaluate against.",
            "Point --data at an eval JSONL file, or at a dataset key containing one.",
        )
    samples: list[Sample] = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except ValueError as exc:
            raise NawatError(
                f"Line {number} of {path.name} is not JSON.",
                'Each line must be an object like {"reference": "...", "prompt": "...", "image": "..."}.',
            ) from exc
        if "reference" not in raw:
            raise NawatError(
                f'Line {number} of {path.name} has no "reference".',
                "Every sample needs the ground-truth text to score against.",
            )
        image = raw.get("image")
        if image and not image.startswith("data:"):
            image = _data_url(path.parent / image)
        samples.append(Sample(reference=str(raw["reference"]), prompt=str(raw.get("prompt", DEFAULT_PROMPT)), image=image))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise NotFound(f"{path} holds no samples.", "Add at least one line to the eval set.")
    return samples


def find_eval_file(directory: Path, name: str | None = None) -> Path:
    """The eval set inside a staged dataset."""
    if name:
        return directory / name
    for candidate in ("eval.jsonl", "test.jsonl", "val.jsonl"):
        if (directory / candidate).is_file():
            return directory / candidate
    raise NotFound(
        f"{directory} holds no eval.jsonl, test.jsonl or val.jsonl.",
        "Name the file explicitly with --file.",
    )


def _data_url(path: Path) -> str:
    if not path.is_file():
        raise NotFound(f"{path} does not exist.", "Check the image paths inside the eval set.")
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


# -- the runner --------------------------------------------------------------


@dataclass
class EvalResult:
    label: str
    model: str
    samples: int
    cer: float
    wer: float
    at: float
    per_sample: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "samples": self.samples,
            "cer": self.cer,
            "wer": self.wer,
            "at": self.at,
            "per_sample": self.per_sample,
        }


Predict = Callable[[Sample], str]


def score(samples: Iterable[Sample], predict: Predict, *, label: str, model: str,
          progress: Callable[[int, int], None] | None = None) -> EvalResult:
    """Run every sample through ``predict`` and aggregate."""
    corpus = Corpus()
    per_sample: list[dict[str, Any]] = []
    samples = list(samples)
    for index, sample in enumerate(samples, start=1):
        prediction = predict(sample)
        sample_cer, sample_wer = corpus.add(sample.reference, prediction)
        per_sample.append({
            "reference": sample.reference,
            "prediction": prediction,
            "cer": sample_cer,
            "wer": sample_wer,
        })
        if progress:
            progress(index, len(samples))
    return EvalResult(
        label=label, model=model, samples=corpus.samples,
        cer=corpus.cer, wer=corpus.wer, at=time.time(), per_sample=per_sample,
    )


def session_predictor(base_url: str, model: str, *, timeout: float = 300.0) -> Predict:
    """Predictions from the running inference session's OpenAI endpoint."""

    def predict(sample: Sample) -> str:
        content: Any
        if sample.image:
            content = [
                {"type": "image_url", "image_url": {"url": sample.image}},
                {"type": "text", "text": sample.prompt},
            ]
        else:
            content = sample.prompt
        request = urllib.request.Request(
            base_url + "/v1/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
            }).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read())
        except Exception as exc:
            raise NawatError(
                f"The inference server failed a sample: {type(exc).__name__}.",
                "Check the session with: nawat session --log",
            ) from exc
        return str(body["choices"][0]["message"]["content"])

    return predict


class Evaluator:
    """Evaluate a run's adapter and write the result into its record."""

    def __init__(self, cache, runs, sessions) -> None:
        self.cache = cache
        self.runs = runs
        self.sessions = sessions

    def resolve_data(self, data: str, file_name: str | None = None) -> Path:
        """``--data`` is a dataset key or a local path."""
        candidate = Path(data).expanduser()
        if candidate.exists():
            return candidate if candidate.is_file() else find_eval_file(candidate, file_name)
        directory = self.cache.resolve(Key.parse(data), lease=False)
        return find_eval_file(directory, file_name)

    def evaluate_run(
        self,
        run_id: str,
        data: str,
        *,
        base: str | None = None,
        file_name: str | None = None,
        limit: int | None = None,
        label: str | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> EvalResult:
        record = self.runs.get(run_id)
        adapter_key = next((k for k in record.artifacts if str(k).endswith("/adapter")), None)
        if adapter_key is None:
            raise NotFound(
                f"Run {run_id} produced no adapter to evaluate.",
                "Evaluate a run whose script saved one under NAWAT_OUT_DIR/adapter.",
            )
        base_key = base or (str(record.spec.model) if record.spec.model else None)
        if base_key is None:
            raise NotFound(
                f"Run {run_id} recorded no base model and none was given.",
                "Pass --base models/<repo> to serve the adapter against.",
            )

        eval_path = self.resolve_data(data, file_name)
        samples = load_samples(eval_path, limit=limit)

        session = self.sessions.start(base_key)
        name = f"eval-{run_id}"
        if name not in session.adapters:
            session = self.sessions.load_adapter(adapter_key, name=name)

        result = score(
            samples,
            session_predictor(session.base_url, name),
            label=label or eval_path.stem,
            model=str(adapter_key),
            progress=progress,
        )
        self._record(record, result)
        return result

    def _record(self, record, result: EvalResult) -> None:
        directory = self.runs.directory(record.id)
        (directory / f"eval-{result.label}.json").write_text(json.dumps(result.to_json(), indent=2) + "\n")

        # Onto the same axis as training progress: the eval point lands at the
        # run's final training step (FR-7.5).
        from . import metrics as metrics_module

        points = metrics_module.read_points(self.runs.metrics_path(record.id))
        final_step = max((p.get("step", 0) for p in points), default=None)
        with self.runs.metrics_path(record.id).open("a") as handle:
            point = {"t": result.at, "event": "eval", "cer": result.cer, "wer": result.wer,
                     "samples": result.samples}
            if final_step is not None:
                point["step"] = final_step
            handle.write(json.dumps(point) + "\n")

        self._republish(record)

    def _republish(self, record) -> None:
        import shutil
        import tempfile

        from .executor import CANCEL_SENTINEL

        directory = self.runs.directory(record.id)
        sources = [p for p in directory.iterdir() if p.is_file() and p.name != CANCEL_SENTINEL]
        if not sources:
            return
        with tempfile.TemporaryDirectory(dir=str(self.cache.config.staging_root)) as scratch:
            for path in sources:
                shutil.copy2(path, Path(scratch) / path.name)
            try:
                self.cache.store.publish(Path(scratch), Key.parse(f"runs/{record.id}/record"))
            except NawatError:
                pass  # the local record is intact; replication retries next publish
