"""Evaluation: CER/WER against a held-out set, written into the run record.

Phase 5's exit criterion — adapter quality comparable across runs without
leaving the platform — closes the file: two runs evaluated, compared on one
axis through the same endpoint the trace uses.
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace as dataclass_replace

import pytest

from nawat.errors import NawatError, NotFound
from nawat.evaluate import (
    Corpus,
    Evaluator,
    Sample,
    cer,
    edit_distance,
    load_samples,
    score,
    session_predictor,
    wer,
)
from nawat.keys import Key
from nawat.runs import RunSpec, RunState
from nawat.sessions import SessionManager

from .conftest import FakeBackend

CACHE_CEILING = 50 * 10**6

#: A one-pixel PNG, for image-bearing samples.
PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)


# -- the arithmetic ----------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,distance",
    [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3), ("same", "same", 0), ("ab", "ba", 2)],
)
def test_edit_distance_is_levenshtein(a, b, distance):
    assert edit_distance(list(a), list(b)) == distance


def test_cer_counts_character_edits_over_reference_length():
    assert cer("نواة", "نواة") == 0.0
    assert cer("abcd", "abcx") == 0.25
    assert cer("abcd", "") == 1.0
    assert cer("", "") == 0.0
    assert cer("", "noise") == 1.0


def test_wer_counts_word_edits():
    assert wer("the quick brown fox", "the quick brown fox") == 0.0
    assert wer("the quick brown fox", "the quick brown cat") == 0.25
    assert wer("one two", "one two three") == 0.5


def test_corpus_rates_weigh_by_length_not_by_sample():
    corpus = Corpus()
    corpus.add("a" * 99, "a" * 99)   # long and perfect
    corpus.add("b", "x")             # short and wrong
    assert corpus.cer == pytest.approx(1 / 100)
    assert corpus.samples == 2


# -- the samples -------------------------------------------------------------


def test_samples_load_with_prompts_references_and_images(tmp_path):
    (tmp_path / "img.png").write_bytes(PIXEL)
    data = tmp_path / "eval.jsonl"
    data.write_text(
        json.dumps({"reference": "text one", "prompt": "read it", "image": "img.png"}) + "\n"
        + json.dumps({"reference": "text two"}) + "\n"
    )

    samples = load_samples(data)

    assert samples[0].reference == "text one"
    assert samples[0].image.startswith("data:image/png;base64,")
    assert samples[1].prompt, "a default prompt fills in"


def test_a_sample_without_a_reference_is_refused_with_the_shape(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text(json.dumps({"prompt": "no truth here"}) + "\n")
    with pytest.raises(NawatError, match='"reference"'):
        load_samples(data)


def test_a_missing_eval_file_says_what_to_point_at(tmp_path):
    with pytest.raises(NotFound, match="--data"):
        load_samples(tmp_path / "absent.jsonl")


def test_limit_caps_the_set(tmp_path):
    data = tmp_path / "eval.jsonl"
    data.write_text("".join(json.dumps({"reference": f"r{i}"}) + "\n" for i in range(10)))
    assert len(load_samples(data, limit=3)) == 3


def test_score_aggregates_predictions(tmp_path):
    samples = [Sample(reference="abcd", prompt="p"), Sample(reference="wxyz", prompt="p")]
    result = score(samples, lambda s: s.reference, label="perfect", model="m")
    assert result.cer == 0.0 and result.wer == 0.0 and result.samples == 2


# -- against a live session --------------------------------------------------


@pytest.fixture
def served(cache, cached, free_port):
    cache.config = dataclass_replace(cache.config, serve_port=free_port, serve_startup_timeout=20.0)
    cached("models/base", 4096)
    manager = SessionManager(cache, backend=FakeBackend())
    yield manager
    manager.stop()


def test_predictions_come_from_the_openai_endpoint(served, tmp_path):
    session = served.start("models/base")
    predict = session_predictor(session.base_url, "models/base")

    assert predict(Sample(reference="x", prompt="ECHO hello from the adapter")) == "hello from the adapter"


def make_finished_run(executor, script, cached, run_id, *, model="models/base"):
    """A succeeded run with an adapter artifact and a short loss series."""
    body = """
import os, pathlib
import nawat.metrics as m
out = pathlib.Path(os.environ["NAWAT_OUT_DIR"])
(out / "adapter").mkdir(parents=True, exist_ok=True)
(out / "adapter" / "adapter_model.safetensors").write_bytes(b"w" * 512)
for step in range(1, 4):
    m.log(step=step, loss=2.0 / step)
"""
    record = executor.runs.create(
        RunSpec(script=script(body, name=f"{run_id}.py"), model=Key.parse(model)), run_id
    )
    finished = executor.execute(record.id)
    assert finished.state is RunState.SUCCEEDED
    return finished


@pytest.fixture
def eval_set(tmp_path):
    data = tmp_path / "eval.jsonl"
    lines = [
        {"reference": "clean transcription", "prompt": "ECHO clean transcription"},  # perfect
        {"reference": "abcd", "prompt": "ECHO abcx"},                                # 1 char off
    ]
    data.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return data


def test_evaluating_a_run_writes_cer_into_its_record(executor, script, cached, served, eval_set, cache, store):
    record = make_finished_run(executor, script, cached, "eval-run")
    evaluator = Evaluator(cache, executor.runs, served)

    result = evaluator.evaluate_run(record.id, str(eval_set))

    # 1 char edit / 23 reference chars, corpus-level
    assert result.cer == pytest.approx(1 / 23)
    assert result.samples == 2

    # into the metric series, at the final training step, for the shared axis
    from nawat import metrics
    points = metrics.read_points(executor.runs.metrics_path(record.id))
    eval_point = next(p for p in points if p.get("event") == "eval")
    assert eval_point["step"] == 3
    assert eval_point["cer"] == pytest.approx(1 / 23)

    # the full per-sample record beside the log, and republished to the store
    on_disk = json.loads((executor.runs.directory(record.id) / "eval-eval.json").read_text())
    assert on_disk["per_sample"][1]["prediction"] == "abcx"
    assert "eval-eval.json" in store.list_prefix(Key.parse(f"runs/{record.id}/record"))


def test_evaluation_serves_the_base_and_hot_loads_the_adapter(executor, script, cached, served, eval_set, cache):
    record = make_finished_run(executor, script, cached, "eval-adapter")
    evaluator = Evaluator(cache, executor.runs, served)

    evaluator.evaluate_run(record.id, str(eval_set))

    session = served.current()
    assert str(session.model) == "models/base"
    assert f"runs/{record.id}/adapter" in session.adapters.values()


def test_a_run_without_an_adapter_is_refused(executor, script, cached, served, eval_set, cache):
    body = "import os, pathlib\npathlib.Path(os.environ['NAWAT_OUT_DIR'], 'notes').mkdir(parents=True)\n" \
           "pathlib.Path(os.environ['NAWAT_OUT_DIR'], 'notes', 'x.txt').write_text('no adapter')\n"
    record = executor.runs.create(RunSpec(script=script(body, name="no-adapter.py")))
    executor.execute(record.id)

    with pytest.raises(NotFound, match="no adapter"):
        Evaluator(cache, executor.runs, served).evaluate_run(record.id, str(eval_set))


def test_a_run_without_a_base_model_names_the_flag(executor, script, cached, served, eval_set, cache, monkeypatch):
    body = """
import os, pathlib
out = pathlib.Path(os.environ["NAWAT_OUT_DIR"])
(out / "adapter").mkdir(parents=True, exist_ok=True)
(out / "adapter" / "x.safetensors").write_bytes(b"w")
"""
    record = executor.runs.create(RunSpec(script=script(body, name="no-base.py")))
    executor.execute(record.id)

    with pytest.raises(NotFound, match="--base"):
        Evaluator(cache, executor.runs, served).evaluate_run(record.id, str(eval_set))


def test_adapter_quality_is_comparable_across_runs(executor, script, cached, served, eval_set, cache):
    """Phase 5's exit criterion: two runs, one axis, no shell arithmetic."""
    from nawat import metrics

    first = make_finished_run(executor, script, cached, "compare-a")
    second = make_finished_run(executor, script, cached, "compare-b")
    evaluator = Evaluator(cache, executor.runs, served)
    evaluator.evaluate_run(first.id, str(eval_set))
    evaluator.evaluate_run(second.id, str(eval_set))

    by_run = {}
    for run_id in (first.id, second.id):
        points = metrics.read_points(executor.runs.metrics_path(run_id))
        by_run[run_id] = metrics.series(points).get("cer", [])
    assert all(entries for entries in by_run.values())
    assert by_run[first.id][0]["value"] == by_run[second.id][0]["value"] == pytest.approx(1 / 23)
