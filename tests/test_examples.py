"""The shipped examples, checked against the library they demonstrate.

An example that no longer runs is worse than no example, and these are the first
thing a new user copies. Nothing here needs a GPU, Unsloth or a notebook host:
the notebook is read as JSON, the script is parsed, and every ``nawat.<name>``
either reaches something that exists or fails here.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import nawat

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
NOTEBOOK = EXAMPLES / "latex_ocr_qwen3_5_vision.ipynb"
SCRIPT = EXAMPLES / "train_latex_ocr.py"


def cells(kind: str) -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == kind]


def python_cells() -> list[tuple[int, str]]:
    """Code cells that are Python, with their index.

    Cells carrying a ``%%`` cell magic or a ``!`` shell line are IPython input,
    not Python, and are excluded rather than mangled into something parseable.
    """
    out = []
    for index, source in enumerate(cells("code")):
        lines = source.splitlines()
        if any(line.lstrip().startswith(("!", "%")) for line in lines):
            continue
        out.append((index, source))
    return out


def nawat_attributes(source: str) -> set[str]:
    """Every ``nawat.<name>`` referenced in a chunk of Python."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "nawat":
                found.add(node.attr)
    return found


def test_the_notebook_is_a_valid_notebook():
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "an empty notebook demonstrates nothing"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)
        assert cell.get("id"), "nbformat 4.5 requires a cell id"


def test_every_python_cell_parses():
    checked = 0
    for index, source in python_cells():
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"code cell {index} does not parse: {exc}")
        checked += 1
    assert checked >= 10, "almost every cell should be plain Python"


def test_the_notebook_only_calls_library_functions_that_exist():
    referenced: set[str] = set()
    for _, source in python_cells():
        referenced |= nawat_attributes(source)
    assert referenced, "the notebook is supposed to demonstrate the library"
    missing = sorted(name for name in referenced if not hasattr(nawat, name))
    assert not missing, f"the notebook calls nawat.{missing} which does not exist"


def test_the_notebook_uses_the_run_object_it_opens():
    """The three lines that differ from the stock Unsloth notebook."""
    code = "\n".join(cells("code"))
    assert "nawat.begin_run(" in code
    assert "run.model_dir" in code, "the model must come from the cache, not a repo id"
    assert "run.dataset_dir" in code, "the dataset must come from the cache, not a repo id"
    assert 'run.artifact_dir("adapter")' in code, "the adapter must be saved where it gets published"
    assert "run.finish()" in code, "without this nothing is published or released"
    assert "run.callback()" in code, "the loss trace is one argument; use it"


def test_the_notebook_keeps_trainer_checkpoints_out_of_the_published_tree():
    """A directory of checkpoints under out_dir would publish as an artifact."""
    code = "\n".join(cells("code"))
    assert 'run.scratch_dir("trainer")' in code
    assert 'output_dir = "outputs"' not in code


def test_the_script_parses_and_calls_only_what_exists():
    source = SCRIPT.read_text()
    ast.parse(source)
    missing = sorted(name for name in nawat_attributes(source) if not hasattr(nawat, name))
    assert not missing, f"the example script calls nawat.{missing} which does not exist"


def test_the_script_reads_the_environment_rather_than_opening_its_own_run():
    """Under `nawat submit` the executor owns the run; the script must not."""
    source = SCRIPT.read_text()
    assert "nawat.begin_run(" not in source
    assert ".finish()" not in source
    for accessor in ("nawat.model_dir()", "nawat.dataset_dir()", "nawat.artifact_dir(", "nawat.param("):
        assert accessor in source, f"{accessor} is the portable form; use it"


def test_the_two_examples_train_the_same_thing():
    """The script is the notebook's body. If they drift, the story is a lie."""
    notebook_code = "\n".join(cells("code"))
    script = SCRIPT.read_text()
    for shared in (
        "FastVisionModel.from_pretrained",
        "FastVisionModel.get_peft_model",
        "UnslothVisionDataCollator",
        "skip_prepare_dataset",
        "Write the LaTeX representation for this image.",
    ):
        assert shared in notebook_code, f"{shared} missing from the notebook"
        assert shared in script, f"{shared} missing from the script"
