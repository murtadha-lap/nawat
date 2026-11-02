# Examples

| File | What it is |
| --- | --- |
| [latex_ocr_experiment.ipynb](latex_ocr_experiment.ipynb) | Look at the LaTeX OCR data, plan the `vlm-sft` experiment (Qwen3.5-0.8B on `unsloth/LaTeX_OCR`), approve it, run it, and read its report, traces and loss curve |

Training happens only as an experiment — a fixed backbone with declared
parameters, approved by a person — so the examples are experiments too, not
training scripts. The notebook is where you look at data and results; the
experiment is what trains.

Set the environment up once first —
[Installation](../README.md#installation) covers Nawāt, Unsloth (to train) and
vLLM (to serve) — then:

```bash
cp .env.example .env && $EDITOR .env      # endpoint, bucket, credentials, ceiling
nawat check --create-bucket
uv run nawat jupyter
```

The same experiment from a terminal:

```bash
nawat exp plan vlm-sft --name latex-ocr-smoke --set train.max_steps=3
nawat exp approve vlm-sft/latex-ocr-smoke
nawat exp run vlm-sft/latex-ocr-smoke
nawat exp report vlm-sft/latex-ocr-smoke
```
