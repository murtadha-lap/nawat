# Examples

Training examples and storage diagnostics. Start with the notebook; move to the
script when the experiment is worth a longer unattended run.

| File | What it is |
| --- | --- |
| [latex_ocr_qwen3_5_vision.ipynb](latex_ocr_qwen3_5_vision.ipynb) | The Unsloth vision notebook — Qwen3.5-0.8B on `unsloth/LaTeX_OCR` — driven from a kernel as a recorded run |
| [train_latex_ocr.py](train_latex_ocr.py) | The same body as a script, for `nawat submit` |
| [network_monitor.py](network_monitor.py) | Benchmark the configured RustFS connection with automatic test-object cleanup |

The training examples do not have an install cell. Set the environment up once —
[Installation](../README.md#installation) covers Nawāt, Unsloth (to train) and
vLLM (to serve) — then configure the store:

```bash
cp .env.example .env && $EDITOR .env      # endpoint, bucket, credentials, ceiling
nawat check --create-bucket
```

After that the notebook opens and runs. Its first cell is a preflight: it reports
the versions it found and the GPU it is on, and stops with a clear message rather
than a deep `ImportError` if something is missing.

## Test RustFS network speed

From the repository root, run the safe default test (three 256 MiB rounds):

```bash
python examples/network_monitor.py
```

Choose a larger payload or more rounds when needed:

```bash
python examples/network_monitor.py --size 1GiB --rounds 5
```

The test reports TCP and authenticated S3 latency plus upload/download speed. It
uses a unique `runs/nawat-speed-test-*` object and deletes it in a `finally`
cleanup, including when the transfer is interrupted.

## The three changed lines

Everything Unsloth-specific is untouched. What changes is only the lines that
name a location:

```python
model, tokenizer = FastVisionModel.from_pretrained(run.model_dir)   # was "unsloth/Qwen3.5-0.8B"
dataset          = load_dataset(run.dataset_dir, split="train")     # was "unsloth/LaTeX_OCR"
model.save_pretrained(run.artifact_dir("adapter"))                  # was "qwen_lora"
```

`run.model_dir` resolves through local cache → object storage → Hugging Face, in
that order, and the last of those happens at most once in the artifact's life.
`run.artifact_dir("adapter")` is an ordinary directory that becomes
`runs/<id>/adapter` — uploaded, verified file by file, and reclaimed from local
disk — when `run.finish()` returns.

## Notebook or script?

They are the same code. `nawat.model_dir()`, `nawat.param()` and
`nawat.artifact_dir()` read the environment when the executor set one and the
open kernel run otherwise, so a cell moves between the two without an edit. The
difference is who opens and closes the run:

```python
# in a notebook — you do
run = nawat.begin_run(model=..., dataset=..., params={...})
...
run.finish()
```

```bash
# as a script — the executor does, around the process
nawat submit train_latex_ocr.py --model ... --dataset ... --param max_steps=500
```

Use the notebook to decide what to run: the images are in front of you, the loss
trace plots inline, and a failed cell costs nothing. Use `nawat submit` to
actually run it — the trainer survives a closed laptop, the queue serialises
against the one GPU, and `nawat logs -f` reaches it from anywhere.

## While it runs

```bash
nawat runs                 # history, newest first
nawat logs    <id> -f      # the trainer's output, live
nawat metrics <id> -f      # the loss trace as a terminal chart, live
nawat status               # occupancy against the cache ceiling
```

## After it finishes

Serve it — the adapter hot-loads onto the base, nothing is merged:

```bash
nawat serve models/unsloth/Qwen3.5-0.8B     # stage the base, start vLLM
nawat adapter runs/<id>/adapter --name latex-ocr
# ... an OpenAI-compatible endpoint on :8001
nawat session --stop
```
