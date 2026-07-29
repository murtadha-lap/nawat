# Nawāt (نواة)

**The local-first training workspace for AI researchers.**

Nawāt is a self-hosted platform for fine-tuning and serving open-weight models on a
single GPU workstation whose local storage is far smaller than its working corpus.
Object storage is the source of truth for every model, dataset and run artifact;
local disk is a disposable cache managed automatically. You write an ordinary
Unsloth notebook or training script, and never touch a file to make room for it.

It is a Python library first — `import nawat` in a Colab-style notebook and keep
working the way you already do — with a CLI and an HTTP control plane over the
same implementation.

Working examples: [examples/](examples/) — the Unsloth Qwen3.5-0.8B vision
notebook, and the same fine-tune as a submitted script.

## Researcher quick path

Nawāt keeps durable models, datasets, adapters, logs, and run metadata in your own S3-compatible store while a bounded local cache holds only the current training or inference working set.

1. Configure storage and verify it: `cp .env.example .env`, edit the cache ceiling and S3 settings, then run `nawat check --create-bucket`.
2. Open [`examples/latex_ocr_qwen3_5_vision.ipynb`](examples/latex_ocr_qwen3_5_vision.ipynb), the complete Unsloth Qwen3.5-0.8B vision fine-tune on `unsloth/LaTeX_OCR`.
3. For a background run, submit [`examples/train_latex_ocr.py`](examples/train_latex_ocr.py) with the model, dataset, and hyperparameters shown below.
4. Start the base with `nawat serve`, then hot-load `runs/<run-id>/adapter` with `nawat adapter`; vLLM serves it without a merge or restart.

```text
Hugging Face (first use only)
            ↓
S3-compatible object storage  ← durable source of truth
            ↓
bounded local cache           ← active working set
            ↓
      Unsloth or vLLM
```

The local disk only needs one working set, not the full research corpus. Resolution follows local cache → object storage → Hugging Face, and the first internet fetch writes through to your store. In-use, unreplicated, or unverifiable artifacts are never evicted.
For image, audio, or document corpora with many small files, `nawat shard /data/corpus datasets/lab/corpus-v1 --shard-size 512MB` creates WebDataset-compatible tar shards for efficient sequential reads.


---

## The problem

A single-GPU workstation — 16 GB of VRAM, ~200 GB of NVMe — working against
terabytes of models and data fails the same way every week:

1. A run dies partway through because the disk filled with checkpoints.
2. `rm -rf` frees space — and occasionally destroys an adapter that was never
   backed up anywhere.
3. The next experiment re-downloads the same 16 GB base model from the internet,
   because the local copy was deleted to make room.
4. Merged FP16 exports and GGUF quantizations pile up silently until the disk
   fills again.
5. Which script, which data, which hyperparameters produced which adapter lives
   in shell history and file names.

Experiment throughput ends up bounded by storage housekeeping, not GPU time —
and results are hard to reproduce weeks later.

## What Nawāt does about it

| Failure | What Nawāt does |
| --- | --- |
| Disk fills mid-run | A configurable cache ceiling; least-recently-used artifacts are evicted automatically to make room |
| `rm -rf` loses work | Nothing is deleted until its replica in object storage is verified **file by file, by name and size, at the moment of deletion**. Unreplicated, in-use or unverifiable artifacts are never touched — if space cannot be freed safely, you get a refusal that says exactly what is holding it |
| Repeat downloads | A model or dataset is fetched from the internet **once, ever**: the first fetch writes through to object storage, and every later use — including after eviction — comes from there |
| Exports accumulate | Every artifact class a run writes (`adapter/`, `merged/`, `gguf/`) is uploaded, verified, and reclaimed from local disk the moment the run succeeds |
| Lost provenance | Every run records its script, inputs, parameters, log and resulting artifact keys, durably, whether it succeeds or fails |

Plus: serve any base model with vLLM and **hot-load a trained LoRA in seconds —
no merge, no restart** — so testing a fine-tune costs nothing; and after seeding,
training runs with hub access disabled, so a run can never silently download.

---

## System requirements

Nawāt itself asks for very little. What it orchestrates does.

| | Requirement | Why |
| --- | --- | --- |
| **OS** | Linux | Leases are keyed to `/proc` boot id and process start time |
| **Python** | 3.11+ | |
| **GPU** | NVIDIA, compute capability 7.0+ | Training and serving both; a T4 (7.5) is enough for the example |
| **Driver** | CUDA 12.x capable | Matches the `torch==2.8.0` cu12 build below |
| **Disk** | Enough for one working set, not your corpus | The point of the project. 200 GB is comfortable |
| **Object store** | Any S3-compatible endpoint | The source of truth: RustFS, MinIO, SeaweedFS, Ceph, AWS S3 |

Three separate pieces of software, installed once each. Nawāt depends on none of
them at import time — it runs both as subprocesses — so a missing one costs you
that capability and nothing else:

| Component | Needed for | Without it |
| --- | --- | --- |
| **Nawāt** | Everything | — |
| **Unsloth** | Training | `nawat submit` and the notebook fail; storage and serving still work |
| **vLLM** | Serving, and hot-loading adapters | `nawat serve` / `nawat adapter` fail; training still works |

## Installation

Do this once. The notebook has no install cell — it assumes the environment
below already exists, so every session after this one starts at the first real
cell instead of spending minutes re-resolving packages that never changed.

Use [uv](https://docs.astral.sh/uv/) throughout: it keeps a single wheel cache
for the whole machine (`~/.cache/uv`), so the *next* environment you build
hardlinks Torch from local disk instead of downloading it again — which matters
here, because you are about to build two of them.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if you do not have it
```

### 1. Nawāt

Small — `boto3` and nothing else — because the heavy machinery stays in your
environment, where you control the versions.

```bash
python3 -m venv ~/nawat/.venv && source ~/nawat/.venv/bin/activate
uv pip install "nawat[notebook] @ git+https://github.com/murtadha-lap/nawat.git"
```

| Extra | Pulls in | For |
| --- | --- | --- |
| *(none)* | `boto3` | The cache, the CLI, everything storage |
| `hub` | `huggingface_hub` | Fetching models and datasets not yet in your store |
| `notebook` | `hub` + `matplotlib` | Notebook use, with the loss trace plotted inline |
| `api` | `fastapi`, `uvicorn`, `httpx` | `nawat api` — control plane, queue, OpenAI-compatible `/v1` |
| `agent` | `claude-agent-sdk` | `nawat agent` — proposed script changes, diff and approval |

### 2. Unsloth — to train

Into the same environment as Nawāt, since the notebook imports both. The pinned
set is the one Unsloth's own notebook uses:

```bash
uv pip install "torch==2.8.0" "triton>=3.3.0" numpy pillow torchvision \
    bitsandbytes xformers==0.0.32.post2 \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
uv pip install --no-deps "torchcodec==0.7.0"
uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
uv pip install transformers==5.2.0
uv pip install --no-build-isolation flash-linear-attention "causal_conv1d==1.6.0"
uv pip install --no-deps --upgrade "torchao>=0.16.0"
```

`causal_conv1d` builds against `torch==2.8.0`; on a newer Torch it compiles from
source and takes several minutes.

**Ampere or newer only** (compute capability 8.0+ — A100, RTX 30xx/40xx, L4).
Check with `nvidia-smi --query-gpu=compute_cap --format=csv`:

```bash
uv pip install --no-deps "apache-tvm-ffi==0.1.9" "tilelang==0.1.8"
```

On anything older — a T4 at 7.5 — skip it and set `FLA_TILELANG=0` before Unsloth
is imported, or the import fails. The notebook's preflight cell detects this and
sets it for you.

### 3. vLLM — to serve

**In its own virtualenv.** vLLM and Unsloth both pin Torch, and not to the same
version; putting them together is how you end up with an environment that
resolves but crashes at load. You do not have to choose, because Nawāt never
imports vLLM — it launches the `vllm` binary it finds on `PATH` as a subprocess:

```bash
python3 -m venv ~/nawat/.venv-vllm
~/nawat/.venv-vllm/bin/pip install -U uv
~/nawat/.venv-vllm/bin/uv pip install vllm

# Put just that binary on PATH, without activating the environment:
mkdir -p ~/.local/bin && ln -sf ~/nawat/.venv-vllm/bin/vllm ~/.local/bin/vllm
vllm --version                     # must answer from any shell Nawāt runs in
```

If you would rather install vLLM into the same environment, nothing stops you —
`uv pip install vllm` — but expect to pin Torch by hand afterwards.

What Nawāt runs, so you can reproduce it by hand when debugging:

```bash
vllm serve <staged path> --served-model-name <key> --port 8001 \
     --host 127.0.0.1 --enable-lora
```

with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` in the environment — that is the
switch that makes `nawat adapter` able to load a LoRA into a running server
instead of restarting it. Tune the rest through `NAWAT_SERVE_EXTRA_ARGS`
(`--gpu-memory-utilization`, `--max-model-len`, quantization, parallelism); see
the serving block in `.env.example`.

vLLM must support the architecture you are serving. When it does not, `nawat
serve` fails with vLLM's own message in `nawat session --log`.

That is the whole setup. From here the notebook opens and runs.

## Getting started, step by step

**1. Have an S3-compatible object store.** Anything that speaks the protocol:
[RustFS](https://docs.rustfs.com/installation/linux/quick-start), MinIO, SeaweedFS,
Ceph, or AWS S3 itself. This is where your models, datasets and results actually
live; the local disk is only a cache in front of it.

**2. Configure.** Copy the template and fill in the endpoint, bucket, credentials
and how much local disk Nawāt may use:

```bash
cp .env.example .env && $EDITOR .env
```

The four that matter:

```bash
NAWAT_S3_ENDPOINT=http://192.168.0.155:9000   # your object store
NAWAT_S3_BUCKET=ai-model
NAWAT_CACHE_CEILING=120GB                     # how much local disk Nawāt may use
NAWAT_MIN_FREE=10GB                           # headroom it will never eat into
```

`nawat` finds `.env` on its own — the nearest one at or above the working
directory — so there is nothing to source, and it works the same from a notebook
kernel started anywhere in the tree. Exported variables still win, for one-off
overrides.

**3. Prove the pairing works.** Not a ping: this writes, lists, verifies and
deletes a probe object, because those are the four things everything else
depends on.

```bash
nawat check --create-bucket
```

```text
ok    configuration              /home/you/nawat/.env
ok    cache root                 /home/you/nawat/cache
ok    cache ceiling              120 GB ceiling + 10.0 GB reserve on a 250 GB filesystem
ok    object storage reachable   http://192.168.0.155:9000 · bucket ai-model
ok    object storage round trip  write, list, verify and delete all succeeded

Ready. Object storage is reachable and this host can publish, verify and reclaim.
```

**4. Pull something in.** The first fetch goes to Hugging Face and writes through
to your store on the way past; every later one — including after the local copy
is evicted — comes from your store.

```bash
nawat resolve models/unsloth/Qwen3.5-0.8B     # prints the local path
nawat status                                  # what that cost you
```

**5. Train.** Either open
[examples/latex_ocr_qwen3_5_vision.ipynb](examples/latex_ocr_qwen3_5_vision.ipynb)
and run the cells, or submit the script version. Submitted scripts live in the
workspace — `~/nawat/workspace` by default, `NAWAT_WORKSPACE` to move it — and
anything outside it is refused, so copy it in first:

```bash
cp examples/train_latex_ocr.py ~/nawat/workspace/
nawat submit train_latex_ocr.py \
  --model   models/unsloth/Qwen3.5-0.8B \
  --dataset datasets/unsloth/LaTeX_OCR \
  --param   max_steps=60
```

**6. Watch it.** From any other shell, at any time:

```bash
nawat runs                 # history, newest first
nawat logs    <id> -f      # the trainer's output, live
nawat metrics <id> -f      # the loss trace as a terminal chart, live
```

**7. Use the result.** Serve the base once and hot-load the adapter onto it —
seconds, no merge, and no second copy of the weights (needs vLLM from step 3):

```bash
nawat serve models/unsloth/Qwen3.5-0.8B     # stages the weights, starts vLLM
nawat adapter runs/<id>/adapter --name latex-ocr
nawat session                               # what is up, on which port
nawat session --log --tail 50               # vLLM's own output, if it did not start
nawat session --stop                        # or walk away: idle teardown does it
```

## Keys

One name per artifact, mapping 1:1 onto an object-storage prefix and a local path:

```text
models/unsloth/Qwen3.5-0.8B      ← the tail IS the Hugging Face repo id
datasets/unsloth/LaTeX_OCR
runs/2026-07-28-a91f/adapter
```

That is the whole trick behind "downloaded once, ever": resolving
`models/unsloth/Qwen3.5-0.8B` checks local disk, then object storage, then — only
if neither has it — Hugging Face, writing through to object storage before
returning. No mapping table, no registration step.

---

## Using it with Unsloth, in a notebook

Take the standard Unsloth vision notebook — Qwen3.5-0.8B fine-tuned on
`unsloth/LaTeX_OCR`. What changes is the lines that name a location, plus two
that open and close the run. Everything Unsloth-specific — the model setup, the
collator, the trainer arguments — stays exactly as the notebook has it.

```python
import nawat

run = nawat.begin_run(
    model   = "models/unsloth/Qwen3.5-0.8B",
    dataset = "datasets/unsloth/LaTeX_OCR",
    params  = {"max_steps": 30, "learning_rate": 2e-4},
    notes   = "LaTeX OCR baseline",
)

from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(   # ← was "unsloth/Qwen3.5-0.8B"
    run.model_dir, load_in_4bit = False, use_gradient_checkpointing = "unsloth",
)
model = FastVisionModel.get_peft_model(model, r = run.param("rank", 16), ...)

from datasets import load_dataset
dataset = load_dataset(run.dataset_dir, split = "train")   # ← was "unsloth/LaTeX_OCR"

trainer = SFTTrainer(
    model = model, tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    callbacks = [run.callback()],                 # live loss trace, one argument
    args = SFTConfig(
        max_steps     = run.param("max_steps", 30),
        learning_rate = run.param("learning_rate", 2e-4),
        # Checkpoints are scratch, not artifacts — kept out of the published tree:
        output_dir = str(run.scratch_dir("trainer")),
        ...
    ),
)
trainer.train()

model.save_pretrained(run.artifact_dir("adapter"))    # ← was "qwen_lora"
tokenizer.save_pretrained(run.artifact_dir("adapter"))

run.finish()    # upload, verify file by file, reclaim the disk, close the record
```

`begin_run` stages both inputs — cache, then object storage, then Hugging Face —
and leases them to the kernel's own pid. The lease dies with the kernel, crash
included, so a forgotten notebook cannot wedge the cache and a live one cannot
lose its weights to an eviction between cells. While the run is open the hub is
switched off, so a typo in a repo id fails loudly instead of quietly pulling
gigabytes onto a disk with no room for them.

`run.finish()` publishes every directory under `run.out_dir` as its own artifact
— `runs/<id>/adapter`, and `merged/` or `gguf/` if you saved them — verifies each
in object storage file by file, reclaims the local copy, and releases the inputs.
A run that raises instead publishes nothing and keeps its log and record:

```python
with nawat.begin_run(model=..., dataset=...) as run:
    ...          # an exception here records the failure and re-raises
```

The full notebook is [examples/latex_ocr_qwen3_5_vision.ipynb](examples/latex_ocr_qwen3_5_vision.ipynb).

| Notebook API | What it is |
| --- | --- |
| `run.model_dir` / `run.dataset_dir` | Staged input paths, ready for `from_pretrained` / `load_dataset` |
| `run.artifact_dir(name)` | A directory published as `runs/<id>/<name>` on `finish()` |
| `run.scratch_dir(name)` | Space that is *not* published, deleted at the end — put checkpoints here |
| `run.param(name, default)` | A hyperparameter, typed from its default, overridable by `--param` |
| `run.callback()` | A `transformers` callback streaming the metric series |
| `run.finish()` / `run.fail(e)` / `run.cancel()` | Close the record; `run.close()` just lets go |
| `nawat.history()` / `nawat.trace(id)` | Run history and metric series, for plotting inline |

Without a run record — just the cache — the two calls that matter are still one
import each:

```python
model_dir = nawat.resolve("models/unsloth/Qwen3.5-0.8B")   # staged + held
nawat.publish(out_dir, "runs/2026-07-28-a91f/adapter")     # upload, verify, reclaim
```

`resolve` holds the artifact for the lifetime of the kernel; `with
nawat.holding(...)` scopes it tighter.

## The same run, submitted

A notebook is for deciding what to run; `nawat submit` is for running it for six
hours without a browser tab open. The body is the same code — `nawat.model_dir()`,
`nawat.param()` and `nawat.artifact_dir()` read the environment the executor
injected when there is one, and the open kernel run otherwise, so cells move
between the two without an edit. Only `begin_run`/`finish` go away, because the
executor does both around the process.

**`~/nawat/workspace/train_latex_ocr.py`:**

```python
# The Unsloth LaTeX-OCR notebook as a Nawāt run. Every accessor has an
# os.environ equivalent, so this script also runs unmodified outside the
# platform. Full version: examples/train_latex_ocr.py
import tempfile

import nawat

from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    nawat.model_dir(),                         # ← was "unsloth/Qwen3.5-0.8B"
    load_in_4bit = False,
    use_gradient_checkpointing = "unsloth",
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16, lora_alpha = 16, lora_dropout = 0,
    bias = "none", random_state = 3407,
)

from datasets import load_dataset
dataset = load_dataset(nawat.dataset_dir(), split = "train")   # ← was "unsloth/LaTeX_OCR"

instruction = "Write the LaTeX representation for this image."
def convert(sample):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "text",  "text":  instruction},
            {"type": "image", "image": sample["image"]}]},
        {"role": "assistant", "content": [
            {"type": "text",  "text":  sample["text"]}]},
    ]}
converted_dataset = [convert(s) for s in dataset]

from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model)
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = converted_dataset,
    callbacks = [nawat.metrics.trainer_callback()],   # the live loss trace
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # Hyperparameters arrive from `nawat submit --param ...`, with the
        # notebook's values as defaults:
        max_steps     = nawat.param("max_steps", 30),
        learning_rate = nawat.param("learning_rate", 2e-4),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        # Intermediate checkpoints are scratch, not artifacts — keep them out of
        # the output directory so they are not published:
        output_dir = tempfile.mkdtemp(prefix = "trainer-"),
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)
trainer.train()

# Everything written under the output directory is published as its own artifact
# class when the run exits 0 — uploaded, verified file by file, then reclaimed.
adapter = nawat.artifact_dir("adapter")            # ← was "qwen_lora"
model.save_pretrained(adapter)
tokenizer.save_pretrained(adapter)

# Only on request, for deployment — each becomes runs/<id>/<name>:
# model.save_pretrained_merged(nawat.artifact_dir("merged"), tokenizer)
# model.save_pretrained_gguf(nawat.artifact_dir("gguf"), tokenizer, quantization_method = "q4_k_m")
```

**Submit it:**

```bash
nawat submit train_latex_ocr.py \
  --model   models/unsloth/Qwen3.5-0.8B \
  --dataset datasets/unsloth/LaTeX_OCR \
  --param   max_steps=60 --param learning_rate=2e-4 \
  --notes   "LaTeX OCR baseline"
```

What happens, in order:

1. **Stage.** Model and dataset resolve from cache → object storage → Hugging Face
   (first time only; written through to the store so it never happens again).
   Space is made safely first if the ceiling requires it.
2. **Hold.** Both inputs are leased to the trainer's own pid — nothing can evict
   them mid-run, and the lease dies with the process, crash included.
3. **Train offline.** The subprocess runs with `HF_HUB_OFFLINE=1` forced. The log
   streams to your terminal (`nawat logs <id> -f` from any other shell).
4. **Publish.** `runs/<id>/adapter` (and `merged/`, `gguf/` if you saved them) is
   uploaded, verified by name and size, and the local copy reclaimed. The log and
   run record go to `runs/<id>/record`. A failed run publishes nothing but keeps
   both.

**What changed vs. the stock Unsloth notebook** — nothing else did:

| Unsloth notebook | In a kernel | Submitted |
| --- | --- | --- |
| `from_pretrained("unsloth/Qwen3.5-0.8B")` | `from_pretrained(run.model_dir)` | `from_pretrained(nawat.model_dir())` |
| `load_dataset("unsloth/LaTeX_OCR")` | `load_dataset(run.dataset_dir)` | `load_dataset(nawat.dataset_dir())` |
| `output_dir = "outputs"` | `run.scratch_dir("trainer")` | `tempfile.mkdtemp()` |
| `save_pretrained("qwen_lora")` | `save_pretrained(run.artifact_dir("adapter"))` | `save_pretrained(nawat.artifact_dir("adapter"))` |
| `push_to_hub(..., token=...)` | `run.finish()` | automatic on exit 0 |
| manual `rm -rf` between runs | automatic, refuses when unsafe | same |

The middle and right columns are the same functions: `nawat.model_dir()` reads
the environment when the executor set one and the open kernel run otherwise. Code
written against the module-level form runs in both places unchanged.

**Watch the run from a chart, not scrollback.** One argument to the trainer —

```python
trainer = SFTTrainer(..., callbacks = [nawat.metrics.trainer_callback()])
```

— and every logging step (loss, learning rate, grad norm, computed steps/second,
epoch boundaries) streams into the run's metric series. Scripts that don't use a
Trainer call `nawat.metrics.log(step=..., loss=...)` directly, and any language
can append JSON lines to `$NAWAT_METRICS_PATH`. Then:

```text
$ nawat metrics <id>            # -f to stream live
run metrics-smoke · 33 points

loss   █▇▆▆▅▅▄▄▃▃▃▃▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁  2.256 → 0.1088   min 0.1088 @ 30
lr     ███▇▇▇▇▆▆▆▆▅▅▅▅▄▄▄▄▃▃▃▃▂▂▂▂▁▁▁  0.0001933 → 0    min 0 @ 30

event  step 10  epoch_end
```

The series lives beside the run log — never in the cache, so it is never
evicted — and is published to object storage with the run record: the chart
renders identically long after the weights themselves are gone. Over HTTP:
`/runs/{id}/metrics`, `/runs/{id}/metrics/stream` (server-sent events), and
`/metrics/compare?run=a&run=b&name=loss` for overlaying runs on shared axes.

In a kernel, `nawat.trace(run.id)` returns the same series grouped by name,
ready to hand to matplotlib inline.

Notebooks submit too (`nawat submit explore.ipynb ...`): the notebook is executed
with nbconvert and the executed copy is archived as the run record. Drop the
`begin_run`/`finish` cells first — under the executor the run already exists —
and the pip-install cell is unnecessary on a host with Unsloth already installed.

**Test the adapter — no merge, under two minutes:**

```bash
nawat serve models/unsloth/Qwen3.5-0.8B          # stages weights, starts vLLM
nawat adapter runs/<id>/adapter --name latex-ocr # hot-loads onto the running base

curl http://127.0.0.1:8001/v1/chat/completions -d '{
  "model": "latex-ocr",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "Write the LaTeX representation for this image."}]}]
}'

nawat session --stop    # or just walk away: idle teardown releases GPU + disk
```

A ~200 MB LoRA against a multi-GB base: merging to test would multiply the
storage cost per experiment by two orders of magnitude. Serving the base once and
swapping adapters at runtime makes testing effectively free.

---

## Exports: merged FP16 and GGUF

Not everything can load an adapter. `llama.cpp`, Ollama and most edge runtimes
want one self-contained file, which means merging the LoRA back into the base and
converting the result. Both outputs are roughly the size of the base model, and
on an ordinary setup both accumulate forever, because nothing ever decides they
can go.

Here they are artifact classes like any other: written into the run's output
directory, published and verified on success, then reclaimed from local disk.

```python
# GGUF conversion consumes merged weights, so it implies this step:
model.save_pretrained_merged(str(run.artifact_dir("merged")), tokenizer)

model.save_pretrained_gguf(
    str(run.artifact_dir("gguf")), tokenizer,
    quantization_method = "q4_k_m",        # or "q8_0", "q5_k_m", "f16"
)
run.finish()                               # → runs/<id>/merged, runs/<id>/gguf
```

Several quantizations at once is far cheaper than one at a time — llama.cpp is
built and the FP16 intermediate produced once, not once per format:

```python
model.save_pretrained_gguf(str(run.artifact_dir("gguf")), tokenizer,
                           quantization_method = ["q4_k_m", "q5_k_m", "q8_0"])
```

Pass `str(...)`: Unsloth's `save_pretrained_*` join paths as text, and a `Path`
trips them up.

**When it refuses.** GGUF conversion asks llama.cpp to recognise the
architecture, and support for vision encoders lags well behind support for text
models. If it fails, that is llama.cpp's answer, not something Nawāt can work
around — so guard it and let the run keep what it already earned:

```python
try:
    model.save_pretrained_gguf(str(run.artifact_dir("gguf")), tokenizer,
                               quantization_method = "q4_k_m")
except Exception as exc:
    run.log(f"gguf conversion failed: {exc}")   # adapter and merged still publish
```

An artifact directory left empty is skipped rather than published, so a failed
conversion leaves no trace but the log line.

**Using one later.** It is in object storage, not on your disk. `nawat resolve`
brings it back, making room first if the ceiling requires it:

```bash
llama-cli -m "$(nawat resolve runs/<id>/gguf)"/*.gguf -p "..."

# Promote it to a named deployment artifact, separate from the run that made it:
nawat publish "$(nawat resolve runs/<id>/gguf)" exports/latex-ocr-q4
nawat keep exports/latex-ocr-q4        # exempt it from reclamation

# Ollama:
printf 'FROM %s\n' "$(nawat resolve exports/latex-ocr-q4)"/*.gguf > Modelfile
ollama create latex-ocr -f Modelfile
```

In the submitted script the exports are behind a parameter, off by default:

```bash
nawat submit train_latex_ocr.py --model ... --dataset ... \
  --param export=gguf --param quantization=q4_k_m
```

---

## Commands

| Command | What it does |
| --- | --- |
| `nawat status` | Occupancy against the ceiling, disk, what is held |
| `nawat ls` | What is on local disk — `K` kept · `L` in use · `R` in object storage |
| `nawat resolve KEY` | Make it present locally and print the path |
| `nawat keep KEY` / `release KEY` | Exempt from reclamation, or stop |
| `nawat free [--need 16GB]` | Reclaim least-recently-used space; `--dry-run` to look first |
| `nawat publish DIR KEY` | Upload, verify, reclaim |
| `nawat add DIR KEY` | Adopt a directory already on disk (seed your own datasets) |
| `nawat verify KEY` | Compare the local copy against its replica |
| `nawat rm KEY` | Remove one artifact (refuses if unreplicated) |
| `nawat registry` | What object storage holds, cached or not |
| `nawat leases` | What is in use, and by whom |
| `nawat submit SCRIPT ...` | Run a training script as a recorded run |
| `nawat runs` / `run ID` / `logs ID [-f]` / `cancel ID` | History, record, log, stop |
| `nawat metrics ID [-f]` | The metric series as a terminal trace, or streamed live |
| `nawat serve KEY` / `adapter KEY` / `session --stop` | Serve, hot-load, tear down |
| `nawat eval ID --data KEY` | Score a run's adapter — CER/WER into its record |
| `nawat agent "…" [--run ID]` | Propose a script change; diff + approval, never autonomous |
| `nawat describe ID` | Store a plain-language account of a run in its record |
| `nawat estimate --model KEY` | Will it fit? VRAM estimate before spending GPU hours |
| `nawat shard DIR KEY` | Pack a small-file corpus into streamable tar shards |
| `nawat api` | The control plane over HTTP (schema at `/docs`) |
| `nawat check` | Prove this host can store, verify and reclaim |
| `nawat config` | Configuration in force, credentials redacted |

Exit codes are stable: `2` invalid key, `3` not found, `4` store unavailable,
`5` verification failed, `6` insufficient space, `7` offline, `8` protected.

`nawat api` serves everything above over HTTP — FIFO run queue, server-sent-event
log streaming, and a stable OpenAI-compatible `/v1` that forwards to whichever
inference session is current, so a client configured once survives restarts and
model changes. `NAWAT_API_TOKEN` enables bearer-token auth; `/health` stays open.

---

## The Python API

Everything the CLI does, the library does — one implementation, three front
doors. Beyond the run object above:

| Call | What it does |
| --- | --- |
| `nawat.resolve(key)` | Stage an artifact and hold it for the kernel; returns the path |
| `nawat.holding(*keys)` | The same, scoped to a `with` block |
| `nawat.publish(dir, key)` | Upload, verify file by file, reclaim the local copy |
| `nawat.keep(key)` / `release(key)` | Exempt from reclamation, or stop |
| `nawat.free_space(need)` | Reclaim least-recently-used space, refusing when unsafe |
| `nawat.status()` / `artifacts()` | Occupancy against the ceiling; what is on disk |
| `nawat.verify(key)` | Compare the local copy against its replica |
| `nawat.history()` / `run_record(id)` / `trace(id)` | Run history, one record, its metric series |

Errors are typed and carry a remedy: `NotFound`, `InsufficientSpace`,
`StoreUnavailable`, `VerificationFailed`, `Protected`, `Offline`, `InvalidKey`,
all deriving from `NawatError` with `.cause` and `.remedy`.

## The agent

The agent is optional and never autonomous (`NAWAT_AGENT_BACKEND=claude` for the
Claude Agent SDK — confined read-only to the workspace — or `local` for any
OpenAI-compatible endpoint, fully on-premises). Every proposal passes a syntax
gate and a VRAM estimate before it is even offered, and reaches the workspace
only through your approval, committed to git with its prompt and backend:

```bash
nawat agent "the run OOMed at step 40; halve the memory it needs" --run <id>
```

---

## The rule everything follows

> Object storage is truth. Local disk is disposable. Verify before deleting,
> always. If space cannot be freed safely, refuse and say why.

- Eviction re-verifies the replica **at the moment of deletion** — never from a
  cached flag. Store unreachable → nothing is deleted; you get a full disk and an
  explanation, not a gamble.
- Artifacts in use are held by a **lease keyed to a live process** (boot id, pid,
  start time), not a timeout: a six-hour run cannot have its weights evicted, and
  a crashed run cannot deadlock the cache.
- Downloads land in staging and are renamed into place atomically; an interrupted
  transfer never looks complete.
- Each artifact directory carries a `.nawat-artifact.json` marker, so the cache
  describes itself on disk — delete the state database and it rebuilds.

## Running it as a service

`nawat api` in the foreground is enough to try. To keep it up, a unit file of
about ten lines:

```ini
# /etc/systemd/system/nawat-api.service
[Unit]
Description=Nawat control plane
After=network-online.target

[Service]
User=you
WorkingDirectory=/home/you/nawat
ExecStart=/home/you/nawat/.venv/bin/nawat api
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Recovery is unattended by design, so there is nothing to do after a reboot: the
cache reconciles against what is actually on disk, leases from before the boot
are cleared (they carry the boot id), and runs that did not survive are marked
failed rather than left claiming to be running.

**What breaks, and what happens.**

| Situation | What happens | What you do |
| --- | --- | --- |
| Host reboot | Cache reconciles, stale leases cleared, interrupted runs marked failed | Nothing |
| Trainer crashes | Exit code recorded, log kept locally and in `runs/<id>/record` | `nawat logs <id>` |
| Object store down | **Nothing is evicted** — verification cannot complete, so deletion is refused. Fetches and publishes fail naming the endpoint; a run already training continues | Restore the store, then `nawat publish` |
| State database lost | Cache rebuilds from the `.nawat-artifact.json` marker in each artifact; run history rebuilds from `run.json` | `nawat status` triggers it |
| Disk filling | `nawat status` and `GET /health` both warn at 90% of the ceiling | `nawat free --need 20GB`, or raise the ceiling |

**The one state where data loss is possible** is an artifact that exists only on
this disk — a publish that failed, or something added but never uploaded.
`nawat status` reports it as *N GB exists only on this disk*, and `nawat ls`
flags it. Everything else is replicated by definition, because nothing is deleted
until its replica is verified.

**Security.** The API binds loopback only; put a reverse proxy in front of it
rather than binding wider. Set `NAWAT_API_TOKEN` before exposing it at all —
`/health` stays open for monitoring, everything else then needs the bearer token.
Credentials live in `.env` and never reach a log, a run record or an API
response; `nawat config` prints `"set"` in their place. Training scripts run with
your privileges and are not sandboxed, so do not point it at code you would not
run yourself.

## Working on it

```bash
git clone https://github.com/murtadha-lap/nawat.git && cd nawat
python3 -m venv .venv && .venv/bin/pip install -e ".[notebook,api]"
```

An editable install, so edits to `nawat/` take effect in the next kernel without
reinstalling.

## Known limits

- Mutating cache operations take a host-wide lock; simultaneous stages serialise.
- Verification compares name and size, not content hashes (as specified) —
  checksums would catch silent corruption at the cost of reading every byte back.
- vLLM must support the architecture being served; adapter hot-load requires the
  vLLM backend.

## Licence

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for anyone to use, study,
modify and share, for any **noncommercial** purpose. Research, teaching, personal
projects, hobby work, and use by charities, schools, universities, public research
bodies and government all count as permitted, whatever their funding.

Commercial use is not granted. If you want it, ask.

Two things to be clear about:

- **This is not an open-source licence** in the OSI sense, which forbids
  restrictions on the field of use. GitHub will not show it as a recognised
  open-source licence, and it is incompatible with GPL-family code. That is the
  price of the noncommercial condition, not an oversight.
- **The examples are LGPL-3.0, not this.** `examples/latex_ocr_qwen3_5_vision.ipynb`
  and `examples/train_latex_ocr.py` are derived from Unsloth's notebook, which is
  LGPL-3.0, and that licence does not permit adding a noncommercial restriction
  downstream. They therefore stay under LGPL-3.0 and may be used commercially on
  its terms — see [examples/LICENSE](examples/LICENSE).

Nawāt does not bundle Unsloth, vLLM or llama.cpp; each is installed separately
under its own licence.
