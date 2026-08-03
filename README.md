<div align="center">

<img src="docs/image.png" alt="Nawāt logo" width="280">

### Train locally. Store durably. Reproduce every run.

**A local-first training and inference workspace for AI researchers using Unsloth, LoRA, vLLM, and S3-compatible object storage.**

<p>
  <a href="https://github.com/murtadha-lap/nawat"><img alt="Repository" src="https://img.shields.io/badge/GitHub-murtadha--lap%2Fnawat-181717?logo=github"></a>
  <a href="LICENSE"><img alt="License: PolyForm Noncommercial 1.0.0" src="https://img.shields.io/badge/license-PolyForm%20Noncommercial-16a34a"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="Platform: Linux" src="https://img.shields.io/badge/platform-Linux-FCC624?logo=linux&logoColor=black">
</p>

<p>
  <img alt="Unsloth" src="https://img.shields.io/badge/training-Unsloth-7C3AED">
  <img alt="NVIDIA CUDA" src="https://img.shields.io/badge/compute-NVIDIA%20CUDA-76B900?logo=nvidia&logoColor=white">
  <img alt="vLLM" src="https://img.shields.io/badge/inference-vLLM-0EA5E9">
  <img alt="LoRA" src="https://img.shields.io/badge/adapters-LoRA-14B8A6">
</p>

<p>
  <img alt="S3-compatible storage" src="https://img.shields.io/badge/storage-S3--compatible-FF9900?logo=amazons3&logoColor=white">
  <a href="https://rustfs.com"><img alt="RustFS" src="https://img.shields.io/badge/storage-RustFS-00AEEF?logo=rust&logoColor=white"></a>
  <img alt="Local-first" src="https://img.shields.io/badge/workflow-local--first-22C55E">
</p>

<p>
  <a href="#installation"><strong>Setup</strong></a> ·
  <a href="#python-examples"><strong>Python examples</strong></a> ·
  <a href="#first-experiment-qwen35-08b"><strong>Train a model</strong></a> ·
  <a href="#use-nawāt-in-an-unsloth-notebook"><strong>Unsloth notebook</strong></a> ·
  <a href="#inference-with-nawāt-and-vllm"><strong>Run with vLLM</strong></a> ·
  <a href="#local-storage-operations"><strong>Manage storage</strong></a> ·
  <a href="#troubleshooting"><strong>Troubleshooting</strong></a>
</p>

</div>

---

Nawāt lets one GPU workstation work with more models and datasets than fit on
its local disk. It stages only the active working set, protects files used by
live jobs, records every experiment, publishes verified artifacts to durable
object storage, and safely reclaims local space.

Your training code remains ordinary Python. Use an existing Unsloth notebook or
script, replace hub identifiers with Nawāt-managed local paths, and keep the
same workflow from a three-step smoke test through repeatable LoRA training and
vLLM inference.

> The Python package and CLI command are `nawat`. **Nawāt** (نواة) means
> “nucleus”—the small core coordinating the research workflow.

## The research problem

AI research on a local GPU is often limited by storage and workflow friction,
not only compute. Models and datasets are downloaded repeatedly, experiments
leave large checkpoints behind, and a workstation disk quickly fills as the
number of models, datasets, and adapters grows. Researchers then have to move
files manually and remember which script, parameters, dataset, and base model
produced each result.

That creates five recurring problems:

- **Local storage does not scale with the research library.** A workstation may
  have enough space for the current run, but not every model, dataset, and adapter.
- **Downloads and manual copies waste time.** The same large artifacts are
  fetched again or copied between machines without a durable catalog.
- **Experiments are difficult to reproduce.** Parameters, logs, metrics, and
  outputs can become separated from the exact inputs and script that made them.
- **Disk cleanup is risky.** It is easy to delete an active or unreplicated
  artifact, while cautious manual cleanup leaves expensive storage unused.
- **Training and inference feel disconnected.** A completed LoRA still has to
  be located, matched to its base model, staged, and served correctly.

Nawāt turns object storage into the durable research library and the local disk
into a bounded working set. It stages inputs when needed, leases files while
they are active, records the full run, verifies outputs remotely, reclaims safe
local copies, and makes trained adapters available to vLLM through stable
artifact keys.

## Why Nawāt

| | Capability | Research benefit |
| --- | --- | --- |
| 💾 | **Bounded local cache** | Work with a large model library without filling the workstation disk |
| ☁️ | **Durable object storage** | Keep models, datasets, adapters, logs, and metrics in an S3-compatible source of truth |
| 🧪 | **Reproducible runs** | Record the exact script, model, dataset, parameters, metrics, and outputs |
| 🔒 | **Lease-safe files** | Prevent active training or inference artifacts from being evicted |
| ⚡ | **Unsloth-native training** | Run ordinary scripts and notebooks against local staged paths |
| 🔌 | **Hot-loaded LoRA** | Load compatible adapters into vLLM without merging weights or restarting |
| ♻️ | **Verified reclamation** | Delete local copies only after the remote artifact is verified |
| 📦 | **Large-dataset tooling** | Publish directories or pack small-file corpora into sequential tar shards |

## Research workflow

```text
models + datasets → bounded cache → Unsloth training → verified adapter → vLLM
                           ↕
               S3-compatible object storage
```

## What Nawāt manages

| Research task | Nawāt behavior |
| --- | --- |
| Load a model or dataset | Local cache → object storage → Hugging Face |
| Avoid repeated downloads | The first hub fetch writes through to object storage |
| Prevent a full disk | Enforces a cache ceiling and safe LRU reclamation |
| Protect active work | Leases inputs so live training and inference cannot lose files |
| Preserve an experiment | Records the script, inputs, parameters, logs, metrics, and outputs |
| Survive a failed run | Keeps its checkpoints and resumes the next run from them |
| Find it again later | Asks what to call the run; the name is its folder in the bucket |
| Save an adapter | Uploads and verifies it before reclaiming the local copy |
| Compare configurations | Stores structured parameters with every run |
| Test a LoRA | Starts one vLLM base and hot-loads compatible adapters |
| Handle many small files | Packs corpora into WebDataset-compatible tar shards |

```text
Hugging Face (first use only)
            │
            ▼
S3-compatible object storage  ← durable source of truth
            │
            ▼
bounded local cache           ← current working set
            │
       ┌────┴────┐
       ▼         ▼
    Unsloth     vLLM
```

Nawāt refuses to evict an artifact that is active, unreplicated, or unverifiable.
If it cannot free space safely, it stops and explains what is holding the disk.

## Included examples

| File | Purpose |
| --- | --- |
| [`examples/latex_ocr_qwen3_5_vision.ipynb`](examples/latex_ocr_qwen3_5_vision.ipynb) | Interactive Qwen3.5 vision fine-tuning notebook |
| [`examples/train_latex_ocr.py`](examples/train_latex_ocr.py) | The same LaTeX OCR experiment as a submitted run |
| [`examples/network_monitor.py`](examples/network_monitor.py) | Safe RustFS latency and upload/download throughput test |

The example uses `unsloth/LaTeX_OCR` and can train Qwen3.5-0.8B, Qwen3.5-2B,
or another compatible Qwen3.5 vision model by changing only the model key.

## Requirements

- Linux and Python 3.11+ — `uv` installs the interpreter if the machine lacks one
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.5 or newer:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- An NVIDIA GPU supported by Torch and the selected model
- A working NVIDIA driver (`nvidia-smi` must succeed)
- CUDA 12.8 toolkit for building the documented Qwen3.5 extensions
- An S3-compatible store: RustFS, MinIO, SeaweedFS, Ceph, or AWS S3
- Enough local disk for one working set, not the full research archive

The tested workstation has an RTX 5060 Ti with 16 GB VRAM. Unsloth documents
approximate BF16 LoRA use of 3 GB for Qwen3.5-0.8B, 5 GB for 2B, and 10 GB for
4B; real usage also depends on sequence length, batch size, and trainable layers.

## Installation

Follow these steps once, in order. The training examples later in this README
reuse this environment and configuration instead of repeating the installation.

### 1. Clone Nawāt

```bash
git clone https://github.com/murtadha-lap/nawat.git
cd nawat
```

### 2. Create the Python environment and install Nawāt

Nawāt is a [uv](https://docs.astral.sh/uv/) project. One command reads
`pyproject.toml`, `uv.lock` and `.python-version`, fetches the right interpreter
if the machine does not have it, and builds `.venv` to match the lockfile
exactly:

```bash
uv sync --extra notebook
```

There is no `activate` step — `uv run` uses the project environment wherever you
are:

```bash
uv run nawat --help
```

This prints the available commands without starting a server or changing any
files. Use `uv run nawat COMMAND --help`, for example `uv run nawat resolve
--help`, to see a command's arguments. The full command guide is in
[CLI help](#cli-help).

| Command | What it does |
| --- | --- |
| `uv sync` | Build `.venv` from `uv.lock`, exactly |
| `uv sync --extra notebook` | Add `huggingface_hub` and `matplotlib` for notebook work |
| `uv sync --extra api` | Add FastAPI and Uvicorn for the control plane |
| `uv run nawat ...` | Run the CLI in the project environment |
| `uv add PACKAGE` | Add a dependency and update the lockfile |
| `uv lock --upgrade` | Re-resolve everything to the newest compatible versions |
| `uv build` | Build the wheel and source distribution into `dist/` |

`uv.lock` is committed, so the same command on another Linux box produces the
same environment down to the version of every transitive dependency. Activating
`.venv` by hand and calling `nawat` directly still works if you prefer it.

To use the storage CLI on a machine that is not doing the training — a laptop
that inspects the bucket, a second workstation — install it as a standalone tool
instead of cloning:

```bash
uv tool install git+https://github.com/murtadha-lap/nawat.git
nawat status
```

Or run it once without installing anything at all:

```bash
uvx --from git+https://github.com/murtadha-lap/nawat.git nawat registry
```

### 3. Install Unsloth

Torch, Unsloth and vLLM are deliberately **not** dependencies of Nawāt. They are
gigabytes of CUDA-specific wheels whose correct combination depends on the card
in the machine, and nothing in the package imports them — the trainer and the
inference server are subprocesses. So they are installed into the same
environment rather than locked with the project:

Installation reference: [official Unsloth pip guide](https://docs.unsloth.ai/get-started/installing-%2B-updating/pip-install).

The following is the tested dependency set for the included Qwen3.5 example:

```bash
uv pip install "torch==2.8.0" "triton>=3.3.0" numpy pillow torchvision \
  bitsandbytes xformers==0.0.32.post2 \
  "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
  "unsloth[base] @ git+https://github.com/unslothai/unsloth"

uv pip install --no-deps "torchcodec==0.7.0"
uv pip install --upgrade --no-deps \
  "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
uv pip install transformers==5.2.0
uv pip install --no-build-isolation \
  flash-linear-attention "causal_conv1d==1.6.0"
uv pip install --no-deps --upgrade "torchao>=0.16.0"
```

For Ampere or newer GPUs:

```bash
uv pip install --no-deps "apache-tvm-ffi==0.1.9" "tilelang==0.1.8"
```

On an older GPU such as a T4, skip TileLang and set `FLA_TILELANG=0` before
importing Unsloth. The included notebook preflight handles this automatically.

> **Keep the two worlds from fighting.** `uv pip install` puts packages in the
> environment without touching `uv.lock`, which is what you want here — but a
> later `uv sync` moves every package that *is* in the lock to its locked
> version, and some of them are shared. vLLM pins `fastapi`, `uvicorn` and
> `websockets`; Torch and Unsloth care about `numpy` and `fsspec`. On a machine
> that trains, prefer:
>
> ```bash
> uv sync --inexact            # never remove what uv pip installed
> uv pip install -e .          # or just refresh Nawāt itself, moving nothing
> ```
>
> `uv sync --dry-run` prints exactly what would change before anything does.
> This does not apply to a machine that only manages storage, where a plain
> `uv sync` is the right command.

### 4. Verify the GPU

```bash
nvidia-smi
nvcc --version

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
PY
```

Do not train until `torch.cuda.is_available()` is `True`.

### 5. Start RustFS with Docker

RustFS provides the S3-compatible durable store used in the examples. See the
[official RustFS Docker installation guide](https://docs.rustfs.com/installation/docker/)
for production, Docker Compose, TLS, and multi-node deployments.

For a local single-node setup:

```bash
mkdir -p rustfs/data rustfs/logs
sudo chown -R 10001:10001 rustfs/data rustfs/logs

docker run -d \
  --name nawat-rustfs \
  -p 9000:9000 \
  -p 9001:9001 \
  -v "$PWD/rustfs/data:/data" \
  -v "$PWD/rustfs/logs:/logs" \
  -e RUSTFS_ACCESS_KEY=rustfsadmin \
  -e RUSTFS_SECRET_KEY=change-me-before-first-use \
  -e RUSTFS_CONSOLE_ENABLE=true \
  rustfs/rustfs:latest \
  /data

docker logs nawat-rustfs
```

RustFS runs as UID/GID `10001:10001`, which is why the mounted directories are
given that owner. The S3 endpoint is `http://127.0.0.1:9000`; the web console is
`http://127.0.0.1:9001`. Replace the example secret before any non-local use.

If an S3-compatible service is already available, skip this step and use its
endpoint and credentials below.

### 6. Create `.env`

```bash
cp .env.example .env
$EDITOR .env
```

At minimum:

```dotenv
NAWAT_S3_ENDPOINT=http://127.0.0.1:9000
NAWAT_S3_BUCKET=nawat
NAWAT_S3_ACCESS_KEY=rustfsadmin
NAWAT_S3_SECRET_KEY=change-me-before-first-use

NAWAT_CACHE_ROOT=/home/you/nawat/cache
NAWAT_WORKSPACE=/home/you/nawat/workspace
NAWAT_CACHE_CEILING=120GB
NAWAT_MIN_FREE=10GB
```

- `NAWAT_CACHE_CEILING` limits staged artifacts.
- `NAWAT_MIN_FREE` reserves space for checkpoints, temporary files, and the OS.
- Keep `.env` private and never commit it.

```bash
chmod 600 .env
```

### 7. Make `.env` visible from the workspace

Nawāt loads the nearest `.env` at or above the current directory. If the
workspace is outside the cloned repository, link the same configuration above
it:

```bash
mkdir -p ~/nawat/workspace
ln -s /absolute/path/to/nawat/.env ~/nawat/.env
```

If the link already exists and is correct, leave it alone. An alternative for a
shell or service is:

```bash
export NAWAT_ENV_FILE=/absolute/path/to/nawat/.env
```

### 8. Test storage from the workspace

```bash
cd ~/nawat/workspace
nawat config
nawat check --create-bucket
```

Confirm the displayed `env_file`, endpoint, bucket, cache root, and workspace.
`nawat check` performs a real write/list/verify/delete round trip.

## Python examples

Each feature below is a standalone example. Run it after completing the setup
steps above. Artifact keys always start with `models/`, `datasets/`, or `runs/`.

### Load a model

`nawat.resolve()` checks the local cache, then RustFS or another configured
object store, then Hugging Face. It returns an ordinary local path.

```python
import nawat
from unsloth import FastVisionModel

model_path = nawat.resolve("models/unsloth/Qwen3.5-0.8B")
model, tokenizer = FastVisionModel.from_pretrained(
    str(model_path),
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)
```

The first resolution downloads and publishes the model. Later resolutions use
the local cache or restore the verified copy from object storage.

### Load a dataset

Dataset keys follow the same resolution order, but are fetched from Hugging
Face as dataset repositories.

```python
import nawat
from datasets import load_dataset

dataset_path = nawat.resolve("datasets/unsloth/LaTeX_OCR")
dataset = load_dataset(str(dataset_path), split="train")
print(dataset[0])
```

### Publish (push) a model or adapter

`nawat.publish()` uploads a directory, verifies every file, and removes the
source directory only after verification. Set `keep_local=True` to adopt the
source into Nawāt's managed cache instead of reclaiming it.

```python
import nawat

result = nawat.publish(
    "./trained-model",
    "models/my-lab/qwen3.5-latex-v1",
    keep_local=True,
)
print(f"Published {result.key}: {result.verification.local_bytes} bytes")
```

Use a run key for an adapter:

```python
import nawat

result = nawat.publish(
    "./qwen-lora",
    "runs/latex-ocr-v1/adapter",
)
print(f"Published {result.key}")
```

The equivalent CLI command is:

```bash
nawat publish ./trained-model models/my-lab/qwen3.5-latex-v1 --keep
```

### Publish a dataset

```python
import nawat

result = nawat.publish(
    "./prepared-dataset",
    "datasets/my-lab/latex-ocr-v1",
    keep_local=True,
)
print(f"Published {result.key}")
```

For a directory containing many small files, use `nawat shard` as shown in
[Large datasets](#large-datasets).

## First experiment: Qwen3.5-0.8B

### 1. Copy the training script

```bash
cp /absolute/path/to/nawat/examples/train_latex_ocr.py \
  ~/nawat/workspace/train_latex_ocr.py

cd ~/nawat/workspace
```

### 2. Run a three-step smoke test

```bash
nawat submit train_latex_ocr.py \
  --model models/unsloth/Qwen3.5-0.8B \
  --dataset datasets/unsloth/LaTeX_OCR \
  --param max_steps=3 \
  --param learning_rate=2e-4 \
  --param rank=16 \
  --param export=none \
  --notes "Qwen3.5-0.8B LaTeX OCR smoke test"
```

The first use downloads the model and dataset from Hugging Face, publishes them
to object storage, and trains from Nawāt-managed local paths. Later runs reuse
the local cache or restore from object storage.

### 3. Monitor it

```bash
nawat runs
nawat logs <run-id> -f
nawat metrics <run-id> -f
```

### 4. Confirm the result

```bash
nawat run <run-id>
```

Expected:

```text
state      succeeded (exit 0)
model      models/unsloth/Qwen3.5-0.8B
artifact   runs/<run-id>/adapter
```

## Second experiment: Qwen3.5-2B

This reuses the same script and cached dataset while staging a distinct base
model. It is a useful test that Nawāt separates models, runs, and adapters.

### 1. Estimate memory

```bash
nawat estimate \
  --model models/unsloth/Qwen3.5-2B \
  --method lora \
  --bits 16 \
  --batch 2 \
  --seq 2048
```

### 2. Run the requested 2B smoke test

```bash
nawat submit train_latex_ocr.py \
  --model models/unsloth/Qwen3.5-2B \
  --dataset datasets/unsloth/LaTeX_OCR \
  --param max_steps=3 \
  --param learning_rate=2e-4 \
  --param rank=16 \
  --param export=none \
  --notes "Qwen3.5-2B LaTeX OCR smoke test"
```

### 3. Inspect and compare

```bash
nawat runs
nawat logs <new-run-id> -f
nawat metrics <new-run-id> -f
nawat run <new-run-id>
nawat status
nawat ls
```

You should see separate base models and adapter keys:

```text
models/unsloth/Qwen3.5-0.8B
models/unsloth/Qwen3.5-2B
runs/<0.8b-run-id>/adapter
runs/<2b-run-id>/adapter
```

Both experiments reuse:

```text
datasets/unsloth/LaTeX_OCR
```

### 4. Run a longer 2B experiment

After the smoke test succeeds:

```bash
nawat submit train_latex_ocr.py \
  --model models/unsloth/Qwen3.5-2B \
  --dataset datasets/unsloth/LaTeX_OCR \
  --param max_steps=100 \
  --param learning_rate=1e-4 \
  --param rank=32 \
  --param export=none \
  --notes "Qwen3.5-2B LaTeX OCR baseline"
```

## Complete `nawat submit` reference

The general command shape is:

```bash
nawat submit SCRIPT [OPTIONS]
```

### Submission options

| Argument | Accepted value | Required for this example | Purpose |
| --- | --- | --- | --- |
| `SCRIPT` | A `.py` or `.ipynb` path inside `NAWAT_WORKSPACE` | Yes | Training program to execute |
| `--model KEY` | One `models/...` artifact key | Yes | Base model staged as `nawat.model_dir()` |
| `--dataset KEY` | A `datasets/...` key; option may be repeated | At least one | Dataset paths exposed through `nawat.dataset_dir()` and `dataset_dirs()` |
| `--input KEY` | Any additional artifact key; option may be repeated | No | Stage extra inputs that are neither the primary model nor datasets |
| `--param NAME=VALUE` | A name/value pair; option may be repeated | No | Record a parameter and expose it through `nawat.param()` |
| `--notes TEXT` | Any quoted text | No | Explain why the run exists |
| `--run-id ID` | A unique run identifier | No | Use a chosen ID instead of an automatically generated one |
| `--queue` | Flag with no value | No | Enqueue the run for the control plane instead of running immediately |

`--dataset`, `--input`, and `--param` are repeatable. If the same parameter name
is supplied more than once, the last value wins. Unknown parameter names are
recorded in the run but ignored unless the training script reads them.

### Parameters supported by `train_latex_ocr.py`

| Parameter | Type | Default | Accepted values | Effect |
| --- | --- | ---: | --- | --- |
| `max_steps` | Integer | `30` | Positive integer; useful examples: `3`, `30`, `100`, `500` | Maximum optimizer steps. Higher values take longer and usually improve convergence until overfitting begins. |
| `learning_rate` | Float | `2e-4` | Positive float; common range `1e-5` to `2e-4` | AdamW learning rate. The baseline uses `1e-4`. |
| `rank` | Integer | `16` | Positive integer; commonly `8`, `16`, `32`, or `64` | LoRA rank. The script also sets `lora_alpha` to the same value. Higher ranks use more VRAM and create larger adapters. |
| `export` | String | Empty (adapter only) | `none`, `merged`, `gguf`, or `merged,gguf` | Selects large deployment artifacts in addition to the always-published adapter. |
| `quantization` | String | `q4_k_m` | One GGUF choice listed below | GGUF quantization method; read only when `export` includes `gguf`. |

Values arrive through the CLI as strings. `nawat.param()` automatically converts
`max_steps` and `rank` to integers and `learning_rate` to a float because their
defaults in the script have those types.

### `export` choices

| Value | Published artifacts | When to use it |
| --- | --- | --- |
| Omitted, empty, or `none` | `adapter` | Recommended for Nawāt + vLLM; smallest and fastest output |
| `merged` | `adapter`, `merged` | A standalone Hugging Face-format merged model is required |
| `gguf` | `adapter`, `merged`, `gguf` | llama.cpp or Ollama needs a GGUF file; GGUF conversion requires merged weights first |
| `merged,gguf` | `adapter`, `merged`, `gguf` | Accepted explicit form; produces the same artifact classes as `gguf` |

Merged and GGUF exports can each be approximately as large as the base model.
The adapter is always saved, even if optional export conversion fails.

### `quantization` choices

The installed Unsloth version accepts these values. Use lowercase spelling:

```text
not_quantized  fast_quantized  quantized
f32            bf16            f16             q8_0
q6_k           q5_k_m          q5_k_s          q5_k
q5_0           q5_1            q4_k_m          q4_k_s
q4_k           q4_0            q4_1            q3_k_l
q3_k_m         q3_k_s          q3_k_xs         q2_k
q2_k_l
```

The three preset names map to a concrete format:

| Preset | Actual format | Trade-off |
| --- | --- | --- |
| `not_quantized` | Model-native `f16` or `bf16` | Highest fidelity, largest file |
| `fast_quantized` | `q8_0` | Fast conversion and high quality, but larger |
| `quantized` | `q4_k_m` | Smaller file and fast inference |

`q4_k` is an alias for `q4_k_m`; `q5_k` is an alias for `q5_k_m`. For this
example, start with the default `q4_k_m`. GGUF support depends on the model
architecture and the installed Unsloth/llama.cpp version; a GGUF conversion
error is reported but does not discard the successfully trained adapter.

### Export examples

Keep the baseline adapter-only by using `--param export=none`. To publish merged
weights instead, use:

```bash
--param export=merged
```

To publish GGUF with the default recommended quantization, use both:

```bash
--param export=gguf \
--param quantization=q4_k_m
```

## Why training parameters belong to the run

In a notebook:

```python
run = nawat.begin_run(
    model="models/unsloth/Qwen3.5-2B",
    dataset="datasets/unsloth/LaTeX_OCR",
    params={
        "max_steps": 30,
        "learning_rate": 2e-4,
        "rank": 16,
    },
)
```

Read them in the trainer:

```python
max_steps = run.param("max_steps", 30)
learning_rate = run.param("learning_rate", 2e-4)
rank = run.param("rank", 16)
```

Benefits:

- The exact values are preserved with the model, dataset, script, logs, metrics,
  and adapter.
- One script can run many experiments without being edited.
- Run records can be compared reliably.
- Defaults still make the code usable outside submitted runs.

Parameters are optional. Hard-coded Unsloth settings still work, but Nawāt cannot
record them as structured experiment parameters.

## Use Nawāt in an Unsloth notebook

Open the included notebook:

```bash
cd /absolute/path/to/nawat
uv run nawat lab
```

The storage-specific pattern is small:

```python
import nawat
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel

run = nawat.begin_run(
    model="models/unsloth/Qwen3.5-2B",
    dataset="datasets/unsloth/LaTeX_OCR",
    params={"max_steps": 30, "learning_rate": 2e-4, "rank": 16},
    notes="2B LaTeX OCR baseline",
)

model, tokenizer = FastVisionModel.from_pretrained(
    run.model_dir,
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)

dataset = load_dataset(run.dataset_dir, split="train")

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    callbacks=[run.callback()],
    args=SFTConfig(
        max_steps=run.param("max_steps", 30),
        learning_rate=run.param("learning_rate", 2e-4),
        **nawat.checkpoint_args(save_steps=250),
        # Add the remaining SFTConfig options for your model here.
    ),
)
trainer.train(resume_from_checkpoint=run.resume_from)

model.save_pretrained(run.artifact_dir("adapter"))
tokenizer.save_pretrained(run.artifact_dir("adapter"))
run.finish()
```

- `run.model_dir` and `run.dataset_dir` are ordinary local paths.
- `run.checkpoint_dir` is durable and survives a failed run; re-running the cell
  continues from the last saved step. `nawat.checkpoint_args()` points the
  trainer at it and saves every 250 steps instead of once an epoch.
- `run.scratch_dir()` is temporary, is not published, and is deleted when the
  run ends — for working files, never for checkpoints.
- `run.artifact_dir("adapter")` becomes `runs/<id>/adapter`.
- `run.callback()` streams Trainer metrics.
- `run.finish()` uploads, verifies, reclaims local output, and closes the record.
  `run.finish(name="arabic-ocr/2-epochs")` files it under that folder in object
  storage instead of under the run id.

For automatic exception recording:

```python
with nawat.begin_run(model="models/org/model", dataset="datasets/org/data") as run:
    ...
```

## Surviving a failed run

A long run fails for reasons that have nothing to do with the model: one corrupt
image in the eleven-thousandth batch, an OOM, a reboot. Nawāt keeps the
checkpoints of every run that does not succeed, so the cost of that is the
minutes since the last save rather than the whole run.

Two lines in the training script are all it needs:

```python
args = SFTConfig(
    **nawat.checkpoint_args(save_steps=250, save_total_limit=3),
    ...,
)
trainer.train(resume_from_checkpoint=nawat.resume_from())
```

`checkpoint_args()` sets `output_dir` to a durable directory, and saves every
250 *steps* rather than once an epoch — an epoch over a large corpus can be a
day wide, and a crash lands between two of them. `resume_from()` returns the
newest checkpoint, or `None` for a fresh start, which is exactly what the
Trainer expects in both cases. Both work under `nawat submit`, in a notebook,
and under plain `python train.py`.

Then resubmitting is resuming:

```bash
nawat submit train.py --model models/org/model --dataset datasets/org/data
# ... fails at step 9522 after 60 hours

nawat resume 2026-08-01-8075        # continues from checkpoint-9522
```

Checkpoints are grouped into a **lineage** named for the script and its inputs,
so any submission of the same command continues the same training, and a
different experiment cannot resume into it by accident. `nawat run <id>` shows
which lineage a run used and which checkpoint it reached.

```bash
nawat checkpoints                   # what can be resumed, and what it costs
nawat checkpoints <lineage>         # every saved step in one lineage
nawat checkpoints --prune --keep 1  # free all but the newest of each
nawat checkpoints <lineage> --rm    # remove one outright
```

| Situation | What happens to the checkpoints |
| --- | --- |
| The run succeeds | Replicated to object storage, then removed from local disk |
| The script fails | Replicated **and** kept locally, recorded, resumed by the next submission |
| The run is cancelled | Kept |
| The machine reboots mid-run | Kept, and attached to the record when the platform restarts |
| Publishing itself fails | Kept |

Every ending replicates, because the disk holding a half-trained model is the one
thing here with no second copy. The local copy is what a resume reads, so
continuing a failed run never pulls gigabytes back down.

Submit-time flags change that policy: `--keep-checkpoints` keeps them on local
disk after success, `--fresh` deletes the lineage and starts at step 0,
`--no-publish-checkpoints` skips the upload when it costs more than the insurance
is worth, and `--checkpoint-lineage NAME` shares or isolates a lineage
explicitly.

Checkpoints live in `<cache root>/checkpoints`, outside the cache ceiling and
outside eviction: nothing reclaims a half-trained model to make room, because it
is the one thing here that cannot be downloaded again. That makes them yours to
manage — `nawat status` reports what they hold, and `nawat checkpoints --prune`
gives it back. Set `NAWAT_CHECKPOINT_ROOT` to put them on another filesystem.

## Naming a run

A bucket full of `2026-08-01-8075` tells you nothing a year later. When a run
ends, Nawāt asks what to call it, and the answer is the folder its artifacts
occupy in object storage:

```text
run 2026-08-03-f3d4 exited 1.
Name it for object storage — artifacts publish under runs/<name>/.
name [2026-08-03-f3d4]: arabic-ocr/2-epochs
```

```text
ai-model/runs/arabic-ocr/2-epochs/adapter/
ai-model/runs/arabic-ocr/2-epochs/checkpoint/
ai-model/runs/arabic-ocr/2-epochs/reports/
ai-model/runs/arabic-ocr/2-epochs/record/
```

The question comes at the end rather than the start because that is when you
know what the run turned out to be. Nothing waits on you indefinitely: press
enter, let it time out, or pass `--no-name`, and it publishes under the run id
instead. `--name NAME` at submit skips the question entirely, which is what a
queued or scripted run does. A name may contain `/`, so related runs file
themselves together under one prefix.

`nawat resume` carries the name across — a resumed run is the same experiment,
so its artifacts belong in the same folder.

**Whatever the outcome, the artifacts go to object storage.** A run that fails
still publishes what it wrote — the partial reports, an emergency adapter, its
last checkpoint — because that is everything it has to show for the GPU time,
and a local disk is not where it should be left. The run record marks it failed,
so a half-trained adapter is never mistaken for a finished one.

## Inference with Nawāt and vLLM

vLLM loads the full base model once. Nawāt then hot-loads a compatible LoRA
adapter at runtime.

| Part | Example |
| --- | --- |
| Base model | `models/unsloth/Qwen3.5-2B` |
| Adapter | `runs/<run-id>/adapter` |
| API model name | `latex-ocr-2b` |

### 1. Install vLLM separately

Unsloth and vLLM may require different Torch versions, so use another virtual
environment. Nawāt launches the executable and does not import it.

```bash
python3 -m venv ~/nawat/.venv-vllm
~/nawat/.venv-vllm/bin/pip install -U uv
~/nawat/.venv-vllm/bin/uv pip install vllm

mkdir -p ~/.local/bin
ln -sf ~/nawat/.venv-vllm/bin/vllm ~/.local/bin/vllm
export PATH="$HOME/.local/bin:$PATH"
vllm --version
```

Optional `.env` settings for a 16 GB GPU:

```dotenv
NAWAT_SERVE_PORT=8001
NAWAT_SERVE_STARTUP_TIMEOUT=600
NAWAT_SERVE_IDLE_TIMEOUT=900
NAWAT_SERVE_EXTRA_ARGS=--gpu-memory-utilization 0.85 --max-model-len 4096
```

### 2. Reaching the server from another machine

The vLLM server always binds loopback, and there is no setting to change that on
purpose: it authenticates nobody, so exposing it directly would let anything that
can route to the port spend your GPU and load adapters. vLLM's own `--api-key` is
not a fix either — it guards every path under `/v1`, which is where Nawāt posts
`load_lora_adapter` without a bearer token, so a key breaks `nawat adapter`.

The network-facing address is the control plane, which proxies `/v1` behind
`NAWAT_API_TOKEN`:

```dotenv
NAWAT_API_HOST=0.0.0.0
NAWAT_API_PORT=8081
NAWAT_API_TOKEN=a-long-random-string
```

```bash
nawat api
```

Clients then use the control plane's address and send the token, and that URL
keeps working across model swaps and restarts because Nawāt resolves the current
session per request:

```bash
curl http://192.168.0.207:8081/v1/completions \
  -H "Authorization: Bearer a-long-random-string" \
  -H 'Content-Type: application/json' \
  -d '{"model":"models/unsloth/Qwen3.5-2B","prompt":"2+2=","max_tokens":10}'
```

Proxied requests also refresh the idle timer, so a session in active use is not
reclaimed underneath you.

### 3. Start the matching base

Do not serve while a training job is using the GPU.

```bash
nawat runs
nawat serve models/unsloth/Qwen3.5-2B
nawat session
```

If startup fails:

```bash
nawat session --log --tail 200
```

### 4. Hot-load the adapter

```bash
nawat adapter runs/<2b-run-id>/adapter --name latex-ocr-2b
nawat session
```

The adapter must have been trained from the running base. The name passed to
`--name` becomes the OpenAI-compatible API `model` value.

### 5. Send an image

```bash
IMAGE_PATH=/absolute/path/to/equation.png
test -f "$IMAGE_PATH" || { echo "Image not found: $IMAGE_PATH" >&2; exit 1; }
IMAGE_B64=$(base64 -w0 "$IMAGE_PATH")

curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{
    \"model\": \"latex-ocr-2b\",
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,$IMAGE_B64\"}},
        {\"type\": \"text\", \"text\": \"Write the LaTeX representation for this image.\"}
      ]
    }],
    \"temperature\": 0,
    \"max_tokens\": 512
  }"
```

The generated LaTeX is in `choices[0].message.content`.

### 6. Swap or stop

```bash
nawat adapter --unload latex-ocr-2b
nawat adapter runs/<another-2b-run-id>/adapter --name latex-ocr-2b
```

Stop and release GPU memory:

```bash
nawat adapter --unload latex-ocr-2b
nawat session --stop
```

## Local storage operations

Artifact keys map directly to object-store prefixes and managed local paths:

```text
models/unsloth/Qwen3.5-2B
datasets/unsloth/LaTeX_OCR
runs/<run-id>/adapter
```

Useful commands:

```bash
nawat status                  # cache and disk occupancy
nawat ls                      # artifacts currently cached
nawat registry                # artifacts held in object storage
nawat resolve KEY             # stage an artifact and print its local path
nawat path KEY                # print its path only if already cached
nawat keep KEY                # prevent automatic eviction
nawat release KEY             # make it reclaimable again
nawat verify KEY              # compare local files with object storage
nawat free --dry-run          # preview safe reclamation
nawat free                    # reclaim to the configured ceiling
nawat rm KEY                  # remove a verified local copy
```

Flags printed by `nawat ls`:

```text
K  kept locally
L  leased by a live process
R  replicated in object storage
```

Never manually delete files under `NAWAT_CACHE_ROOT`. Use `nawat rm` or
`nawat free` so remote verification and active leases remain effective.

## Large datasets

Publish an existing dataset:

```bash
nawat publish /data/my-dataset datasets/my-lab/my-dataset-v1
```

Pack many small files into approximately 512 MB sequential tar shards:

```bash
nawat shard /data/ocr-corpus datasets/my-lab/ocr-v3-sharded \
  --shard-size 512MB
```

The output is plain POSIX tar, WebDataset-compatible, with an `index.json`.
Sorted neighboring files stay together, which helps keep image/label pairs in
the same shard.

## Hugging Face cache

Hugging Face and Nawāt use different caches:

```text
~/.cache/huggingface   Hugging Face download cache
NAWAT_CACHE_ROOT       Nawāt-managed working set
S3-compatible store   durable source of truth
```

A small Hugging Face cache does not interfere with Nawāt. Submitted training
runs use the staged Nawāt paths and run with hub access disabled. Inspect both:

```bash
du -sh ~/.cache/huggingface 2>/dev/null
nawat status
nawat ls
```

## Troubleshooting

### `nvidia-smi` cannot communicate with the driver

```bash
uname -r
nvidia-smi
lsmod | grep nvidia
dpkg -l | grep linux-modules-nvidia
```

On Ubuntu, a kernel update can leave only a module for the previous kernel.
Install the matching signed module through the driver metapackage, then reboot.
Use the driver series selected for your machine rather than copying a version
number blindly.

### PyTorch has CUDA but an extension cannot find `nvcc`

The runtime bundled with PyTorch is not the compiler toolkit:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

For Torch `2.8.0+cu128` and an RTX 50-series GPU, select the real CUDA 12.8
toolkit before building extensions:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
hash -r
nvcc --version
```

### The inference server exits with `FlashInfer requires GPUs with sm75 or higher`

The message is misleading: it also appears when the GPU is far newer than sm75.
FlashInfer compiles its sampler on demand against the CUDA toolkit on `PATH`, and
a recent card needs a recent toolkit (sm120, the RTX 50 series, needs CUDA 12.9
or later). When the toolkit is older, FlashInfer finds no architecture it can
target and reports that empty list as the sm75 error. The line above it in the
session log names the real cause:

```text
Failed to get device capability: SM 12.x requires CUDA >= 12.9.
```

Nawāt therefore starts vLLM with `VLLM_USE_FLASHINFER_SAMPLER=0`, which selects
vLLM's built-in sampler and needs no compiler. Sampled output is unchanged. If
your toolkit does match your GPU, opt back in from the real environment:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=1
nawat serve models/unsloth/Qwen3.5-2B
```

This one is not a `NAWAT_` setting, so `.env` alone will not carry it: Nawāt
reads that file for its own configuration and does not export it. Putting it in
`.env` works only if you also source the file, as its header describes:

```bash
set -a && . ./.env && set +a
```

### `workspace` exists but is not a directory

Inspect it first:

```bash
ls -ld ~/nawat/workspace
readlink -f ~/nawat/workspace
```

If and only if it is a broken symbolic link:

```bash
rm ~/nawat/workspace
mkdir -p ~/nawat/workspace
```

Do not recursively delete a real workspace containing research files.

### Training exits 0 but `.env` was not found

Training succeeded but final publication failed because the workspace was
outside the `.env` search tree.

```bash
ln -s /absolute/path/to/nawat/.env ~/nawat/.env
cd ~/nawat/workspace
nawat config
nawat check
```

The adapter normally remains under the cache root shown by `nawat config`:

```bash
nawat publish \
  "<cache-root>/runs/<id>/adapter" \
  runs/<id>/adapter
```

The historical record remains failed because publication failed at that time,
but the recovered adapter key is valid. Run another three-step smoke test for a
clean `succeeded` record.

### Out of memory

Try these in order:

1. Set `per_device_train_batch_size=1` in the training script.
2. Reduce `max_length` from 2048.
3. Keep `use_gradient_checkpointing="unsloth"` enabled.
4. Use Qwen3.5-0.8B or 2B instead of 4B.
5. Stop any inference session with `nawat session --stop`.

## CLI help

Start with:

```bash
nawat --help
```

This command is read-only. It shows the global options and all top-level
commands. Add `--help` after a command for its positional arguments and flags:

```bash
nawat resolve --help
nawat publish --help
nawat submit --help
```

The global options apply before the subcommand:

| Option | Meaning |
| --- | --- |
| `--cache-root PATH` | Override `NAWAT_CACHE_ROOT` for this invocation |
| `--ceiling SIZE` | Override `NAWAT_CACHE_CEILING`, for example `80GB` |
| `--env-file PATH` | Load a specific configuration file |
| `--no-env-file` | Ignore `.env` and use only the process environment |

Commands are grouped here by task so the list is easier to scan:

| Task | Commands | What they do |
| --- | --- | --- |
| Setup | `config`, `check` | Show effective configuration and verify storage end to end |
| Find artifacts | `status`, `ls`, `registry`, `path` | Inspect disk usage and local or remote artifacts |
| Move artifacts | `resolve`, `add`, `publish`, `verify` | Stage, adopt, upload, or verify model and dataset directories |
| Protect space | `keep`, `release`, `leases`, `free`, `rm`, `checkpoints` | Protect live data and safely reclaim local storage |
| Execute work | `hold`, `submit`, `resume`, `scripts` | Stage inputs and run commands, scripts, or notebooks |
| Inspect runs | `runs`, `run`, `logs`, `metrics`, `cancel` | Monitor history, output, metrics, or stop a run |
| Inference and evaluation | `serve`, `session`, `adapter`, `eval` | Serve a base, hot-load LoRA, and score a run |
| Utilities | `estimate`, `shard`, `lab`, `api` | Estimate VRAM, shard data, open JupyterLab, or run the API |

`pin`, `unpin`, and `collect` remain available as aliases for `keep`, `release`,
and `free` respectively.

Two of those are new. `nawat resume ID` runs a failed run again from its last
checkpoint, and `nawat checkpoints` lists what can be resumed — with `--prune`
to give the disk back and `--rm` to drop a lineage outright.

## Safety model

Local files are reclaimed only when:

1. The artifact is not protected by `nawat keep`.
2. No live training or inference process holds a lease.
3. A matching object-storage replica is verified by file name and size at the
   moment of deletion.

Run records, logs, and metrics are durable provenance rather than disposable
cache. A failed run publishes what it produced rather than discarding it — the
record carries the failure, so a partial artifact is never mistaken for a
finished one.

Checkpoints are exempt from reclamation entirely. They are the only state on the
host that cannot be fetched again, so nothing evicts them to make room; they are
removed when a run succeeds, or when you ask.

## Development

```bash
uv sync --all-extras          # every optional dependency, plus the dev group
uv run pyflakes nawat         # lint
uv run nawat check            # exercise the real storage path
uv build                      # wheel and sdist into dist/
```

`uv sync` installs the `dev` dependency group by default; `--no-dev` leaves it
out. Dependencies live in `pyproject.toml` and are resolved into `uv.lock` —
edit through `uv add` / `uv remove` rather than by hand, and commit the lockfile
with the change.

## License

Nawāt uses the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use
is not granted.

The Unsloth-derived notebook and training script use the license in
[`examples/LICENSE`](examples/LICENSE).
