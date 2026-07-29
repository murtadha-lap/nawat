<div align="center">
  
<p style="text-align: center">
  <img align="center" src="docs/image.png#gh-light-mode-only" alt="Nawāt"><img align="center" src="docs/image.png#gh-light-mode-only" alt="Nawāt">
</p>
  
# Nawāt

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
  <img alt="S3-compatible storage" src="https://img.shields.io/badge/storage-S3--compatible-FF9900?logo=amazons3&logoColor=white">
  <img alt="Local-first" src="https://img.shields.io/badge/workflow-local--first-22C55E">
</p>

<p>
  <a href="#quick-start"><strong>Quick start</strong></a> ·
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

## Quick start

```bash
# Install Nawāt from this checkout
python3 -m venv .venv
source .venv/bin/activate
pip install -U uv
uv pip install -e ".[notebook]"

# Configure and verify storage
cp .env.example .env
$EDITOR .env
nawat check --create-bucket

# Submit a cheap smoke test from your workspace
nawat submit train_latex_ocr.py \
  --model models/unsloth/Qwen3.5-0.8B \
  --dataset datasets/unsloth/LaTeX_OCR \
  --param max_steps=3 \
  --param learning_rate=2e-4 \
  --param rank=16 \
  --param export=none
```

The sections below cover the complete installation, GPU checks, storage setup,
training workflow, adapter publishing, vLLM inference, cache management, and
common failure modes.

## What Nawāt manages

| Research task | Nawāt behavior |
| --- | --- |
| Load a model or dataset | Local cache → object storage → Hugging Face |
| Avoid repeated downloads | The first hub fetch writes through to object storage |
| Prevent a full disk | Enforces a cache ceiling and safe LRU reclamation |
| Protect active work | Leases inputs so live training and inference cannot lose files |
| Preserve an experiment | Records the script, inputs, parameters, logs, metrics, and outputs |
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

The example uses `unsloth/LaTeX_OCR` and can train Qwen3.5-0.8B, Qwen3.5-2B,
or another compatible Qwen3.5 vision model by changing only the model key.

## Requirements

- Linux and Python 3.11+
- An NVIDIA GPU supported by Torch and the selected model
- A working NVIDIA driver (`nvidia-smi` must succeed)
- CUDA 12.8 toolkit for building the documented Qwen3.5 extensions
- An S3-compatible store: RustFS, MinIO, SeaweedFS, Ceph, or AWS S3
- Enough local disk for one working set, not the full research archive

The tested workstation has an RTX 5060 Ti with 16 GB VRAM. Unsloth documents
approximate BF16 LoRA use of 3 GB for Qwen3.5-0.8B, 5 GB for 2B, and 10 GB for
4B; real usage also depends on sequence length, batch size, and trainable layers.

## Install from this repository

### 1. Create the training environment

```bash
cd /path/to/nawat

python3 -m venv .venv
source .venv/bin/activate
pip install -U uv
uv pip install -e ".[notebook]"
```

### 2. Install Unsloth

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

### 3. Verify the GPU

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

## Configure storage

### 1. Create `.env`

```bash
cp .env.example .env
$EDITOR .env
```

At minimum:

```dotenv
NAWAT_S3_ENDPOINT=http://127.0.0.1:9000
NAWAT_S3_BUCKET=nawat
NAWAT_S3_ACCESS_KEY=your-access-key
NAWAT_S3_SECRET_KEY=your-secret-key

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

### 2. Make `.env` visible from the workspace

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

### 3. Test storage from the workspace

```bash
cd ~/nawat/workspace
nawat config
nawat check --create-bucket
```

Confirm the displayed `env_file`, endpoint, bucket, cache root, and workspace.
`nawat check` performs a real write/list/verify/delete round trip.

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
source .venv/bin/activate
nawat lab
```

The storage-specific pattern is small:

```python
import nawat

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
    ...,
    callbacks=[run.callback()],
    args=SFTConfig(
        max_steps=run.param("max_steps", 30),
        learning_rate=run.param("learning_rate", 2e-4),
        output_dir=str(run.scratch_dir("trainer")),
        ...,
    ),
)
trainer.train()

model.save_pretrained(run.artifact_dir("adapter"))
tokenizer.save_pretrained(run.artifact_dir("adapter"))
run.finish()
```

- `run.model_dir` and `run.dataset_dir` are ordinary local paths.
- `run.scratch_dir()` is temporary and is not published.
- `run.artifact_dir("adapter")` becomes `runs/<id>/adapter`.
- `run.callback()` streams Trainer metrics.
- `run.finish()` uploads, verifies, reclaims local output, and closes the record.

For automatic exception recording:

```python
with nawat.begin_run(model="models/org/model", dataset="datasets/org/data") as run:
    ...
```

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

## Command reference

| Command | Purpose |
| --- | --- |
| `nawat config` | Effective configuration with credentials redacted |
| `nawat check` | Validate cache and object-storage operations |
| `nawat status` | Cache, disk, and protected-byte summary |
| `nawat ls` | Locally cached artifacts |
| `nawat registry` | Object-storage artifacts |
| `nawat resolve KEY` | Stage an artifact and print its path |
| `nawat keep KEY` / `release KEY` | Pin or unpin a local artifact |
| `nawat free` | Safe LRU reclamation |
| `nawat publish DIR KEY` | Upload, verify, and reclaim |
| `nawat shard DIR KEY` | Pack a small-file corpus into tar shards |
| `nawat submit SCRIPT ...` | Run a recorded training job |
| `nawat runs` / `run ID` | List runs or inspect one record |
| `nawat logs ID -f` | Follow trainer output |
| `nawat metrics ID -f` | Follow the metric trace |
| `nawat serve KEY` | Start vLLM for a base model |
| `nawat adapter KEY --name NAME` | Hot-load a compatible LoRA |
| `nawat session` | Inspect or stop the inference server |

## Safety model

Local files are reclaimed only when:

1. The artifact is not protected by `nawat keep`.
2. No live training or inference process holds a lease.
3. A matching object-storage replica is verified by file name and size at the
   moment of deletion.

Run records, logs, and metrics are durable provenance rather than disposable
cache. Failed runs retain their diagnostics but do not publish incomplete model
artifacts.

## Development

```bash
git clone https://github.com/murtadha-lap/nawat.git
cd nawat
python3 -m venv .venv
.venv/bin/pip install -e ".[notebook,api]"
```

Run the CLI directly from the editable checkout:

```bash
.venv/bin/nawat --help
```

## License

Nawāt uses the [PolyForm Noncommercial License 1.0.0](LICENSE). Commercial use
is not granted.

The Unsloth-derived notebook and training script use the license in
[`examples/LICENSE`](examples/LICENSE).
