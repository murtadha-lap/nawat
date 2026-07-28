# Nawāt (نواة)

**Fine-tune on a small disk as if it were a big one.**

Nawāt is a self-hosted platform for fine-tuning and serving open-weight models on a
single GPU workstation whose local storage is far smaller than its working corpus.
Object storage is the source of truth for every model, dataset and run artifact;
local disk is a disposable cache managed automatically. You write an ordinary
Unsloth training script, submit it, and never touch a file to make room for it.

All seven PRD phases are built: the storage core, the control plane, metrics, the
web interface, evaluation, agent-assisted authoring, and hardening. Operational
procedures live in [docs/OPERATIONS.md](docs/OPERATIONS.md); the control plane
runs as a systemd service via [deploy/nawat-api.service](deploy/nawat-api.service).

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

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env && $EDITOR .env    # endpoint, bucket, credentials, cache ceiling
nawat check --create-bucket
```

`nawat` finds `.env` on its own (nearest one at or above the working directory);
nothing to source. Exported variables still win for one-off overrides.

Nawāt speaks to any S3-compatible object store. This deployment uses
[RustFS](https://docs.rustfs.com/installation/linux/quick-start) running as a
systemd service; point `NAWAT_S3_ENDPOINT` at it. `nawat check` proves the pairing
works — it writes, lists, verifies and deletes a probe object rather than pinging.

```text
ok    configuration              /home/lap/lap/tr/.env
ok    cache root                 /home/lap/nawat/cache
ok    cache ceiling              120 GB ceiling + 10.0 GB reserve on a 250 GB filesystem
ok    object storage reachable   http://192.168.0.155:9000 · bucket ai-model
ok    object storage round trip  write, list, verify and delete all succeeded

Ready. Object storage is reachable and this host can publish, verify and reclaim.
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

## Using it with Unsloth

Take the standard Unsloth vision notebook — Qwen3.5-0.8B fine-tuned on
`unsloth/LaTeX_OCR`. Three lines change, all of them the lines that name a
location. Everything Unsloth-specific stays exactly as the notebook has it.

**`~/nawat/workspace/train_latex_ocr.py`:**

```python
# The Unsloth LaTeX-OCR notebook as a Nawāt run. The NAWAT_* variables are just
# paths and strings, so this script also runs unmodified outside the platform.
import os, pathlib, tempfile

MODEL_DIR = os.environ["NAWAT_MODEL_DIR"]      # staged models/unsloth/Qwen3.5-0.8B
DATA_DIR  = os.environ["NAWAT_DATASET_DIR"]    # staged datasets/unsloth/LaTeX_OCR
OUT       = pathlib.Path(os.environ["NAWAT_OUT_DIR"])

from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_DIR,                                 # ← was "unsloth/Qwen3.5-0.8B"
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
dataset = load_dataset(DATA_DIR, split = "train")   # ← was "unsloth/LaTeX_OCR"

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
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # Hyperparameters arrive from `nawat submit --param ...`, with the
        # notebook's values as defaults:
        max_steps     = int(os.environ.get("NAWAT_PARAM_MAX_STEPS", "30")),
        learning_rate = float(os.environ.get("NAWAT_PARAM_LEARNING_RATE", "2e-4")),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        # Intermediate checkpoints are scratch, not artifacts — keep them out of
        # NAWAT_OUT_DIR so they are not published:
        output_dir = tempfile.mkdtemp(prefix = "trainer-"),
        report_to = "none",
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    ),
)
trainer.train()   # add callbacks=[nawat.metrics.trainer_callback()] above for a live loss trace

# Everything written under NAWAT_OUT_DIR is published as its own artifact class
# when the run exits 0 — uploaded, verified file by file, then reclaimed.
model.save_pretrained(OUT / "adapter")             # ← was "qwen_lora"
tokenizer.save_pretrained(OUT / "adapter")

# Only on request, for deployment — each becomes runs/<id>/<name>:
# model.save_pretrained_merged(OUT / "merged", tokenizer)
# model.save_pretrained_gguf(OUT / "gguf", tokenizer, quantization_method = "q4_k_m")
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

**What changed vs. the notebook** — nothing else did:

| Notebook | Under Nawāt |
| --- | --- |
| `from_pretrained("unsloth/Qwen3.5-0.8B")` | `from_pretrained(os.environ["NAWAT_MODEL_DIR"])` |
| `load_dataset("unsloth/LaTeX_OCR")` | `load_dataset(os.environ["NAWAT_DATASET_DIR"])` |
| `save_pretrained("qwen_lora")` | `save_pretrained(OUT / "adapter")` |
| `push_to_hub(..., token=...)` | automatic verified publish to your own store |
| manual `rm -rf` between runs | automatic, refuses when unsafe |

**Watch the run from a chart, not scrollback.** Add one line to the trainer —

```python
import nawat.metrics
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

Notebooks submit too (`nawat submit explore.ipynb ...`): the notebook is executed
with nbconvert and the executed copy is archived as the run record. Adapt the
load cells to the environment variables first, same as above — the pip-install
cell is unnecessary on a host with Unsloth already installed.

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

## In a notebook, interactively

```python
import nawat

model_dir = nawat.resolve("models/unsloth/Qwen3.5-0.8B")   # staged + held
data_dir  = nawat.resolve("datasets/unsloth/LaTeX_OCR")
# ... train exactly as above ...
nawat.publish(out_dir, "runs/2026-07-28-a91f/adapter")     # upload, verify, reclaim
```

`resolve` holds the artifact for the lifetime of the kernel; `with
nawat.holding(...)` scopes it tighter.

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
| `nawat api` | The control plane and web UI over HTTP (docs at `/docs`) |
| `nawat check` | Prove this host can store, verify and reclaim |
| `nawat config` | Configuration in force, credentials redacted |

Exit codes are stable: `2` invalid key, `3` not found, `4` store unavailable,
`5` verification failed, `6` insufficient space, `7` offline, `8` protected.

`nawat api` serves everything above over HTTP — FIFO run queue, server-sent-event
log streaming, and a stable OpenAI-compatible `/v1` that forwards to whichever
inference session is current, so a client configured once survives restarts and
model changes. `NAWAT_API_TOKEN` enables bearer-token auth; `/health` stays open.

---

## The interface and the agent

`nawat api` serves a benchtop-instrument web UI at `/ui` — storage, registry, runs
with a live gold-on-graticule loss trace, submission, serving with chat and image
input, cross-run comparison, and an Agent view. A failed run's **Diagnose** button
leads to propose → review diff → apply → resubmit, all on one screen.

The agent is optional and never autonomous (`NAWAT_AGENT_BACKEND=claude` for the
Claude Agent SDK — confined read-only to the workspace — or `local` for any
OpenAI-compatible endpoint, fully on-premises). Every proposal passes a syntax
gate and a VRAM estimate before it is even offered, and reaches the workspace
only through your approval, committed to git with its prompt and backend.

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

## Tests

```bash
.venv/bin/pip install -e ".[dev]" && .venv/bin/python -m pytest
```

The eviction tests are a release gate: unreplicated, kept, in-use and
unverifiable artifacts are never removed, and a refusal deletes nothing.
The S3 backend is driven over real HTTP (pagination past 1000 keys, multipart,
prefix isolation), and sessions are tested against a real subprocess server —
no GPU needed to run the suite.

## Known limits

- Mutating cache operations take a host-wide lock; simultaneous stages serialise.
- Verification compares name and size, not content hashes (as specified) —
  checksums would catch silent corruption at the cost of reading every byte back.
- vLLM must support the architecture being served; adapter hot-load requires the
  vLLM backend.
