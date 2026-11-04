# Changelog

## Unreleased

- Evaluate a model without training it (#9): the `eval-only` backbone and the
  `llm-eval` preset load a model — a base model, or a finished run's
  `runs/<group>/<experiment>/adapter` — and run every eval block without an
  optimizer step. Its plan and report say "Training: none". Dataset and
  template blocks of any training method fit it.
- `model.chat_template_kwargs` (`llm-lora`, `vlm-lora`) passes variables to the
  chat template for generation prompts, e.g. Qwen3's `enable_thinking=false`,
  so base and fine-tuned models answer under the same prompt. When set, it is
  part of every eval fingerprint.

- Runs take turns on the one GPU (#10). Every run holds a GPU lock (a file
  lock in the workspace, shared with the sandbox) from staging to its last
  upload, and runs start in the order they were submitted. A plain
  `nawat exp run` waits as `queued` while the GPU is busy, saying in its log
  what it waits for; Ctrl-C takes it out of the queue. `--queue` no longer
  needs `nawat api`: a worker (`nawat exp queue --work`) is started when nothing
  else serves the queue. New: `nawat exp approve --queue`, and `nawat exp queue`
  to see the queue with estimated start times. `nawat exp cancel` takes a
  queued run out of the queue. An experiment can have only one attempt queued
  or running, and `nawat sweep run` skips experiments already queued or running.

- `nawat exp report`, `exp ls` and `sweep ls` showed an experiment that was
  still training as "interrupted" (#8). An attempt without a run record now
  takes its state from the run store — queued, staging, running, publishing —
  and is "interrupted" only when the run store does not know it, or its trainer
  process is gone.

Fixes from a review of 0.3.0b1. Each has a regression test in
`tests/test_regressions.py`.

- `nawat registry` (and the API's `/registry`) failed with InvalidKey once any
  experiment had run: the runner left `code/__pycache__` in the experiment
  folder and it was uploaded with it. Bytecode caches are no longer written
  there, never uploaded, and the registry lists artifacts — the shallowest
  folder holding files — rather than every folder inside them.
- `nawat estimate` no longer needs the `api` extra; the GPU reading lives in
  `nawat.gpu`.
- The `notebook` extra includes JupyterLab, so `uv sync --extra notebook` and
  `uv run nawat jupyter` work as documented.
- Inside the sandbox, an approval that cannot be verified (there is no key
  there) is refused instead of accepted as "unverified". A run in the sandbox is
  started from the host with `nawat exp run`, which checks the signature and
  then launches it in the container. `AGENTS.md` and `docs/lab.md` say so.
- `nawat exp logs/metrics/cancel` on the host now reach an experiment that ran
  in the sandbox: live, through the container; afterwards, from the copy in
  `attempts/<run>/`.
- Ctrl-C during `nawat exp run` cancels the run — the trainer is stopped, its
  checkpoints kept, its record and experiment folder published — instead of
  leaving it running unwatched. Cancelled runs are finalised like any other; a
  cancel that arrives while inputs are staging stops the trainer from starting.
- `nawat exp rm <exp> --parts experiment` removes the local folder; before, it
  did so only together with every other part.
- `nawat shard --no-publish` works when the corpus and the cache are on
  different filesystems.
- `nawat exp traces --clear` deletes the remote copy first, so a refused delete
  leaves the local one in place.
- `nawat exp plan --save-preset` saves the preset before planning, so a taken
  name fails before an experiment exists.
- Bad numbers or sizes in `.env` are reported as configuration errors (exit 2)
  rather than tracebacks.
- The API's `/health` shows only liveness without the token; the token check is
  constant-time. A run the queue cannot execute is marked failed instead of
  being retried forever.
- `env-<run>.json` records every installed distribution (`installed`);
  `pip freeze` recorded nothing in a uv environment.
- Messages that still named `nawat submit`, `resume`, `scripts` and
  `serve status` name the commands that exist. The README's troubleshooting and
  install sections match the experiment workflow and list
  `sentence-transformers`, `snac` and `torchaudio`, which the embedding, TTS
  and STT backbones need (a test keeps the README and the sandbox image in
  step). `.env.example` documents the control-plane variables and no longer
  lists the removed agent ones.
- Removed what the `submit` era left behind: the notebook `Run` class and
  kernel-run accessors (`nawat.Run`, `nawat.current_run`), the `.ipynb`
  executor path, the run-naming prompt and the training-script scanner in
  `nawat.estimator`.

## 0.3.0b1 — beta (2026-11-02)

Training becomes experiments. Every training run is now an experiment: a fixed
backbone (the official Unsloth notebook for its method), swappable blocks and
declared parameters, approved by a person before it runs. Everything an
experiment produced is kept together in object storage.

This is a **beta**. The platform around training is complete and covered by
tests; several training methods have not yet run on a GPU (see *Status* below).

### New

- **The lab** (`nawat preset`, `nawat block`, `nawat exp`, `nawat sweep`).
  - Backbones for SFT, CPT, DPO and GRPO, with DAPO, GSPO and Dr. GRPO as GRPO
    presets, plus TTS (Orpheus), STT (Whisper), embeddings and typed decision
    models. There are 12 built-in presets.
  - Blocks (model, dataset, template, rewards, eval, export, runtime) are
    swapped with `--use`; parameters are changed with `--set`.
  - Every number a block uses is declared, shown in the plan, and part of the
    approval.
  - Each experiment is frozen when planned: a copy of the exact code goes into
    its `code/` folder, and a used block is locked against in-place edits.
    Blocks and presets are forked, never overwritten.
- **Approval.** `nawat exp approve` needs a person at a terminal and the code
  printed under the plan. The approval is bound to a hash of the code,
  parameters and inputs, and signed with a key kept outside the workspace.
  Inside the sandbox, approval is refused.
- **Traces.** Every GRPO/DAPO completion is recorded with each reward and its
  group advantage; other methods record probe answers over training.
  `nawat exp traces` shows several experiments side by side.
- **Evals.** GSM8K, MATH-500, MMLU, ARC-Challenge and TruthfulQA (fetched on
  first use and kept in object storage); `evalset` for your own test sets;
  held-out generation, loss, retrieval and decision evals.
  - Every eval has a fingerprint, and `nawat exp compare` puts numbers side by
    side only when they were measured the same way.
- **Reports.** `nawat exp report` and `nawat exp compare` produce plain
  Markdown or JSON for a person or a local agent. A comparison shows only what
  differs.
- **One folder per experiment** in object storage, under
  `runs/<group>/<name>/`: adapter, merged, gguf, checkpoint, record,
  experiment and traces. `nawat exp pull` brings it to another machine.
- **Storage control:** `nawat store ls/du/get/rm`, `nawat exp du/rm` and
  `nawat exp traces --clear`. Deleting is confirmed, refuses what a live run
  uses, and is refused inside the sandbox.
- **Sweeps.** `nawat sweep plan --vary …` plans a grid of experiments, approved
  once with a single code.
- **OpenCode sandbox** (optional): `nawat sandbox build/up/opencode`. It is a
  Docker container with the GPU, the network and the workspace, but it cannot
  approve or delete. `scripts/install-docker-nvidia.sh` installs Docker and the
  NVIDIA Container Toolkit.
- **Guarantee tests in CI** (`tests/`, `.github/workflows/guarantees.yml`):
  89 CPU-only tests. They cover one managed path, approval, no hidden
  parameters, fair evals, immutability, and a full run through the real
  executor.
- **Docs:** `docs/lab.md`, plus architecture diagrams in `docs/`
  (`nawat-overview.svg`, `nawat-stack.svg`, `nawat-lab.svg`).

### Breaking changes

An experiment is the only way to train. Code written outside it cannot become a
recorded run.

| Removed or changed | Instead |
| --- | --- |
| `nawat submit`, `nawat resume`, `nawat hold`, `nawat scripts` | `nawat exp plan` → `exp approve` → `exp run`; running a failed experiment again continues from its checkpoint |
| `nawat eval` and the API's `/runs/{id}/evaluate` | eval blocks, e.g. `--use eval=evalset --set eval.evalset.dataset=…` |
| API `POST /runs`, `GET /scripts` | `POST /experiments/{group}/{name}/run` (an approved experiment) |
| `nawat.begin_run()` in notebooks | refuses; plan an experiment, and use the notebook to look at data and results |
| `nawat publish` / `add` to `runs/…` | refused: only experiments write there; publish under `models/` or `datasets/` |
| `nawat lab` (JupyterLab) | `nawat jupyter` |
| `nawat logs/metrics/cancel <run-id>` | still available; `nawat exp logs/metrics/cancel <experiment>` do the same by name |
| `examples/train_latex_ocr.py`, `examples/latex_ocr_qwen3_5_vision.ipynb` | the `vlm-sft` preset and `examples/latex_ocr_experiment.ipynb` |

### Requirements

- `python3.12-dev` and `build-essential`: Triton compiles GPU kernels at run
  time and needs `Python.h`.
- For GRPO fast inference (`model.fast_inference=true`), install
  `vllm==0.11.0` into the training environment. It is built for Torch 2.8 and
  keeps the pinned Unsloth set; a newer vLLM replaces Torch and Transformers.
  The `llm-lora` and `vlm-lora` blocks include a compatibility shim for vLLM
  0.11 with Transformers 5. Use `uv sync --inexact` so a sync does not remove
  it. Serving keeps its own newer vLLM in a separate environment.

### Licensing

Backbones and blocks derived from Unsloth's notebooks are under LGPL-3.0, as
those notebooks are. `nawat/lab/library/LICENSE` lists them. The rest of the
repository stays under PolyForm Noncommercial 1.0.0.

### Status in this beta

| Verified | How |
| --- | --- |
| The platform: planning, approval, gate, traces, evals, fingerprints, compare, sweeps, storage, publishing | 89 tests on CPU, plus a full run through the real executor and RustFS |
| Text SFT backbone (`llm-sft`) | on the GPU (RTX 5060 Ti, Qwen3-0.6B) and on Unsloth's CPU backend |
| vLLM 0.11 fast generation inside Unsloth training | loaded and generated on the GPU |
| Trainer settings of every backbone | checked field by field against the installed TRL / Transformers / sentence-transformers |

Not yet run on a GPU: GRPO/DAPO/GSPO end to end, vision SFT and GRPO, DPO, CPT,
TTS, STT, embedding and decision. Smoke experiments for GRPO + vLLM, vision SFT
and embedding are planned. Expect fixes in the next beta.

## 0.2.0

Native uv project; checkpoints kept and resumed across failed runs; runs named
by folder; outputs published whatever the outcome.
