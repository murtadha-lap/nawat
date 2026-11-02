# Changelog

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
