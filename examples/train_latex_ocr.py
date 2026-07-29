"""The LaTeX-OCR notebook as a submitted run.

Derived from Unsloth's Qwen3.5-0.8B vision fine-tuning notebook and, like it,
licensed LGPL-3.0 — not under the noncommercial licence covering the rest of
this repository. See examples/LICENSE.

    nawat submit train_latex_ocr.py \
      --model   models/unsloth/Qwen3.5-0.8B \
      --dataset datasets/unsloth/LaTeX_OCR \
      --param   max_steps=500 --param learning_rate=2e-4 \
      --notes   "LaTeX OCR, long run"

Add `--param export=gguf --param quantization=q4_k_m` to also produce a
llama.cpp/Ollama build; `--param export=merged` for FP16 weights alone. Both are
off by default because each is roughly the size of the base model, and vLLM
serves the adapter without either.

The same cells as latex_ocr_qwen3_5_vision.ipynb, minus the two the executor
does for you: it opens the run before this process starts and closes it when
this process exits. `nawat.model_dir()`, `nawat.param()` and
`nawat.artifact_dir()` read the environment it injected — the identical calls in
the notebook read the open kernel run instead, which is why the body is
unchanged between the two.

Nothing here needs the platform to be importable, either: every accessor has an
`os.environ` equivalent, so this file is ordinary Python that also happens to be
reproducible.
"""

import nawat

MAX_STEPS = nawat.param("max_steps", 30)
LEARNING_RATE = nawat.param("learning_rate", 2e-4)
RANK = nawat.param("rank", 16)

# -- model --------------------------------------------------------------------

from unsloth import FastVisionModel  # noqa: E402

model, tokenizer = FastVisionModel.from_pretrained(
    nawat.model_dir(),                          # ← was "unsloth/Qwen3.5-0.8B"
    load_in_4bit=False,
    use_gradient_checkpointing="unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=RANK,
    lora_alpha=RANK,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# -- data ---------------------------------------------------------------------

from datasets import load_dataset  # noqa: E402

dataset = load_dataset(nawat.dataset_dir(), split="train")   # ← was "unsloth/LaTeX_OCR"

INSTRUCTION = "Write the LaTeX representation for this image."


def convert_to_conversation(sample):
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": INSTRUCTION},
                {"type": "image", "image": sample["image"]}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": sample["text"]}]},
        ]
    }


converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# -- train --------------------------------------------------------------------

import tempfile  # noqa: E402

from trl import SFTConfig, SFTTrainer  # noqa: E402
from unsloth.trainer import UnslothVisionDataCollator  # noqa: E402

FastVisionModel.for_training(model)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=converted_dataset,
    # Every logging step into the run's metric series: nawat metrics <id> -f
    callbacks=[nawat.metrics.trainer_callback()],
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        # Intermediate checkpoints are scratch, not artifacts — keep them out of
        # the output directory so they are not published.
        output_dir=tempfile.mkdtemp(prefix="trainer-"),
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
    ),
)

trainer.train()

# -- save ---------------------------------------------------------------------
#
# Everything under the output directory is published as its own artifact class
# when this process exits 0 — uploaded, verified file by file, then reclaimed.
# That is what keeps merged and GGUF exports from silently filling the disk:
# they go to object storage and come back on demand.

adapter = nawat.artifact_dir("adapter")         # ← was "qwen_lora"
model.save_pretrained(adapter)
tokenizer.save_pretrained(adapter)

# --export merged / --export gguf, off by default: both are the size of the base
# model, and vLLM can serve the adapter without either (see the README).
EXPORTS = {value.strip() for value in nawat.param("export", "").split(",") if value.strip()}

if EXPORTS & {"merged", "gguf"}:
    # GGUF conversion consumes merged weights, so it implies this step.
    model.save_pretrained_merged(str(nawat.artifact_dir("merged")), tokenizer)

if "gguf" in EXPORTS:
    # Builds llama.cpp on first use. The likely failure is llama.cpp not
    # supporting the architecture — vision encoders especially. Let the run
    # succeed with what it has rather than throwing away a finished training.
    try:
        model.save_pretrained_gguf(
            str(nawat.artifact_dir("gguf")),
            tokenizer,
            quantization_method=nawat.param("quantization", "q4_k_m"),
        )
    except Exception as exc:  # noqa: BLE001 - a failed export is not a failed run
        print(f"[nawat] gguf conversion failed, publishing without it: "
              f"{type(exc).__name__}: {exc}", flush=True)
