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