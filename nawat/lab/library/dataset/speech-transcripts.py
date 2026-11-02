"""Audio + transcript pairs for Whisper, prepared as the Whisper notebook prepares them.

Audio is cast to 16 kHz, turned into log-mel features by the model's feature
extractor, and the transcript tokenised as labels. A share is held out (6 % in
the notebook) for WER during training and for the held-out eval.
"""

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Audio + transcript pairs → Whisper features and labels",
         "backbones": ("stt",), "models": ("stt",)}

PARAMS = {
    "split": P("train", "Dataset split to read"),
    "subset": P("", "Dataset configuration name"),
    "audio_column": P("audio", "Column with the audio"),
    "text_column": P("text", "Column with the transcript"),
    "sampling_rate": P(16000, "Audio is resampled to this rate (Whisper: 16000)"),
    "max_samples": P(0, "Use at most N rows (0 = all)"),
    "eval_size": P(0.06, "Held out for WER: a fraction (< 1) or a row count"),
    "probe_count": P(4, "Held-out clips used as probes"),
    "seed": P(3407, "Seed for the held-out split"),
}


def load(ctx, p, model, tokenizer):
    from datasets import Audio

    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    ds = ds.cast_column(p.audio_column, Audio(sampling_rate=p.sampling_rate))
    ds = support.limit(ds, p.max_samples, p.seed)
    train, held = support.split_off(ds, p.eval_size, p.seed)
    ac, tc = p.audio_column, p.text_column

    def features(example):
        audio = example[ac]
        extracted = tokenizer.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"])
        return {"input_features": extracted.input_features[0], "labels": tokenizer.tokenizer(example[tc]).input_ids}

    raw_held = support.rows(held)
    return {
        "train": [features(e) for e in train],
        "eval": [features(e) for e in raw_held] if raw_held else None,
        "heldout": [{"audio": e[ac], "input": "", "reference": e[tc]} for e in raw_held],
        "probes": [{"audio": e[ac], "input": "", "reference": e[tc]} for e in raw_held[: p.probe_count]],
    }
