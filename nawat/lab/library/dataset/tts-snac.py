"""Speech + text rows → Orpheus training sequences, through the SNAC audio codec.

Exactly the Orpheus notebook's preparation: audio resampled to 24 kHz and
encoded by SNAC into 7 codes per frame (offset into the audio vocabulary),
repeated frames removed, then ``[human] text [/human] [ai] [speech] codes
[/speech] [/ai]``. The codec is an input of the experiment
(``models/hubertsiuzdak/snac_24khz`` by default), staged like any model, never
downloaded by name at training time.
"""

import json

from nawat.lab import P

BLOCK = {"kind": "dataset", "summary": "Audio + text → SNAC codes → Orpheus token sequences",
         "backbones": ("tts",), "models": ("tts",)}

PARAMS = {
    "split": P("train", "Dataset split to read"),
    "subset": P("", "Dataset configuration name"),
    "audio_column": P("audio", "Column with the audio"),
    "text_column": P("text", "Column with the text spoken"),
    "source_column": P("source", "Speaker column for multi-speaker data (used when present)"),
    "codec_model": P("models/hubertsiuzdak/snac_24khz", "SNAC codec, staged as an input"),
    "codec_sample_rate": P(24000, "Rate SNAC expects"),
    "tokeniser_length": P(128256, "Size of the text vocabulary; special and audio tokens come after it"),
    "start_of_text": P(128000, "Start-of-text token id"),
    "end_of_text": P(128009, "End-of-text token id"),
    "remove_duplicate_frames": P(True, "Drop consecutive frames with the same first code"),
    "max_samples": P(0, "Use at most N clips (0 = all)"),
    "eval_size": P(0.0, "Held out for eval loss: a fraction (< 1) or a row count"),
    "seed": P(3407, "Seed for sampling and the held-out split"),
}


def inputs(p):
    return [p.codec_model]


def _snac(directory):
    import torch
    from pathlib import Path

    from snac import SNAC

    directory = Path(directory)
    config = directory / "config.json"
    model = SNAC.from_config(str(config)) if hasattr(SNAC, "from_config") else SNAC(**json.loads(config.read_text()))
    weights = directory / "pytorch_model.bin"
    model.load_state_dict(torch.load(str(weights), map_location="cpu"))
    return model.eval()


def load(ctx, p, model, tokenizer):
    import torch
    import torchaudio.transforms as T

    from nawat.lab import support

    ds = support.load_dataset_dir(ctx.dataset_dir, split=p.split, subset=p.subset or None)
    ds = support.limit(ds, p.max_samples, p.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = _snac(ctx.input_dir(p.codec_model)).to(device)
    ds_sample_rate = ds[0][p.audio_column]["sampling_rate"]
    start = p.tokeniser_length + 10  # audio_tokens_start
    resample = T.Resample(orig_freq=ds_sample_rate, new_freq=p.codec_sample_rate)

    def tokenise_audio(waveform):
        waveform = torch.from_numpy(waveform).unsqueeze(0).to(dtype=torch.float32)
        waveform = resample(waveform).unsqueeze(0).to(device)
        with torch.inference_mode():
            codes = snac_model.encode(waveform)
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item() + start)
            all_codes.append(codes[1][0][2 * i].item() + start + 4096)
            all_codes.append(codes[2][0][4 * i].item() + start + (2 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 1].item() + start + (3 * 4096))
            all_codes.append(codes[1][0][(2 * i) + 1].item() + start + (4 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 2].item() + start + (5 * 4096))
            all_codes.append(codes[2][0][(4 * i) + 3].item() + start + (6 * 4096))
        return all_codes

    ac = p.audio_column

    def add_codes(example):
        codes_list = None
        try:
            audio = example.get(ac)
            if audio and "array" in audio:
                codes_list = tokenise_audio(audio["array"])
        except Exception as exc:  # noqa: BLE001 - the notebook skips bad rows the same way
            print(f"Skipping row due to error: {exc}")
        return {"codes_list": codes_list}

    ds = ds.map(add_codes, remove_columns=[ac])
    ds = ds.filter(lambda x: x["codes_list"] is not None and len(x["codes_list"]) > 0)

    def dedupe(example):
        vals = example["codes_list"]
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")
        result = vals[:7]
        for i in range(7, len(vals), 7):
            if vals[i] != result[-7]:
                result.extend(vals[i:i + 7])
        return {"codes_list": result}

    if p.remove_duplicate_frames:
        ds = ds.map(dedupe)

    L = p.tokeniser_length
    tc, sc, eot = p.text_column, p.source_column, p.end_of_text

    def create_input_ids(example):
        text_prompt = f"{example[sc]}: {example[tc]}" if sc in example and example[sc] else example[tc]
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True) + [eot]
        input_ids = [L + 3] + text_ids + [L + 4] + [L + 5] + [L + 1] + example["codes_list"] + [L + 2] + [L + 6]
        return {"input_ids": input_ids, "labels": input_ids, "attention_mask": [1] * len(input_ids)}

    ds = ds.map(create_input_ids)
    ds = ds.remove_columns([c for c in ds.column_names if c not in ("input_ids", "labels", "attention_mask")])
    snac_model.to("cpu")  # give the codec's VRAM back before training starts
    train, held = support.split_off(ds, p.eval_size, p.seed)
    return {"train": train, "eval": held, "heldout": [], "probes": []}
