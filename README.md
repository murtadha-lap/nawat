# Nawāt (نواة)

Self-hosted fine-tuning and serving platform for storage-constrained GPU hosts.

**Phase 1 — storage core.** Local disk is a disposable cache over an S3-compatible
object store (RustFS), managed automatically. Nothing is deleted until it has been
verified present in the store, at the moment of deletion.

Phases 2–7 (control plane, metrics, web interface, evaluation, agent authoring,
hardening) are not built yet; see `nawat-prd.md` §11.

---

## The rule everything follows

> Object storage is truth. Local disk is disposable. Verify before deleting, always.
> If space cannot be freed safely, refuse and say why.

Concretely:

- An artifact is evicted only after its replica has been listed and compared **file by
  file, by name and size**, at the moment of deletion — never from a cached flag.
- If the store is unreachable, nothing is evicted. You get a full disk and an
  explanation, not a gamble.
- Artifacts in use are held by a **lease keyed to a live process**, not a timeout, so a
  six-hour run cannot have its weights pulled out from under it and a crashed run
  cannot deadlock the cache.
- Downloads land in a staging directory and are renamed into place atomically. An
  interrupted transfer never appears complete.

---

## Bring-up

### 1. RustFS

RustFS runs as its own systemd service on this host.

```bash
curl -O https://rustfs.com/install_rustfs.sh && bash install_rustfs.sh
sudo $EDITOR /etc/default/rustfs      # RUSTFS_ACCESS_KEY, RUSTFS_SECRET_KEY, RUSTFS_VOLUMES
sudo systemctl restart rustfs
sudo systemctl status rustfs --no-pager
```

Defaults: S3 API on `:9000`, console on `:9001`, data under `/data/rustfs0`. Point
`RUSTFS_VOLUMES` at the 8 TB volume.

### 2. Nawāt

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env && $EDITOR .env  # endpoint, bucket, credentials, cache ceiling
```

`nawat` reads `.env` on its own — the nearest one at or above the working
directory, the way git finds its root — so there is nothing to source and nothing
to re-export in a new shell. Real environment variables still win over the file,
so `NAWAT_CACHE_CEILING=40GB nawat free` overrides for one command. Use
`--env-file PATH` to point elsewhere, or `--no-env-file` to ignore it entirely.
`nawat check` prints which file is in force, because misconfiguration is usually
the wrong file rather than the wrong value.

### 3. Confirm

```bash
nawat check --create-bucket
```

This does not ping the endpoint — it writes, lists, verifies and deletes a probe
object, because reachable credentials that cannot delete still cannot run this
platform.

```
ok    configuration              /home/lap/lap/tr/.env
ok    cache root                 /home/lap/nawat/cache
ok    state directory            /home/lap/nawat/cache/.nawat
ok    state database             /home/lap/nawat/cache/.nawat/nawat.sqlite3
ok    cache ceiling              120 GB ceiling + 10.0 GB reserve on a 250 GB filesystem (216 GB free)
ok    object storage reachable   http://127.0.0.1:9000 · bucket nawat
ok    object storage round trip  write, list, verify and delete all succeeded

Ready. Object storage is reachable and this host can publish, verify and reclaim.
```

---

## Keys

One name for a thing, mapping 1:1 onto an object-storage prefix and a local path.

```
models/unsloth/Qwen2.5-VL-7B-Instruct
datasets/ocr-arabic-v3
runs/2026-07-28-a91f/adapter
exports/qwen-ocr-v3-gguf
```

For `models/` and `datasets/`, the part after the kind **is** the upstream hub repo id.
That is what makes "downloaded from the internet exactly once, ever" fall out without a
mapping table: fetch it once, it is written through to object storage, and every later
resolution — including after eviction — comes from the store.

---

## Running an experiment from the shell

```bash
nawat hold \
  --model   models/unsloth/Qwen2.5-VL-7B-Instruct \
  --dataset datasets/ocr-arabic-v3 \
  --out     runs/2026-07-28-a91f/adapter \
  -- python train.py
```

That single command stages the inputs from object storage (evicting whatever it must,
safely, to make room), holds them under lease for the lifetime of the trainer, runs it,
then uploads the outputs, verifies them file by file, and reclaims the local copy.
Nothing is deleted by hand at any point.

The script receives its configuration through the environment, so it runs unmodified
outside the platform:

| Variable | Meaning |
|---|---|
| `NAWAT_RUN_ID` | Identifier for this run |
| `NAWAT_OUT_DIR` | Where to write artifacts |
| `NAWAT_MODEL_DIR` | Staged base model |
| `NAWAT_DATASET_DIR` | Staged dataset (first, if several) |
| `NAWAT_DATASET_DIRS` | JSON array of every staged dataset |
| `NAWAT_INPUTS` | JSON object mapping every input key to its path |

```python
import os, pathlib

model = pathlib.Path(os.environ["NAWAT_MODEL_DIR"])
data  = pathlib.Path(os.environ["NAWAT_DATASET_DIR"])
out   = pathlib.Path(os.environ["NAWAT_OUT_DIR"])
```

Outputs are published only when the command exits 0. A failed run leaves them on disk
for inspection.

---

## In a notebook

```python
import nawat

model = nawat.resolve("models/unsloth/Qwen2.5-VL-7B-Instruct")   # staged, leased
data  = nawat.resolve("datasets/ocr-arabic-v3")
...
nawat.publish(out_dir, "runs/2026-07-28-a91f/adapter")           # uploaded, verified, reclaimed
```

`resolve` holds the artifact for the lifetime of the kernel process. To scope it
tighter:

```python
with nawat.holding("models/base", "datasets/ocr-arabic-v3") as staged:
    ...
```

---

## Commands

| Command | What it does |
|---|---|
| `nawat status` | Occupancy against the ceiling, disk, what is held |
| `nawat ls` | What is on local disk, with flags: `K` kept · `L` in use · `R` in object storage |
| `nawat resolve KEY` | Make it present locally and print the path |
| `nawat keep KEY` / `release KEY` | Exempt from reclamation, or stop |
| `nawat free [--need 16GB]` | Reclaim least-recently-used space; `--dry-run` to look first |
| `nawat publish DIR KEY` | Upload, verify, reclaim |
| `nawat add DIR KEY` | Adopt a directory already on disk as an artifact |
| `nawat verify KEY` | Compare the local copy against its replica |
| `nawat rm KEY` | Remove one artifact (refuses if unreplicated) |
| `nawat registry` | What object storage holds, cached or not |
| `nawat leases` | What is in use, and by whom |
| `nawat hold ... -- CMD` | Stage, run, publish |
| `nawat check` | Bring-up verification |
| `nawat config` | Configuration in force, credentials redacted |

Exit codes are stable: `2` invalid key, `3` not found, `4` store unavailable,
`5` verification failed, `6` insufficient space, `7` offline, `8` protected.

---

## What is on disk

```
$NAWAT_CACHE_ROOT/
  models/unsloth/Qwen2.5-VL-7B-Instruct/
    .nawat-artifact.json          # key, size, file count, fetch time
    config.json
    model.safetensors
  .nawat-staging/                 # in-flight downloads, renamed into place on completion
  .nawat/
    nawat.sqlite3                 # artifact record and leases
    cache.lock
```

`.nawat-artifact.json` is written into each artifact directory so the cache describes
itself: delete the database and `nawat status` rebuilds it from disk. It is a dotfile,
never uploaded, and never counted towards an artifact's size.

---

## Tests

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
```

Per PRD §12 the eviction tests are a release gate — `tests/test_eviction.py` asserts
that unreplicated, kept, in-use and unverifiable artifacts are never removed, and that a
refusal deletes nothing. `tests/test_store_s3.py` drives the S3 backend over real HTTP
(list pagination past 1000 keys, parallel multipart, prefix isolation) against an
in-process endpoint, so the RustFS path is exercised without RustFS running.

---

## Known limits in Phase 1

- Mutating operations take a host-wide lock, so two simultaneous fetches serialise.
  Acceptable on a single-GPU host with serial runs; revisit if it bites.
- An upstream fetch reserves space from the hub's reported size; if that is unavailable
  the reserve falls back to `NAWAT_MIN_FREE` and the difference is reclaimed after the
  fetch rather than before it.
- Verification compares name and size, not content hashes — as specified (NFR-1.2).
  Checksums would catch silent corruption at the cost of reading every byte back.
