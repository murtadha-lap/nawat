# Operating Nawāt

The operator's view: bring-up, recovery, backup, and sizing. The premise of the
platform is that object storage is truth and local disk is disposable — every
procedure below leans on that.

## Bring-up, from bare host

```bash
# 1. RustFS — its own systemd service, owns the 8 TB volume
curl -O https://rustfs.com/install_rustfs.sh && bash install_rustfs.sh
sudo $EDITOR /etc/default/rustfs     # RUSTFS_ACCESS_KEY, RUSTFS_SECRET_KEY, RUSTFS_VOLUMES
sudo systemctl restart rustfs

# 2. Nawāt
cd /home/lap/lap/tr
python3 -m venv .venv && .venv/bin/pip install -e ".[api,hub,agent]"
cp .env.example .env && $EDITOR .env
.venv/bin/nawat check --create-bucket   # must end with "Ready."

# 3. The control plane as a service
sudo cp deploy/nawat-api.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nawat-api
```

`nawat check` is the acceptance test: it writes, lists, verifies and deletes a
probe object. Reachable-but-cannot-delete credentials fail here, not mid-run.

## What restarts how

| Failure | What happens | Operator action |
| --- | --- | --- |
| Host reboot | Both services restart via systemd. Nawāt reconciles the cache against disk, clears leases from before the boot, and marks interrupted runs failed. | None |
| Control plane crash mid-run | systemd restarts it; the orphaned trainer's lease dies with its process; the run record is marked failed with "did not survive a restart". Outputs stay on disk for inspection. | Resubmit the run |
| Trainer crash | Exit code recorded, run failed, log kept locally and in `runs/<id>/record`. | `nawat logs <id>`, then `nawat agent --run <id>` to diagnose |
| Inference server dies | The session record notices the dead pid, the weights' lease is released, the cache may reclaim them. | `nawat serve <model>` again |
| RustFS down | Nothing is evicted (verification cannot complete); fetches and publishes fail with the endpoint named. Runs already training continue; publish fails at the end and outputs stay local. | Restore RustFS, re-run `nawat publish` |
| State database lost | The cache rebuilds from `.nawat-artifact.json` markers on disk; run history rebuilds from `run.json` files. Leases and pins are lost — pins must be re-applied. | `nawat status` triggers recovery |

## Backup policy

**The RustFS volume is the single point of failure by design** (PRD §12).
Everything else on the host is disposable; that volume is not.

- Back up the directory named by `RUSTFS_VOLUMES` (default `/data/rustfs0`)
  with any file-level tool — restic, borg, rsync to a second machine. Objects
  are ordinary files; snapshot-friendly.
- Frequency to match tolerance: run records and adapters change after every
  run; base models and datasets change rarely. A daily incremental captures
  both cheaply.
- The local cache, the state directory and the workspace need no backup — the
  cache is a mirror, state rebuilds from disk, and the workspace is a git
  repository (push it somewhere if the scripts matter).
- Test restores the same way the platform tests itself: point a scratch
  `NAWAT_S3_ENDPOINT` at the restored volume and run `nawat check`.

## Sizing the cache ceiling

Size for the working set, not the disk (PRD §12: a ceiling too low for the
working set causes thrashing — runs stall re-fetching what was just evicted).

```text
one base model (fp16, 7B)        ~16 GB
a second base for comparison     ~16 GB
active dataset                    corpus-dependent
adapters, evals, scratch          ~5 GB
trainer checkpoints (outside the cache, in /tmp)  budget NAWAT_MIN_FREE for it
```

On the 233 GB NVMe: `NAWAT_CACHE_CEILING=120GB`, `NAWAT_MIN_FREE=10GB` leaves
~100 GB for the OS, the venv and tempfile scratch. Watch `nawat status`:
sustained >90% occupancy with frequent evictions means the ceiling is too low
for the working set — raise it or trim the set.

## Storage-pressure signals

Surfaced in two places, same thresholds: `nawat status` (stderr warnings) and
`GET /health` (`warnings` array — point external monitoring here; it needs no
token).

- *cache at N% of its ceiling* — eviction is imminent; fine if replicated.
- *N GB exists only on this disk* — unreplicated artifacts; the one state in
  which data loss is possible. Publish, or investigate why publish failed.
- *only N GB free on the filesystem* — the disk itself, independent of the
  ceiling. Something outside the cache is eating the disk.

## Small-file corpora

A corpus of many small files starves the dataloader when streamed from object
storage (NFR-2.5). Pack it once:

```bash
nawat shard /data/raw-ocr-corpus datasets/ocr-arabic-v3-sharded
```

Sorted packing keeps a sample's files (image + label) in the same shard; the
result is plain tar shards plus `index.json`, streamable with `webdataset` or
`datasets`. Shards are verified after packing and again by `nawat verify`
against object storage like any artifact.

## Security posture

- The API binds to `NAWAT_API_HOST` (default loopback). Exposure beyond the
  local network is a reverse proxy you configure, never a wider bind (NFR-4.3).
- Set `NAWAT_API_TOKEN` before binding beyond loopback; `/health` stays open
  for monitoring, everything else then requires the bearer token (NFR-4.2).
- Credentials live in `.env` (gitignored) and are never written to logs, run
  records or API responses; `nawat config` shows `"set"` in their place (NFR-4.1).
- Training scripts run with the submitting user's privileges. There is no
  sandbox; do not point the platform at untrusted scripts (NFR-4.4).
- The optional agent backend receives training code and run metrics only.
  Pointing `NAWAT_AGENT_BACKEND=local` at an on-host endpoint keeps even that
  on-premises.
