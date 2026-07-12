# MP3D Revisit Dataset — Pipeline Progress & Issues Log

_Last updated: 2026-07-11._

End-to-end status of the MP3D revisit dataset (`mp3d_revisit_v0`): generation → packing →
LingBot KV-cache precompute, plus the issues hit along the way and how they were resolved.

## Pipeline overview

| Stage | Script | Output |
|-------|--------|--------|
| 1. Generate | `MemNavData/generate_twoleg.py` via `data_download/generate_mp3d_all.sbatch` | `mp3d_revisit_v0/vln_n1/traj_data/mp3d_{2,3}leg/<scene>/episode_XXXX/` |
| 2. Pack (0–53) | `mksquashfs` → `data_download/pack_mp3d_pt1.*` | `datasets/_overlays/mp3d_revisit_v0_pt1.sqf` |
| 3. Precompute | `InternNav/scripts/dataset_converters/precompute_lingbot_features.py` via `data_download/precompute_mp3d_pt1.sbatch` | `mp3d_revisit_v0_feat/vln_n1/traj_data/.../{lingbot_cache,lingbot_cam_cache}.npz` |

**90 scenes total** (array `0-89`), each targeting **15 two-leg + 25 three-leg** episodes.
Episode layout: `data/chunk-000/episode_000000.parquet`, `meta/gen_meta.json`,
`videos/chunk-000/observation.images.{rgb/*.jpg,depth/*.png}`, `goal_*.jpg`.

## Key paths

```
Scenes (source glb):   /scratch/lg154/Research/datasets/mp3d/mp3d/<scene>/<scene>.glb
Generated frames:      /scratch/lg154/Research/datasets/mp3d_revisit_v0/vln_n1/traj_data
Packed 0–53 (ro):      /scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
  in-overlay path:     /mp3d_revisit_v0/vln_n1/traj_data          (via apptainer --overlay)
Precomputed caches:    /scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data
Env (gen):             /scratch/lg154/conda-envs/habitat          (habitat-sim 0.3.1 headless)
Env (precompute):      /scratch/lg154/conda-envs/memnav           (torch 2.8)
LingBot weights:       NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt
```

## Current status (2026-07-11)

- **Generation:** scenes 0–89 complete. All non-scene-limited scenes at 15/25.
- **Packing:** 0–53 packed into `pt1.sqf`; source pruned (to reclaim inodes).
- **Precompute (0–53, `--skip_scale`):** ~846 / 1944 trajectories cached (~285 MB/traj,
  ~0.5 TB projected). Remaining shards in flight.
- **Validation:** 54–89 → 1371/1371 clean; 0–53 pack → 1944/1944 structurally clean.

### Integrity contract (used by the validator)

Per episode: `gen_meta.n_frames == #rgb .jpg == #depth .png == parquet num_rows`.
The parquet is written **last** in `save_traj`, so if it exists with `rows == n_frames` the
frames `0..n_frames-1` are guaranteed complete and consistent.

### Scene-limited scenes (NOT failures — accept as-is)

Judged by `navigable_area` in the `[make_sim]` log line; too small to satisfy the
revisit/covisibility/geodesic criteria (3-leg needs more room than 2-leg):

| Scene | idx | navmesh | result |
|-------|-----|---------|--------|
| gZ6f7yhEvPG | 62 | 10.9 m² | 2/0 |
| i5noydFURQK | 64 | 39.5 m² | 8/1 |
| HxpKQynjfin | 23 | 2.8 m²  | 0/0 |
| D7G3Y4RVNrH | 17 | 6.1 m²  | 0/0 |
| RPmz2sHmrrY | 31 | 12.1 m² | 0/0 |
| 2t7WUuJeko7 | 6  | 13.6 m² | 0/0 |

Do **not** keep resubmitting these — they reproduce the same low count.

## Issues encountered

### 1. `/scratch` inode quota blown during generation
Per-frame jpg/png hit the 5M-inode cap. **Fix:** pack finished scenes into a squashfs `.sqf`
and prune the source. See `data_download/pack_mp3d_pt1.*`.

### 2. Read-only squashfs can't be mounted with raw `squashfuse_ll`
On compute nodes: `fuse: failed to exec fusermount3`. **Fix:** mount the NYU-supported way —
```
apptainer exec --nv --overlay mp3d_revisit_v0_pt1.sqf:ro -B /scratch/lg154 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif  bash -c '<cmd>'
```
Inside the overlay the tree is at `/mp3d_revisit_v0/vln_n1/traj_data` (packed with
`-keep-as-directory`). Writes to `/scratch/lg154/...` land on the real fs (overlay is `:ro`).
**Training must run inside this same overlay** so recorded absolute rgb paths resolve.

### 3. Orphan-tail frames on regeneration (generator bug) — FIXED
`save_traj` did `os.makedirs(exist_ok=True)` without clearing the dir, so re-running a scene
whose new trajectory was **shorter** than a prior one left stale tail frames (idx ≥ n_frames)
from the old run. Data stayed correct (training indexes by the parquet), but it wasted inodes.
**Fix:** added `shutil.rmtree(out_dir, ignore_errors=True)` at the top of `save_traj`. Cleaned
up 33,712 orphan frames across 27 episodes in 5 regenerated scenes via an mtime-hardened
surgical delete (kept `0..n_frames-1`, dropped strictly-older orphans) — no full regen needed.

### 4. Pack captured a partial regen (data loss) — RESOLVED (accept partial)
`YFuZgdQ5vWj` (idx48) and `YmJkqBEsHnH` (idx50) were generated FULL (15/25) in round-1, but a
later refresh job re-ran them, yielded only partial counts (9/8, 7/0) before being cut short, and
**overwrote** the full data. The pack was then built from that partial state — losing ~56 episodes.
**Root cause:** `run_legs` regenerates the whole leg from `episode_0000` whenever a scene is `< n`
complete (it is all-or-skip, not incremental fill), so a cut-short rerun reduces the count; then it
got packed **before completeness was validated.**
**Decision:** accept the partial 48/50 already in `pt1.sqf` — they are valid episodes (passed the
integrity contract), just fewer (~1.5% of the dataset, 2 of 90 scenes). We did **not** regenerate
(regen kept hitting black-hole nodes; and putting a full copy elsewhere would either duplicate
against pt1's copy or need a load-time exclude-set — not worth it for 2 scenes). The on-disk
partial regen folders were deleted. **`pt2.sqf` contains 54-89 only** (48/50 explicitly excluded).
**Lesson (now standing rule): always run the completeness validator BEFORE packing.**

### 5. Black-hole GPU nodes (ongoing) — MITIGATED via partition switch
`ga015`, `ga035`, `ga040` are broken but report `mix` (not `down`), so the scheduler keeps
assigning them our jobs, which crash in ~30–50 s (empty `.err`, fast SIGSEGV-like exit). They
caused ~10 job failures (gen scene 62 attempts, precompute shards 1/4/5/6/7, regen idx48).
NYU's submit filter **blocks `--exclude`**, so we can't dodge them by hand. **Mitigation:** both
sbatch scripts now default to **`--partition=h100_tandon`** (nodes `gh###`), a different pool
that avoids the broken `ga###` nodes entirely. Draining them permanently is an HPC request.

## How to resume / verify

```bash
# Generate (resumable; skips scene-legs already at N complete):
N2=15 N3=25 sbatch --array=<idx> MemNavData/data_download/generate_mp3d_all.sbatch

# Precompute a shard set (idempotent; skips trajs whose npz already exist):
sbatch --array=0-7%4 --export=ALL,NUM_SHARDS=8 MemNavData/data_download/precompute_mp3d_pt1.sbatch

# Validate integrity (run in the memnav env; pass the traj_data root as argv[1]):
python MemNavData/analysis/validate_mp3d.py /path/to/vln_n1/traj_data
#   -> reports "N/N pass the count contract", scenes below target, and any bad episodes.
#   For packed 0–53, run it inside the apptainer overlay (see issue #2) with root
#   /mp3d_revisit_v0/vln_n1/traj_data.

# Clean orphan-tail frames after a regen (dry-run by default; --apply to delete):
python MemNavData/analysis/mp3d_cleanup_orphans.py            # edit the BENIGN list first
```

Generation is deterministic given `args.seed` (= scene index) **and** config; changing config
(`window`, `anchor_margin`, `goal_jitter`, …) forces a full regen and changes counts. Completed
scenes with unchanged config are frozen (they `[skip]`).
