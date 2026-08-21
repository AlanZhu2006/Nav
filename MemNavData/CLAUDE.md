# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file covers `MemNavData/` specifically; the repo-root `/home/asus/Research/Nav-graph-blind/CLAUDE.md` covers the sibling `NavDP/` (IsaacSim benchmark) and `InternNav/` (training) trees. MemNavData is **Habitat-native, not IsaacSim** — the root file's IsaacSim setup does not apply here.

## What this directory is

Research working directory for the MemNav / **Certified Episodic Compass (CEC)** project: image-goal navigation with causal online episodic memory on top of a frozen NavDP controller. It contains episode generation, frozen benchmark manifests, closed-loop Habitat evaluation clients, statistical summarizers/verifiers, forensic audits, and HPC orchestration. It is intentionally **flat** (~500 Python files, ~200 shell/sbatch, ~140 markdown): one file per experiment stage, suffixed `_<YYYYMMDD>`. The five small subdirs (`analysis/`, `manifests/`, `posthoc/`, `viz/`, `data_download/`) are pre-2026-08-06 legacy.

**Before doing anything substantive, read the newest `STATUS_*.md` ledger** (Chinese-language running ledgers; as of 2026-08-17 the head is `STATUS_20260817_PI3X_LEARNED_RELOCALIZER_FINAL14.md`, which lists its predecessors). Each ledger's opening section names which file supersedes it. `HPC_SHARED_SSH_OPERATIONS_20260816.md` is the mandatory SSH operating procedure. Older method context: `STATUS_20260814_PAPER_EVAL.md` (CEC architecture, evidence table, what may / may not be claimed in the paper).

Results use a strict status vocabulary — `confirmed` / `strong internal` / `underpowered` / `mechanism (oracle)` / `prospective` / `null-negative` / `infrastructure failure`. Never upgrade a label, never present an infrastructure failure or an oracle as a method result.

## Commands

Two conda interpreters, always referenced by **absolute path** (never `conda activate`):

- `MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python` — torch + LingBot + LightGlue: policy servers, all tests, audits, summarizers.
- `HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python` — habitat-sim 0.3.3, python 3.9, EGL: anything that renders (generation, closed-loop eval client).
- HPC equivalents live under `/scratch/lg154/conda-envs/{memnav,habitat}`.

**Tests** (193 `test_*.py`, unittest-style, CPU-only, no Habitat/GPU/network needed; each non-test module has a same-named `test_` twin). Imports are `from MemNavData.<module> import …` via namespace packages, so run **from the repo root**, never from inside MemNavData:

```bash
cd /home/asus/Research/Nav-graph-blind
/home/asus/miniconda3/envs/memnav/bin/python -m unittest MemNavData.test_deterministic_eval_protocol            # single test module
/home/asus/miniconda3/envs/memnav/bin/python -m pytest -q -p no:cacheprovider MemNavData/test_<name>.py         # pytest also works
```

**Episode generation** — `generate_twoleg.py` (documented in `README.md` here): genuine 2-leg/3-leg multi-stop episodes on MP3D in InternData-N1 layout, one scene per process, deterministic per `--seed`, Slurm-array parallelism over scenes. `--window 32` must match the LingBot precompute.

**Closed-loop eval** — always two Flask servers first, then the Habitat client over HTTP:

```bash
# server side (MEMNAV_PY): NavDP/baselines/memnav/memnav_server.py + NavDP/baselines/navdp/navdp_server.py
# client side (HAB_PY):   eval_2leg_habitat.py / eval_3leg_habitat.py  →  --out/{summary.json,metric.csv}
bash run_certified_mixed_role_safety_gate_local.sh   # any run_*.sh / *_local.sh wires ports, env vars, arms for you
```

Local runners are the 13 `*_local.sh` and the 8 `*_5090.sh` (the latter run Pi3X work on the shared `/home/cv/...` RTX 5090 box and hard-assert the GPU name). **`run_*.sh` does NOT mean local** — most are the inner per-scene stage executed inside Slurm.

**HPC** (NYU **Torch** cluster, not Greene): ssh alias `alantorch` → `yz11502@login.torch.hpc.nyu.edu`, account `torch_pr_769_tandon_advanced`, no `module load`; GPU stages run inside `singularity exec --nv /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif` with datasets as read-only squashfs `--overlay`s (omitting the overlay is a known incident class). The chain is:

```
submit_*_hpc.sh (local: py_compile + tests + bash -n → content-addressed immutable
  source bundle <task>_<sha16> → rsync → sbatch chain with afterok dependencies)
    → slurm_*.sbatch (re-verify receipts/SHAs, start server pair on collision-checked ports)
        → run_*.sh (one scene: Latin-square arm order, cross-arm invariant checks)
            → eval_*_habitat.py
```

Bundles land in `/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/`, run roots in `/scratch/yz11502/Research/Nav-axis-uturn-results/<experiment>/<RUN_TAG>` (must not pre-exist), logs in `…/slurm_logs/`. Never reuse an SSH socket that resolves to another user; a hung no-PTY mux channel does not mean auth is dead (see `HPC_SHARED_SSH_OPERATIONS_20260816.md`).

## Architecture

### The evidence pipeline (file-prefix taxonomy)

Prefixes encode a **role in a pre-registered evidence chain**, not a topic:
`freeze_` (seal an outcome-blind population) → `build_`/`materialize_` (construct manifests / roll out online Goal-A traces) → `finalize_` (merge to a sealed population) → `validate_` (fail-closed input gates) → `eval_` (closed-loop client) → `summarize_` (paired stats: exact McNemar, scene-cluster bootstrap CI) → `verify_`/`independent_verify_` (second recount from raw files, never trusting the summarizer) → `audit_` (renderer-free forensic checks) — plus `diag_` (mechanism probes) and `analyze_` (offline stratification).

Document lifecycle mirrors it: `*_PROTOCOL_<date>.md` is **frozen before any outcome is read** (with a machine-readable JSON twin whose SHA-256 is embedded downstream) → `*_RECEIPT_<date>.json` (submission record with job IDs and boolean guards) → `*_INCIDENT_<date>.json` (failed attempts are recorded, numbered `ATTEMPT<n>`, never deleted) → `*_RESULT_<date>.md`. Corrections are new `*_AMENDMENT_*` / `*_REPAIR_*` documents naming their predecessor (`authorization_inherited_from`, `repair_provenance`) — **never edit a frozen document**. Everything is content-addressed; scripts abort on hash mismatch and `chmod a-w` their outputs.

Consequences when changing code:
- A file used by an experiment must be listed in the `required=(...)` array of its `submit_*`/`package_*` script and have its `test_*.py` twin, or bundling fails.
- Local tests are a release gate: nothing reaches Slurm without them.
- Do not re-tune thresholds, re-filter denominators, or re-sample populations after an outcome has been read; failed attempts do not consume the frozen population.

### Runtime model (what actually runs)

`NavDP/baselines/memnav/policy_agent.py` (`MemNavAgent`) is the live agent, but it loads the model **from InternNav**, not from the local `policy_network.py`/`policy_backbone.py` (those are a stale v1 sketch — do not extend them). Authoritative model code: `InternNav/internnav/model/basemodel/memnav/memnav_policy.py` — frozen LingBot-Map GCT streaming backbone (S=8 scale frames, W=32 window, anchor margin 39), trainable retrieval/novel/revisit heads with a gate that becomes a cross-attention bias, NavDP DDPM decoder, no critic (geometric collision selection at eval). `memnav_server.py` also imports optional research modules **from MemNavData** (`phase_b_runtime`, `lingbot_pnp_localization`, `cdec_pairwise_runtime`, `pi3x_online_relocalizer`) — the dependency between the two trees is bidirectional, all via repo-root `sys.path`.

Eval clients speak only HTTP (`/navigator_reset`, `/memory_step`, `/imagegoal_step`, `/certified_relocalize`, …). `eval_3leg_habitat.py` deliberately does `import eval_2leg_habitat as base` and reuses its audited internals — follow that pattern rather than duplicating HTTP/waypoint/collision logic. Contract modules that must not drift: `deterministic_eval_protocol.py` (seed algebra, leg-1 trace schema — what makes Goal-A byte-identical across arms), `phase_b_feature_schema.py` (feature order is checkpoint ABI), `navdp_goal_switch.py` (FIFO-reset semantics that protect the long-term-memory variable).

Standard comparison arms recur everywhere: `native` (frozen NavDP), `raw_direct`/`raw_fixed_bearing` (DINO, no certificate), `geometry_fixed`, `certified` (CEC). Success is distance-only (default 1.0 m); waypoint decode must match the InternNav label convention pinned in `eval_2leg_habitat.py`'s docstring.

### Paths and weights (the sharp edge)

Default paths across scripts span **three checkout names** (`/home/asus/Research/Nav`, `…/Nav-axis-uturn`, this tree `…/Nav-graph-blind`) and two scratch users (`lg154` data/envs, `yz11502` results/bundles). Locally there is no `lingbot-map/` under `NavDP/baselines/memnav/`, so runs need explicit `LINGBOT_REPO` / `LINGBOT_WEIGHTS` (commit-pinned), `MEMNAV_CKPT` (e.g. `gatecurr600.memnav.ckpt`), `NAVDP_CKPT`, and where applicable `ASSET_ROOT_OVERRIDE` / `EPISODE_ROOT_OVERRIDE`. Server env knobs that must match the checkpoint/precompute: `MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_GATE_FUSION=complementary` etc.

Outputs never go in this directory or relative to CWD: local results go to `/home/asus/Research/Nav-graph-blind/.diagnostics/<experiment>_<date>/` (also holds vendored external baselines and throwaway envs), HPC results to `/scratch/yz11502/Research/Nav-axis-uturn-results/`, 5090 results to `/home/cv/memnav_eval/results/`. Frozen benchmark manifests (`*.json` at top level, e.g. `strict_graph_blind_20260806.json`) carry the dataset roots.
