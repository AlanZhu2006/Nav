# Repository consolidation checks — 2026-09-17

The pending navigation source passed Python parsing, JSON parsing and Shell
syntax checks: 334 Python files, 26 JSON files and 79 Shell/Slurm scripts.
Entry-document local links and `git diff --check` were also checked.

The existing GEM, relocalization, RGB transport, multipart, depth-raster and
image-controller checks produced **323 passed, 6 skipped**. The six skipped
checks require CUDA; this run used CPU only. No scene replay or navigation
experiment was launched.

The matching environment on this machine is `lingbot-map`, with `LINGBOT_REPO`
and `PYTHONPATH` pointing to the retained dependency at
`/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map`. Its attention source
matches the pinned GEM integration hash. A different research checkout at
`/home/asus/Research/lingbot-map` has modified attention code and is not an
interchangeable dependency for these checks.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  LINGBOT_REPO=/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map \
  PYTHONPATH=/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map \
  /home/asus/miniconda3/envs/lingbot-map/bin/python -m pytest -q \
  MemNavData/test_gem_*.py \
  MemNavData/test_certified_relocalization_runtime.py \
  MemNavData/test_xnavdp_revisit_server.py \
  MemNavData/test_xnavdp_rgb_contract.py \
  MemNavData/test_multipart_crlf_repair.py \
  MemNavData/test_lingbot_depth_raster.py \
  MemNavData/test_image_controller_goal_adapter.py
```

The local `.workspace-maintenance/20260917-git-sync/` directory retains source
snapshots, validation output and final synchronization receipts. Large data and
nested repositories are excluded by tracked ignore rules.
