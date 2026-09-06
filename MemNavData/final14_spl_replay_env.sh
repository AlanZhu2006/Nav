#!/usr/bin/env bash
# Shared immutable runtime/data lineage for the Table-III measurement replay.
export FROZEN_BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
export BENCH_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T115533Z_attempt7_handoff/benchmarks/natural_direction
export EXPECTED_MANIFEST_SHA=7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a
export BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
export SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
export BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
export HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
export MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
export MODEL_ROOT=/scratch/yz11502/Research/Nav-axis-uturn
export INTERNNAV_ROOT=${MODEL_ROOT}/InternNav
export MEMNAV_CKPT=${MODEL_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
export NAVDP_CKPT=${MODEL_ROOT}/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
export LIGHTGLUE_REPO=${FROZEN_BASE}/third_party/LightGlue
export DEPENDENCY_ROOT=${FROZEN_BASE}/third_party/python
export TORCH_HOME=${FROZEN_BASE}/torch_home
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
