#!/usr/bin/env bash
# Explicit HPC paths; local runner defaults remain unchanged.
: "${REPAIRED_BUNDLE:?}"
export REPAIRED_MEM_PY=/scratch/lg154/conda-envs/memnav/bin/python
export REPAIRED_HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
export REPAIRED_HAB_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor
export REPAIRED_HAB_EXTRA=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_hab_cv2_20260908
export REPAIRED_MEM_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
export REPAIRED_NAV_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
export REPAIRED_LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export REPAIRED_INTERNNAV=${REPAIRED_BUNDLE}/InternNav
export REPAIRED_DEPENDENCIES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2/third_party/python
export REPAIRED_LIGHTGLUE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2/third_party/LightGlue
export TORCH_HOME=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2/torch_home
export REPAIRED_SOURCE_RECEIPT=${REPAIRED_BUNDLE}/SOURCE_BUNDLE.sha256
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PATH=/scratch/lg154/conda-envs/habitat/bin:${PATH}
export PYTHONPATH=${REPAIRED_BUNDLE}:${REPAIRED_BUNDLE}/MemNavData:${REPAIRED_INTERNNAV}:${REPAIRED_INTERNNAV}/src/diffusion-policy:${REPAIRED_DEPENDENCIES}:${REPAIRED_LIGHTGLUE}:${REPAIRED_HAB_VENDOR}:${REPAIRED_HAB_EXTRA}
