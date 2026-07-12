#!/usr/bin/env bash
# Create the `habitat` conda env used by MemNavData generators (generate_twoleg.py,
# revisit_sweep_gen.py, mp3d_loadtest.py). Headless habitat-sim build for HPC (EGL
# offscreen rendering; needs a GPU node to render RGB). Path-based env to match the
# other /scratch/lg154/conda-envs/* envs.
set -euo pipefail

ENV_PREFIX=/scratch/lg154/conda-envs/habitat
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh

echo "===== creating habitat-sim env at ${ENV_PREFIX} ====="
conda create -y -p "${ENV_PREFIX}" -c conda-forge -c aihabitat \
    python=3.9 habitat-sim=0.3.1 headless

conda activate "${ENV_PREFIX}"
echo "===== pip: generator python deps ====="
pip install --no-input numpy-quaternion scipy pandas pillow pyarrow

echo "===== import smoke test (no GPU needed for import) ====="
python - <<'PY'
import habitat_sim, magnum, quaternion, scipy, pandas, numpy, PIL
print("habitat_sim", habitat_sim.__version__)
print("numpy", numpy.__version__, "quaternion OK", "scipy", scipy.__version__)
print("IMPORTS_OK")
PY
echo "===== DONE: ${ENV_PREFIX} ====="
