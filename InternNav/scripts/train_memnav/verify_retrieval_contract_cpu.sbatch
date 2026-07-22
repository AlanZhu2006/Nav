#!/bin/bash
#SBATCH --job-name=ret_contract
#SBATCH --output=/scratch/yz11502/Research/diagnostics/legacy_retrieval/logs/contract-%j.out
#SBATCH --error=/scratch/yz11502/Research/diagnostics/legacy_retrieval/logs/contract-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --export=ALL
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yz11502/Research/Nav-memnav-1c7aee0-deploy/InternNav}
DIAG_ROOT=${DIAG_ROOT:-/scratch/yz11502/Research/diagnostics/legacy_retrieval}
SCRIPT=${SCRIPT:-${DIAG_ROOT}/eval_legacy_memnav_retrieval.py}
IMG=${IMG:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}
BASE_SIF=${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}
CONDA_SH=${CONDA_SH:-/scratch/lg154/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/scratch/yz11502/conda_envs/memnav-runtime-8414d2d}
FEATURE_ROOT=${FEATURE_ROOT:-/scratch/yz11502/Research/datasets/mp3d_revisit_v0_feat_kf320_1371557/vln_n1/traj_data}
LINGBOT_REPO=${LINGBOT_REPO:-/scratch/yz11502/Research/lingbot-map-7ff6f3e-clean}
EXPECTED_CACHE_SIGNATURE=${EXPECTED_CACHE_SIGNATURE:-b25fb60ed3abd8bca31da996274eba22e1eec669d968a5b5a9d96fe6487fc809}
OUTPUT=${OUTPUT:-${DIAG_ROOT}/current8-contract-${SLURM_JOB_ID}.json}

mkdir -p "${DIAG_ROOT}/logs"
for path in "${REPO_ROOT}" "${SCRIPT}" "${IMG}" "${BASE_SIF}" \
            "${CONDA_SH}" "${CONDA_ENV}" "${FEATURE_ROOT}" "${LINGBOT_REPO}"; do
    [[ -e "${path}" ]] || { echo "ABORT missing dependency: ${path}" >&2; exit 1; }
done
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == \
   "1c7aee077252fa3d23fda6532c5660a21ddca702" ]] || {
    echo "ABORT unexpected deployment commit" >&2
    exit 1
}

export REPO_ROOT SCRIPT FEATURE_ROOT LINGBOT_REPO EXPECTED_CACHE_SIGNATURE OUTPUT
export CONDA_SH CONDA_ENV
apptainer exec --overlay "${IMG}:ro" -B /scratch "${BASE_SIF}" bash -c '
    set -euo pipefail
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src/diffusion-policy:${PYTHONPATH:-}"
    cd "${REPO_ROOT}"
    python -m pip check
    python -c "import numpy, pyarrow, torch; print(\"contract_dependencies=PASS\", numpy.__version__, pyarrow.__version__, torch.__version__)"
    exec python "${SCRIPT}" \
        --root-dir /mp3d_revisit_v0/vln_n1/traj_data \
        --feature-root "${FEATURE_ROOT}" \
        --checkpoint /dev/null --output "${OUTPUT}" \
        --lingbot-repo "${LINGBOT_REPO}" --lingbot-weights /dev/null \
        --window 32 --num-scale 8 --max-frame-num 4096 \
        --sampling-seeds 0 --panels current8 --data-split val \
        --validation-fraction 0.1 --split-seed 0 \
        --strict-feature-coverage --require-versioned-cache \
        --expected-cache-signature "${EXPECTED_CACHE_SIGNATURE}" \
        --contract-verification-only
'
test -s "${OUTPUT}"
sha256sum "${OUTPUT}"
