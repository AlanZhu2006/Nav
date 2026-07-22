#!/bin/bash
#SBATCH --job-name=legacy_ret_audit
#SBATCH --output=/scratch/yz11502/Research/diagnostics/legacy_retrieval/logs/audit-%j.out
#SBATCH --error=/scratch/yz11502/Research/diagnostics/legacy_retrieval/logs/audit-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
# REQUIRED: complete scripts/train_memnav/README.md before every submission.
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/scratch/yz11502/Research/Nav-memnav-1c7aee0-deploy/InternNav}
DIAG_ROOT=${DIAG_ROOT:-/scratch/yz11502/Research/diagnostics/legacy_retrieval}
SCRIPT=${SCRIPT:-${DIAG_ROOT}/eval_legacy_memnav_retrieval.py}
IMG=${IMG:-/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf}
BASE_SIF=${BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}
CONDA_SH=${CONDA_SH:-/scratch/lg154/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-/scratch/yz11502/conda_envs/memnav-runtime-8414d2d}
CHECKPOINT=${CHECKPOINT:-/scratch/yz11502/Research/Nav/InternNav/checkpoints/mn38_aux000_s0_e1_20260715/ckpts/checkpoint-759/memnav.ckpt}
EXPECTED_CHECKPOINT_SHA=${EXPECTED_CHECKPOINT_SHA:-cf781e82da27f3e64afc8445def7bfdb332401ee7612b888536c122c2df2500c}
EXPECTED_CODE_COMMIT=${EXPECTED_CODE_COMMIT:-1c7aee077252fa3d23fda6532c5660a21ddca702}
EXPECTED_CACHE_SIGNATURE=${EXPECTED_CACHE_SIGNATURE:-b25fb60ed3abd8bca31da996274eba22e1eec669d968a5b5a9d96fe6487fc809}

ROOT_DIR=${ROOT_DIR:-/mp3d_revisit_v0/vln_n1/traj_data}
FEATURE_ROOT=${FEATURE_ROOT:-/scratch/yz11502/Research/datasets/mp3d_revisit_v0_feat_kf320_1371557/vln_n1/traj_data}
LINGBOT_REPO=${LINGBOT_REPO:-/scratch/yz11502/Research/lingbot-map-7ff6f3e-clean}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt}
MAX_SAMPLES=${MAX_SAMPLES:-32}
SAMPLING_SEEDS=${SAMPLING_SEEDS:-0 1}
PANELS=${PANELS:-legacy current8 current39}
VERIFY_CURRENT_GETITEM=${VERIFY_CURRENT_GETITEM:-0}
NAME=${NAME:-legacy-ret-smoke}
OUTPUT=${OUTPUT:-${DIAG_ROOT}/${NAME}-${SLURM_JOB_ID}.json}

mkdir -p "${DIAG_ROOT}/logs" "$(dirname "${OUTPUT}")"
for path in "${REPO_ROOT}" "${SCRIPT}" "${IMG}" "${BASE_SIF}" \
            "${CONDA_SH}" "${CONDA_ENV}" "${CHECKPOINT}" \
            "${FEATURE_ROOT}" "${LINGBOT_REPO}" "${LINGBOT_WEIGHTS}"; do
    [[ -e "${path}" ]] || { echo "ABORT missing dependency: ${path}" >&2; exit 1; }
done

actual_checkpoint_sha=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
[[ "${actual_checkpoint_sha}" == "${EXPECTED_CHECKPOINT_SHA}" ]] || {
    echo "ABORT checkpoint SHA ${actual_checkpoint_sha} != ${EXPECTED_CHECKPOINT_SHA}" >&2
    exit 1
}
actual_code_commit=$(git -C "${REPO_ROOT}" rev-parse HEAD)
[[ "${actual_code_commit}" == "${EXPECTED_CODE_COMMIT}" ]] || {
    echo "ABORT code commit ${actual_code_commit} != ${EXPECTED_CODE_COMMIT}" >&2
    exit 1
}
[[ -z "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]] || {
    echo "ABORT deployed repository has tracked modifications" >&2
    exit 1
}

echo "host=$(hostname) start=$(date --iso-8601=seconds)"
echo "code_commit=${actual_code_commit} lingbot_commit=$(git -C "${LINGBOT_REPO}" rev-parse HEAD)"
echo "checkpoint_sha=${actual_checkpoint_sha} script_sha=$(sha256sum "${SCRIPT}" | awk '{print $1}')"
echo "max_samples=${MAX_SAMPLES} sampling_seeds=${SAMPLING_SEEDS} panels=${PANELS} output=${OUTPUT}"
nvidia-smi -L

export REPO_ROOT DIAG_ROOT SCRIPT CHECKPOINT ROOT_DIR FEATURE_ROOT
export LINGBOT_REPO LINGBOT_WEIGHTS EXPECTED_CACHE_SIGNATURE OUTPUT
export CONDA_SH CONDA_ENV MAX_SAMPLES SAMPLING_SEEDS PANELS
export VERIFY_CURRENT_GETITEM

apptainer exec --nv --overlay "${IMG}:ro" -B /scratch "${BASE_SIF}" bash -c '
    set -euo pipefail
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src/diffusion-policy:${PYTHONPATH:-}"
    cd "${REPO_ROOT}"
    command -v python
    python -V
    python -m pip check
    echo "pip_check=PASS"
    python -c "import cv2, diffusers, numpy, pyarrow, torch, transformers; image=numpy.zeros((2,2,3), dtype=numpy.uint8); assert cv2.resize(image, (1,1)).shape == (1,1,3); tensor=torch.ones(2, device=\"cuda\"); assert float(tensor.sum()) == 2.0; print(\"retrieval_dependency_imports=PASS\", \"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"numpy\", numpy.__version__, \"cv2\", cv2.__version__, \"transformers\", transformers.__version__, \"diffusers\", diffusers.__version__, \"pyarrow\", pyarrow.__version__)"
    python "${SCRIPT}" --help >/dev/null
    read -r -a seeds <<< "${SAMPLING_SEEDS}"
    read -r -a panels <<< "${PANELS}"
    extra=()
    (( MAX_SAMPLES > 0 )) && extra+=(--max-samples "${MAX_SAMPLES}")
    [[ "${VERIFY_CURRENT_GETITEM}" == 1 ]] && extra+=(--verify-current-getitem)
    exec python "${SCRIPT}" \
        --root-dir "${ROOT_DIR}" \
        --feature-root "${FEATURE_ROOT}" \
        --checkpoint "${CHECKPOINT}" \
        --output "${OUTPUT}" \
        --lingbot-repo "${LINGBOT_REPO}" \
        --lingbot-weights "${LINGBOT_WEIGHTS}" \
        --window 32 --num-scale 8 --max-frame-num 4096 \
        --batch-size 8 --sampling-seeds "${seeds[@]}" \
        --panels "${panels[@]}" --data-split val \
        --validation-fraction 0.1 --split-seed 0 \
        --raw-temperature 0.01 --strict-feature-coverage \
        --require-versioned-cache \
        --expected-cache-signature "${EXPECTED_CACHE_SIGNATURE}" \
        --save-records "${extra[@]}"
'

test -s "${OUTPUT}"
sha256sum "${OUTPUT}"
echo "completed=$(date --iso-8601=seconds)"
