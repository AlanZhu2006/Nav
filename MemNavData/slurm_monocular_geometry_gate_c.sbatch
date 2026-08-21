#!/usr/bin/env bash
# Frozen short-horizon RGB-only geometry distillation.  This job performs
# offline shard extraction plus scene-grouped Gate C; it never runs Habitat SR.
#SBATCH --job-name=mdtec_gatec
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable source bundle}"
: "${SOURCE_RECEIPT_SHA:?set SOURCE_BUNDLE.sha256 file hash}"
: "${OUT_ROOT:?set new output root}"
: "${SCENE_SPLIT:?set frozen scene split JSON}"
: "${TEACHER_DEPTH_AUDIT:?set frozen teacher-depth population audit JSON}"
: "${TEACHER_DEPTH_AUDIT_SHA:?set frozen teacher-depth audit SHA256}"

readonly OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
readonly OVERLAY_BYTES=128854888448
readonly BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
readonly FEATURE_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data
readonly NAVDP_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
readonly NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
readonly LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
readonly LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
readonly LINGBOT_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409

MAX_EPISODES_PER_SCENE=${MAX_EPISODES_PER_SCENE:-4}
STATES_PER_EPISODE=${STATES_PER_EPISODE:-4}
SCALE_PREFIX_FRAMES=${SCALE_PREFIX_FRAMES:-40}
MIN_SCENES=${MIN_SCENES:-40}
EPOCHS=${EPOCHS:-30}
BATCH_SIZE=${BATCH_SIZE:-4}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
SCENE_FIELD=${SCENE_FIELD:-train}
VALIDATION_FRACTION=${VALIDATION_FRACTION:-0.2}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ ! -e "${OUT_ROOT}" ]] || fail "OUT_ROOT already exists"
[[ -f "${SOURCE_ROOT}/SOURCE_BUNDLE.sha256" ]] || fail "source receipt missing"
[[ "$(sha256sum "${SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
    "${SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 --quiet) || \
  fail "source bundle changed"
[[ "$(stat -c %s "${OVERLAY}")" -eq "${OVERLAY_BYTES}" ]] || \
  fail "PT1 overlay changed"
[[ -d "${FEATURE_ROOT}" ]] || fail "feature root missing"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == "${NAVDP_SHA}" ]] || \
  fail "NavDP checkpoint changed"
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == "${LINGBOT_SHA}" ]] || \
  fail "LingBot weights changed"
[[ -f "${SCENE_SPLIT}" ]] || fail "scene split missing"
[[ -f "${TEACHER_DEPTH_AUDIT}" ]] || fail "teacher-depth audit missing"
[[ "$(sha256sum "${TEACHER_DEPTH_AUDIT}" | awk '{print $1}')" == \
    "${TEACHER_DEPTH_AUDIT_SHA}" ]] || fail "teacher-depth audit changed"

mkdir -p "${OUT_ROOT}"/shards "${OUT_ROOT}"/gate_c
if [[ -n "${RESUME_SHARDS_ROOT:-}" ]]; then
  : "${RESUME_SHARDS_RECEIPT_SHA:?set partial-shard receipt SHA256}"
  [[ -f "${RESUME_SHARDS_ROOT}/PARTIAL_SHARDS.sha256" ]] || \
    fail "partial-shard receipt missing"
  [[ "$(sha256sum "${RESUME_SHARDS_ROOT}/PARTIAL_SHARDS.sha256" | awk '{print $1}')" == \
      "${RESUME_SHARDS_RECEIPT_SHA}" ]] || fail "partial-shard receipt changed"
  (cd "${RESUME_SHARDS_ROOT}" && \
    sha256sum -c PARTIAL_SHARDS.sha256 --quiet) || fail "partial shards changed"
  cp "${RESUME_SHARDS_ROOT}"/shards/*.pt "${OUT_ROOT}"/shards/
  echo "preseeded $(find "${OUT_ROOT}/shards" -maxdepth 1 -name '*.pt' | wc -l) verified shards"
fi
readonly JOB_TMP_ROOT=/tmp/${USER}/mdtec_${SLURM_JOB_ID}
mkdir -p "${JOB_TMP_ROOT}"/tmp "${JOB_TMP_ROOT}"/xdg

srun --export=ALL --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  apptainer exec --nv \
    --overlay "${OVERLAY}:ro" \
    -B /scratch/lg154 -B /scratch/yz11502 -B "${SOURCE_ROOT}" \
    "${BASE_SIF}" \
    bash -c '
      set -euo pipefail
      source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
      conda activate /scratch/lg154/conda-envs/memnav
      export PYTHONPATH="'"${SOURCE_ROOT}"'/InternNav:${PYTHONPATH:-}"
      export PYTHONPYCACHEPREFIX="'"${JOB_TMP_ROOT}"'/pycache"
      export TMPDIR="'"${JOB_TMP_ROOT}"'/tmp"
      export XDG_CACHE_HOME="'"${JOB_TMP_ROOT}"'/xdg"
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
      cd "'"${SOURCE_ROOT}"'"

      python - <<PY
import torch
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(),
       "gpu": torch.cuda.get_device_name(0)})
PY
      test -d /mp3d_revisit_v0/vln_n1/traj_data

      python MemNavData/build_monocular_geometry_shards.py \
        --data-root /mp3d_revisit_v0/vln_n1/traj_data \
        --feature-root "'"${FEATURE_ROOT}"'" \
        --scene-split "'"${SCENE_SPLIT}"'" \
        --scene-field "'"${SCENE_FIELD}"'" \
        --max-episodes-per-scene "'"${MAX_EPISODES_PER_SCENE}"'" \
        --states-per-episode "'"${STATES_PER_EPISODE}"'" \
        --scale-prefix-frames "'"${SCALE_PREFIX_FRAMES}"'" \
        --teacher-depth-audit "'"${TEACHER_DEPTH_AUDIT}"'" \
        --min-scenes "'"${MIN_SCENES}"'" \
        --resume \
        --navdp-checkpoint "'"${NAVDP_CKPT}"'" \
        --lingbot-repo "'"${LINGBOT_REPO}"'" \
        --lingbot-weights "'"${LINGBOT_WEIGHTS}"'" \
        --output-dir "'"${OUT_ROOT}"'/shards"

      python MemNavData/train_monocular_geometry_adapter.py \
        --shard-root "'"${OUT_ROOT}"'/shards" \
        --output-dir "'"${OUT_ROOT}"'/gate_c" \
        --navdp-checkpoint "'"${NAVDP_CKPT}"'" \
        --epochs "'"${EPOCHS}"'" \
        --batch-size "'"${BATCH_SIZE}"'" \
        --eval-batch-size "'"${EVAL_BATCH_SIZE}"'" \
        --validation-fraction "'"${VALIDATION_FRACTION}"'"
    '

python - "${OUT_ROOT}" "${SOURCE_RECEIPT_SHA}" "${SLURM_JOB_ID}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
receipt = json.loads((root / "gate_c/gate_c_receipt.json").read_text())
summary = {
    "status": "complete",
    "slurm_job_id": sys.argv[3],
    "source_receipt_sha256": sys.argv[2],
    "scene_count": receipt["source_scene_count"],
    "sample_count": receipt["source_sample_count"],
    "selected_sample_count": json.loads(
        (root / "shards/manifest.json").read_text()
    )["selected_sample_count"],
    "invalid_teacher_state_count": json.loads(
        (root / "shards/manifest.json").read_text()
    )["invalid_teacher_state_count"],
    "teacher_depth_audit_sha256": json.loads(
        (root / "shards/manifest.json").read_text()
    )["teacher_depth_audit"]["sha256"],
    "scale_evidence_contract": json.loads(
        (root / "shards/manifest.json").read_text()
    )["scale_evidence_contract"],
    "scale_prefix_frames": json.loads(
        (root / "shards/manifest.json").read_text()
    )["scale_prefix_frames"],
    "gate_c": receipt["gate_c"],
    "not_closed_loop_or_sr": True,
}
(root / "RUN_RECEIPT.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
files = [root / "shards/manifest.json", root / "gate_c/gate_c_receipt.json",
         root / "gate_c/geometry_adapter_gate_c.pt", root / "RUN_RECEIPT.json"]
with (root / "OUTPUTS.sha256").open("w") as handle:
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path.relative_to(root)}\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
