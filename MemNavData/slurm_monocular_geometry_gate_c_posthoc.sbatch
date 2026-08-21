#!/usr/bin/env bash
# Read-only post-hoc closure for a completed MDTEC Gate C formal run.
# Submission must use --dependency=afterok:<formal-job-id>.
# It independently verifies frozen Gate C arithmetic and adds diagnostics; it
# never trains, selects a checkpoint, runs Habitat, or changes authorization.
#SBATCH --job-name=mdtec_c_audit
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable audit source bundle}"
: "${SOURCE_RECEIPT_SHA:?set SOURCE_BUNDLE.sha256 file hash}"
: "${FORMAL_ROOT:?set completed formal Gate C output root}"
: "${FORMAL_JOB_ID:?set formal Gate C Slurm job id}"

readonly BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
readonly NAVDP_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
readonly NAVDP_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ -f "${SOURCE_ROOT}/SOURCE_BUNDLE.sha256" ]] || fail "source receipt missing"
[[ "$(sha256sum "${SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
    "${SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 --quiet) || \
  fail "audit source bundle changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == "${NAVDP_SHA}" ]] || \
  fail "NavDP checkpoint changed"
[[ -f "${FORMAL_ROOT}/OUTPUTS.sha256" ]] || fail "formal outputs are incomplete"
(cd "${FORMAL_ROOT}" && sha256sum -c OUTPUTS.sha256 --quiet) || \
  fail "formal outputs changed"
[[ ! -e "${FORMAL_ROOT}/independent_gate_c_verification.json" ]] || \
  fail "independent verifier output already exists"
[[ ! -e "${FORMAL_ROOT}/gate_c/stratified_audit.json" ]] || \
  fail "stratified audit output already exists"

readonly JOB_TMP_ROOT=/tmp/${USER}/mdtec_audit_${SLURM_JOB_ID}
mkdir -p "${JOB_TMP_ROOT}"/tmp "${JOB_TMP_ROOT}"/xdg

srun --export=ALL --ntasks=1 --ntasks-per-node=1 \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" --kill-on-bad-exit=1 \
  apptainer exec --nv \
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

      python MemNavData/independent_verify_monocular_geometry_gate_c.py \
        --receipt "'"${FORMAL_ROOT}"'/gate_c/gate_c_receipt.json" \
        --manifest "'"${FORMAL_ROOT}"'/shards/manifest.json" \
        --output "'"${FORMAL_ROOT}"'/independent_gate_c_verification.json"

      python MemNavData/audit_monocular_geometry_gate_c_strata.py \
        --shard-root "'"${FORMAL_ROOT}"'/shards" \
        --receipt "'"${FORMAL_ROOT}"'/gate_c/gate_c_receipt.json" \
        --adapter-checkpoint "'"${FORMAL_ROOT}"'/gate_c/geometry_adapter_gate_c.pt" \
        --navdp-checkpoint "'"${NAVDP_CKPT}"'" \
        --output "'"${FORMAL_ROOT}"'/gate_c/stratified_audit.json" \
        --batch-size 8
    '

python - "${FORMAL_ROOT}" "${FORMAL_JOB_ID}" "${SLURM_JOB_ID}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
verification = json.loads((root / "independent_gate_c_verification.json").read_text())
audit = json.loads((root / "gate_c/stratified_audit.json").read_text())
summary = {
    "schema": "monocular_geometry_gate_c_posthoc_receipt_v1_20260818",
    "status": "complete",
    "formal_slurm_job_id": sys.argv[2],
    "audit_slurm_job_id": sys.argv[3],
    "independent_verified": verification["verified"],
    "authorized": verification["authorized"],
    "choice": verification.get("choice"),
    "aggregate_reproduction_all_within_tolerance": audit[
        "aggregate_reproduction_all_within_tolerance"
    ],
    "posthoc_diagnostic_not_authorization": True,
    "not_closed_loop_or_sr": True,
}
(root / "POSTHOC_RECEIPT.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n"
)
files = [
    root / "independent_gate_c_verification.json",
    root / "gate_c/stratified_audit.json",
    root / "POSTHOC_RECEIPT.json",
]
with (root / "POSTHOC_OUTPUTS.sha256").open("w") as handle:
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f"{digest}  {path.relative_to(root)}\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
