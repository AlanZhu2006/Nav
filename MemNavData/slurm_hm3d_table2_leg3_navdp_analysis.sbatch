#!/usr/bin/env bash
#SBATCH --job-name=h3T2L3Ana
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${MODE:?aggregate or verify}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${FORMAL_RUN_ROOT:?}" "${BENCH_ROOT:?}"
: "${CONSTRUCTION_VERIFICATION:?}"
: "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA:?}"
PY=/scratch/lg154/conda-envs/memnav/bin/python
RUN_ROOT=${FORMAL_RUN_ROOT}/formal/navdp
SUMMARY=${RUN_ROOT}/navdp_table2_leg3_summary.json
VERIFICATION=${RUN_ROOT}/navdp_table2_leg3_independent_verification.json
CLAIM_SCOPE=conference_table2_hm3d_conditional_leg3_after_actual_mono_AB
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"
[[ "$(sha256sum "${CONSTRUCTION_VERIFICATION}" | awk '{print $1}')" == \
   "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}" ]] || \
  fail "construction verification changed"
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData

case "${MODE}" in
  aggregate)
    [[ ! -e "${SUMMARY}" ]] || fail "summary already exists"
    "${PY}" -u "${TASK_ROOT}/MemNavData/aggregate_hm3d_table1_navdp_pair.py" \
      --run-root "${RUN_ROOT}" --benchmark-root "${BENCH_ROOT}" \
      --construction-verification "${CONSTRUCTION_VERIFICATION}" \
      --claim-scope "${CLAIM_SCOPE}" --dataset HM3D_TABLE2 \
      --out "${SUMMARY}"
    sha256sum "${SUMMARY}" >"${SUMMARY}.sha256"
    ;;
  verify)
    [[ -f "${SUMMARY}" && ! -e "${VERIFICATION}" ]] || \
      fail "verification precondition failed"
    "${PY}" -u \
      "${TASK_ROOT}/MemNavData/independent_verify_hm3d_table1_navdp_pair.py" \
      --run-root "${RUN_ROOT}" --benchmark-root "${BENCH_ROOT}" \
      --construction-verification "${CONSTRUCTION_VERIFICATION}" \
      --summary "${SUMMARY}" --dataset HM3D_TABLE2 \
      --out "${VERIFICATION}"
    sha256sum "${VERIFICATION}" >"${VERIFICATION}.sha256"
    "${PY}" - "${VERIFICATION}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
if payload.get("verified") is not True or payload.get("authorized") is not True:
 raise SystemExit("Table-2 independent verification failed")
if payload.get("unconditional_three_leg_joint_sr_reported") is not False:
 raise SystemExit("Table-2 reporting boundary changed")
PY
    printf '%s\n' verified=true >"${RUN_ROOT}/SEALED"
    chmod -R a-w "${RUN_ROOT}"
    ;;
  *) fail "invalid MODE=${MODE}" ;;
esac
echo "COMPLETE mode=${MODE}"
