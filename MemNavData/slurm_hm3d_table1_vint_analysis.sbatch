#!/usr/bin/env bash
#SBATCH --job-name=h3T1ViNAna
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
DATASET=${DATASET:-HM3D}
CLAIM_SCOPE=${CLAIM_SCOPE:-conference_table_hm3d_fresh_query_scene_overlap}
PY=/scratch/lg154/conda-envs/memnav/bin/python
RUN_ROOT=${FORMAL_RUN_ROOT}/formal/vint
SUMMARY=${RUN_ROOT}/vint_table1_summary.json
VERIFICATION=${RUN_ROOT}/vint_table1_independent_verification.json
MANIFEST=${BENCH_ROOT}/manifest.json
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${DATASET}" == HM3D || "${DATASET}" == MP3D ]] || \
  fail "unsupported Table-1 dataset"

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"
[[ "$(sha256sum "${CONSTRUCTION_VERIFICATION}" | awk '{print $1}')" == \
   "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}" ]] || \
  fail "construction verification changed"
readarray -t counts < <("${PY}" - "${CONSTRUCTION_VERIFICATION}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
if p.get("verified") is not True or p.get("formal_policy_evaluation_authorized") is not True:
 raise SystemExit("construction gate did not pass")
print(p["histories"]); print(p["scene_clusters"])
PY
)
[[ "${#counts[@]}" -eq 2 ]] || fail "construction counts missing"
export PYTHONPATH=${TASK_ROOT}:${TASK_ROOT}/MemNavData

case "${MODE}" in
  aggregate)
    [[ ! -e "${SUMMARY}" ]] || fail "summary already exists"
    "${PY}" -u \
      "${TASK_ROOT}/MemNavData/aggregate_vint_controller_native_hm3d.py" \
      --run-root "${RUN_ROOT}" --benchmark-manifest "${MANIFEST}" \
      --expected-histories "${counts[0]}" --expected-scenes "${counts[1]}" \
      --claim-scope "${CLAIM_SCOPE}" --dataset "${DATASET}" \
      --expected-grant-alignment first_certified_bounded \
      --out "${SUMMARY}"
    sha256sum "${SUMMARY}" >"${SUMMARY}.sha256"
    ;;
  verify)
    [[ -f "${SUMMARY}" && ! -e "${VERIFICATION}" ]] || \
      fail "verification precondition failed"
    "${PY}" -u \
      "${TASK_ROOT}/MemNavData/independent_verify_vint_controller_native_hm3d.py" \
      --run-root "${RUN_ROOT}" --summary "${SUMMARY}" \
      --benchmark-manifest "${MANIFEST}" --dataset "${DATASET}" \
      --out "${VERIFICATION}"
    sha256sum "${VERIFICATION}" >"${VERIFICATION}.sha256"
    "${PY}" - "${VERIFICATION}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1]))
if payload.get("verified") is not True:
 raise SystemExit("ViNT independent verification failed")
PY
    printf '%s\n' verified=true >"${RUN_ROOT}/SEALED"
    chmod -R a-w "${RUN_ROOT}"
    ;;
  *) fail "invalid MODE=${MODE}" ;;
esac
echo "COMPLETE mode=${MODE}"
