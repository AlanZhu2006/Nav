#!/usr/bin/env bash
#SBATCH --job-name=h3OracleAna
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
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}"
MANIFEST=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table3_causal_survey_expansion_20260831/expansion_20260830T165729Z_a8dd232e/merged_query_population/role_pairs/manifest.json
MANIFEST_SHA=cbc518cea991fd252893f97fd5e730c277e4d899369932536a745351d47e7451
PY=/scratch/lg154/conda-envs/memnav/bin/python
SUMMARY=${RUN_ROOT}/summary/longrange_oracle_attribution_result.json
VERIFICATION=${RUN_ROOT}/summary/longrange_oracle_attribution_independent_verification.json
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${MANIFEST_SHA}" ]] || \
  fail "benchmark manifest changed"
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData
mkdir -p "${RUN_ROOT}/summary"

case "${MODE}" in
  aggregate)
    [[ ! -e "${SUMMARY}" ]] || fail "summary already exists"
    "${PY}" -u \
      "${SOURCE_ROOT}/MemNavData/analyze_hm3d_longrange_oracle_attribution.py" \
      --run-root "${RUN_ROOT}" --manifest "${MANIFEST}" \
      --expected-manifest-sha256 "${MANIFEST_SHA}" --out "${SUMMARY}"
    ;;
  verify)
    [[ -f "${SUMMARY}" && ! -e "${VERIFICATION}" ]] || \
      fail "verification precondition failed"
    "${PY}" -u \
      "${SOURCE_ROOT}/MemNavData/independent_verify_hm3d_longrange_oracle_attribution.py" \
      --run-root "${RUN_ROOT}" --manifest "${MANIFEST}" \
      --expected-manifest-sha256 "${MANIFEST_SHA}" \
      --summary "${SUMMARY}" --out "${VERIFICATION}"
    "${PY}" - "${VERIFICATION}" <<'PY'
import json,sys
assert json.load(open(sys.argv[1])).get('verified') is True
PY
    printf '%s\n' verified=true >"${RUN_ROOT}/SEALED"
    chmod -R a-w "${RUN_ROOT}"
    ;;
  *) fail "invalid MODE=${MODE}" ;;
esac
echo "COMPLETE mode=${MODE}"
