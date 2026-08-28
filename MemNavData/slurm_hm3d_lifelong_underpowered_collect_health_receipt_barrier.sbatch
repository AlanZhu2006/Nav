#!/usr/bin/env bash
#SBATCH --job-name=h3CHealthGate
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${AUDIT_ROOT:?}" "${AUDIT_RECEIPT:?}" "${EXPECTED_AUDIT_RECEIPT_SHA:?}"
: "${PROTOCOL:?}" "${BASE_PROTOCOL:?}" "${RUN_ROOT:?}"
: "${SOURCE_REPAIR_ROOT:?}" "${OUTPUT_ROOT:?}" "${GATE_MODE:?smoke|ready}"
case "${GATE_MODE}" in smoke|ready) ;; *) echo "bad GATE_MODE" >&2; exit 2;; esac

[[ "$(sha256sum "${AUDIT_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AUDIT_RECEIPT_SHA}" ]]
(cd "${AUDIT_ROOT}" && sha256sum -c --quiet "${AUDIT_RECEIPT}")
[[ -f "${SOURCE_REPAIR_ROOT}/post_archive_audit.json" \
   && -f "${SOURCE_REPAIR_ROOT}/post_archive_audit.json.sha256" ]]
(cd "${SOURCE_REPAIR_ROOT}" && \
  sha256sum -c --quiet post_archive_audit.json.sha256)

mkdir -p "${OUTPUT_ROOT}"
if [[ "${GATE_MODE}" == smoke ]]; then
  phase=smoke_ready
  output=${OUTPUT_ROOT}/smoke_ready_audit.json
else
  phase=ready_to_seal
  output=${OUTPUT_ROOT}/ready_to_seal_audit.json
  [[ -f "${OUTPUT_ROOT}/smoke_ready_audit.json" \
     && -f "${OUTPUT_ROOT}/smoke_ready_audit.json.sha256" ]]
  (cd "${OUTPUT_ROOT}" && sha256sum -c --quiet smoke_ready_audit.json.sha256)
fi
[[ ! -e "${output}" && ! -e "${output}.sha256" ]]
/scratch/lg154/conda-envs/memnav/bin/python \
  "${AUDIT_ROOT}/MemNavData/audit_hm3d_lifelong_underpowered_collect_repair_attempt2.py" \
  --protocol "${PROTOCOL}" --base-protocol "${BASE_PROTOCOL}" \
  --run-root "${RUN_ROOT}" --phase "${phase}" --out "${output}" >/dev/null
sha256sum "${output}" >"${output}.sha256"
printf 'HEALTH_RECEIPT_GATE=%s\nAUDIT=%s\n' "${GATE_MODE}" "${output}"
