#!/usr/bin/env bash
# Submit generation -> identity manifest -> paired eval -> read-only summary.
# Run on the HPC login node against an immutable, hash-receipted source root.

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable SOURCE_ROOT}"
: "${RUN_TAG:?set unique RUN_TAG}"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG" >&2; exit 2; }

RESULT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_fresh_confirmation_20260811
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}
SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
GEN=${SOURCE_ROOT}/MemNavData/slurm_revisit_fresh_generate.sbatch
MANIFEST=${SOURCE_ROOT}/MemNavData/slurm_revisit_fresh_manifest.sbatch
EVAL=${SOURCE_ROOT}/MemNavData/slurm_revisit_fresh_eval.sbatch
SUMMARY=${SOURCE_ROOT}/MemNavData/slurm_revisit_fresh_summary.sbatch
for command_name in sbatch sha256sum python; do
  command -v "${command_name}" >/dev/null || {
    echo "ABORT: ${command_name} unavailable" >&2; exit 2; }
done
for path in "${SOURCE_RECEIPT}" "${GEN}" "${MANIFEST}" "${EVAL}" "${SUMMARY}"; do
  test -r "${path}" || { echo "ABORT: missing ${path}" >&2; exit 2; }
done
[[ ! -e "${RUN_ROOT}" ]] || { echo "ABORT: RUN_ROOT exists: ${RUN_ROOT}" >&2; exit 2; }
EXPECTED_SOURCE_RECEIPT_SHA=$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')
(cd / && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null
mkdir -p "${RUN_ROOT}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp "${SOURCE_RECEIPT}" "${RUN_ROOT}/source_bundle.sha256"
chmod a-w "${RUN_ROOT}/source_bundle.sha256"
exports="ALL,SOURCE_ROOT=${SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA}"

sbatch --test-only --array=0 --export="${exports}" "${GEN}" >/dev/null
generation_job=$(sbatch --parsable --array=0-19%4 --export="${exports}" "${GEN}")
[[ "${generation_job}" =~ ^[0-9]+([;].*)?$ ]] || { echo "bad generation job" >&2; exit 2; }
generation_id=${generation_job%%;*}

sbatch --test-only --dependency="afterok:${generation_id}" --kill-on-invalid-dep=yes \
  --export="${exports}" "${MANIFEST}" >/dev/null
manifest_job=$(sbatch --parsable --dependency="afterok:${generation_id}" \
  --kill-on-invalid-dep=yes --export="${exports}" "${MANIFEST}")
manifest_id=${manifest_job%%;*}

sbatch --test-only --array=0 --dependency="afterok:${manifest_id}" \
  --kill-on-invalid-dep=yes --export="${exports}" "${EVAL}" >/dev/null
evaluation_job=$(sbatch --parsable --array=0-19%4 \
  --dependency="afterok:${manifest_id}" --kill-on-invalid-dep=yes \
  --export="${exports}" "${EVAL}")
evaluation_id=${evaluation_job%%;*}

sbatch --test-only --dependency="afterok:${evaluation_id}" --kill-on-invalid-dep=yes \
  --export="${exports}" "${SUMMARY}" >/dev/null
summary_job=$(sbatch --parsable --dependency="afterok:${evaluation_id}" \
  --kill-on-invalid-dep=yes --export="${exports}" "${SUMMARY}")
summary_id=${summary_job%%;*}

python - "${RUN_ROOT}/submission.json" "${RUN_TAG}" \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" "${generation_id}" "${manifest_id}" \
  "${evaluation_id}" "${summary_id}" <<'PY'
import json, sys
path, tag, source_sha, generation, manifest, evaluation, summary = sys.argv[1:]
with open(path, "x", encoding="utf-8") as handle:
    json.dump({
        "run_tag": tag,
        "source_receipt_sha256": source_sha,
        "jobs": {
            "generation_array": int(generation),
            "manifest": int(manifest),
            "evaluation_array": int(evaluation),
            "summary": int(summary),
        },
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "generation=${generation_id} manifest=${manifest_id} evaluation=${evaluation_id} summary=${summary_id}"
