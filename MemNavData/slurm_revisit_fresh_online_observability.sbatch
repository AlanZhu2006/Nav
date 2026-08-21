#!/usr/bin/env bash
# Read-only supplemental audit: render actual online-A traces, then stratify the
# already frozen fresh160 outcomes.  This does not rerun a navigation policy.
#SBATCH --job-name=revfresh_obs
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --partition=l40s_public,h100_tandon,h200_public,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable supplemental source root}"
: "${SOURCE_RECEIPT:?set SOURCE_BUNDLE.sha256 path}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA256}"
: "${EXPECTED_LAUNCHER_SHA:?set launcher SHA256}"
: "${ORIGINAL_RUN_ROOT:?set frozen fresh160 run root}"
: "${AUDIT_ROOT:?set unique supplemental output root}"
RUN_CONTRACT=${RUN_CONTRACT:-fresh_confirmation}

BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
AUDITOR=${SOURCE_ROOT}/MemNavData/audit_revisit_fresh_online_observability.py
MANIFEST=${ORIGINAL_RUN_ROOT}/data_manifest.json
OUT=${AUDIT_ROOT}/online_observability_audit.json
EXPECTED_MANIFEST_SHA=8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${RUN_CONTRACT}" =~ ^(fresh_confirmation|certified_relocalization)$ ]] || \
  fail "invalid RUN_CONTRACT=${RUN_CONTRACT}"
for command_name in singularity sha256sum nvidia-smi; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
for path in "$0" "${BASE_SIF}" "${AUDITOR}" \
    "${SOURCE_RECEIPT}" "${MANIFEST}" "${MANIFEST}.sha256" \
    "${ORIGINAL_RUN_ROOT}/report.json"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/symlink input: ${path}"
done
resolved_hab_py=$(readlink -f "${HAB_PY}") || \
  fail "cannot resolve Habitat interpreter: ${HAB_PY}"
[[ -x "${HAB_PY}" && -f "${resolved_hab_py}" && ! -L "${resolved_hab_py}" ]] || \
  fail "missing/broken Habitat interpreter: ${HAB_PY}"
[[ "$(sha256sum "$0" | awk '{print $1}')" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "spooled launcher SHA mismatch"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt SHA mismatch"
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256) >/dev/null || \
  fail "supplemental source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "frozen fresh160 manifest changed"
(cd "$(dirname "${MANIFEST}")" &&
  sha256sum -c "$(basename "${MANIFEST}").sha256") >/dev/null || \
  fail "manifest sidecar failed"
[[ ! -e "${AUDIT_ROOT}" ]] || fail "audit output already exists: ${AUDIT_ROOT}"

trace_count=$(find "${ORIGINAL_RUN_ROOT}/scenes" -type f \
  -path '*/trace_source/episode_*_leg1_trace.json' | wc -l)
[[ "${trace_count}" -eq 160 ]] || fail "expected 160 shared-A traces, got ${trace_count}"
arms=(geometry_router known_revisit_direct native)
if [[ "${RUN_CONTRACT}" == certified_relocalization ]]; then
  arms=(certified_relocalization known_revisit_direct geometry_router native)
fi
for arm in "${arms[@]}"; do
  metric_count=$(find "${ORIGINAL_RUN_ROOT}/scenes" -type f \
    -path "*/${arm}/metric.csv" | wc -l)
  [[ "${metric_count}" -eq 20 ]] || \
    fail "expected 20 ${arm} metric files, got ${metric_count}"
done

mkdir -p "${AUDIT_ROOT}" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp "${SOURCE_RECEIPT}" "${AUDIT_ROOT}/source_bundle.sha256"
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/revfresh_obs_pycache_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"

echo "host=$(hostname) job=${SLURM_JOB_ID} run_contract=${RUN_CONTRACT} audit_root=${AUDIT_ROOT}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" \
  env PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -m py_compile "${AUDITOR}"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" \
  env PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -u "${AUDITOR}" \
    --manifest "${MANIFEST}" \
    --run-root "${ORIGINAL_RUN_ROOT}" \
    --run-contract "${RUN_CONTRACT}" \
    --bootstrap-seed 20260813 \
    --bootstrap-resamples 100000 \
    --require-goal-render-hash \
    --out "${OUT}"

"${HAB_PY}" - "${OUT}" "${RUN_CONTRACT}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
expected = {
    "fresh_confirmation": "revisit_fresh_online_a_observability_v1_20260813",
    "certified_relocalization":
        "certified_relocalization_online_a_observability_v1_20260813",
}[sys.argv[2]]
if report.get("audit_id") != expected:
    raise SystemExit("wrong audit schema")
if report.get("summary", {}).get("episodes") != 160:
    raise SystemExit("audit did not contain 160 episodes")
if set(report.get("stratified_outcomes", {})) != {
    "shared_a_success_all",
    "actual_online_supported_ge_0p20_given_a",
    "actual_online_strong_ge_0p50_given_a",
    "actual_online_unsupported_lt_0p20_given_a_diagnostic",
}:
    raise SystemExit("stratified outcome populations are incomplete")
print(json.dumps({"postflight": "ok", "summary": report["summary"]},
                 sort_keys=True))
PY
sha256sum "${OUT}" > "${OUT}.sha256"
chmod a-w "${OUT}" "${OUT}.sha256" "${AUDIT_ROOT}/source_bundle.sha256"
echo "COMPLETE ${OUT}"
