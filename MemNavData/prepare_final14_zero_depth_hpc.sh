#!/usr/bin/env bash
# Freeze and stage the Final14 zero-depth arm.  SUBMIT=1 starts the DAG.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
SUBMIT=${SUBMIT:-0}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_PY=/scratch/lg154/conda-envs/memnav/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_zero_depth_20260828
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_SOURCE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_SOURCE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
REFERENCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_mono_factorial_20260819/formal_20260819T124820Z_5690569a
BENCH_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T115533Z_attempt7_handoff/benchmarks/natural_direction
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
EXPECTED_SOURCE_OVERLAY_BYTES=128854888448
cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "authoritative shared SSH master is not responsive"
remote_user=$(timeout 20 ssh -n -T -o BatchMode=yes \
  -o ControlMaster=no -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  'id -un' 2>/dev/null || true)
[[ "${remote_user}" == "${EXPECTED_SSH_USER}" ]] || \
  fail "authoritative shared SSH identity is ${remote_user:-unavailable}, expected ${EXPECTED_SSH_USER}"
remote() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

[[ "${SUBMIT}" =~ ^[01]$ ]] || fail "SUBMIT must be 0 or 1"
[[ -x "${LOCAL_PY}" ]] || fail "local Python missing"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "shared SSH socket missing"

files=(
  MemNavData/final14_zero_depth.py
  MemNavData/test_final14_zero_depth.py
  MemNavData/run_final14_zero_depth_episode.py
  MemNavData/run_final14_zero_depth_history.sh
  MemNavData/slurm_port_pair.sh
  MemNavData/test_slurm_port_pair.sh
  MemNavData/summarize_final14_zero_depth.py
  MemNavData/independent_verify_final14_zero_depth.py
  MemNavData/slurm_final14_zero_depth.sbatch
  MemNavData/slurm_final14_zero_depth_analysis.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/prepare_final14_zero_depth_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}" "${LOCAL_PY}" \
  -m unittest -q MemNavData.test_final14_zero_depth
bash MemNavData/test_slurm_port_pair.sh
"${LOCAL_PY}" -m py_compile \
  MemNavData/final14_zero_depth.py \
  MemNavData/run_final14_zero_depth_episode.py \
  MemNavData/summarize_final14_zero_depth.py \
  MemNavData/independent_verify_final14_zero_depth.py
bash -n \
  MemNavData/run_final14_zero_depth_history.sh \
  MemNavData/slurm_port_pair.sh \
  MemNavData/test_slurm_port_pair.sh \
  MemNavData/slurm_final14_zero_depth.sbatch \
  MemNavData/slurm_final14_zero_depth_analysis.sbatch \
  MemNavData/prepare_final14_zero_depth_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template MemNavData/slurm_final14_zero_depth.sbatch
  lint_sbatch_template MemNavData/slurm_final14_zero_depth_analysis.sbatch
)

scratch=$(mktemp -d /tmp/f14_zero_prepare.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do
  install -m 0644 "${path}" "${scratch}/root/MemNavData/$(basename "${path}")"
done
"${LOCAL_PY}" - "${scratch}/root/source_bundle_manifest.json" <<'PY'
import json, sys
payload = {
    "schema_version": "final14_zero_depth_bundle_v1_20260828",
    "scope": "consumed Final14 query-leg depth attribution",
    "history_count": 21,
    "query_count": 42,
    "arm": "zero_native",
    "depth_source": "explicit_zero",
    "controller_checkpoint_changed": False,
    "population_seed_or_budget_changed": False,
    "reference_factorial_reused": True,
    "runtime_role_visibility": "none",
}
open(sys.argv[1], "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
repair_root=${REMOTE_BUNDLES}/final14_zero_depth_${receipt_sha:0:16}
repair_stage=${repair_root}.partial.$$
run_root=${REMOTE_RESULTS}/formal_${receipt_sha:0:16}
smoke_root=${REMOTE_RESULTS}/smoke_${receipt_sha:0:16}

remote "set -euo pipefail; \
  test \"\$(sha256sum '${BASE_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}'; \
  cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet source_inputs.sha256; \
  test \"\$(sha256sum '${BENCH_ROOT}/manifest.json' | awk '{print \$1}')\" = 7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a; \
  test \"\$(sha256sum '${REFERENCE_ROOT}/POSTHOC/final14_mono_factorial_summary.json' | awk '{print \$1}')\" = ae24138b3cd7fecb737dffe8454eb91aaae8f5aa19d4e679e67e039d793e17b7; \
  test \"\$(sha256sum '${REFERENCE_ROOT}/POSTHOC/final14_mono_factorial_independent_verification.json' | awk '{print \$1}')\" = 7bf7e496c1a9cc53f3dc9ef0ff0194cce61f03d554a84678210320f699fbb35f; \
  test \"\$(stat -c '%s' '${SOURCE_OVERLAY}')\" = '${EXPECTED_SOURCE_OVERLAY_BYTES}'"

if remote "test -d '${repair_root}'"; then
  remote "test \"\$(sha256sum '${repair_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${receipt_sha}' && cd '${repair_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${repair_stage}' && mkdir -p '${repair_stage}'"
  timeout 240 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${repair_stage}/"
  remote "cd '${repair_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${repair_stage}' && mv '${repair_stage}' '${repair_root}'"
fi
repair_receipt=${repair_root}/SOURCE_BUNDLE.sha256
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${repair_root}:${BASE_SOURCE_ROOT}' '${REMOTE_PY}' -m unittest -q MemNavData.test_final14_zero_depth"
remote "ROOT='${repair_root}' bash '${repair_root}/MemNavData/test_slurm_port_pair.sh'"

common="ALL,REPAIR_ROOT=${repair_root},REPAIR_RECEIPT=${repair_receipt},EXPECTED_REPAIR_RECEIPT_SHA=${receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT=${BASE_SOURCE_RECEIPT},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA},REFERENCE_ROOT=${REFERENCE_ROOT},BENCH_ROOT=${BENCH_ROOT},SOURCE_OVERLAY=${SOURCE_OVERLAY},EXPECTED_SOURCE_OVERLAY_BYTES=${EXPECTED_SOURCE_OVERLAY_BYTES}"
gpu_script=${repair_root}/MemNavData/slurm_final14_zero_depth.sbatch
analysis_script=${repair_root}/MemNavData/slurm_final14_zero_depth_analysis.sbatch
remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0 --export='${common},RUN_ROOT=${smoke_root},SMOKE=1,MAX_STEPS=80' '${gpu_script}' >/dev/null"
remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common},RUN_ROOT=${run_root}' '${analysis_script}' >/dev/null"

prepare_receipt=MemNavData/FINAL14_ZERO_DEPTH_PREPARATION_RECEIPT_20260828.json
if [[ ! -e "${prepare_receipt}" ]]; then
  "${LOCAL_PY}" - "${prepare_receipt}" "${repair_root}" "${receipt_sha}" \
    "${run_root}" "${smoke_root}" <<'PY'
import json, sys
path, bundle, digest, run, smoke = sys.argv[1:]
payload = {
    "schema_version": "final14_zero_depth_preparation_v1_20260828",
    "source_bundle": bundle,
    "source_bundle_receipt_sha256": digest,
    "run_root": run,
    "smoke_root": smoke,
    "history_count": 21,
    "query_count": 42,
    "arm": "zero_native",
    "same_population_as_final14_factorial": True,
    "controller_checkpoint_changed": False,
    "submitted": False,
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
fi

if [[ "${SUBMIT}" == 0 ]]; then
  printf 'PREPARED_ONLY=1\nSOURCE_ROOT=%s\nRUN_ROOT=%s\n' \
    "${repair_root}" "${run_root}"
  exit 0
fi
remote "test ! -e '${run_root}' && test ! -e '${smoke_root}'"
smoke_job=$(remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --array=0 --export='${common},RUN_ROOT=${smoke_root},SMOKE=1,MAX_STEPS=80' '${gpu_script}'" | job_id)
formal_job=$(remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --dependency=afterok:${smoke_job} --kill-on-invalid-dep=yes --array=0-20%2 --export='${common},RUN_ROOT=${run_root},SMOKE=0,MAX_STEPS=600' '${gpu_script}'" | job_id)
analysis_job=$(remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${formal_job} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${run_root}' '${analysis_script}'" | job_id)
for value in "${smoke_job}" "${formal_job}" "${analysis_job}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || fail "bad submitted job id"
done
receipt=MemNavData/FINAL14_ZERO_DEPTH_SUBMISSION_RECEIPT_20260828.json
[[ ! -e "${receipt}" ]] || fail "submission receipt exists"
"${LOCAL_PY}" - "${receipt}" "${repair_root}" "${receipt_sha}" \
  "${run_root}" "${smoke_root}" "${smoke_job}" "${formal_job}" \
  "${analysis_job}" <<'PY'
import json, sys
path,bundle,digest,run,smoke,smoke_job,formal,analysis=sys.argv[1:]
payload={
 "schema_version":"final14_zero_depth_submission_v1_20260828",
 "source_bundle":bundle,"source_bundle_receipt_sha256":digest,
 "run_root":run,"smoke_root":smoke,
 "smoke_job":int(smoke_job),"formal_array_job":int(formal),
 "formal_array":"0-20%2","analysis_and_verifier_job":int(analysis),
 "history_count":21,"query_count":42,"outcomes_read_before_submission":False,
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
PY
