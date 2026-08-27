#!/usr/bin/env bash
# Repair only the renderer-free v4 materialization seal and its verifier.
# The 54 successful GPU materialization fragments remain immutable.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
AUDIT_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_power_v3_20260826/natural_b_audit_formal_20260827T125748Z_e2832e17
ORIGINAL_V4_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_v4_d85fc50df19b1384
PROTOCOL=${ORIGINAL_V4_ROOT}/MemNavData/hm3d_fullmono_lifelong_direct_natural_protocol_20260827.json
EXPECTED_PROTOCOL_SHA=2bfc62c08cbee1dffd3c5a3f627b1cb58a7c5076a9c9b2e2554e28843564492f
AUDIT_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_b_audit_e2832e17231534e3
AUDIT_SOURCE_RECEIPT=${AUDIT_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_AUDIT_SOURCE_RECEIPT_SHA=e2832e17231534e38db1b3b507ddf68881ee8bd56548c3151235e52a397f3121
SOURCE_TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SOURCE_TASK_RECEIPT=${SOURCE_TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SOURCE_TASK_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
VERIFY_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_v4_verifier_307a4aedd3c3c73e
VERIFY_RECEIPT=${VERIFY_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_VERIFY_RECEIPT_SHA=307a4aedd3c3c73ee304ee25615be9b5e221c0f89f10fb06c9bc3b0129ce7a31
MATERIALIZE_JOB=16465105
FAILED_SEAL_JOB=16465110
CANCELED_VERIFY_JOB=16465183
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
remote_query() {
  local attempt output
  for attempt in 1 2 3; do
    if output=$(timeout 180 ssh -n -T -o BatchMode=yes \
      -o ControlMaster=no -o ServerAliveInterval=15 \
      -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"); then
      printf '%s\n' "${output}"
      return 0
    fi
    sleep 2
  done
  return 1
}
job_id() {
  tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}

[[ -x "${LOCAL_PY}" ]] || fail "local Python missing"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"

files=(
  MemNavData/finalize_hm3d_fullmono_lifelong_ab.py
  MemNavData/test_finalize_hm3d_fullmono_lifelong_ab_lightweight.py
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/submit_hm3d_fullmono_lifelong_natural_v4_seal_repair_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || \
    fail "missing physical input ${path}"
done
bash -n \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize.sbatch \
  MemNavData/slurm_safe_submit.sh \
  MemNavData/submit_hm3d_fullmono_lifelong_natural_v4_seal_repair_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize.sbatch
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}/MemNavData" \
  "${LOCAL_PY}" -m unittest -q \
  test_finalize_hm3d_fullmono_lifelong_ab_lightweight

scratch=$(mktemp -d /tmp/h3life_v4_sealfix.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do
  cp -p -- "${path}" "${scratch}/root/MemNavData/$(basename "${path}")"
done
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
repair_root=${REMOTE_BUNDLES}/hm3d_lifelong_natural_v4_sealfix_${receipt_sha:0:16}
repair_stage=${repair_root}.partial.$$

preflight=$(remote_query "set -euo pipefail; \
  echo IDENTITY=\$(id -un); \
  echo PROTOCOL=\$(sha256sum '${PROTOCOL}' | awk '{print \$1}'); \
  echo AUDIT=\$(sha256sum '${AUDIT_SOURCE_RECEIPT}' | awk '{print \$1}'); \
  echo TASK=\$(sha256sum '${SOURCE_TASK_RECEIPT}' | awk '{print \$1}'); \
  echo BASE=\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}'); \
  echo VERIFY=\$(sha256sum '${VERIFY_RECEIPT}' | awk '{print \$1}'); \
  test ! -e '${RUN_ROOT}/ab_population'; \
  test \"\$(find '${RUN_ROOT}/construct_ab/scenes' -mindepth 2 -maxdepth 2 -name completion.json -type f | wc -l)\" -eq 54; \
  test \"\$(sacct -j '${FAILED_SEAL_JOB}' -X -n -o State | awk 'NF{print \$1; exit}')\" = FAILED; \
  echo COMPLETE")
value_for() {
  local key=$1
  printf '%s\n' "${preflight}" | tr -d '\r' | \
    awk -F= -v key="${key}" '$1==key {print $2; exit}'
}
[[ "$(value_for IDENTITY)" == yz11502 ]] || fail "wrong remote identity"
[[ "$(value_for PROTOCOL)" == "${EXPECTED_PROTOCOL_SHA}" ]] || fail "protocol changed"
[[ "$(value_for AUDIT)" == "${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA}" ]] || fail "audit source changed"
[[ "$(value_for TASK)" == "${EXPECTED_SOURCE_TASK_RECEIPT_SHA}" ]] || fail "task source changed"
[[ "$(value_for BASE)" == "${EXPECTED_BASE_RECEIPT_SHA}" ]] || fail "base source changed"
[[ "$(value_for VERIFY)" == "${EXPECTED_VERIFY_RECEIPT_SHA}" ]] || fail "verifier source changed"
printf '%s\n' "${preflight}" | tr -d '\r' | grep -qx COMPLETE || fail "preflight incomplete"

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
seal_script=${repair_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize.sbatch
common="ALL,V4_ROOT=${repair_root},AUDIT_SOURCE_ROOT=${AUDIT_SOURCE_ROOT},SOURCE_TASK_ROOT=${SOURCE_TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},V4_RECEIPT=${repair_receipt},EXPECTED_V4_RECEIPT_SHA=${receipt_sha},AUDIT_SOURCE_RECEIPT=${AUDIT_SOURCE_RECEIPT},EXPECTED_AUDIT_SOURCE_RECEIPT_SHA=${EXPECTED_AUDIT_SOURCE_RECEIPT_SHA},SOURCE_TASK_RECEIPT=${SOURCE_TASK_RECEIPT},EXPECTED_SOURCE_TASK_RECEIPT_SHA=${EXPECTED_SOURCE_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA}"

# This is the exact CPU runtime that failed before.  The fresh subprocess test
# proves the seal no longer imports renderer/quaternion dependencies.
remote "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${repair_root}/MemNavData:$(dirname "${PROTOCOL}"):${AUDIT_SOURCE_ROOT}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python -m unittest -q test_finalize_hm3d_fullmono_lifelong_ab_lightweight"
remote "env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${repair_root}/MemNavData:$(dirname "${PROTOCOL}"):${AUDIT_SOURCE_ROOT}/MemNavData:${SOURCE_TASK_ROOT}:${SOURCE_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python -c \"from pathlib import Path; from hm3d_fullmono_lifelong import load_protocol; load_protocol(Path('${PROTOCOL}'))\""
remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${seal_script}' >/dev/null"
seal_id=$(remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --export='${common}' '${seal_script}'" | job_id)
[[ "${seal_id}" =~ ^[0-9]+$ ]] || fail "bad replacement seal job id"

verify_script=${VERIFY_ROOT}/MemNavData/slurm_independent_verify_hm3d_fullmono_lifelong_natural_v4_materialization.sbatch
verify_common="ALL,SOURCE_ROOT=${VERIFY_ROOT},SOURCE_RECEIPT=${VERIFY_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_VERIFY_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},AUDIT_RUN_ROOT=${AUDIT_RUN_ROOT},PROTOCOL=${PROTOCOL},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA}"
remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${verify_common}' '${verify_script}' >/dev/null"
verify_id=$(remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${seal_id} --kill-on-invalid-dep=yes --export='${verify_common}' '${verify_script}'" | job_id)
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad replacement verifier job id"

receipt=MemNavData/HM3D_FULLMONO_LIFELONG_NATURAL_V4_SEAL_REPAIR_R2_SUBMISSION_20260827.json
"${LOCAL_PY}" - "${receipt}" "${repair_root}" "${receipt_sha}" \
  "${seal_id}" "${verify_id}" <<'PY'
import json, sys
path, root, digest, seal, verify = sys.argv[1:]
payload = {
    "schema_version": "hm3d_fullmono_lifelong_natural_v4_seal_repair_r2_v1_20260827",
    "failure": {
        "materialize_job": 16465105,
        "materialize_tasks_completed": 54,
        "failed_seal_job": 16465110,
        "error": "ModuleNotFoundError: No module named 'quaternion'",
        "navigation_outcomes_read": False,
    },
    "supersedes_failed_repair": {
        "replacement_seal_job": 16469893,
        "replacement_verifier_job": 16469948,
        "error": "older source-task parser rejected frozen v4 protocol schema",
        "navigation_outcomes_read": False,
    },
    "repair": {
        "strategy": "renderer-free CPU finalizer reuses exact frozen fragment contract",
        "source_bundle": root,
        "source_bundle_receipt_sha256": digest,
        "replacement_seal_job": int(seal),
        "replacement_verifier_job": int(verify),
    },
    "gpu_materialization_rerun": False,
    "scientific_thresholds_changed": False,
    "factual_B_executed_at_submission": False,
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
timeout 120 scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" "${receipt}" \
  "${SSH_ALIAS}:${RUN_ROOT}/seal_import_repair_r2_submission.json"
remote "sha256sum '${RUN_ROOT}/seal_import_repair_r2_submission.json' >'${RUN_ROOT}/seal_import_repair_r2_submission.json.sha256'; squeue -j '${seal_id},${verify_id}' -o '%.18i %.24j %.2t %.10M %.40R'"
printf 'REPAIR_ROOT=%s\nREPLACEMENT_SEAL_JOB=%s\nREPLACEMENT_VERIFY_JOB=%s\n' \
  "${repair_root}" "${seal_id}" "${verify_id}"
