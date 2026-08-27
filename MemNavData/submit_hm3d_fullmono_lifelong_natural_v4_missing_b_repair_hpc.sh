#!/usr/bin/env bash
# Submit the identity-only 4/99 factual-B repair, then resume the frozen DAG.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_PY=/scratch/lg154/conda-envs/memnav/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_v4_parserfix_14316838b2bec0c9
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=14316838b2bec0c9e2c4714ffc8aae247650aa3c796a75f1ace288e86b1b9d60
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
ORIGINAL_V4_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_v4_d85fc50df19b1384
PROTOCOL=${ORIGINAL_V4_ROOT}/MemNavData/hm3d_fullmono_lifelong_direct_natural_protocol_20260827.json
EXPECTED_PROTOCOL_SHA=2bfc62c08cbee1dffd3c5a3f627b1cb58a7c5076a9c9b2e2554e28843564492f
AB_MANIFEST=${RUN_ROOT}/ab_population/role_pairs/manifest.json
EXPECTED_AB_MANIFEST_SHA=e5c97dad42b26c67032c5b42a8467d857ce57f4b627a3f4851616a159ecef978
SHARD_MANIFEST=${RUN_ROOT}/factual_b_schedule/shards.json
EXPECTED_SHARD_MANIFEST_SHA=5b89096c613893a3963d34079b382140d1a8cd4e1fb648968da65f93f6eafbef
REPAIR_TAG=missingfix1
REPAIR_ARRAY=31,37%1
SUPERSEDED_FACTUAL_B_JOB=16472222
BLOCKING_GPU_ARRAY_JOB=${BLOCKING_GPU_ARRAY_JOB:-16482393}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() {
  tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}

[[ -x "${LOCAL_PY}" ]] || fail "local MemNav Python missing"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"
[[ "${REPAIR_ARRAY}" == 31,37%1 ]] || fail "repair population changed"
[[ "${REPAIR_TAG}" == missingfix1 ]] || fail "repair tag changed"

files=(
  MemNavData/repair_hm3d_fullmono_lifelong_natural_v4_factual_b.py
  MemNavData/test_repair_hm3d_fullmono_lifelong_natural_v4_factual_b.py
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_missing_b_repair.sbatch
  MemNavData/submit_hm3d_fullmono_lifelong_natural_v4_missing_b_repair_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}/MemNavData" \
  "${LOCAL_PY}" -m unittest -q \
  test_repair_hm3d_fullmono_lifelong_natural_v4_factual_b
bash -n \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_missing_b_repair.sbatch \
  MemNavData/submit_hm3d_fullmono_lifelong_natural_v4_missing_b_repair_hpc.sh

scratch=$(mktemp -d /tmp/h3life_v4_missing_b.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do
  install -m 0644 "${path}" "${scratch}/root/MemNavData/$(basename "${path}")"
done
"${LOCAL_PY}" - "${scratch}/root/source_bundle_manifest.json" <<'PY'
import json, sys
payload = {
    "schema_version": "hm3d_fullmono_lifelong_natural_v4_missing_b_repair_bundle_v1_20260828",
    "repair_shards": [31, 37],
    "repair_history_indices": [51, 52, 62, 63],
    "expected_completed_before_repair": 95,
    "superseded_factual_B_job": 16472222,
    "selection_reads_navigation_outcomes": False,
    "protocol_or_threshold_changed": False,
    "collector_or_controller_changed": False,
    "partial_outputs_preserved": True,
}
open(sys.argv[1], "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
repair_receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
repair_root=${REMOTE_BUNDLES}/hm3d_lifelong_natural_v4_missing_b_${repair_receipt_sha:0:16}
repair_stage=${repair_root}.partial.$$

remote "set -euo pipefail; \
  test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'; \
  cd '${TASK_ROOT}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256; \
  test \"\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}'; \
  test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'; \
  test \"\$(sha256sum '${PROTOCOL}' | awk '{print \$1}')\" = '${EXPECTED_PROTOCOL_SHA}'; \
  test \"\$(sha256sum '${AB_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_AB_MANIFEST_SHA}'; \
  test \"\$(sha256sum '${SHARD_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_SHARD_MANIFEST_SHA}'; \
  test -z \"\$(squeue -h -j '${SUPERSEDED_FACTUAL_B_JOB}' -o '%i')\"; \
  test ! -e '${RUN_ROOT}/prefix_fragments'; \
  test ! -e '${RUN_ROOT}/population'; \
  test ! -e '${RUN_ROOT}/deferred_submission/natural_v4_prefix.json'; \
  test ! -e '${RUN_ROOT}/failed_attempts/factual_b_${REPAIR_TAG}_shard031'; \
  test ! -e '${RUN_ROOT}/failed_attempts/factual_b_${REPAIR_TAG}_shard037'; \
  test ! -e '${RUN_ROOT}/runtime/lifelong_b_28_${REPAIR_TAG}_shard031'; \
  test ! -e '${RUN_ROOT}/runtime/lifelong_b_32_${REPAIR_TAG}_shard037'"

if remote "test -d '${repair_root}'"; then
  remote "test \"\$(sha256sum '${repair_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${repair_receipt_sha}' && cd '${repair_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${repair_stage}' && mkdir -p '${repair_stage}'"
  timeout 240 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${repair_stage}/"
  remote "cd '${repair_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${repair_stage}' && mv '${repair_stage}' '${repair_root}'"
fi

repair_receipt=${repair_root}/SOURCE_BUNDLE.sha256
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${repair_root}/MemNavData' '${REMOTE_PY}' -m unittest -q test_repair_hm3d_fullmono_lifelong_natural_v4_factual_b"
remote "'${REMOTE_PY}' -u '${repair_root}/MemNavData/repair_hm3d_fullmono_lifelong_natural_v4_factual_b.py' audit --run-root '${RUN_ROOT}' --manifest '${AB_MANIFEST}' --schedule '${SHARD_MANIFEST}' --expected-manifest-sha256 '${EXPECTED_AB_MANIFEST_SHA}' --expected-schedule-sha256 '${EXPECTED_SHARD_MANIFEST_SHA}'"

common="ALL,REPAIR_ROOT=${repair_root},REPAIR_RECEIPT=${repair_receipt},EXPECTED_REPAIR_RECEIPT_SHA=${repair_receipt_sha},REPAIR_TAG=${REPAIR_TAG},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},AB_MANIFEST=${AB_MANIFEST},EXPECTED_AB_MANIFEST_SHA=${EXPECTED_AB_MANIFEST_SHA},SHARD_MANIFEST=${SHARD_MANIFEST},EXPECTED_SHARD_MANIFEST_SHA=${EXPECTED_SHARD_MANIFEST_SHA}"
repair_script=${repair_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_missing_b_repair.sbatch
remote "source '${TASK_ROOT}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=31 --export='${common}' '${repair_script}' >/dev/null"
repair_job=$(remote "source '${TASK_ROOT}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --array='${REPAIR_ARRAY}' --export='${common}' '${repair_script}'" | job_id)
[[ "${repair_job}" =~ ^[0-9]+$ ]] || fail "bad factual-B repair job"

deferred=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_deferred_prefix.sbatch
deferred_common="ALL,TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},VERIFY_ROOT=${TASK_ROOT},VERIFY_RECEIPT=${TASK_RECEIPT},EXPECTED_VERIFY_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},EXPECTED_PROTOCOL_SHA=${EXPECTED_PROTOCOL_SHA},SHARD_MANIFEST=${SHARD_MANIFEST},EXPECTED_SHARD_MANIFEST_SHA=${EXPECTED_SHARD_MANIFEST_SHA},FACTUAL_B_ARRAY_JOB=${repair_job},PREFIX_CONCURRENCY=4"
dependency=afterok:${repair_job},afterany:${BLOCKING_GPU_ARRAY_JOB}
remote "source '${TASK_ROOT}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --dependency='${dependency}' --export='${deferred_common}' '${deferred}' >/dev/null"
deferred_job=$(remote "source '${TASK_ROOT}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency='${dependency}' --export='${deferred_common}' '${deferred}'" | job_id)
[[ "${deferred_job}" =~ ^[0-9]+$ ]] || fail "bad deferred prefix launcher job"

receipt=MemNavData/HM3D_FULLMONO_LIFELONG_NATURAL_V4_MISSING_B_REPAIR_SUBMISSION_20260828.json
[[ ! -e "${receipt}" ]] || fail "local submission receipt exists"
"${LOCAL_PY}" - "${receipt}" "${repair_root}" "${repair_receipt_sha}" \
  "${repair_job}" "${deferred_job}" "${dependency}" <<'PY'
import json, sys
path, bundle, digest, repair, deferred, dependency = sys.argv[1:]
payload = {
    "schema_version": "hm3d_fullmono_lifelong_natural_v4_missing_b_repair_submission_v1_20260828",
    "run_root": "/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d",
    "repair_bundle": bundle,
    "repair_bundle_receipt_sha256": digest,
    "superseded_factual_B_job": 16472222,
    "completed_before_repair": 95,
    "repair_shards": [31, 37],
    "repair_history_indices": [51, 52, 62, 63],
    "repair_array": "31,37%1",
    "repair_job": int(repair),
    "deferred_prefix_launcher_job": int(deferred),
    "deferred_dependency": dependency,
    "blocking_gpu_array_job": 16482393,
    "selection_reads_navigation_outcomes": False,
    "protocol_or_threshold_changed": False,
    "collector_or_controller_changed": False,
    "completed_output_overwrite_allowed": False,
    "partial_outputs_preserved": True,
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
timeout 120 scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" "${receipt}" \
  "${SSH_ALIAS}:${RUN_ROOT}/factual_b_missing_repair_submission.json"
remote "sha256sum '${RUN_ROOT}/factual_b_missing_repair_submission.json' >'${RUN_ROOT}/factual_b_missing_repair_submission.json.sha256'; chmod a-w '${RUN_ROOT}/factual_b_missing_repair_submission.json' '${RUN_ROOT}/factual_b_missing_repair_submission.json.sha256'; squeue -j '${repair_job},${deferred_job},${BLOCKING_GPU_ARRAY_JOB}' -o '%.18i %.24j %.2t %.10M %.40R'"
printf 'REPAIR_JOB=%s\nDEFERRED_PREFIX=%s\nREPAIR_BUNDLE=%s\n' \
  "${repair_job}" "${deferred_job}" "${repair_root}"
