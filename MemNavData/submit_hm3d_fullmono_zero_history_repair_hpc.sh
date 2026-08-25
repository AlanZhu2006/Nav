#!/usr/bin/env bash
# Submit the exact, result-blind repair for parent-certified zero-history scenes.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_20260824/formal_20260824T041000Z_cbef63fd
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
ORIGINAL_TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_cbef63fd46d88451
ORIGINAL_TASK_RECEIPT=${ORIGINAL_TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_ORIGINAL_TASK_RECEIPT_SHA=cbef63fd46d88451296fbfcb88ee605861497795c916c28deffbac2f1fdee909
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
ORIGINAL_BUILD_JOB=16265026
ORIGINAL_SEAL_JOB=16265034
REPAIR_INDICES=11,15,34,40,44
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() {
  tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}
upload_bundle() {
  local source=$1 destination=$2 attempt
  for attempt in 1 2 3; do
    if timeout 240 rsync -a --partial \
      --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
      -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
      "${source}/" "${SSH_ALIAS}:${destination}/"; then
      return 0
    fi
  done
  return 1
}

[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" >/dev/null 2>&1 \
  || fail "shared SSH master is not responsive"
[[ -x "${LOCAL_HAB_PY}" ]] || fail "local Habitat Python missing"

files=(
  MemNavData/construct_hm3d_fullmono_lifelong_ab.py
  MemNavData/hm3d_fullmono_lifelong.py
  MemNavData/hm3d_fullmono_lifelong_protocol_20260824.json
  MemNavData/hm3d_fullmono_zero_history_repair_manifest_20260824.json
  MemNavData/slurm_hm3d_fullmono_lifelong_zero_history_repair.sbatch
  MemNavData/test_final14_role_pair_construction.py
  MemNavData/submit_hm3d_fullmono_zero_history_repair_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing repair input ${path}"
done
"${LOCAL_HAB_PY}" -m json.tool \
  MemNavData/hm3d_fullmono_zero_history_repair_manifest_20260824.json >/dev/null
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/construct_hm3d_fullmono_lifelong_ab.py \
  MemNavData/hm3d_fullmono_lifelong.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_HAB_PY}" -m unittest -q \
  MemNavData.test_final14_role_pair_construction \
  MemNavData.test_hm3d_fullmono_lifelong
bash -n MemNavData/slurm_hm3d_fullmono_lifelong_zero_history_repair.sbatch

staging=$(mktemp -d /tmp/h3life_abfix_bundle.XXXXXX)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
mkdir -p "${staging}/root"
for path in "${files[@]}"; do
  mkdir -p "${staging}/root/$(dirname "${path}")"
  cp -p -- "${path}" "${staging}/root/${path}"
done
(
  cd "${staging}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
repair_receipt_sha=$(sha256sum "${staging}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${repair_receipt_sha:0:16}
repair_root=${REMOTE_BUNDLES}/hm3d_fullmono_abfix_${bundle_key}
repair_stage=${repair_root}.partial.$$

remote "set -euo pipefail
test \"\$(sha256sum '${ORIGINAL_TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_ORIGINAL_TASK_RECEIPT_SHA}'
cd '${ORIGINAL_TASK_ROOT}' && sha256sum -c --quiet '${ORIGINAL_TASK_RECEIPT}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test \"\$(sha256sum '${RUN_ROOT}/sealed_inputs/parent_manifest.json' | awk '{print \$1}')\" = a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
test \"\$(sha256sum '${RUN_ROOT}/sealed_inputs/parent_population_receipt.json' | awk '{print \$1}')\" = 4dd6b8dcb759dff1c0835bef8e755e7291a5f049c9adec88b954d1fda62e30d5"

if remote "test -d '${repair_root}'"; then
  remote "test \"\$(sha256sum '${repair_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${repair_receipt_sha}' && cd '${repair_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${repair_stage}' && mkdir -p '${repair_stage}'"
  upload_bundle "${staging}/root" "${repair_stage}" || fail "repair upload failed"
  remote "cd '${repair_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${repair_stage}' && mv '${repair_stage}' '${repair_root}'"
fi

protocol=${repair_root}/MemNavData/hm3d_fullmono_lifelong_protocol_20260824.json
repair_manifest=${repair_root}/MemNavData/hm3d_fullmono_zero_history_repair_manifest_20260824.json
repair_receipt=${repair_root}/SOURCE_BUNDLE.sha256
repair_sbatch=${repair_root}/MemNavData/slurm_hm3d_fullmono_lifelong_zero_history_repair.sbatch
vendor=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor

remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${repair_root}:${repair_root}/MemNavData:${ORIGINAL_TASK_ROOT}:${ORIGINAL_TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${vendor}' /scratch/lg154/conda-envs/habitat/bin/python -m unittest -q MemNavData.test_final14_role_pair_construction MemNavData.test_hm3d_fullmono_lifelong"

common="ALL,REPAIR_ROOT=${repair_root},ORIGINAL_TASK_ROOT=${ORIGINAL_TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${protocol},REPAIR_MANIFEST=${repair_manifest},REPAIR_RECEIPT=${repair_receipt},EXPECTED_REPAIR_RECEIPT_SHA=${repair_receipt_sha},ORIGINAL_TASK_RECEIPT=${ORIGINAL_TASK_RECEIPT},EXPECTED_ORIGINAL_TASK_RECEIPT_SHA=${EXPECTED_ORIGINAL_TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA}"
remote "sbatch --test-only --array=${REPAIR_INDICES} --export='${common}' '${repair_sbatch}' >/dev/null"
repair_id=$(remote "sbatch --parsable --array=${REPAIR_INDICES}%5 --export='${common}' '${repair_sbatch}'" | job_id)
[[ "${repair_id}" =~ ^[0-9]+$ ]] || fail "bad zero-history repair job id"

# The original array is expected to terminate non-zero on exactly these five
# legacy-wrapper indices.  The existing seal must wait for all original tasks
# and all exact repairs, then independently require all 54 completion receipts.
remote "scontrol update JobId=${ORIGINAL_SEAL_JOB} Dependency=afterany:${ORIGINAL_BUILD_JOB},afterok:${repair_id}"
dependency=$(remote "scontrol show job -o '${ORIGINAL_SEAL_JOB}'" | tr -d '\r' | sed -n 's/.*Dependency=\([^ ]*\).*/\1/p')
[[ "${dependency}" == *"afterany:${ORIGINAL_BUILD_JOB}"* \
   && "${dependency}" == *"afterok:${repair_id}"* ]] \
  || fail "seal dependency update did not persist: ${dependency}"

receipt=MemNavData/HM3D_FULLMONO_ZERO_HISTORY_REPAIR_SUBMISSION_${bundle_key}.json
"${LOCAL_HAB_PY}" - "${receipt}" "${repair_root}" "${repair_receipt_sha}" \
  "${repair_id}" "${dependency}" <<'PY'
import json, sys
path, bundle, bundle_sha, repair, dependency = sys.argv[1:]
payload = {
  "schema_version": "hm3d_fullmono_zero_history_repair_submission_v1_20260824",
  "repair_bundle": bundle,
  "repair_bundle_receipt_sha256": bundle_sha,
  "repair_job_id": int(repair),
  "repair_indices": [11, 15, 34, 40, 44],
  "selection_reads_navigation_outcomes": False,
  "scientific_thresholds_changed": False,
  "original_build_job_id": 16265026,
  "updated_original_seal_job_id": 16265034,
  "updated_seal_dependency": dependency,
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
timeout 120 scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" "${receipt}" \
  "${SSH_ALIAS}:${RUN_ROOT}/zero_history_repair_submission.json" \
  || fail "repair receipt upload failed"
remote "sha256sum '${RUN_ROOT}/zero_history_repair_submission.json' >'${RUN_ROOT}/zero_history_repair_submission.json.sha256'; squeue -j '${repair_id},${ORIGINAL_SEAL_JOB}' -o '%.18i %.22j %.2t %.12M %.40R'"
printf 'REPAIR_ROOT=%s\nREPAIR_JOB=%s\nSEAL_JOB=%s\nDEPENDENCY=%s\n' \
  "${repair_root}" "${repair_id}" "${ORIGINAL_SEAL_JOB}" "${dependency}"
