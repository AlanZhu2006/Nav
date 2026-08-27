#!/usr/bin/env bash
# Resume a sealed full-mono population after an external smoke cancellation.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_mixed_role_0e587874d5b89531
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_mixed_role_20260820/formal_20260819T182657Z_0e587874
SMOKE_ROOT=${RUN_ROOT}_smoke_repair1
PARENT_MANIFEST=${RUN_ROOT}/sealed_inputs/parent_hm3d_manifest.json
PROTOCOL=${TASK_ROOT}/MemNavData/hm3d_fullmono_mixed_role_protocol_20260820.json
BENCH_ROOT=${RUN_ROOT}/benchmarks/natural_direction
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
TASK_RECEIPT_SHA=0e587874d5b8953176a49a77345f6bde8733c2d31930ab3547d8c238a857b372
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
EXPECTED_MANIFEST_SHA=b70d53557d5fb656badf1311d19b3f643583a1ab5fad782b542b66e4647c5948
INCIDENT=${ROOT}/MemNavData/HM3D_FULLMONO_MIXED_ROLE_SMOKE_UID0_INCIDENT_20260820.json
RECEIPT=${ROOT}/MemNavData/HM3D_FULLMONO_MIXED_ROLE_SMOKE_REPAIR_SUBMISSION_RECEIPT_20260820.json

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  ssh -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master socket missing"
[[ -f "${INCIDENT}" && ! -e "${RECEIPT}" ]] || fail "incident missing or repair receipt exists"
[[ "$(remote 'id -un' | tr -d '\r')" == yz11502 ]] || fail "wrong remote identity"

remote "set -euo pipefail
test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${TASK_RECEIPT_SHA}'
cd '${TASK_ROOT}' && sha256sum -c --quiet '${TASK_RECEIPT}'
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${BASE_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test -f '${RUN_ROOT}/benchmarks/SEALED'
cd '${RUN_ROOT}/benchmarks' && sha256sum -c --quiet BENCHMARK_FILES.sha256
test \"\$(sha256sum '${BENCH_ROOT}/manifest.json' | awk '{print \$1}')\" = '${EXPECTED_MANIFEST_SHA}'
test ! -e '${SMOKE_ROOT}'
test \"\$(find '${RUN_ROOT}/runtime' -maxdepth 1 -type d -name 'eval_*' | wc -l)\" -eq 0
test ! -e '${RUN_ROOT}/hm3d_fullmono_mixed_role_summary.json'
test ! -e '${RUN_ROOT}/hm3d_fullmono_mixed_role_independent_verification.json'
mkdir -p '${SMOKE_ROOT}/logs'"

common="ALL,TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},PARENT_MANIFEST=${PARENT_MANIFEST},PROTOCOL=${PROTOCOL},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${BASE_RECEIPT_SHA}"
eval_script=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_eval.sbatch
analysis_script=${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_analysis.sbatch

remote "sbatch --test-only --array=0 --export='${common},RUN_ROOT=${SMOKE_ROOT},BENCH_ROOT=${BENCH_ROOT},MODE=smoke,MAX_STEPS=80' '${eval_script}' >/dev/null"
smoke_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0 --export='${common},RUN_ROOT=${SMOKE_ROOT},BENCH_ROOT=${BENCH_ROOT},MODE=smoke,MAX_STEPS=80' '${eval_script}'" | tr -d '\r')
smoke_id=${smoke_raw%%;*}; [[ "${smoke_id}" =~ ^[0-9]+$ ]] || fail "bad repair smoke job"
eval_raw=$(remote "sbatch --parsable --qos=gpu48 --array=0-8%4 --dependency=afterok:${smoke_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${RUN_ROOT},BENCH_ROOT=${BENCH_ROOT},MODE=eval,MAX_STEPS=600' '${eval_script}'" | tr -d '\r')
eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad repair eval job"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${RUN_ROOT},BENCH_ROOT=${BENCH_ROOT},MODE=summary' '${analysis_script}'" | tr -d '\r')
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad repair summary job"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${common},RUN_ROOT=${RUN_ROOT},BENCH_ROOT=${BENCH_ROOT},MODE=verify' '${analysis_script}'" | tr -d '\r')
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad repair verify job"

/home/asus/miniconda3/envs/memnav/bin/python - "${RECEIPT}" \
  "${SMOKE_ROOT}" "${smoke_id}" "${eval_id}" "${summary_id}" "${verify_id}" <<'PY'
import json, sys
path, smoke_root, smoke, evaluation, summary, verify = sys.argv[1:]
payload = {
    "schema_version": "hm3d_fullmono_mixed_role_smoke_repair_submission_v1_20260820",
    "repair_scope": "externally_cancelled_zero_artifact_smoke_only",
    "original_cancelled_smoke": "16066788_0",
    "run_root": "/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_mixed_role_20260820/formal_20260819T182657Z_0e587874",
    "isolated_smoke_root": smoke_root,
    "sealed_population_reused": True,
    "goal_a_or_construction_rerun": False,
    "query_outcomes_read_before_repair": False,
    "jobs": {
        "query_smoke": int(smoke),
        "query_evaluation_array": int(evaluation),
        "summary": int(summary),
        "independent_verification": int(verify),
    },
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

scp -q -o BatchMode=yes "${INCIDENT}" "${SSH_ALIAS}:${RUN_ROOT}/smoke_uid0_incident.json"
scp -q -o BatchMode=yes "${RECEIPT}" "${SSH_ALIAS}:${RUN_ROOT}/smoke_repair_submission.json"
remote "sha256sum '${RUN_ROOT}/smoke_uid0_incident.json' >'${RUN_ROOT}/smoke_uid0_incident.json.sha256'; sha256sum '${RUN_ROOT}/smoke_repair_submission.json' >'${RUN_ROOT}/smoke_repair_submission.json.sha256'"
printf 'SMOKE_REPAIR=%s\nEVAL=%s\nSUMMARY=%s\nVERIFY=%s\n' \
  "${smoke_id}" "${eval_id}" "${summary_id}" "${verify_id}"
