#!/usr/bin/env bash
# Freeze, preflight, and optionally dependency-submit the underpowered HM3D
# continual-memory amendment.  SUBMIT=0 is non-mutating on Slurm.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
SUBMIT=${SUBMIT:-0}
UPSTREAM_AFTEROK=${UPSTREAM_AFTEROK:-}
SUBMISSION_RECEIPT=${SUBMISSION_RECEIPT:-MemNavData/HM3D_LIFELONG_UNDERPOWERED_SUBMISSION_RECEIPT_20260828.json}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_PY=/scratch/lg154/conda-envs/memnav/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_TASK_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
POPULATION_SHA=ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5
POPULATION_VERIFIER_SHA=d9ce97df4b0687969090e710ef719f6da56fc5d39a0535a7e8afd6c5d852499b
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
EXPECTED_SOURCE_OVERLAY_BYTES=128854888448
HAB_REQUESTS_VENDOR=/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/pip/_vendor

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SUBMIT}" =~ ^[01]$ ]] || fail "SUBMIT must be 0 or 1"
[[ -z "${UPSTREAM_AFTEROK}" || "${UPSTREAM_AFTEROK}" =~ ^[0-9]+$ ]] || \
  fail "UPSTREAM_AFTEROK must be empty or numeric"
[[ "${SUBMISSION_RECEIPT}" == MemNavData/*.json ]] || \
  fail "submission receipt must be a JSON file under MemNavData"
if [[ "${SUBMIT}" == 1 && -z "${UPSTREAM_AFTEROK}" ]]; then
  fail "formal submission requires UPSTREAM_AFTEROK to preserve one-array residency"
fi

SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "authoritative shared SSH master is not responsive"
remote_user=$(timeout 20 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
  -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" 'id -un' 2>/dev/null || true)
[[ "${remote_user}" == "${EXPECTED_SSH_USER}" ]] || \
  fail "shared SSH identity is ${remote_user:-unavailable}"
remote() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

files=(
  MemNavData/hm3d_fullmono_lifelong_underpowered_amendment_20260828.json
  MemNavData/HM3D_FULLMONO_LIFELONG_UNDERPOWERED_AMENDMENT_20260828.md
  MemNavData/audit_hm3d_lifelong_underpowered_amendment.py
  MemNavData/test_audit_hm3d_lifelong_underpowered_amendment.py
  MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch
  MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
  MemNavData/slurm_port_pair.sh
  MemNavData/test_slurm_port_pair.sh
  MemNavData/slurm_safe_submit.sh
  MemNavData/prepare_hm3d_lifelong_underpowered_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

"${LOCAL_PY}" -m json.tool \
  MemNavData/hm3d_fullmono_lifelong_underpowered_amendment_20260828.json \
  >/dev/null
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}:${ROOT}/MemNavData" \
  "${LOCAL_PY}" -m pytest -q -p no:cacheprovider \
  MemNavData/test_audit_hm3d_lifelong_underpowered_amendment.py
bash MemNavData/test_slurm_port_pair.sh
"${LOCAL_PY}" -m py_compile \
  MemNavData/audit_hm3d_lifelong_underpowered_amendment.py
bash -n MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch \
  MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch \
  MemNavData/slurm_port_pair.sh MemNavData/test_slurm_port_pair.sh \
  MemNavData/prepare_hm3d_lifelong_underpowered_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template \
    MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch
  lint_sbatch_template MemNavData/slurm_hm3d_fullmono_shared_c_arm.sbatch
)

stage=$(mktemp -d /tmp/h3_underpowered_prepare.XXXXXX)
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
for path in "${files[@]}"; do
  mkdir -p "${stage}/root/$(dirname "${path}")"
  install -m 0644 "${path}" "${stage}/root/${path}"
done
(
  cd "${stage}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
bundle_sha=$(sha256sum "${stage}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle=${REMOTE_BUNDLES}/hm3d_lifelong_underpowered_${bundle_sha:0:16}
partial=${bundle}.partial.$$
bundle_receipt=${bundle}/SOURCE_BUNDLE.sha256
protocol=${bundle}/MemNavData/hm3d_fullmono_lifelong_underpowered_amendment_20260828.json
deferred=${bundle}/MemNavData/slurm_hm3d_lifelong_underpowered_deferred.sbatch
smoke_root=${RUN_ROOT}/underpowered_smoke_${bundle_sha:0:16}

remote "set -euo pipefail; \
  test \"\$(sha256sum '${TASK_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_TASK_RECEIPT_SHA}'; \
  cd '${TASK_ROOT}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256; \
  test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'; \
  cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet source_inputs.sha256; \
  test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'; \
  test \"\$(sha256sum '${RUN_ROOT}/population/population.json' | awk '{print \$1}')\" = '${POPULATION_SHA}'; \
  test \"\$(sha256sum '${RUN_ROOT}/independent_natural_v4_population_verification.json' | awk '{print \$1}')\" = '${POPULATION_VERIFIER_SHA}'; \
  test \"\$(stat -c '%s' '${SOURCE_OVERLAY}')\" = '${EXPECTED_SOURCE_OVERLAY_BYTES}'"

if remote "test -d '${bundle}'"; then
  remote "test \"\$(sha256sum '${bundle_receipt}' | awk '{print \$1}')\" = '${bundle_sha}' && cd '${bundle}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${partial}' && mkdir -p '${partial}'"
  timeout 240 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${stage}/root/" "${SSH_ALIAS}:${partial}/"
  remote "cd '${partial}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${partial}' && mv '${partial}' '${bundle}'"
fi

remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${bundle}:${bundle}/MemNavData' '${REMOTE_PY}' -m pytest -q -p no:cacheprovider '${bundle}/MemNavData/test_audit_hm3d_lifelong_underpowered_amendment.py'"
remote "ROOT='${bundle}' bash '${bundle}/MemNavData/test_slurm_port_pair.sh'"
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${bundle}:${bundle}/MemNavData' '${REMOTE_PY}' '${bundle}/MemNavData/audit_hm3d_lifelong_underpowered_amendment.py' --protocol '${protocol}' --run-root '${RUN_ROOT}' --require-pristine"
remote "singularity exec --overlay '${SOURCE_OVERLAY}:ro' -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${HAB_REQUESTS_VENDOR}' /scratch/lg154/conda-envs/habitat/bin/python -c 'import collect_hm3d_lifelong_shared_c,eval_hm3d_lifelong_shared_c_b2' --episode_root /contract/dry/episodes --scene /contract/dry/scene.glb --scene_identity scene0 --out /contract/dry/out --host 127.0.0.1 --port 18888 --novel_port 18889 --server_backend cec_portability --success_dist 1.0 --max_steps 600 --exec_horizon 8 --trajectory_selector server --trajectory_selector_scope all --leg1_mode shared_trace --leg1_goal_source own --seed 0 --terminal_uturn off --terminal_visual_refine off --deterministic_plan_seeds --retrieval_override off --certified_cdec_rescue off --certified_stagnation_graph off --revisit_controller navdp_mixed --role_pair_scope consumed_integration --revisit_adapter legacy_metric --navdp_depth_source monocular_sidecar --hybrid_route phase --navdp_goal_switch_reset before_c --shared_leg1_trace_root /contract/dry/run --double_revisit_c_history initial_leg_only --shared_online_nnr_arm cec_portability --lifelong_history_scope all_prior"

common="ALL,DEFERRED_MODE=collect,AMENDMENT_ROOT=${bundle},AMENDMENT_RECEIPT=${bundle_receipt},EXPECTED_AMENDMENT_RECEIPT_SHA=${bundle_sha},TASK_ROOT=${TASK_ROOT},TASK_RECEIPT=${TASK_RECEIPT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PROTOCOL=${protocol},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},SMOKE_ROOT=${smoke_root},SOURCE_POPULATION_SHA=${POPULATION_SHA},SOURCE_POPULATION_COUNT=22,SOURCE_SCENE_COUNT=15,EVAL_CONCURRENCY=2,GPU_TIME_LIMIT=01:00:00,UPSTREAM_POPULATION_SEAL_JOB_ID=16489720"
remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common}' '${deferred}' >/dev/null"

preparation=MemNavData/HM3D_LIFELONG_UNDERPOWERED_PREPARATION_${bundle_sha:0:16}.json
if [[ ! -e "${preparation}" ]]; then
  "${LOCAL_PY}" - "${preparation}" "${bundle}" "${bundle_sha}" \
    "${RUN_ROOT}" "${smoke_root}" <<'PY'
import json,sys
path,bundle,digest,run,smoke=sys.argv[1:]
payload={
 "schema_version":"hm3d_lifelong_underpowered_preparation_v1_20260828",
 "source_bundle":bundle,"source_bundle_receipt_sha256":digest,
 "run_root":run,"smoke_root":smoke,"histories":22,"scene_clusters":15,
 "underpowered":True,"query_outcomes_read_before_preparation":False,
 "remote_tests_passed":True,"slurm_test_only_passed":True,"submitted":False,
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
fi

if [[ "${SUBMIT}" == 0 ]]; then
  printf 'PREPARED_ONLY=1\nBUNDLE=%s\nRUN_ROOT=%s\n' "${bundle}" "${RUN_ROOT}"
  exit 0
fi

remote "test ! -e '${RUN_ROOT}/shared_c_collection' && test ! -e '${smoke_root}'"
launcher=$(remote "source '${bundle}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --partition=cpu_short --dependency=afterok:${UPSTREAM_AFTEROK} --kill-on-invalid-dep=yes --export='${common}' '${deferred}'" | job_id)
[[ "${launcher}" =~ ^[0-9]+$ ]] || fail "bad deferred launcher job id"
receipt=${SUBMISSION_RECEIPT}
[[ ! -e "${receipt}" ]] || fail "submission receipt already exists"
"${LOCAL_PY}" - "${receipt}" "${bundle}" "${bundle_sha}" "${RUN_ROOT}" \
  "${smoke_root}" "${UPSTREAM_AFTEROK}" "${launcher}" <<'PY'
import json,sys
path,bundle,digest,run,smoke,upstream,launcher=sys.argv[1:]
payload={
 "schema_version":"hm3d_lifelong_underpowered_submission_v1_20260828",
 "source_bundle":bundle,"source_bundle_receipt_sha256":digest,
 "run_root":run,"smoke_root":smoke,"histories":22,"scene_clusters":15,
 "underpowered":True,"powered_confirmation_claim":False,
 "upstream_afterok_job":int(upstream),"deferred_launcher_job":int(launcher),
 "one_large_GPU_array_resident_per_stage":True,"submitted":True,
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'SUBMITTED=1\nDEFERRED_LAUNCHER=%s\nUPSTREAM_AFTEROK=%s\n' \
  "${launcher}" "${UPSTREAM_AFTEROK}"
