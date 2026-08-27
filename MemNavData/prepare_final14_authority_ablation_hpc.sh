#!/usr/bin/env bash
# Freeze and stage the Final14 matched-proposal authority ablation.
# SUBMIT=1 starts smoke -> formal array -> independent analysis.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
EXPECTED_SSH_USER=${EXPECTED_SSH_USER:-yz11502}
SUBMIT=${SUBMIT:-0}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_PY=/scratch/lg154/conda-envs/memnav/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_authority_20260828
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_SOURCE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_SOURCE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
BENCH_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T115533Z_attempt7_handoff/benchmarks/natural_direction
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
EXPECTED_SOURCE_OVERLAY_BYTES=128854888448

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }

select_control_path() {
  local candidate
  local remote_user
  local candidates=()
  if [[ -n "${SSH_CONTROL_PATH:-}" ]]; then
    candidates+=("${SSH_CONTROL_PATH}")
  fi
  while IFS= read -r candidate; do candidates+=("${candidate}"); done < <(
    find /home/asus/.ssh -maxdepth 1 -type s -name 'cm-*' \
      -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk '{print $2}')
  for candidate in "${candidates[@]}"; do
    [[ -S "${candidate}" ]] || continue
    remote_user=$(timeout 12 ssh -n -T -o BatchMode=yes \
      -o ControlMaster=no -S "${candidate}" "${SSH_ALIAS}" \
      'id -un' 2>/dev/null || true)
    if [[ "${remote_user}" == "${EXPECTED_SSH_USER}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

SSH_CONTROL_PATH=$(select_control_path) || \
  fail "no shared SSH socket is both responsive and authenticated as ${EXPECTED_SSH_USER}"
remote() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
job_id() { tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'; }

[[ "${SUBMIT}" =~ ^[01]$ ]] || fail "SUBMIT must be 0 or 1"
[[ -x "${LOCAL_PY}" ]] || fail "local Python missing"
files=(
  MemNavData/certified_relocalization_runtime.py
  MemNavData/test_certified_relocalization_runtime.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/final14_authority_ablation.py
  MemNavData/test_final14_authority_ablation.py
  MemNavData/run_final14_authority_ablation_episode.py
  MemNavData/run_final14_authority_ablation_history.sh
  MemNavData/summarize_final14_authority_ablation.py
  MemNavData/independent_verify_final14_authority_ablation.py
  MemNavData/final14_cec_authority_ablation_protocol_20260828.json
  MemNavData/FINAL14_CEC_AUTHORITY_ABLATION_PROTOCOL_20260828.md
  MemNavData/slurm_final14_authority_ablation.sbatch
  MemNavData/slurm_final14_authority_ablation_analysis.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/prepare_final14_authority_ablation_hpc.sh
  MemNavData/test_policy_agent_graph.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/memnav_server.py
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}:${ROOT}/MemNavData" \
  "${LOCAL_PY}" -m pytest -q -p no:cacheprovider \
    MemNavData/test_certified_relocalization_runtime.py \
    MemNavData/test_final14_authority_ablation.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}:${ROOT}/MemNavData" \
  "${LOCAL_PY}" -m unittest -q MemNavData.test_policy_agent_graph
"${LOCAL_PY}" -m py_compile \
  MemNavData/certified_relocalization_runtime.py \
  MemNavData/eval_2leg_habitat.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/final14_authority_ablation.py \
  MemNavData/run_final14_authority_ablation_episode.py \
  MemNavData/summarize_final14_authority_ablation.py \
  MemNavData/independent_verify_final14_authority_ablation.py \
  NavDP/baselines/memnav/policy_agent.py \
  NavDP/baselines/memnav/memnav_server.py
bash -n \
  MemNavData/run_final14_authority_ablation_history.sh \
  MemNavData/slurm_final14_authority_ablation.sbatch \
  MemNavData/slurm_final14_authority_ablation_analysis.sbatch \
  MemNavData/prepare_final14_authority_ablation_hpc.sh
"${LOCAL_PY}" -m json.tool \
  MemNavData/final14_cec_authority_ablation_protocol_20260828.json >/dev/null
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template MemNavData/slurm_final14_authority_ablation.sbatch
  lint_sbatch_template \
    MemNavData/slurm_final14_authority_ablation_analysis.sbatch
)

scratch=$(mktemp -d /tmp/f14_authority_prepare.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
for path in "${files[@]}"; do
  mkdir -p "${scratch}/root/$(dirname "${path}")"
  install -m 0644 "${path}" "${scratch}/root/${path}"
done
"${LOCAL_PY}" - "${scratch}/root/source_bundle_manifest.json" <<'PY'
import json, sys
payload = {
    "schema_version": "final14_cec_authority_ablation_bundle_v1_20260828",
    "scope": "consumed Final14 matched-proposal authority-only ablation",
    "history_count": 21,
    "queries_per_arm": 42,
    "arms": ["mono_cec", "mono_unthresholded_witness"],
    "sole_intervention": "operational_authority_policy",
    "same_process_pairing": True,
    "runtime_role_visibility": "none",
    "controller_checkpoint_changed": False,
    "population_seed_or_budget_changed": False,
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
repair_root=${REMOTE_BUNDLES}/final14_cec_authority_${receipt_sha:0:16}
repair_stage=${repair_root}.partial.$$
run_root=${REMOTE_RESULTS}/formal_${receipt_sha:0:16}
smoke_root=${REMOTE_RESULTS}/smoke_${receipt_sha:0:16}

remote "set -euo pipefail; \
  test \"\$(sha256sum '${BASE_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}'; \
  cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet source_inputs.sha256; \
  test \"\$(sha256sum '${BENCH_ROOT}/manifest.json' | awk '{print \$1}')\" = 7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a; \
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
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${repair_root}:${repair_root}/MemNavData:${repair_root}/NavDP/baselines/memnav:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav' '${REMOTE_PY}' -m pytest -q -p no:cacheprovider '${repair_root}/MemNavData/test_certified_relocalization_runtime.py' '${repair_root}/MemNavData/test_final14_authority_ablation.py'"
remote "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${repair_root}:${repair_root}/MemNavData:${repair_root}/NavDP/baselines/memnav:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav' '${REMOTE_PY}' -m unittest -q MemNavData.test_policy_agent_graph"
for route in certified_relocalization certified_unthresholded_witness; do
  remote "singularity exec --overlay '${SOURCE_OVERLAY}:ro' -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${repair_root}:${repair_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python '${repair_root}/MemNavData/eval_shared_online_role_pairs.py' --contract_dry_run --episode_root /contract/dry/scene0 --scene /contract/dry/scene.glb --scene_identity scene0 --out /contract/dry/out --host 127.0.0.1 --port 18888 --novel_port 18889 --server_backend hybrid_pose --success_dist 1.0 --max_steps 600 --exec_horizon 8 --trajectory_selector server --trajectory_selector_scope all --leg1_mode shared_trace --leg1_goal_source own --seed 0 --terminal_uturn off --terminal_visual_refine off --deterministic_plan_seeds --retrieval_override off --certified_cdec_rescue off --certified_stagnation_graph off --revisit_controller navdp_mixed --role_pair_scope consumed_integration --revisit_adapter verified_bearing_v1 --navdp_depth_source monocular_sidecar --hybrid_route '${route}'"
done

common="ALL,REPAIR_ROOT=${repair_root},REPAIR_RECEIPT=${repair_receipt},EXPECTED_REPAIR_RECEIPT_SHA=${receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT=${BASE_SOURCE_RECEIPT},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA},BENCH_ROOT=${BENCH_ROOT},SOURCE_OVERLAY=${SOURCE_OVERLAY},EXPECTED_SOURCE_OVERLAY_BYTES=${EXPECTED_SOURCE_OVERLAY_BYTES}"
gpu_script=${repair_root}/MemNavData/slurm_final14_authority_ablation.sbatch
analysis_script=${repair_root}/MemNavData/slurm_final14_authority_ablation_analysis.sbatch
remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0 --export='${common},RUN_ROOT=${smoke_root},SMOKE=1,MAX_STEPS=80' '${gpu_script}' >/dev/null"
remote "source '${repair_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --partition=cpu_short --export='${common},RUN_ROOT=${run_root}' '${analysis_script}' >/dev/null"

prepare_receipt=MemNavData/FINAL14_CEC_AUTHORITY_ABLATION_PREPARATION_RECEIPT_20260828.json
if [[ ! -e "${prepare_receipt}" ]]; then
  "${LOCAL_PY}" - "${prepare_receipt}" "${repair_root}" "${receipt_sha}" \
    "${run_root}" "${smoke_root}" "${SSH_CONTROL_PATH}" <<'PY'
import json, sys
path, bundle, digest, run, smoke, socket = sys.argv[1:]
payload = {
    "schema_version": "final14_cec_authority_ablation_preparation_v1_20260828",
    "source_bundle": bundle,
    "source_bundle_receipt_sha256": digest,
    "run_root": run,
    "smoke_root": smoke,
    "history_count": 21,
    "queries_per_arm": 42,
    "arms": ["mono_cec", "mono_unthresholded_witness"],
    "same_process_pairing": True,
    "runtime_role_visibility": "none",
    "shared_ssh_socket_used": socket,
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
receipt=MemNavData/FINAL14_CEC_AUTHORITY_ABLATION_SUBMISSION_RECEIPT_20260828.json
[[ ! -e "${receipt}" ]] || fail "submission receipt exists"
"${LOCAL_PY}" - "${receipt}" "${repair_root}" "${receipt_sha}" \
  "${run_root}" "${smoke_root}" "${smoke_job}" "${formal_job}" \
  "${analysis_job}" <<'PY'
import json, sys
path,bundle,digest,run,smoke,smoke_job,formal,analysis=sys.argv[1:]
payload={
 "schema_version":"final14_cec_authority_ablation_submission_v1_20260828",
 "source_bundle":bundle,"source_bundle_receipt_sha256":digest,
 "run_root":run,"smoke_root":smoke,
 "smoke_job":smoke_job,"formal_array_job":formal,
 "analysis_job":analysis,"history_count":21,"queries_per_arm":42,
 "same_process_pairing":True,"submitted":True,
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'SUBMITTED=1\nSMOKE_JOB=%s\nFORMAL_JOB=%s\nANALYSIS_JOB=%s\nRUN_ROOT=%s\n' \
  "${smoke_job}" "${formal_job}" "${analysis_job}" "${run_root}"
