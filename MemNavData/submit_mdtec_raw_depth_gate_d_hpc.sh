#!/usr/bin/env bash
# Local release gate, immutable bundle upload, and formal Slurm submission.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_raw_depth_gate_d_20260819
LOCAL_PY=/home/asus/miniconda3/envs/memnav/bin/python
cd "${ROOT}"

"${LOCAL_PY}" -m py_compile \
  MemNavData/monocular_depth_runtime.py \
  MemNavData/mdtec_raw_depth_gate_d.py \
  MemNavData/eval_mdtec_raw_depth_gate_d_habitat.py \
  MemNavData/summarize_mdtec_raw_depth_gate_d.py \
  MemNavData/independent_verify_mdtec_raw_depth_gate_d.py \
  MemNavData/eval_2leg_habitat.py \
  NavDP/baselines/memnav/memnav_server.py \
  NavDP/baselines/memnav/policy_agent.py \
  NavDP/baselines/navdp/navdp_server.py
"${LOCAL_PY}" -m unittest \
  MemNavData.test_mdtec_raw_depth_gate_d \
  MemNavData.test_monocular_depth_runtime \
  MemNavData.test_navdp_memory_replay \
  MemNavData.test_policy_agent_graph \
  MemNavData.test_deterministic_eval_protocol
bash -n MemNavData/run_mdtec_raw_depth_gate_d_scene.sh \
  MemNavData/slurm_mdtec_raw_depth_gate_d.sbatch \
  MemNavData/slurm_mdtec_raw_depth_gate_d_summary.sbatch

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
files=(
  MemNavData/MDTEC_RAW_DEPTH_GATE_D_PROTOCOL_20260819.md
  MemNavData/MDTEC_RAW_DEPTH_GATE_D_ANALYSIS_AMENDMENT_20260819.md
  MemNavData/mdtec_raw_depth_gate_d_protocol_20260819.json
  MemNavData/mdtec_raw_depth_gate_d_analysis_20260819.json
  MemNavData/expanded_navdp_router_eval_20260805.json
  MemNavData/novel_a_bearing_inputs_20260808.json
  MemNavData/monocular_depth_runtime.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/eval_mdtec_raw_depth_gate_d_habitat.py
  MemNavData/summarize_mdtec_raw_depth_gate_d.py
  MemNavData/independent_verify_mdtec_raw_depth_gate_d.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/generate_twoleg.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/arrival_shadow.py
  MemNavData/bearing_diagnostics.py
  MemNavData/navdp_goal_switch.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/revisit_action_shadow.py
  MemNavData/shared_online_double_revisit_runtime.py
  MemNavData/xnavdp_revisit_contract.py
  MemNavData/test_mdtec_raw_depth_gate_d.py
  MemNavData/test_monocular_depth_runtime.py
  MemNavData/test_navdp_memory_replay.py
  MemNavData/test_policy_agent_graph.py
  MemNavData/test_deterministic_eval_protocol.py
  MemNavData/run_mdtec_raw_depth_gate_d_scene.sh
  MemNavData/slurm_mdtec_raw_depth_gate_d.sbatch
  MemNavData/slurm_mdtec_raw_depth_gate_d_summary.sbatch
  MemNavData/submit_mdtec_raw_depth_gate_d_hpc.sh
)
while IFS= read -r path; do files+=("${path}"); done < <(
  find MemNavData -maxdepth 1 -type f -name '*.py' -print | sort)
while IFS= read -r path; do files+=("${path}"); done < <(
  find NavDP/baselines/memnav NavDP/baselines/navdp -maxdepth 1 -type f \
    -name '*.py' -print | sort)
while IFS= read -r path; do files+=("${path}"); done < <(
  find NavDP/baselines/navdp/depth_anything/depth_anything_v2 -type f \
    ! -path '*/__pycache__/*' ! -name '*.pyc' -print | sort)
while IFS= read -r path; do files+=("${path}"); done < <(
  find InternNav/internnav InternNav/scripts/train/configs \
       InternNav/src/diffusion-policy -type f \
    ! -path '*/__pycache__/*' ! -name '*.pyc' -print | sort)
printf '%s\n' "${files[@]}" | sort -u >"${staging}/file_list.txt"
while IFS= read -r path; do
  [[ -r "${path}" ]] || { echo "missing bundle input ${path}" >&2; exit 1; }
  mkdir -p "${staging}/root/$(dirname "${path}")"
  cp -p -- "${path}" "${staging}/root/${path}"
done <"${staging}/file_list.txt"
(
  cd "${staging}/root"
  PYTHONPATH="${staging}/root" "${LOCAL_PY}" -m unittest \
    MemNavData.test_mdtec_raw_depth_gate_d \
    MemNavData.test_monocular_depth_runtime \
    MemNavData.test_navdp_memory_replay \
    MemNavData.test_policy_agent_graph \
    MemNavData.test_deterministic_eval_protocol
  PYTHONPATH="${staging}/root" "${LOCAL_PY}" -m py_compile \
    MemNavData/eval_mdtec_raw_depth_gate_d_habitat.py \
    MemNavData/summarize_mdtec_raw_depth_gate_d.py \
    MemNavData/independent_verify_mdtec_raw_depth_gate_d.py \
    NavDP/baselines/memnav/memnav_server.py \
    NavDP/baselines/navdp/navdp_server.py
)
find "${staging}/root" -type d -name __pycache__ -prune -exec rm -rf -- {} +
(
  cd "${staging}/root"
  find . -type f -print0 | sort -z | xargs -0 sha256sum \
    > "${staging}/source_inputs.sha256"
)
mv "${staging}/source_inputs.sha256" "${staging}/root/source_inputs.sha256"
receipt_sha=$(sha256sum "${staging}/root/source_inputs.sha256" | awk '{print $1}')
bundle_tag=mdtec_gate_d_${receipt_sha:0:16}
remote_root=${REMOTE_BUNDLES}/${bundle_tag}
run_tag=formal_$(date -u +%Y%m%dT%H%M%SZ)_${receipt_sha:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}

ssh "${SSH_ALIAS}" "test ! -e '${remote_root}' && mkdir -p '${remote_root}' '${REMOTE_RESULTS}' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs"
rsync -a "${staging}/root/" "${SSH_ALIAS}:${remote_root}/"
ssh "${SSH_ALIAS}" "cd '${remote_root}' && sha256sum -c --quiet source_inputs.sha256 && test \"\$(sha256sum source_inputs.sha256 | cut -d' ' -f1)\" = '${receipt_sha}' && chmod -R a-w '${remote_root}'"

array_job=$(ssh "${SSH_ALIAS}" "sbatch --parsable --array=0-19%10 --export=ALL,SOURCE_ROOT='${remote_root}',RUN_ROOT='${run_root}',SOURCE_RECEIPT='${remote_root}/source_inputs.sha256',EXPECTED_SOURCE_RECEIPT_SHA='${receipt_sha}' '${remote_root}/MemNavData/slurm_mdtec_raw_depth_gate_d.sbatch'")
summary_job=$(ssh "${SSH_ALIAS}" "sbatch --parsable --dependency=afterok:${array_job} --export=ALL,SOURCE_ROOT='${remote_root}',RUN_ROOT='${run_root}',SOURCE_RECEIPT='${remote_root}/source_inputs.sha256',EXPECTED_SOURCE_RECEIPT_SHA='${receipt_sha}' '${remote_root}/MemNavData/slurm_mdtec_raw_depth_gate_d_summary.sbatch'")
printf 'SOURCE_ROOT=%s\nRUN_ROOT=%s\nSOURCE_RECEIPT_SHA=%s\nARRAY_JOB=%s\nSUMMARY_JOB=%s\n' \
  "${remote_root}" "${run_root}" "${receipt_sha}" "${array_job}" "${summary_job}"
