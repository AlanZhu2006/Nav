#!/usr/bin/env bash
# Freeze and submit the consumed five-query ViNT/CEC direction mechanism test.
set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/vint_cec_bearing_alignment_loss5_20260828}
RUN_TAG=${RUN_TAG:-mechanism_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
FRESH_ROOT=${FRESH_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6}
PARENT_FORMAL_SUMMARY=${PARENT_FORMAL_SUMMARY:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_vint_controller_native_cec_20260828/hm3d_vint_cec_table1_20260828/formal/formal_summary.json}
EXPECTED_PARENT_FORMAL_SUMMARY_SHA=${EXPECTED_PARENT_FORMAL_SUMMARY_SHA:-aaadb96512fd41855732d3e8bd1adf473ddcb19a5b7d528b115b989100f2cc82}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA:-74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98}
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA:-4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e}
PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-envs/controller_portability_a9ec7146bce7_v1}
PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-checkpoints/controller_portability_50387aa89be8}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
DRY_RUN=${DRY_RUN:-0}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 300 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" "$@" | tr -d '\r'
}

[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${REMOTE_HOST}" \
  >/dev/null 2>&1 || fail "authoritative SSH master is not responsive"
[[ "$(remote 'id -un')" == yz11502 ]] || fail "remote identity is not yz11502"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/cec_bearing_alignment.py" \
  "${LOCAL_ROOT}/MemNavData/audit_vint_cec_bearing_alignment_cell.py" \
  "${LOCAL_ROOT}/MemNavData/aggregate_vint_cec_bearing_alignment_loss5.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_vint_cec_bearing_alignment_loss5.py"
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${LOCAL_ROOT}/MemNavData/eval_shared_online_role_pairs.py"
"${MEMNAV_PY}" -m pytest -q \
  "${LOCAL_ROOT}/MemNavData/test_cec_bearing_alignment.py" \
  "${LOCAL_ROOT}/MemNavData/test_audit_vint_cec_bearing_alignment_cell.py" \
  "${LOCAL_ROOT}/MemNavData/test_cec_handoff_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_controller_portability_contract.py" \
  "${LOCAL_ROOT}/MemNavData/test_cec_controller_portability_hub.py"
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_vint_cec_bearing_alignment_loss5.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_vint_cec_bearing_alignment_loss5_analysis.sbatch"
source "${LOCAL_ROOT}/MemNavData/slurm_safe_submit.sh"
lint_sbatch_template \
  "${LOCAL_ROOT}/MemNavData/slurm_vint_cec_bearing_alignment_loss5.sbatch" || \
  fail "cell sbatch lint failed"
lint_sbatch_template \
  "${LOCAL_ROOT}/MemNavData/slurm_vint_cec_bearing_alignment_loss5_analysis.sbatch" || \
  fail "analysis sbatch lint failed"

contract_args=(
  --contract_dry_run --episode_root /contract/dry/scene0
  --scene /contract/dry/scene.glb --scene_identity scene0
  --out /contract/dry/out --host 127.0.0.1 --port 18888
  --server_backend cec_portability --success_dist 1.0 --max_steps 600
  --exec_horizon 8 --trajectory_selector server --trajectory_selector_scope all
  --leg1_mode shared_trace --leg1_goal_source own --seed 0
  --terminal_uturn off --terminal_visual_refine off --deterministic_plan_seeds
  --retrieval_override off --certified_cdec_rescue off
  --certified_stagnation_graph off --revisit_controller navdp_mixed
  --revisit_adapter legacy_metric --navdp_depth_source monocular_sidecar
  --hybrid_route phase --role_pair_scope consumed_integration
  --role_pair_query_role all
  --role_pair_query_manifest MemNavData/vint_cec_direction_loss5_manifest_20260828.json
  --cec_initial_bearing_alignment first_certified
)
(cd "${LOCAL_ROOT}" && PYTHONPATH="${LOCAL_ROOT}/MemNavData:${LOCAL_ROOT}" \
  "${HAB_PY}" MemNavData/eval_shared_online_role_pairs.py "${contract_args[@]}")

staging=$(mktemp -d)
trap 'rm -rf -- "${staging}"' EXIT
mkdir -p "${staging}/MemNavData"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${staging}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f \
  -name '*.py' -print0)
for name in run_cec_controller_portability_smoke_local.sh bundle_selftest.sh \
  slurm_vint_cec_bearing_alignment_loss5.sbatch \
  slurm_vint_cec_bearing_alignment_loss5_analysis.sbatch \
  VINT_CEC_BEARING_ALIGNMENT_LOSS5_PROTOCOL_20260828.md \
  vint_cec_direction_loss5_manifest_20260828.json; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/MemNavData/${name}" \
    "${staging}/MemNavData/${name}"
done
for component in memnav navdp vint; do
  mkdir -p "${staging}/NavDP/baselines/${component}"
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${staging}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
  if [[ -d "${LOCAL_ROOT}/NavDP/baselines/${component}/configs" ]]; then
    mkdir -p "${staging}/NavDP/baselines/${component}/configs"
    while IFS= read -r -d '' path; do
      cp --preserve=mode,timestamps "${path}" \
        "${staging}/NavDP/baselines/${component}/configs/$(basename "${path}")"
    done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}/configs" \
      -maxdepth 1 -type f -print0)
  fi
done

navdp_runtime_support=(
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dpt.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/__init__.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/attention.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/block.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/drop_path.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/layer_scale.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/mlp.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/patch_embed.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/dinov2_layers/swiglu_ffn.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/util/blocks.py
  NavDP/baselines/navdp/depth_anything/depth_anything_v2/util/transform.py
)
for relative in "${navdp_runtime_support[@]}"; do
  mkdir -p "${staging}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${staging}/${relative}"
done

local_head=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${staging}" "${local_head}" "${FRESH_ROOT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
 if path.is_symlink(): raise SystemExit("bundle symlink: "+str(path))
 if path.is_file() and path.name not in {"SOURCE_BUNDLE.sha256","source_bundle_manifest.json"}:
  files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"vint_cec_bearing_alignment_loss5_bundle_v1_20260828",
 "local_git_head_context":sys.argv[2],"fresh_source_root":sys.argv[3],
 "population":"five outcome-aware formal losses",
 "arms":["anchor_unaligned","native_bearing_aligned","anchor_bearing_aligned"],
 "paper_claim_allowed":False,"files":files,
}
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

local_entries=${staging}/selftest_local.entries
"${MEMNAV_PY}" - "${local_entries}" "${MEMNAV_PY}" "${HAB_PY}" <<'PY'
from pathlib import Path
import sys
path,memnav,habitat=sys.argv[1:]
lines=[
 f"{memnav} import MemNavData.cec_bearing_alignment",
 f"{memnav} import MemNavData.audit_vint_cec_bearing_alignment_cell",
 f"{memnav} import MemNavData.aggregate_vint_cec_bearing_alignment_loss5",
 f"{memnav} import MemNavData.independent_verify_vint_cec_bearing_alignment_loss5",
 f"{habitat} run MemNavData/eval_shared_online_role_pairs.py --contract_dry_run --episode_root /contract/dry/scene0 --scene /contract/dry/scene.glb --scene_identity scene0 --out /contract/dry/out --host 127.0.0.1 --port 18888 --server_backend cec_portability --success_dist 1.0 --max_steps 600 --exec_horizon 8 --trajectory_selector server --trajectory_selector_scope all --leg1_mode shared_trace --leg1_goal_source own --seed 0 --terminal_uturn off --terminal_visual_refine off --deterministic_plan_seeds --retrieval_override off --certified_cdec_rescue off --certified_stagnation_graph off --revisit_controller navdp_mixed --revisit_adapter legacy_metric --navdp_depth_source monocular_sidecar --hybrid_route phase --role_pair_scope consumed_integration --role_pair_query_role all --role_pair_query_manifest MemNavData/vint_cec_direction_loss5_manifest_20260828.json --cec_initial_bearing_alignment first_certified",
]
Path(path).write_text("\n".join(lines)+"\n")
PY
SELFTEST_BUNDLE_SUBPATHS=MemNavData bash \
  "${staging}/MemNavData/bundle_selftest.sh" "${staging}" "${local_entries}"

(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
source_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_manifest_sha=$(sha256sum "${staging}/source_bundle_manifest.json" | awk '{print $1}')
selection_manifest_sha=$(sha256sum \
  "${staging}/MemNavData/vint_cec_direction_loss5_manifest_20260828.json" | \
  awk '{print $1}')
remote_bundle=${REMOTE_BUNDLE_BASE}/vint_cec_direction_loss5_${bundle_manifest_sha:0:16}
remote_stage=${remote_bundle}.partial.$$
if [[ "${DRY_RUN}" == 1 ]]; then
  printf 'DRY_RUN_RUN_ROOT=%s\nDRY_RUN_REMOTE_BUNDLE=%s\nDRY_RUN_SOURCE_RECEIPT_SHA=%s\n' \
    "${RUN_ROOT}" "${remote_bundle}" "${source_receipt_sha}"
  exit 0
fi

remote "test \"\$(sha256sum '${PARENT_FORMAL_SUMMARY}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_FORMAL_SUMMARY_SHA}'"
remote "test \"\$(sha256sum '${FRESH_ROOT}/benchmarks/natural_direction/manifest.json' | awk '{print \$1}')\" = 'aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0'"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"
remote "test -r '${PORTABILITY_ENV_ROOT}/environment_receipt.json' && cd '${PORTABILITY_CHECKPOINT_ROOT}' && sha256sum -c --quiet CHECKPOINTS.sha256"
if remote "test -d '${remote_bundle}' && test \"\$(sha256sum '${remote_bundle}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${source_receipt_sha}' && cd '${remote_bundle}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"; then
  echo "Reusing verified bundle ${remote_bundle}"
else
  remote "test ! -e '${remote_bundle}' && test ! -e '${remote_stage}' && mkdir -p '${remote_stage}'"
  timeout 300 rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${REMOTE_HOST}:${remote_stage}/"
  remote "set -euo pipefail
python3 - '${remote_stage}' <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
memnav='/scratch/lg154/conda-envs/memnav/bin/python'
habitat='/scratch/lg154/conda-envs/habitat/bin/python'
lines=[
 f'{memnav} import MemNavData.cec_bearing_alignment',
 f'{memnav} import MemNavData.audit_vint_cec_bearing_alignment_cell',
 f'{memnav} import MemNavData.aggregate_vint_cec_bearing_alignment_loss5',
 f'{memnav} import MemNavData.independent_verify_vint_cec_bearing_alignment_loss5',
 f'{habitat} run MemNavData/eval_shared_online_role_pairs.py --contract_dry_run --episode_root /contract/dry/scene0 --scene /contract/dry/scene.glb --scene_identity scene0 --out /contract/dry/out --host 127.0.0.1 --port 18888 --server_backend cec_portability --success_dist 1.0 --max_steps 600 --exec_horizon 8 --trajectory_selector server --trajectory_selector_scope all --leg1_mode shared_trace --leg1_goal_source own --seed 0 --terminal_uturn off --terminal_visual_refine off --deterministic_plan_seeds --retrieval_override off --certified_cdec_rescue off --certified_stagnation_graph off --revisit_controller navdp_mixed --revisit_adapter legacy_metric --navdp_depth_source monocular_sidecar --hybrid_route phase --role_pair_scope consumed_integration --role_pair_query_role all --role_pair_query_manifest MemNavData/vint_cec_direction_loss5_manifest_20260828.json --cec_initial_bearing_alignment first_certified',
]
Path('/tmp/vint_cec_dir5_remote.entries').write_text('\n'.join(lines)+'\n')
PY
hab_site_packages=\$(/scratch/lg154/conda-envs/habitat/bin/python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')
hab_requests_vendor=\${hab_site_packages}/pip/_vendor
test -r \"\${hab_requests_vendor}/requests/__init__.py\"
SELFTEST_BUNDLE_SUBPATHS=MemNavData SELFTEST_EXTRA_PYTHONPATH=\"\${hab_requests_vendor}\" bash '${remote_stage}/MemNavData/bundle_selftest.sh' '${remote_stage}' /tmp/vint_cec_dir5_remote.entries
cd '${remote_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256
rm -f /tmp/vint_cec_dir5_remote.entries
chmod -R a-w '${remote_stage}' && mv '${remote_stage}' '${remote_bundle}'"
fi

remote "python3 - '${PARENT_FORMAL_SUMMARY}' '${remote_bundle}/MemNavData/vint_cec_direction_loss5_manifest_20260828.json' <<'PY'
import glob,json,sys
summary,selection=sys.argv[1:]
p=json.load(open(summary)); s=json.load(open(selection))
expected={(q['scene'],q['episode'],q['query_id']) for q in s['queries']}
actual=set()
for cell in p['cells']:
 a=json.load(open(cell['audit_path']))
 for q in a['query_results']:
  if q['analysis_role']=='revisit' and q['native_success']==1 and q['grant_success']==0:
   actual.add((q['scene'],q['episode'],q['query_id']))
assert actual==expected and len(actual)==5
PY"
remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}/sealed_inputs' '${RUN_ROOT}/evaluation' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs && cp '${remote_bundle}/MemNavData/VINT_CEC_BEARING_ALIGNMENT_LOSS5_PROTOCOL_20260828.md' '${RUN_ROOT}/sealed_inputs/' && cp '${remote_bundle}/MemNavData/vint_cec_direction_loss5_manifest_20260828.json' '${RUN_ROOT}/sealed_inputs/' && sha256sum '${PARENT_FORMAL_SUMMARY}' >'${RUN_ROOT}/sealed_inputs/parent_formal_summary.sha256' && chmod -R a-w '${RUN_ROOT}/sealed_inputs'"

source_receipt=${remote_bundle}/SOURCE_BUNDLE.sha256
common="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${source_receipt},EXPECTED_SOURCE_RECEIPT_SHA=${source_receipt_sha},FRESH_ROOT=${FRESH_ROOT},RUN_ROOT=${RUN_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},PORTABILITY_ENV_ROOT=${PORTABILITY_ENV_ROOT},PORTABILITY_CHECKPOINT_ROOT=${PORTABILITY_CHECKPOINT_ROOT},SELECTION_MANIFEST_SHA=${selection_manifest_sha}"
cell_sbatch=${remote_bundle}/MemNavData/slurm_vint_cec_bearing_alignment_loss5.sbatch
analysis_sbatch=${remote_bundle}/MemNavData/slurm_vint_cec_bearing_alignment_loss5_analysis.sbatch
remote "sbatch --test-only --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array=0 --export='${common}' '${cell_sbatch}' >/dev/null"
remote "sbatch --test-only --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:20:00 --export='${common}' '${analysis_sbatch}' >/dev/null"

gate_raw=$(remote "sbatch --parsable --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array=0 --export='${common}' '${cell_sbatch}'")
gate_id=${gate_raw%%;*}; [[ "${gate_id}" =~ ^[0-9]+$ ]] || fail "bad gate job"
remaining_raw=$(remote "sbatch --parsable --partition=h100_tandon,a100_tandon --account=torch_pr_769_tandon_advanced --qos=gpu48 --gres=gpu:1 --time=01:00:00 --array=1-4%2 --dependency=afterok:${gate_id} --kill-on-invalid-dep=yes --export='${common}' '${cell_sbatch}'")
remaining_id=${remaining_raw%%;*}; [[ "${remaining_id}" =~ ^[0-9]+$ ]] || fail "bad remaining job"
analysis_raw=$(remote "sbatch --parsable --partition=cpu_short --account=torch_pr_769_tandon_advanced --time=00:20:00 --dependency=afterany:${gate_id}:${remaining_id} --kill-on-invalid-dep=yes --export='${common}' '${analysis_sbatch}'")
analysis_id=${analysis_raw%%;*}; [[ "${analysis_id}" =~ ^[0-9]+$ ]] || fail "bad analysis job"
remote "python3 - '${RUN_ROOT}/submission_receipt.json' '${gate_id}' '${remaining_id}' '${analysis_id}' '${remote_bundle}' '${source_receipt_sha}' <<'PY'
import json,sys
path,gate,remaining,analysis,bundle,receipt=sys.argv[1:]
payload={'schema_version':'vint_cec_bearing_alignment_loss5_submission_v1_20260828','gate_job':int(gate),'remaining_array':int(remaining),'analysis_job':int(analysis),'source_bundle':bundle,'source_receipt_sha256':receipt,'scope':'consumed mechanism; not paper SR'}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
chmod a-w '${RUN_ROOT}/submission_receipt.json'"

printf 'RUN_ROOT=%s\nSOURCE_BUNDLE=%s\nSOURCE_RECEIPT_SHA=%s\nGATE=%s\nREMAINING=%s\nANALYSIS=%s\n' \
  "${RUN_ROOT}" "${remote_bundle}" "${source_receipt_sha}" \
  "${gate_id}" "${remaining_id}" "${analysis_id}"
