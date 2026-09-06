#!/usr/bin/env bash
# Submit the frozen HM3D Table-1 fixed-bearing vs full-metric Revisit gate.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
LOCAL_MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
LOCAL_HAB_PY=/home/asus/miniconda3/envs/habitat/bin/python
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_metric_distance_revisit_20260901
CONSTRUCTION_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_table1_fresh_query_reserve_20260829/construction_20260828T212552Z_bb757914
CONSTRUCTION_VERIFICATION=${CONSTRUCTION_RUN}/hm3d_table1_fresh_query_verification.json
BENCH_ROOT=${CONSTRUCTION_RUN}/population/natural_direction
SOURCE_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
PARENT_MANIFEST=${SOURCE_RUN_ROOT}/sealed_inputs/parent_manifest.json
EXPECTED_PARENT_MANIFEST_SHA=a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5
EXPECTED_MANIFEST_SHA=f82dbcbc6255219aae94b6d77bffdfa454f36835cf803a70df5cf8616193ad01
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
CONCURRENCY=${CONCURRENCY:-2}
RUN_TAG=${RUN_TAG:-formal_$(date -u +%Y%m%dT%H%M%SZ)}
RECEIPT_BASENAME=${RECEIPT_BASENAME:-HM3D_TABLE1_METRIC_DISTANCE_REVISIT_SUBMISSION_20260901.json}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(ssh -G "${SSH_ALIAS}" 2>/dev/null | awk '$1=="controlpath"{value=$2} END{print value}')}
cd "${ROOT}"

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -tt -o BatchMode=yes -o ControlMaster=no \
    -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}

[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency"
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"
required=(
  MemNavData/HM3D_TABLE1_METRIC_DISTANCE_REVISIT_PROTOCOL_20260901.md
  MemNavData/hm3d_table1_metric_distance_revisit_protocol_20260901.json
  MemNavData/run_hm3d_table1_metric_distance_revisit_pair.py
  MemNavData/aggregate_hm3d_table1_metric_distance_revisit_pair.py
  MemNavData/independent_verify_hm3d_table1_metric_distance_revisit_pair.py
  MemNavData/test_hm3d_table1_metric_distance_revisit_pair.py
  MemNavData/test_revisit_bearing_adapter.py
  MemNavData/test_first40_local_pose_scale.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_hm3d_table1_metric_distance_revisit_pair.sbatch
  MemNavData/slurm_hm3d_table1_metric_distance_revisit_analysis.sbatch
  MemNavData/slurm_port_pair.sh
  MemNavData/slurm_safe_submit.sh
)
for path in "${required[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${path}"
done

export PYTHONPATH=${ROOT}:${ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${LOCAL_MEMNAV_PY}" -m py_compile \
  MemNavData/run_hm3d_table1_metric_distance_revisit_pair.py \
  MemNavData/aggregate_hm3d_table1_metric_distance_revisit_pair.py \
  MemNavData/independent_verify_hm3d_table1_metric_distance_revisit_pair.py \
  NavDP/baselines/memnav/policy_agent.py
"${LOCAL_HAB_PY}" -m py_compile \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/eval_2leg_habitat.py
"${LOCAL_MEMNAV_PY}" -m json.tool \
  MemNavData/hm3d_table1_metric_distance_revisit_protocol_20260901.json \
  >/dev/null
"${LOCAL_MEMNAV_PY}" -m pytest -q \
  MemNavData/test_revisit_bearing_adapter.py \
  MemNavData/test_first40_local_pose_scale.py \
  MemNavData/test_hm3d_table1_metric_distance_revisit_pair.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_table1_metric_distance_revisit_pair.sbatch \
  MemNavData/slurm_hm3d_table1_metric_distance_revisit_analysis.sbatch \
  MemNavData/submit_hm3d_table1_metric_distance_revisit_hpc.sh
source MemNavData/slurm_safe_submit.sh
lint_sbatch_template \
  MemNavData/slurm_hm3d_table1_metric_distance_revisit_pair.sbatch || \
  fail "pair sbatch lint failed"
lint_sbatch_template \
  MemNavData/slurm_hm3d_table1_metric_distance_revisit_analysis.sbatch || \
  fail "analysis sbatch lint failed"

echo '[gate] authoritative remote identity and frozen population'
remote_identity=$(remote 'id -un' | tr -d '\r')
[[ "${remote_identity}" == yz11502 ]] || fail "wrong remote identity"
readarray -t gate < <(remote "python - '${CONSTRUCTION_VERIFICATION}' '${BENCH_ROOT}/manifest.json' '${PARENT_MANIFEST}' <<'PY'
import hashlib,json,sys
verification=json.load(open(sys.argv[1])); manifest=sys.argv[2]
if verification.get('verified') is not True or verification.get('formal_policy_evaluation_authorized') is not True:
 raise SystemExit('construction did not authorize policy evaluation')
digest=lambda p: hashlib.sha256(open(p,'rb').read()).hexdigest()
if digest(manifest) != verification.get('benchmark_manifest_sha256'):
 raise SystemExit('benchmark manifest changed')
rows=json.load(open(manifest))['episodes']
if len(rows) != 28 or len({str(row['scene']) for row in rows}) != 21:
 raise SystemExit('frozen denominator changed')
parent=json.load(open(sys.argv[3]))
ranks=sorted({int(row['final14_scene_rank']) for row in rows})
if not ranks or min(ranks)<0 or max(ranks)>=len(parent['scenes']):
 raise SystemExit('scene-rank mapping invalid')
print(len(rows)); print(len({str(row['scene']) for row in rows}))
print(digest(manifest)); print(digest(sys.argv[1])); print(digest(sys.argv[3]))
print(','.join(map(str,ranks))); print(len(parent['scenes']))
PY" | tr -d '\r')
[[ "${#gate[@]}" -eq 7 ]] || fail "population gate receipt incomplete"
histories=${gate[0]}; scenes=${gate[1]}; manifest_sha=${gate[2]}
construction_sha=${gate[3]}; parent_sha=${gate[4]}; scene_spec=${gate[5]}
parent_scenes=${gate[6]}
[[ "${histories}" == 28 && "${scenes}" == 21 ]] || fail "denominator changed"
[[ "${manifest_sha}" == "${EXPECTED_MANIFEST_SHA}" ]] || fail "manifest changed"
[[ "${parent_sha}" == "${EXPECTED_PARENT_MANIFEST_SHA}" ]] || fail "parent changed"
[[ "${scene_spec}" =~ ^[0-9]+(,[0-9]+)*$ ]] || fail "bad scene index set"
[[ "${parent_scenes}" =~ ^[1-9][0-9]*$ ]] || fail "bad parent scene count"

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
mkdir -p "${staging}/MemNavData"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${staging}/MemNavData/$(basename "${path}")"
done < <(find "${ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for path in "${required[@]}"; do
  case "${path}" in
    *.py) ;;
    *) cp --preserve=mode,timestamps "${path}" \
         "${staging}/MemNavData/$(basename "${path}")" ;;
  esac
done
for component in memnav navdp; do
  mkdir -p "${staging}/NavDP/baselines/${component}"
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${staging}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
  if [[ -d "${ROOT}/NavDP/baselines/${component}/configs" ]]; then
    mkdir -p "${staging}/NavDP/baselines/${component}/configs"
    cp -a "${ROOT}/NavDP/baselines/${component}/configs/." \
      "${staging}/NavDP/baselines/${component}/configs/"
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
  cp --preserve=mode,timestamps "${ROOT}/${relative}" "${staging}/${relative}"
done
local_head=$(git -C "${ROOT}" rev-parse HEAD)
"${LOCAL_MEMNAV_PY}" - "${staging}" "${local_head}" "${manifest_sha}" \
  "${construction_sha}" "${scene_spec}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,head,manifest,construction,scene_spec=sys.argv[1:]
base=Path(root); files={}
for path in sorted(base.rglob('*')):
 if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
 if path.is_file() and path.name not in {'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
  files[path.relative_to(base).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 'schema_version':'hm3d_table1_metric_distance_revisit_bundle_v1_20260901',
 'local_git_head_context':head,'benchmark_manifest_sha256':manifest,
 'construction_verification_sha256':construction,
 'histories':28,'scene_clusters':21,'scene_rank_spec':scene_spec,
 'arms':['fixed_2p5m','full_metric'],'runtime_role_visibility':'none',
 'claim_boundary':'consumed-population promotion gate','files':files,
}
(base/'source_bundle_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
task_receipt_sha=$(sha256sum "${staging}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
task_root=${REMOTE_BUNDLES}/hm3d_table1_metric_distance_${task_receipt_sha:0:16}
task_stage=${task_root}.partial.$$
task_receipt=${task_root}/SOURCE_BUNDLE.sha256
run_root=${REMOTE_RESULTS}/${RUN_TAG}

echo '[gate] remote dependencies'
remote "set -euo pipefail
test \"\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_BASE_RECEIPT_SHA}'
cd '${BASE_SOURCE_ROOT}' && sha256sum -c --quiet '${BASE_RECEIPT}'
test \"\$(sha256sum '${PARENT_MANIFEST}' | awk '{print \$1}')\" = '${EXPECTED_PARENT_MANIFEST_SHA}'
test \"\$(sha256sum '${BENCH_ROOT}/manifest.json' | awk '{print \$1}')\" = '${EXPECTED_MANIFEST_SHA}'"

echo '[stage] immutable source bundle'
if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_receipt}' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${staging}/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

protocol=${task_root}/MemNavData/hm3d_table1_metric_distance_revisit_protocol_20260901.json
pair=${task_root}/MemNavData/slurm_hm3d_table1_metric_distance_revisit_pair.sbatch
analysis=${task_root}/MemNavData/slurm_hm3d_table1_metric_distance_revisit_analysis.sbatch
remote "test ! -e '${run_root}' && mkdir -p '${run_root}/sealed_inputs' '${run_root}/logs' '${run_root}/smoke' '${run_root}/formal' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
cp '${CONSTRUCTION_VERIFICATION}' '${run_root}/sealed_inputs/'
cp '${BENCH_ROOT}/manifest.json' '${run_root}/sealed_inputs/benchmark_manifest.json'
cp '${task_root}/MemNavData/HM3D_TABLE1_METRIC_DISTANCE_REVISIT_PROTOCOL_20260901.md' '${run_root}/sealed_inputs/'
cp '${protocol}' '${run_root}/sealed_inputs/'
sha256sum '${CONSTRUCTION_VERIFICATION}' '${BENCH_ROOT}/manifest.json' '${PARENT_MANIFEST}' '${protocol}' >'${run_root}/sealed_inputs/experiment_inputs.sha256'
chmod -R a-w '${run_root}/sealed_inputs'"

common="ALL,TASK_ROOT=${task_root},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},FORMAL_RUN_ROOT=${run_root},BENCH_ROOT=${BENCH_ROOT},PARENT_MANIFEST=${PARENT_MANIFEST},PROTOCOL=${protocol},CONSTRUCTION_VERIFICATION=${CONSTRUCTION_VERIFICATION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${construction_sha},EXPECTED_MANIFEST_SHA=${manifest_sha}"

echo '[gate] exact container imports, tests, and Slurm test-only'
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${task_root}/NavDP/baselines/memnav:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/memnav/bin/python -m pytest -q -p no:cacheprovider '${task_root}/MemNavData/test_revisit_bearing_adapter.py' '${task_root}/MemNavData/test_first40_local_pose_scale.py' '${task_root}/MemNavData/test_hm3d_table1_metric_distance_revisit_pair.py'"
remote "singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${task_root}:${task_root}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -c 'from pathlib import Path; [compile(Path(path).read_bytes(),path,\"exec\") for path in (\"${task_root}/MemNavData/eval_shared_online_role_pairs.py\",\"${task_root}/MemNavData/eval_2leg_habitat.py\",\"${task_root}/MemNavData/run_hm3d_table1_metric_distance_revisit_pair.py\")]'"
remote "sbatch --test-only --array=0 --export='${common},PHASE=smoke' '${pair}' >/dev/null"
remote "sbatch --test-only --array='${scene_spec}%${CONCURRENCY}' --export='${common},PHASE=formal' '${pair}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=aggregate' '${analysis}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=verify' '${analysis}' >/dev/null"

echo '[submit] result-blind smoke -> formal -> aggregate -> independent verify'
smoke_raw=$(remote "sbatch --parsable --array=0 --export='${common},PHASE=smoke' '${pair}'" | tr -d '\r')
smoke=${smoke_raw%%;*}
formal_raw=$(remote "sbatch --parsable --array='${scene_spec}%${CONCURRENCY}' --dependency=afterok:${smoke} --kill-on-invalid-dep=yes --export='${common},PHASE=formal' '${pair}'" | tr -d '\r')
formal=${formal_raw%%;*}
aggregate_raw=$(remote "sbatch --parsable --dependency=afterok:${formal} --kill-on-invalid-dep=yes --export='${common},MODE=aggregate' '${analysis}'" | tr -d '\r')
aggregate=${aggregate_raw%%;*}
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${aggregate} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${analysis}'" | tr -d '\r')
verify=${verify_raw%%;*}
for id in "${smoke}" "${formal}" "${aggregate}" "${verify}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid submitted job id: ${id}"
done

receipt=MemNavData/${RECEIPT_BASENAME}
[[ ! -e "${receipt}" ]] || fail "local submission receipt already exists"
"${LOCAL_MEMNAV_PY}" - "${receipt}" "${run_root}" "${task_root}" \
  "${task_receipt_sha}" "${manifest_sha}" "${construction_sha}" \
  "${scene_spec}" "${smoke}" "${formal}" "${aggregate}" "${verify}" <<'PY'
import json,sys
(path,run,bundle,bundle_sha,manifest,construction,scene_spec,
 smoke,formal,aggregate,verify)=sys.argv[1:]
payload={
 'schema_version':'hm3d_table1_metric_distance_revisit_submission_v1_20260901',
 'run_root':run,'task_bundle':bundle,'task_receipt_sha256':bundle_sha,
 'benchmark_manifest_sha256':manifest,
 'construction_verification_sha256':construction,
 'histories':28,'scene_clusters':21,'formal_scene_rank_spec':scene_spec,
 'arms':['fixed_2p5m','full_metric'],'closed_loop_rollouts':56,
 'partial_policy_outcomes_read_at_submission':False,
 'jobs':{'smoke':int(smoke),'formal':int(formal),
         'aggregate':int(aggregate),'verify':int(verify)},
}
open(path,'x').write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY
scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath="${SSH_CONTROL_PATH}" \
  "${ROOT}/${receipt}" "${SSH_ALIAS}:${run_root}/submission.json"
remote "sha256sum '${run_root}/submission.json' >'${run_root}/submission.json.sha256' && chmod a-w '${run_root}/submission.json' '${run_root}/submission.json.sha256'"
printf 'RUN_ROOT=%s\nTASK_ROOT=%s\nSCENE_SPEC=%s\nSMOKE=%s\nFORMAL=%s\nAGGREGATE=%s\nVERIFY=%s\n' \
  "${run_root}" "${task_root}" "${scene_spec}" "${smoke}" \
  "${formal}" "${aggregate}" "${verify}"
