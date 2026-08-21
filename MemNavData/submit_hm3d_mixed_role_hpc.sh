#!/usr/bin/env bash
# Submit construction, role-unknown paired evaluation, summary, and verification.

set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
RUN_TAG=${RUN_TAG:-hm3d_mixed_role_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_mixed_role_20260818/${RUN_TAG}}
CONSTRUCT_CONCURRENCY=${CONSTRUCT_CONCURRENCY:-6}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-6}

HM3D_RUNTIME_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/shared_online_nnr_11458cb2b75ee334
HM3D_RUNTIME_RECEIPT_SHA=31b3e087b855e0220f6821ad96e6f5e74114bc12dc6c3afa6f7f79150dfb4575
EXEC_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_learned_overlay_575950199e157fdf
EXEC_SOURCE_RECEIPT_SHA=575950199e157fdfc822ef2be71305418b7b746eac65efe813563feee3fe2d39
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
PARENT_RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_heldout_val10_runtime_repair_20260816/hm3d_heldout_val10_rt_20260816T1345Z
PARENT_MANIFEST_SOURCE=${PARENT_RUN_ROOT}/data_manifest.json
PARENT_MANIFEST_SHA=62bc6299203da709e65787c735a531974905f2ab8e940f72e91318914d949c89
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
SOURCE_OVERLAY_BYTES=128854888448
OVERLAY_EPISODE_ROOT=/mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg

files=(
  MemNavData/hm3d_mixed_role_protocol_20260818.json
  MemNavData/hm3d_materialize_existing_online_a.py
  MemNavData/run_hm3d_mixed_role_construct_scene.sh
  MemNavData/finalize_hm3d_mixed_role.py
  MemNavData/summarize_hm3d_mixed_role.py
  MemNavData/verify_hm3d_mixed_role.py
  MemNavData/slurm_hm3d_mixed_role_construct.sbatch
  MemNavData/slurm_hm3d_mixed_role_finalize.sbatch
  MemNavData/slurm_hm3d_mixed_role_analysis.sbatch
)

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() { ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "bad run tag"
[[ "${CONSTRUCT_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad construction concurrency"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad evaluation concurrency"
for file in "${files[@]}"; do
  [[ -f "${LOCAL_ROOT}/${file}" ]] || fail "missing ${file}"
done
bundle_key=$(for file in "${files[@]}"; do
  sha256sum "${LOCAL_ROOT}/${file}"
done | sha256sum | awk '{print substr($1,1,16)}')
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_mixed_role_${bundle_key}
TASK_STAGE=${TASK_ROOT}.partial.$$

[[ "$(remote 'id -un')" == yz11502 ]] || fail "wrong remote identity"
remote "set -euo pipefail
test ! -e '${RUN_ROOT}'
test \"\$(sha256sum '${HM3D_RUNTIME_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${HM3D_RUNTIME_RECEIPT_SHA}'
test \"\$(sha256sum '${EXEC_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${EXEC_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${DEPENDENCY_RECEIPT_SHA}'
test \"\$(sha256sum '${PARENT_MANIFEST_SOURCE}' | awk '{print \$1}')\" = '${PARENT_MANIFEST_SHA}'
test \"\$(stat -c %s '${SOURCE_OVERLAY}')\" -eq '${SOURCE_OVERLAY_BYTES}'"

if ! remote "test -d '${TASK_ROOT}'"; then
  remote "test ! -e '${TASK_STAGE}' && mkdir -p '${TASK_STAGE}'"
  (cd "${LOCAL_ROOT}" && rsync -e "ssh -o BatchMode=yes" \
    -aR --chmod=Fugo=r,Dugo=rx "${files[@]}" \
    "${REMOTE_HOST}:${TASK_STAGE}/")
  remote "set -euo pipefail
cd '${TASK_STAGE}'
find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum > SOURCE_BUNDLE.sha256
sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
chmod -R a-w '${TASK_STAGE}'
mv '${TASK_STAGE}' '${TASK_ROOT}'"
fi
TASK_RECEIPT_SHA=$(remote "sha256sum '${TASK_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")
[[ "${TASK_RECEIPT_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "bad task receipt"
remote "cd '${TASK_ROOT}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"

remote "mkdir -p '${RUN_ROOT}/sealed_inputs' '${RUN_ROOT}/logs'"
remote "cp '${PARENT_MANIFEST_SOURCE}' '${RUN_ROOT}/sealed_inputs/parent_hm3d_manifest.json' && \
  printf '%s  %s\n' '${PARENT_MANIFEST_SHA}' parent_hm3d_manifest.json > \
    '${RUN_ROOT}/sealed_inputs/parent_hm3d_manifest.json.sha256' && \
  chmod -R a-w '${RUN_ROOT}/sealed_inputs'"
PARENT_MANIFEST=${RUN_ROOT}/sealed_inputs/parent_hm3d_manifest.json

CONSTRUCT=${TASK_ROOT}/MemNavData/slurm_hm3d_mixed_role_construct.sbatch
FINALIZE=${TASK_ROOT}/MemNavData/slurm_hm3d_mixed_role_finalize.sbatch
EVAL=${EXEC_SOURCE_ROOT}/MemNavData/slurm_paper_role_pair_eval.sbatch
ANALYSIS=${TASK_ROOT}/MemNavData/slurm_hm3d_mixed_role_analysis.sbatch
common="ALL,TASK_ROOT=${TASK_ROOT},EXPECTED_TASK_RECEIPT_SHA=${TASK_RECEIPT_SHA},HM3D_RUNTIME_ROOT=${HM3D_RUNTIME_ROOT},EXPECTED_HM3D_RUNTIME_RECEIPT_SHA=${HM3D_RUNTIME_RECEIPT_SHA},EXEC_SOURCE_ROOT=${EXEC_SOURCE_ROOT},EXPECTED_EXEC_SOURCE_RECEIPT_SHA=${EXEC_SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},PARENT_RUN_ROOT=${PARENT_RUN_ROOT},PARENT_MANIFEST=${PARENT_MANIFEST}"
eval_exports="ALL,SOURCE_ROOT=${EXEC_SOURCE_ROOT},SOURCE_RECEIPT=${EXEC_SOURCE_ROOT}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${EXEC_SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${DEPENDENCY_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},FINAL14_POPULATION_MODE=1,INCLUDE_LEARNED_PI3X=0,MAX_POPULATION_PER_PROTOCOL=27,SOURCE_OVERLAY=${SOURCE_OVERLAY},EXPECTED_SOURCE_OVERLAY_BYTES=${SOURCE_OVERLAY_BYTES},OVERLAY_EPISODE_ROOT=${OVERLAY_EPISODE_ROOT}"

remote "sbatch --test-only --time=01:00:00 --array=0 --export='${common}' '${CONSTRUCT}' >/dev/null"
remote "sbatch --test-only --export='${common}' '${FINALIZE}' >/dev/null"
remote "sbatch --test-only --time=01:00:00 --array=0 --export='${eval_exports}' '${EVAL}' >/dev/null"
remote "sbatch --test-only --export='${common},MODE=summary' '${ANALYSIS}' >/dev/null"

construct_raw=$(remote "sbatch --parsable --time=01:00:00 --array=0-8%${CONSTRUCT_CONCURRENCY} --export='${common}' '${CONSTRUCT}'")
construct_id=${construct_raw%%;*}; [[ "${construct_id}" =~ ^[0-9]+$ ]] || fail "bad construct job"
finalize_raw=$(remote "sbatch --parsable --dependency=afterok:${construct_id} --kill-on-invalid-dep=yes --export='${common}' '${FINALIZE}'")
finalize_id=${finalize_raw%%;*}; [[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad finalize job"
eval_raw=$(remote "sbatch --parsable --time=01:00:00 --array=0-26%${EVAL_CONCURRENCY} --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${eval_exports}' '${EVAL}'")
eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval job"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${common},MODE=summary' '${ANALYSIS}'")
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${common},MODE=verify' '${ANALYSIS}'")
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verify job"

remote "/scratch/lg154/conda-envs/memnav/bin/python - '${RUN_ROOT}/submission.json' \
  '${TASK_ROOT}' '${TASK_RECEIPT_SHA}' '${construct_id}' '${finalize_id}' \
  '${eval_id}' '${summary_id}' '${verify_id}'" <<'PY'
import json,sys
path,bundle,sha,construct,finalize,evaluation,summary,verify=sys.argv[1:]
with open(path,"x") as handle:
    json.dump({
      "schema_version":"hm3d_mixed_role_submission_v1_20260818",
      "scope":"same-scene HM3D mixed Novel/Revisit role-unknown safety extension",
      "training_free":True,"scene_reuse_disclosed":True,
      "new_scene_generalization_claim":False,
      "query_policy_outcomes_read_at_submission":False,
      "task_bundle":bundle,"task_receipt_sha256":sha,
      "jobs":{"construction_array":int(construct),"population_finalize":int(finalize),
              "paired_evaluation_array":int(evaluation),"summary":int(summary),
              "independent_verification":int(verify)},
      "resources":{"construction_time_limit":"01:00:00",
                   "evaluation_time_limit":"01:00:00"},
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
remote "sha256sum '${RUN_ROOT}/submission.json' > '${RUN_ROOT}/submission.json.sha256'"

echo "RUN_ROOT=${RUN_ROOT}"
echo "TASK_ROOT=${TASK_ROOT}"
echo "construct=${construct_id} finalize=${finalize_id} eval=${eval_id} summary=${summary_id} verify=${verify_id}"
