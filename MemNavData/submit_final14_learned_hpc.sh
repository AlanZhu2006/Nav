#!/usr/bin/env bash
# Submit the one-shot Final14 five-arm confirmation after explicit authorization.

set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
SOURCE_BUNDLE=${SOURCE_BUNDLE:?set immutable remote Final14 source bundle}
EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}
LOCAL_SCENE_BUDGET=${LOCAL_SCENE_BUDGET:-${LOCAL_ROOT}/.diagnostics/mp3d_scene_budget_20260816/scene_budget.json}
RUN_TAG=${RUN_TAG:-final14_learned_$(date -u +%Y%m%dT%H%M%SZ)}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
CONCURRENCY=${CONCURRENCY:-8}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}

BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
PI3X_SNAPSHOT=/scratch/yz11502/Research/model_assets/pi3x_69972d6e1c4492c
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
EXPECTED_SOURCE_OVERLAY_BYTES=128854888448
OVERLAY_EPISODE_ROOT=/mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg
EXPECTED_PI3X_MODEL_SHA=69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a
EXPECTED_PI3X_PROOF_SHA=1a05aaa7cf75296cb68e32f9ea57fba6bcce2b9f57313a8cede05b7c7b0cffdd
EXPECTED_SCENE_BUDGET_SHA=779e2d7d63faa0f9b9e735680b1d620f04428c11a57ac83158933306b62407ef
EXPECTED_PARENT_PROTOCOL_SHA=3d1ebc6ef429fd16df4d550eda52eceb55d7b15fd181a5c00c0b8f971f7aaa32
EXPECTED_AMENDMENT_SHA=21189b7596403ab19d08576e509e2169a3c9a024d42fbe47e5faa3eb402afbf8
BASE_MANIFEST_SHA=b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() { ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "bad RUN_TAG"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad concurrency"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad eval concurrency"

[[ "$(sha256sum "${LOCAL_SCENE_BUDGET}" | awk '{print $1}')" == \
  "${EXPECTED_SCENE_BUDGET_SHA}" ]] || fail "scene budget changed"
[[ "$(sha256sum "${LOCAL_ROOT}/MemNavData/FINAL14_CEC_ROLE_FREE_CONFIRMATION_PROTOCOL_20260816.md" | awk '{print $1}')" == \
  "${EXPECTED_PARENT_PROTOCOL_SHA}" ]] || fail "parent protocol changed"
[[ "$(sha256sum "${LOCAL_ROOT}/MemNavData/FINAL14_LEARNED_RELOCALIZER_PROSPECTIVE_AMENDMENT_20260817.md" | awk '{print $1}')" == \
  "${EXPECTED_AMENDMENT_SHA}" ]] || fail "learned amendment changed"

actual_user=$(remote "id -un")
[[ "${actual_user}" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity differs: ${actual_user}"
remote "set -euo pipefail
test ! -e '${RUN_ROOT}'
test \"\$(sha256sum '${SOURCE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${EXPECTED_SOURCE_RECEIPT_SHA}'
cd '${SOURCE_BUNDLE}'
sha256sum -c SOURCE_BUNDLE.sha256 --quiet
test \"\$(find . -perm /222 | wc -l)\" -eq 0
test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'
test \"\$(sha256sum '${PI3X_SNAPSHOT}/model.safetensors' | awk '{print \$1}')\" = '${EXPECTED_PI3X_MODEL_SHA}'
test \"\$(sha256sum '${SOURCE_BUNDLE}/pi3x_deployment/deployment_manifest.json' | awk '{print \$1}')\" = '${EXPECTED_PI3X_PROOF_SHA}'
test \"\$(sha256sum '${SOURCE_BUNDLE}/MemNavData/strict_graph_blind_20260806.json' | awk '{print \$1}')\" = '${BASE_MANIFEST_SHA}'"
remote "test -r '${SOURCE_BUNDLE}/MemNavData/verify_portable_checksum_manifest.py' && \
  /scratch/lg154/conda-envs/memnav/bin/python \
    '${SOURCE_BUNDLE}/MemNavData/verify_portable_checksum_manifest.py' \
    '${SOURCE_BUNDLE}/pi3x_deployment/OUTPUTS.sha256' --quiet"
remote "test -f '${BASE_SIF}' && test -f '${SOURCE_OVERLAY}' && \
  test \"\$(stat -c %s '${SOURCE_OVERLAY}')\" -eq '${EXPECTED_SOURCE_OVERLAY_BYTES}'"

SOURCE_RECEIPT=${SOURCE_BUNDLE}/SOURCE_BUNDLE.sha256
COLLECT=${SOURCE_BUNDLE}/MemNavData/slurm_paper_online_a_collect.sbatch
SUMMARY=${SOURCE_BUNDLE}/MemNavData/slurm_paper_online_a_summary.sbatch
EVAL=${SOURCE_BUNDLE}/MemNavData/slurm_paper_role_pair_eval.sbatch
PAIR_SUMMARY=${SOURCE_BUNDLE}/MemNavData/slurm_paper_role_pair_summary.sbatch
VERIFY=${SOURCE_BUNDLE}/MemNavData/slurm_paper_role_pair_verify.sbatch
PI3X_ROOT=${SOURCE_BUNDLE}/third_party/Pi3
PI3X_PROOF_MANIFEST=${SOURCE_BUNDLE}/pi3x_deployment/deployment_manifest.json

# Slurm resource/partition validation happens before the ledger is copied or
# interpreted.  No Final14 identity has been opened at this point.
test_exports="ALL,SOURCE_ROOT=${SOURCE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},MANIFEST=/dev/null,EXPECTED_MANIFEST_SHA=${BASE_MANIFEST_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},FINAL14_POPULATION_MODE=1,INCLUDE_LEARNED_PI3X=1,EXPECTED_PI3X_MODEL_SHA=${EXPECTED_PI3X_MODEL_SHA},EXPECTED_PI3X_PROOF_SHA=${EXPECTED_PI3X_PROOF_SHA},PI3X_ROOT=${PI3X_ROOT},PI3X_SNAPSHOT=${PI3X_SNAPSHOT},PI3X_PROOF_MANIFEST=${PI3X_PROOF_MANIFEST},MAX_POPULATION_PER_PROTOCOL=42,SOURCE_OVERLAY=${SOURCE_OVERLAY},EXPECTED_SOURCE_OVERLAY_BYTES=${EXPECTED_SOURCE_OVERLAY_BYTES},OVERLAY_EPISODE_ROOT=${OVERLAY_EPISODE_ROOT}"
remote "sbatch --test-only --array=0 --export='${test_exports}' '${COLLECT}' >/dev/null"
remote "sbatch --test-only --export='${test_exports}' '${SUMMARY}' >/dev/null"
remote "sbatch --test-only --array=0 --export='${test_exports}' '${EVAL}' >/dev/null"
remote "sbatch --test-only --export='${test_exports}' '${PAIR_SUMMARY}' >/dev/null"
remote "sbatch --test-only --export='${test_exports}' '${VERIFY}' >/dev/null"

remote "mkdir -p '${RUN_ROOT}/sealed_inputs' '${RUN_ROOT}/logs'"
rsync -e "ssh -o BatchMode=yes" -a --chmod=Fugo=r \
  "${LOCAL_SCENE_BUDGET}" \
  "${REMOTE_HOST}:${RUN_ROOT}/sealed_inputs/scene_budget.json.partial"
remote "set -euo pipefail
test \"\$(sha256sum '${RUN_ROOT}/sealed_inputs/scene_budget.json.partial' | awk '{print \$1}')\" = '${EXPECTED_SCENE_BUDGET_SHA}'
mv '${RUN_ROOT}/sealed_inputs/scene_budget.json.partial' '${RUN_ROOT}/sealed_inputs/scene_budget.json'
singularity exec --overlay '${SOURCE_OVERLAY}:ro' \
  -B /scratch/lg154 -B /scratch/yz11502 '${BASE_SIF}' \
  env PYTHONPATH='${SOURCE_BUNDLE}:${SOURCE_BUNDLE}/MemNavData' \
  /scratch/lg154/conda-envs/memnav/bin/python \
  '${SOURCE_BUNDLE}/MemNavData/freeze_final14_source_manifest.py' \
  --base-manifest '${SOURCE_BUNDLE}/MemNavData/strict_graph_blind_20260806.json' \
  --scene-budget '${RUN_ROOT}/sealed_inputs/scene_budget.json' \
  --episode-root '${OVERLAY_EPISODE_ROOT}' \
  --out '${RUN_ROOT}/sealed_inputs/final14_source_manifest.json' \
  --receipt '${RUN_ROOT}/sealed_inputs/final14_source_manifest_receipt.json'
chmod -R a-w '${RUN_ROOT}/sealed_inputs'"

MANIFEST=${RUN_ROOT}/sealed_inputs/final14_source_manifest.json
manifest_sha=$(remote "sha256sum '${MANIFEST}' | awk '{print \$1}'")
[[ "${manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "bad Final14 manifest hash"

exports="ALL,SOURCE_ROOT=${SOURCE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},MANIFEST=${MANIFEST},EXPECTED_MANIFEST_SHA=${manifest_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},FINAL14_POPULATION_MODE=1,INCLUDE_LEARNED_PI3X=1,EXPECTED_PI3X_MODEL_SHA=${EXPECTED_PI3X_MODEL_SHA},EXPECTED_PI3X_PROOF_SHA=${EXPECTED_PI3X_PROOF_SHA},PI3X_ROOT=${PI3X_ROOT},PI3X_SNAPSHOT=${PI3X_SNAPSHOT},PI3X_PROOF_MANIFEST=${PI3X_PROOF_MANIFEST},MAX_POPULATION_PER_PROTOCOL=42,SOURCE_OVERLAY=${SOURCE_OVERLAY},EXPECTED_SOURCE_OVERLAY_BYTES=${EXPECTED_SOURCE_OVERLAY_BYTES},OVERLAY_EPISODE_ROOT=${OVERLAY_EPISODE_ROOT}"

collect_raw=$(remote "sbatch --parsable --array=0-13%${CONCURRENCY} --export='${exports}' '${COLLECT}'")
collect_id=${collect_raw%%;*}; [[ "${collect_id}" =~ ^[0-9]+$ ]] || fail "bad collect id"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${collect_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary id"
eval_raw=$(remote "sbatch --parsable --array=0-83%${EVAL_CONCURRENCY} --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}'")
eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval id"
pair_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${PAIR_SUMMARY}'")
pair_id=${pair_raw%%;*}; [[ "${pair_id}" =~ ^[0-9]+$ ]] || fail "bad pair-summary id"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${pair_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verify id"

remote "/scratch/lg154/conda-envs/memnav/bin/python - \
  '${RUN_ROOT}/submission.json' '${SOURCE_BUNDLE}' '${EXPECTED_SOURCE_RECEIPT_SHA}' \
  '${manifest_sha}' '${collect_id}' '${summary_id}' '${eval_id}' \
  '${pair_id}' '${verify_id}' '${CONCURRENCY}' '${EVAL_CONCURRENCY}'" <<'PY'
import json,sys
(path,bundle,source_sha,manifest_sha,collect,construction,evaluation,
 summary,verification,collect_concurrency,eval_concurrency)=sys.argv[1:]
with open(path,"x") as handle:
    json.dump({
        "schema_version":"final14_learned_submission_v1_20260817",
        "scope":"one-shot Final14 natural/standard and hard-support five-arm evaluation",
        "explicit_user_authorization":True,
        "source_bundle":bundle,
        "source_receipt_sha256":source_sha,
        "final14_source_manifest_sha256":manifest_sha,
        "final14_accessed_for_manifest_freeze":True,
        "query_policy_outcomes_read_at_submission":False,
        "arrays":{
            "collection_concurrency":int(collect_concurrency),
            "evaluation_concurrency":int(eval_concurrency),
            "collection":"0-13",
            "evaluation":"0-83",
        },
        "jobs":{
            "collect_array":int(collect),
            "construction_summary":int(construction),
            "evaluation_array":int(evaluation),
            "policy_summary":int(summary),
            "independent_verification":int(verification),
        },
    },handle,indent=2,sort_keys=True)
    handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA}"
echo "FINAL14_MANIFEST_SHA=${manifest_sha}"
echo "collect=${collect_id} construction=${summary_id} eval=${eval_id} summary=${pair_id} verify=${verify_id}"
