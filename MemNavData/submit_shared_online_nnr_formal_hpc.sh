#!/usr/bin/env bash
# Bundle and submit: extract -> construct[22] -> seal -> paired eval -> audit.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_nnr_20260814}
RUN_TAG=${RUN_TAG:-shared_online_nnr_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
GENERATION_ROOT=${GENERATION_ROOT:-/scratch/qw2440/v4_eval/formal_gen}
NATIVE_EVAL_ROOT=${NATIVE_EVAL_ROOT:-/scratch/qw2440/v4_eval/formal_eval/native}
ASSET_ROOT=${ASSET_ROOT:-/scratch/lg154/Research/datasets/mp3d/mp3d}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
NAVDP_CHECKPOINT=${NAVDP_CHECKPOINT:-/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
EXPECTED_NAVDP_CHECKPOINT_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
BUILD_CONCURRENCY=${BUILD_CONCURRENCY:-6}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}

remote() { ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"; }
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"
[[ "${BUILD_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid build concurrency"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid eval concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"

required=(
  MemNavData/build_shared_online_novel_revisit.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_3leg_habitat.py
  MemNavData/eval_shared_online_novel_revisit.py
  MemNavData/extract_native_shared_ab_traces.py
  MemNavData/finalize_shared_online_novel_revisit.py
  MemNavData/generate_twoleg.py
  MemNavData/independent_verify_shared_online_nnr.py
  MemNavData/multigoal_benchmark_contract.py
  MemNavData/multigoal_policy_contract.py
  MemNavData/run_shared_online_novel_revisit_formal_episode.sh
  MemNavData/SHARED_ONLINE_NNR_RETEST_PROTOCOL_20260814.md
  MemNavData/slurm_shared_online_nnr_extract.sbatch
  MemNavData/slurm_shared_online_nnr_build.sbatch
  MemNavData/slurm_shared_online_nnr_finalize.sbatch
  MemNavData/slurm_shared_online_nnr_eval.sbatch
  MemNavData/slurm_shared_online_nnr_summary.sbatch
  MemNavData/slurm_shared_online_nnr_verify.sbatch
  MemNavData/summarize_shared_online_novel_revisit.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/policy_agent.py
)
for relative in "${required[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || \
    fail "missing physical input ${relative}"
done

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/build_shared_online_novel_revisit.py" \
  "${LOCAL_ROOT}/MemNavData/eval_shared_online_novel_revisit.py"
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/extract_native_shared_ab_traces.py" \
  "${LOCAL_ROOT}/MemNavData/finalize_shared_online_novel_revisit.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_shared_online_nnr.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_shared_online_novel_revisit.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py"
"${MEMNAV_PY}" -m unittest \
  MemNavData.test_deterministic_eval_protocol \
  MemNavData.test_multigoal_benchmark_contract \
  MemNavData.test_multigoal_policy_contract \
  MemNavData.test_policy_agent_graph \
  MemNavData.test_shared_online_double_revisit_runtime \
  MemNavData.test_certified_relocalization_runtime
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_shared_online_novel_revisit_formal_episode.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_nnr_extract.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_nnr_build.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_nnr_finalize.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_nnr_eval.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_nnr_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_shared_online_nnr_verify.sbatch"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/memnav" "${STAGING}/NavDP/baselines/navdp"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for relative in \
  MemNavData/SHARED_ONLINE_NNR_RETEST_PROTOCOL_20260814.md \
  MemNavData/run_shared_online_novel_revisit_formal_episode.sh \
  MemNavData/slurm_shared_online_nnr_extract.sbatch \
  MemNavData/slurm_shared_online_nnr_build.sbatch \
  MemNavData/slurm_shared_online_nnr_finalize.sbatch \
  MemNavData/slurm_shared_online_nnr_eval.sbatch \
  MemNavData/slurm_shared_online_nnr_summary.sbatch \
  MemNavData/slurm_shared_online_nnr_verify.sbatch; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${STAGING}/${relative}"
done
for component in memnav navdp; do
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${STAGING}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
done

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" "${GENERATION_ROOT}" \
  "${NATIVE_EVAL_ROOT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"shared_online_nnr_task_bundle_v1_20260814",
 "local_git_head_context":sys.argv[2],
 "strict_v4_generation_root":sys.argv[3],
 "native_eval_root":sys.argv[4],
 "source_population":"native A-and-B successes",
 "expected_source_population":22,
 "construction":"Revisit-C supported only by factual online-A; B endpoint negative; pre-C FIFO reset; A-bounded long memory",
 "navigation_arms":["native","known_direct","certified","certified_budget","certified_graph"],
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
SOURCE_RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/shared_online_nnr_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  exit 0
fi

remote "test -d '${GENERATION_ROOT}' && test -d '${NATIVE_EVAL_ROOT}' && test -d '${ASSET_ROOT}'"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${NAVDP_CHECKPOINT}' | awk '{print \$1}')\" = '${EXPECTED_NAVDP_CHECKPOINT_SHA}'"
if remote "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${REMOTE_BUNDLE}"
else
  remote "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi
remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}/logs'"

SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},GENERATION_ROOT=${GENERATION_ROOT},NATIVE_EVAL_ROOT=${NATIVE_EVAL_ROOT},ASSET_ROOT=${ASSET_ROOT},NAVDP_CHECKPOINT=${NAVDP_CHECKPOINT},EXPECTED_NAVDP_CHECKPOINT_SHA=${EXPECTED_NAVDP_CHECKPOINT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA}"
EXTRACT=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_nnr_extract.sbatch
BUILD=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_nnr_build.sbatch
FINALIZE=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_nnr_finalize.sbatch
EVAL=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_nnr_eval.sbatch
SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_nnr_summary.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_shared_online_nnr_verify.sbatch

remote "sbatch --test-only --export='${exports}' '${EXTRACT}' >/dev/null"
extract_raw=$(remote "sbatch --parsable --export='${exports}' '${EXTRACT}'")
extract_id=${extract_raw%%;*}
[[ "${extract_id}" =~ ^[0-9]+$ ]] || fail "bad extract job"
remote "sbatch --test-only --array=0 --dependency=afterok:${extract_id} --kill-on-invalid-dep=yes --export='${exports}' '${BUILD}' >/dev/null"
build_raw=$(remote "sbatch --parsable --array=0-21%${BUILD_CONCURRENCY} --dependency=afterok:${extract_id} --kill-on-invalid-dep=yes --export='${exports}' '${BUILD}'")
build_id=${build_raw%%;*}
[[ "${build_id}" =~ ^[0-9]+$ ]] || fail "bad build array"
remote "sbatch --test-only --dependency=afterok:${build_id} --kill-on-invalid-dep=yes --export='${exports}' '${FINALIZE}' >/dev/null"
finalize_raw=$(remote "sbatch --parsable --dependency=afterok:${build_id} --kill-on-invalid-dep=yes --export='${exports}' '${FINALIZE}'")
finalize_id=${finalize_raw%%;*}
[[ "${finalize_id}" =~ ^[0-9]+$ ]] || fail "bad finalizer"
remote "sbatch --test-only --array=0 --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}' >/dev/null"
eval_raw=$(remote "sbatch --parsable --array=0-21%${EVAL_CONCURRENCY} --dependency=afterok:${finalize_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}'")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval array"
remote "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}' >/dev/null"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job"
remote "sbatch --test-only --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}' >/dev/null"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verification job"

remote "'/scratch/lg154/conda-envs/memnav/bin/python' - '${RUN_ROOT}/submission.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${extract_id}' '${build_id}' '${finalize_id}' '${eval_id}' '${summary_id}' '${verify_id}' '${BUILD_CONCURRENCY}' '${EVAL_CONCURRENCY}'" <<'PY'
import json,sys
(path,bundle,receipt,extract,build,finalize,evaluation,summary,verification,
 build_concurrency,eval_concurrency)=sys.argv[1:]
with open(path,"x",encoding="utf-8") as f:
 json.dump({
  "schema_version":"shared_online_nnr_submission_v1_20260814",
  "source_bundle":bundle,"source_receipt_sha256":receipt,
  "scope":"internal strict retest; consumed source scenes; not paper confirmation",
  "expected_native_AB_population":22,
  "arrays":{"build_concurrency":int(build_concurrency),"eval_concurrency":int(eval_concurrency)},
  "jobs":{"extract":int(extract),"build_array":int(build),
          "finalize":int(finalize),"evaluation_array":int(evaluation),
          "summary":int(summary),"independent_verification":int(verification)},
 },f,indent=2,sort_keys=True); f.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "extract=${extract_id} build=${build_id} finalize=${finalize_id} eval=${eval_id} summary=${summary_id} verify=${verify_id}"
