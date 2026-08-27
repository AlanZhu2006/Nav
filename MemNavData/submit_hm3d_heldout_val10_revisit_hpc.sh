#!/usr/bin/env bash
# Seal and submit the complete non-MP3D HM3D held-out val10 Revisit evaluation chain.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/home/asus/.ssh/cm-hm3d-yz11502}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_heldout_val10_revisit_20260816}
RUN_TAG=${RUN_TAG:-hm3d_heldout_val10_revisit_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
DATA_ROOT=${DATA_ROOT:-/scratch/yz11502/Research/datasets/hm3d_heldout_val10_v0.2_20260816}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA:-74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98}
GEN_CONCURRENCY=${GEN_CONCURRENCY:-6}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}
PACKAGE_ONLY_DIR=${PACKAGE_ONLY_DIR:-}

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 \
    -o ControlPath="${SSH_CONTROL_PATH}" "${REMOTE_HOST}" "$@"
}
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid run tag"
[[ "${GEN_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid generation concurrency"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid evaluation concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"

required=(
  MemNavData/HM3D_HELDOUT_VAL10_EXTERNAL_REVISIT_PROTOCOL_20260816.md
  MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json
  MemNavData/hm3d_consumed_scene_audit_20260816.json
  MemNavData/generate_twoleg.py
  MemNavData/audit_hm3d_heldout_scene_selection.py
  MemNavData/test_audit_hm3d_heldout_scene_selection.py
  MemNavData/goat_navdp_runtime_pilot_manifest.json
  MemNavData/goat_certified_arrival_manifest.json
  MemNavData/goat_sequential_revisit_formal_manifest_20260815.json
  MemNavData/goat_sequential_revisit_smoke_manifest_20260815.json
  MemNavData/goat_certified_arrival_smoke_manifest.json
  .diagnostics/datasets/goat-bench/hm3d_val_receipts/hm3d-val-habitat-v0.2.members.txt
  MemNavData/build_hm3d_heldout_val10_revisit_manifest.py
  MemNavData/test_build_hm3d_heldout_val10_revisit_manifest.py
  MemNavData/summarize_hm3d_heldout_val10_revisit.py
  MemNavData/test_hm3d_heldout_val10_revisit_summary.py
  MemNavData/test_hm3d_heldout_val10_revisit_integration.py
  MemNavData/verify_hm3d_heldout_val10_revisit.py
  MemNavData/run_hm3d_heldout_val10_revisit_scene.sh
  MemNavData/launch_hm3d_heldout_val10_revisit_hpc.sh
  MemNavData/slurm_hm3d_heldout_val10_prepare.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_generate.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_manifest.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_eval.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch
)
for relative in "${required[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || \
    fail "missing physical task input ${relative}"
done

MEMNAV_PY=/home/asus/miniconda3/envs/memnav/bin/python
[[ -x "${MEMNAV_PY}" ]] || fail "missing local MemNav Python"
export PYTHONPATH=${LOCAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/audit_hm3d_heldout_scene_selection.py" \
  "${LOCAL_ROOT}/MemNavData/generate_twoleg.py" \
  "${LOCAL_ROOT}/MemNavData/build_hm3d_heldout_val10_revisit_manifest.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_hm3d_heldout_val10_revisit.py" \
  "${LOCAL_ROOT}/MemNavData/verify_hm3d_heldout_val10_revisit.py"
"${MEMNAV_PY}" -m pytest -q \
  "${LOCAL_ROOT}/MemNavData/test_audit_hm3d_heldout_scene_selection.py" \
  "${LOCAL_ROOT}/MemNavData/test_build_hm3d_heldout_val10_revisit_manifest.py" \
  "${LOCAL_ROOT}/MemNavData/test_hm3d_heldout_val10_revisit_summary.py" \
  "${LOCAL_ROOT}/MemNavData/test_hm3d_heldout_val10_revisit_integration.py"
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_hm3d_heldout_val10_revisit_scene.sh" \
  "${LOCAL_ROOT}/MemNavData/launch_hm3d_heldout_val10_revisit_hpc.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_prepare.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_generate.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_manifest.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_eval.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
for relative in "${required[@]}"; do
  mkdir -p "${STAGING}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${STAGING}/${relative}"
done
"${MEMNAV_PY}" "${STAGING}/MemNavData/audit_hm3d_heldout_scene_selection.py" \
  --audit "${STAGING}/MemNavData/hm3d_consumed_scene_audit_20260816.json" \
  --member-list "${STAGING}/.diagnostics/datasets/goat-bench/hm3d_val_receipts/hm3d-val-habitat-v0.2.members.txt" \
  --repo-root "${STAGING}" \
  --out "${STAGING}/hm3d_heldout_scene_selection_verification.json" \
  >/dev/null
LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
PROTOCOL_SHA=$(sha256sum \
  "${STAGING}/MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json" | \
  awk '{print $1}')
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" "${PROTOCOL_SHA}" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {
            "source_bundle_manifest.json", "SOURCE_BUNDLE.sha256"}:
        digest=hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda:handle.read(8<<20),b""):
                digest.update(block)
        files[path.relative_to(root).as_posix()]=digest.hexdigest()
payload={
 "schema_version":"hm3d_heldout_val10_revisit_task_bundle_v1_20260816",
 "objective":"non-MP3D external causal-Revisit transfer",
 "local_git_head_context":sys.argv[2],
 "protocol_sha256":sys.argv[3],
 "dataset":"HM3D v0.2 outcome-disjoint val10",
 "scenes":10,"episodes":40,
 "forbidden_dataset":"MP3D",
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
TASK_RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/hm3d_heldout_val10_revisit_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$

if [[ -n "${PACKAGE_ONLY_DIR}" ]]; then
  [[ "${PACKAGE_ONLY_DIR}" = /* ]] || fail "PACKAGE_ONLY_DIR must be absolute"
  [[ ! -e "${PACKAGE_ONLY_DIR}" ]] || fail "package output already exists"
  mkdir -p "$(dirname "${PACKAGE_ONLY_DIR}")"
  cp -a "${STAGING}" "${PACKAGE_ONLY_DIR}"
  echo "PACKAGE_ONLY_DIR=${PACKAGE_ONLY_DIR}"
  echo "PACKAGE_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "PACKAGE_TASK_RECEIPT_SHA=${TASK_RECEIPT_SHA}"
  echo "PACKAGE_PROTOCOL_SHA=${PROTOCOL_SHA}"
  exit 0
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_DATA_ROOT=${DATA_ROOT}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_TASK_RECEIPT_SHA=${TASK_RECEIPT_SHA}"
  echo "DRY_RUN_PROTOCOL_SHA=${PROTOCOL_SHA}"
  exit 0
fi

remote "hostname >/dev/null"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}'"
remote "(test -r /scratch/yz11502/Research/datasets/goat_bench_20260814/downloads/hm3d-val-habitat-v0.2.tar || test -r \"\${HOME}/.config/hm3d/netrc\") && test -x /scratch/lg154/conda-envs/habitat/bin/python && test -x /scratch/lg154/conda-envs/memnav/bin/python && test -r /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
remote "test -r '${BASE_SOURCE_ROOT}/third_party/LightGlue' && test -r '${BASE_SOURCE_ROOT}/third_party/python' && test -r '${BASE_SOURCE_ROOT}/torch_home'"
if remote "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${TASK_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${REMOTE_BUNDLE}"
else
  remote "test ! -e '${REMOTE_BUNDLE}' && test ! -e '${REMOTE_STAGING}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ControlPath=${SSH_CONTROL_PATH}" \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote "cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi
remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}/logs'"

exports="ALL,TASK_ROOT=${REMOTE_BUNDLE},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},DATA_ROOT=${DATA_ROOT},EXPECTED_TASK_RECEIPT_SHA=${TASK_RECEIPT_SHA},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"
PREP=${REMOTE_BUNDLE}/MemNavData/slurm_hm3d_heldout_val10_prepare.sbatch
GENERATE=${REMOTE_BUNDLE}/MemNavData/slurm_hm3d_heldout_val10_revisit_generate.sbatch
MANIFEST=${REMOTE_BUNDLE}/MemNavData/slurm_hm3d_heldout_val10_revisit_manifest.sbatch
EVAL=${REMOTE_BUNDLE}/MemNavData/slurm_hm3d_heldout_val10_revisit_eval.sbatch
SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch

remote "sbatch --test-only --export='${exports}' '${PREP}' >/dev/null"
prep_raw=$(remote "sbatch --parsable --export='${exports}' '${PREP}'")
prep_id=${prep_raw%%;*}
[[ "${prep_id}" =~ ^[0-9]+$ ]] || fail "bad prepare job ID"

remote "sbatch --test-only --array=0 --dependency=afterok:${prep_id} --kill-on-invalid-dep=yes --export='${exports}' '${GENERATE}' >/dev/null"
generate_raw=$(remote "sbatch --parsable --array=0-9%${GEN_CONCURRENCY} --dependency=afterok:${prep_id} --kill-on-invalid-dep=yes --export='${exports}' '${GENERATE}'")
generate_id=${generate_raw%%;*}
[[ "${generate_id}" =~ ^[0-9]+$ ]] || fail "bad generation job ID"

remote "sbatch --test-only --dependency=afterok:${generate_id} --kill-on-invalid-dep=yes --export='${exports}' '${MANIFEST}' >/dev/null"
manifest_raw=$(remote "sbatch --parsable --dependency=afterok:${generate_id} --kill-on-invalid-dep=yes --export='${exports}' '${MANIFEST}'")
manifest_id=${manifest_raw%%;*}
[[ "${manifest_id}" =~ ^[0-9]+$ ]] || fail "bad manifest job ID"

remote "sbatch --test-only --array=0 --dependency=afterok:${manifest_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}' >/dev/null"
eval_raw=$(remote "sbatch --parsable --array=0-9%${EVAL_CONCURRENCY} --dependency=afterok:${manifest_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}'")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad evaluation job ID"

remote "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}' >/dev/null"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job ID"

remote "sbatch --test-only --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}' >/dev/null"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verification job ID"

remote "/scratch/lg154/conda-envs/memnav/bin/python - '${RUN_ROOT}/submission.json' '${REMOTE_BUNDLE}' '${TASK_RECEIPT_SHA}' '${PROTOCOL_SHA}' '${DATA_ROOT}' '${prep_id}' '${generate_id}' '${manifest_id}' '${eval_id}' '${summary_id}' '${verify_id}' '${GEN_CONCURRENCY}' '${EVAL_CONCURRENCY}'" <<'PY'
import json,sys
(path,bundle,receipt,protocol,data_root,prepare,generation,manifest,
 evaluation,summary,verification,gen_concurrency,eval_concurrency)=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
 json.dump({
  "schema_version":"hm3d_heldout_val10_revisit_submission_v1_20260816",
  "objective":"non-MP3D external causal-Revisit transfer",
  "dataset":"HM3D v0.2 outcome-disjoint val10","scene_count":10,"episode_count":40,
  "source_bundle":bundle,"task_receipt_sha256":receipt,
  "protocol_sha256":protocol,"data_root":data_root,
  "guards":{"no_mp3d_evaluation":True,"intention_to_treat":True,
            "no_outcome_filtering":True,"no_hm3d_heldout_val10_tuning":True},
  "concurrency":{"generation":int(gen_concurrency),
                 "evaluation":int(eval_concurrency)},
  "jobs":{"prepare":int(prepare),"generation_array":int(generation),
          "manifest":int(manifest),"evaluation_array":int(evaluation),
          "summary":int(summary),"independent_verification":int(verification)},
 },handle,indent=2,sort_keys=True); handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "TASK_BUNDLE=${REMOTE_BUNDLE}"
echo "TASK_RECEIPT_SHA=${TASK_RECEIPT_SHA}"
echo "PROTOCOL_SHA=${PROTOCOL_SHA}"
echo "prepare=${prep_id} generation=${generate_id} manifest=${manifest_id} eval=${eval_id} summary=${summary_id} verify=${verify_id}"
