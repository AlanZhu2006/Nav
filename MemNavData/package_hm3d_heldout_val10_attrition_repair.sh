#!/usr/bin/env bash
# Build the immutable outcome-blind HM3D construction-attrition repair bundle.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
PACKAGE_ONLY_DIR=${PACKAGE_ONLY_DIR:?set an absolute, nonexistent output path}
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${PACKAGE_ONLY_DIR}" = /* ]] || fail "PACKAGE_ONLY_DIR must be absolute"
[[ ! -e "${PACKAGE_ONLY_DIR}" ]] || fail "package output already exists"

required=(
  MemNavData/HM3D_HELDOUT_VAL10_EXTERNAL_REVISIT_PROTOCOL_20260816.md
  MemNavData/HM3D_HELDOUT_VAL10_CONSTRUCTION_ATTRITION_AMENDMENT_20260816.md
  MemNavData/HPC_SHARED_SSH_OPERATIONS_20260816.md
  MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json
  MemNavData/hm3d_heldout_val10_revisit_attrition_protocol_20260816.json
  MemNavData/hm3d_consumed_scene_audit_20260816.json
  MemNavData/build_hm3d_heldout_val10_revisit_manifest.py
  MemNavData/test_build_hm3d_heldout_val10_revisit_manifest.py
  MemNavData/summarize_hm3d_heldout_val10_revisit.py
  MemNavData/test_hm3d_heldout_val10_revisit_summary.py
  MemNavData/test_hm3d_heldout_val10_revisit_integration.py
  MemNavData/verify_hm3d_heldout_val10_revisit.py
  MemNavData/run_hm3d_heldout_val10_revisit_scene.sh
  MemNavData/launch_hm3d_heldout_val10_attrition_repair_hpc.sh
  MemNavData/slurm_hm3d_heldout_val10_revisit_manifest_attrition_repair.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_eval.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch
)
for relative in "${required[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || \
    fail "missing physical repair input ${relative}"
done

readonly PY=/home/asus/miniconda3/envs/memnav/bin/python
[[ -x "${PY}" ]] || fail "missing local MemNav Python"
export PYTHONPATH=${LOCAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/build_hm3d_heldout_val10_revisit_manifest.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_hm3d_heldout_val10_revisit.py" \
  "${LOCAL_ROOT}/MemNavData/verify_hm3d_heldout_val10_revisit.py"
"${PY}" -m pytest -q \
  "${LOCAL_ROOT}/MemNavData/test_build_hm3d_heldout_val10_revisit_manifest.py" \
  "${LOCAL_ROOT}/MemNavData/test_hm3d_heldout_val10_revisit_summary.py" \
  "${LOCAL_ROOT}/MemNavData/test_hm3d_heldout_val10_revisit_integration.py"
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_hm3d_heldout_val10_revisit_scene.sh" \
  "${LOCAL_ROOT}/MemNavData/launch_hm3d_heldout_val10_attrition_repair_hpc.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_manifest_attrition_repair.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_eval.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch"

staging=$(mktemp -d)
trap 'rm -rf -- "${staging}"' EXIT
for relative in "${required[@]}"; do
  mkdir -p "${staging}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${staging}/${relative}"
done
local_head=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
protocol_sha=$(sha256sum \
  "${staging}/MemNavData/hm3d_heldout_val10_revisit_attrition_protocol_20260816.json" | \
  awk '{print $1}')
"${PY}" - "${staging}" "${local_head}" "${protocol_sha}" <<'PY'
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
 "schema_version":"hm3d_heldout_val10_attrition_repair_bundle_v1_20260816",
 "objective":"outcome-blind construction attrition and external causal-Revisit transfer",
 "local_git_head_context":sys.argv[2],
 "protocol_sha256":sys.argv[3],
 "parent_protocol_sha256":"a019a49248950a537b14c651b7a812ba7ccb421504901f8a8de030d63ae3a230",
 "failed_generation_summary_sha256":"672055791985b4199a6c60e6ce639bfa0e45d4abdbdd840f66e5207e40fc39b7",
 "dataset":"HM3D v0.2 outcome-disjoint val10",
 "selected_scenes":10,"constructible_scenes":9,"episodes":36,
 "evaluation_scene_indices":[0,1,2,3,4,5,6,7,9],
 "guards":{"navigation_outcomes_read":False,"scene_replacement":False,
           "generation_retry":False,"original_scene_indices_preserved":True},
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${staging}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
mkdir -p "$(dirname "${PACKAGE_ONLY_DIR}")"
cp -a "${staging}" "${PACKAGE_ONLY_DIR}"
receipt_sha=$(sha256sum "${PACKAGE_ONLY_DIR}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
manifest_sha=$(sha256sum "${PACKAGE_ONLY_DIR}/source_bundle_manifest.json" | awk '{print $1}')
echo "PACKAGE_ONLY_DIR=${PACKAGE_ONLY_DIR}"
echo "PACKAGE_RECEIPT_SHA=${receipt_sha}"
echo "PACKAGE_MANIFEST_SHA=${manifest_sha}"
echo "REMOTE_BUNDLE_NAME=hm3d_heldout_val10_attrition_repair_${manifest_sha:0:16}"
