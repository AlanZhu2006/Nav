#!/usr/bin/env bash
# Build immutable source-overlay and consumed HM3D smoke-data bundles.

set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
PACKAGE_ONLY_DIR=${PACKAGE_ONLY_DIR:?set absolute nonexistent source output}
SMOKE_PACKAGE_ONLY_DIR=${SMOKE_PACKAGE_ONLY_DIR:?set absolute nonexistent smoke output}
fail() { echo "ABORT: $*" >&2; exit 2; }
for target in "${PACKAGE_ONLY_DIR}" "${SMOKE_PACKAGE_ONLY_DIR}"; do
  [[ "${target}" = /* ]] || fail "package targets must be absolute"
  [[ ! -e "${target}" ]] || fail "package output exists: ${target}"
done

readonly EXPECTED_ADAPTER_SHA=c1f10b3c831f00a5b4742e0b34ac0675f10e161c4795ed1497c74b9551fdaf78
readonly PY=/home/asus/miniconda3/envs/memnav/bin/python
readonly SMOKE_SOURCE=${LOCAL_ROOT}/.diagnostics/hm3d_external_revisit_smoke_20260816/scene_5cd_n4/episode_0001
readonly SMOKE_SCENE=${LOCAL_ROOT}/.diagnostics/datasets/goat-smoke-hm3d-20260814/data/scene_datasets/hm3d/val/00853-5cdEh9F2hJL
readonly PRIOR_RECEIPT=${LOCAL_ROOT}/.diagnostics/hm3d_external_revisit_smoke_20260816/closed_loop_ep1/receipt.json

required=(
  MemNavData/HM3D_HELDOUT_VAL10_RUNTIME_INTERFACE_REPAIR_20260816.md
  MemNavData/HM3D_HELDOUT_VAL10_EXTERNAL_REVISIT_PROTOCOL_20260816.md
  MemNavData/HM3D_HELDOUT_VAL10_CONSTRUCTION_ATTRITION_AMENDMENT_20260816.md
  MemNavData/HPC_SHARED_SSH_OPERATIONS_20260816.md
  MemNavData/revisit_bearing_adapter.py
  MemNavData/test_revisit_bearing_adapter.py
  MemNavData/run_hm3d_runtime_interface_smoke.sh
  MemNavData/run_hm3d_heldout_val10_revisit_scene_runtime_repair.sh
  MemNavData/slurm_hm3d_runtime_interface_eval.sbatch
  MemNavData/launch_hm3d_runtime_interface_repair_hpc.sh
  MemNavData/summarize_hm3d_heldout_val10_revisit.py
  MemNavData/test_hm3d_heldout_val10_revisit_summary.py
  MemNavData/test_hm3d_heldout_val10_revisit_integration.py
  MemNavData/verify_hm3d_heldout_val10_revisit.py
  MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch
  MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch
)
for relative in "${required[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || \
    fail "missing physical source input ${relative}"
done
for path in \
  "${SMOKE_SOURCE}/meta/gen_meta.json" \
  "${SMOKE_SOURCE}/data/chunk-000/episode_000000.parquet" \
  "${SMOKE_SOURCE}/goal_image.jpg" "${SMOKE_SOURCE}/goal_1.jpg" \
  "${SMOKE_SCENE}/5cdEh9F2hJL.basis.glb" \
  "${SMOKE_SCENE}/5cdEh9F2hJL.basis.navmesh" "${PRIOR_RECEIPT}" "${PY}"; do
  test -r "${path}" || fail "missing package input ${path}"
done
[[ "$(sha256sum "${LOCAL_ROOT}/MemNavData/revisit_bearing_adapter.py" | awk '{print $1}')" == \
    "${EXPECTED_ADAPTER_SHA}" ]] || fail "adapter no longer matches repair amendment"

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/revisit_bearing_adapter.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_hm3d_heldout_val10_revisit.py" \
  "${LOCAL_ROOT}/MemNavData/verify_hm3d_heldout_val10_revisit.py"
"${PY}" -m unittest -q \
  MemNavData.test_revisit_bearing_adapter \
  MemNavData.test_hm3d_heldout_val10_revisit_summary \
  MemNavData.test_hm3d_heldout_val10_revisit_integration
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_hm3d_runtime_interface_smoke.sh" \
  "${LOCAL_ROOT}/MemNavData/run_hm3d_heldout_val10_revisit_scene_runtime_repair.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_runtime_interface_eval.sbatch" \
  "${LOCAL_ROOT}/MemNavData/launch_hm3d_runtime_interface_repair_hpc.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch"

source_stage=$(mktemp -d)
smoke_stage=$(mktemp -d)
cleanup() { rm -rf -- "${source_stage}" "${smoke_stage}"; }
trap cleanup EXIT
for relative in "${required[@]}"; do
  mkdir -p "${source_stage}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${source_stage}/${relative}"
done
local_head=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${PY}" - "${source_stage}" "${local_head}" "${EXPECTED_ADAPTER_SHA}" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {
            "source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
payload={
 "schema_version":"hm3d_runtime_interface_repair_bundle_v1_20260816",
 "objective":"runtime parity repair gated by consumed-scene smoke",
 "local_git_head_context":sys.argv[2],
 "overlay":{
   "file":"MemNavData/revisit_bearing_adapter.py",
   "sha256":sys.argv[3],
   "purpose":"expose frozen raw_fixed_bearing_v1 controller ablation",
 },
 "verified_runtime":{
   "bundle_name":"shared_online_nnr_11458cb2b75ee334",
   "receipt_sha256":"31b3e087b855e0220f6821ad96e6f5e74114bc12dc6c3afa6f7f79150dfb4575",
   "evaluator_sha256":"4552a93910d91c4957f170ef311ddd7a9151d6754eea246fc91141b41f349d75",
 },
 "frozen_manifest_sha256":"62bc6299203da709e65787c735a531974905f2ab8e940f72e91318914d949c89",
 "guards":{
   "method_change":False,"population_change":False,
   "analysis_change":False,"formal_depends_on_consumed_smoke":True,
 },
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${source_stage}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)

mkdir -p "${smoke_stage}/scene" "${smoke_stage}/episodes"
cp --preserve=mode,timestamps "${SMOKE_SCENE}/5cdEh9F2hJL.basis.glb" \
  "${SMOKE_SCENE}/5cdEh9F2hJL.basis.navmesh" "${smoke_stage}/scene/"
# The evaluator reconstructs Goal A from the frozen expert RGB stream.  Keep
# the complete already-consumed episode (RGB, depth, metadata, parquet, goals)
# so the smoke tests the same input contract as formal generated episodes.
cp -a "${SMOKE_SOURCE}" "${smoke_stage}/episodes/episode_0001"
cp --preserve=mode,timestamps "${PRIOR_RECEIPT}" \
  "${smoke_stage}/prior_consumed_receipt.json"
"${PY}" - "${smoke_stage}" \
  "${LOCAL_ROOT}/MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); protocol=json.load(open(sys.argv[2]))
prior=json.loads((root/"prior_consumed_receipt.json").read_text())
if not prior.get("passed") or prior.get("episode_id") != "episode_0001":
    raise SystemExit("prior consumed smoke receipt is not the expected pass")
payload={
 "schema_version":"hm3d_consumed_runtime_smoke_data_v1_20260816",
 "scope":"consumed_engineering_smoke_no_efficacy_claim",
 "scene_id":"5cdEh9F2hJL",
 "scene_file":"scene/5cdEh9F2hJL.basis.glb",
 "navmesh_file":"scene/5cdEh9F2hJL.basis.navmesh",
 "episode_root":"episodes",
 "episode_id":"episode_0001",
 "seed":2026081602,
 "max_steps_per_leg":500,
 "heldout_val10_scene_ids":[row["scene_id"] for row in protocol["scenes"]],
 "prior_consumed_receipt_sha256":hashlib.sha256(
     (root/"prior_consumed_receipt.json").read_bytes()).hexdigest(),
 "selection":"pre-existing successful consumed smoke chosen only for interface coverage",
 "efficacy_claim":False,
}
if payload["scene_id"] in payload["heldout_val10_scene_ids"]:
    raise SystemExit("consumed smoke overlaps heldout val10")
(root/"smoke_data_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${smoke_stage}"
  find . -type f ! -name SMOKE_DATA.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SMOKE_DATA.sha256
  sha256sum -c SMOKE_DATA.sha256 >/dev/null
)

mkdir -p "$(dirname "${PACKAGE_ONLY_DIR}")" \
  "$(dirname "${SMOKE_PACKAGE_ONLY_DIR}")"
cp -a "${source_stage}" "${PACKAGE_ONLY_DIR}"
cp -a "${smoke_stage}" "${SMOKE_PACKAGE_ONLY_DIR}"
source_receipt=$(sha256sum "${PACKAGE_ONLY_DIR}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
smoke_receipt=$(sha256sum "${SMOKE_PACKAGE_ONLY_DIR}/SMOKE_DATA.sha256" | awk '{print $1}')
source_manifest=$(sha256sum "${PACKAGE_ONLY_DIR}/source_bundle_manifest.json" | awk '{print $1}')
smoke_manifest=$(sha256sum "${SMOKE_PACKAGE_ONLY_DIR}/smoke_data_manifest.json" | awk '{print $1}')
echo "PACKAGE_ONLY_DIR=${PACKAGE_ONLY_DIR}"
echo "PACKAGE_RECEIPT_SHA=${source_receipt}"
echo "REMOTE_BUNDLE_NAME=hm3d_runtime_interface_repair_${source_manifest:0:16}"
echo "SMOKE_PACKAGE_ONLY_DIR=${SMOKE_PACKAGE_ONLY_DIR}"
echo "SMOKE_DATA_RECEIPT_SHA=${smoke_receipt}"
echo "SMOKE_MANIFEST_SHA=${smoke_manifest}"
echo "REMOTE_SMOKE_BUNDLE_NAME=hm3d_consumed_runtime_smoke_${smoke_receipt:0:16}"
