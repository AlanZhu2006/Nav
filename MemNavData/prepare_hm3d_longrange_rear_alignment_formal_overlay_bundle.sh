#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
PARENT=${LOCAL_BUNDLE_PARENT:-${ROOT}/.diagnostics/source_bundles}
FORMAL=${ROOT}/.diagnostics/source_bundles/hm3d_longrange_route_tangent_751efdcbb436c99b
FORMAL_SHA=751efdcbb436c99b5cc92d69633e812e9d1edc381dcf224af9042113ee8e874e
PROTOCOL_SHA=f61dbd3021df5ff8603b68666ead03863d77a0f91d5f760301f28b2a596dbe4f
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ -x "${PY}" && -x "${HAB_PY}" ]] || fail "local interpreter missing"
[[ -d "${FORMAL}" && -f "${FORMAL}/SOURCE_BUNDLE.sha256" ]] || \
  fail "sealed formal route-tangent bundle missing"
[[ "$(sha256sum "${FORMAL}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${FORMAL_SHA}" ]] || fail "formal route-tangent source receipt changed"
(cd "${FORMAL}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "formal route-tangent bundle content changed"
[[ "$(sha256sum "${ROOT}/MemNavData/hm3d_longrange_rear_alignment_protocol_20260904.json" | awk '{print $1}')" == \
   "${PROTOCOL_SHA}" ]] || fail "frozen rear-alignment protocol changed"

mkdir -p "${PARENT}"
stage=$(mktemp -d "${PARENT}/hm3d_rear_alignment_formal_overlay.partial.XXXXXX")
cleanup() {
  if [[ -n "${stage:-}" && -d "${stage}" ]]; then
    chmod -R u+w "${stage}" || true
    rm -r -- "${stage}"
  fi
}
trap cleanup EXIT

"${PY}" "${ROOT}/MemNavData/build_hm3d_rear_alignment_formal_overlay.py" \
  --workspace-root "${ROOT}" --formal-root "${FORMAL}" --stage-root "${stage}"

export PYTHONDONTWRITEBYTECODE=1
(
  cd "${stage}"
  export PYTHONPATH="${stage}:${stage}/MemNavData"
  "${PY}" -m pytest -q -p no:cacheprovider \
    MemNavData/test_policy_agent_monocular_route_tangent.py \
    MemNavData/test_policy_agent_route_alignment_minimal.py \
    MemNavData/test_cec_bearing_alignment.py \
    MemNavData/test_hm3d_longrange_rear_alignment.py \
    MemNavData/test_analyze_hm3d_longrange_rear_alignment.py \
    MemNavData/test_monocular_route_tangent_runtime.py \
    MemNavData/test_hm3d_longrange_route_tangent_experiment.py
  "${PY}" -m py_compile \
    MemNavData/build_hm3d_rear_alignment_formal_overlay.py \
    MemNavData/hm3d_longrange_rear_alignment.py \
    MemNavData/run_hm3d_longrange_rear_alignment_history.py \
    MemNavData/analyze_hm3d_longrange_rear_alignment.py \
    MemNavData/independent_verify_hm3d_longrange_rear_alignment.py \
    MemNavData/route_alignment_contract.py \
    MemNavData/cec_bearing_alignment.py \
    MemNavData/eval_2leg_habitat.py \
    MemNavData/eval_shared_online_role_pairs.py \
    NavDP/baselines/memnav/policy_agent.py \
    NavDP/baselines/memnav/memnav_server.py
  "${HAB_PY}" -m py_compile \
    MemNavData/eval_2leg_habitat.py \
    MemNavData/eval_shared_online_role_pairs.py
  PYTHONPATH="${stage}:${stage}/MemNavData" \
    "${HAB_PY}" MemNavData/eval_shared_online_role_pairs.py --help >/dev/null
  bash -n \
    MemNavData/run_hm3d_fullmono_server_scene.sh \
    MemNavData/slurm_hm3d_longrange_rear_alignment.sbatch \
    MemNavData/slurm_hm3d_longrange_rear_alignment_result.sbatch \
    MemNavData/submit_hm3d_longrange_rear_alignment_remote.sh \
    MemNavData/prepare_hm3d_longrange_rear_alignment_formal_overlay_bundle.sh
  bash MemNavData/test_slurm_port_pair.sh
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template MemNavData/slurm_hm3d_longrange_rear_alignment.sbatch
  lint_sbatch_template MemNavData/slurm_hm3d_longrange_rear_alignment_result.sbatch
  test "$(sha256sum MemNavData/monocular_adjacent_motion.py | awk '{print $1}')" = \
    "$(sha256sum "${FORMAL}/MemNavData/monocular_adjacent_motion.py" | awk '{print $1}')"
  test "$(sha256sum NavDP/baselines/memnav/memnav_server.py | awk '{print $1}')" = \
    "$(sha256sum "${FORMAL}/NavDP/baselines/memnav/memnav_server.py" | awk '{print $1}')"
  find . -type d -name __pycache__ -prune -exec rm -r -- {} +
  if find . -type d -name __pycache__ -print -quit | grep -q .; then
    fail "bundle-only self-test leaked Python bytecode"
  fi
)

head=$(git -C "${ROOT}" rev-parse HEAD)
"${PY}" - "${stage}" "${head}" "${PROTOCOL_SHA}" "${FORMAL_SHA}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob('*')):
    if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
    if path.is_file() and path.name not in {
            'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
payload={
 'schema_version':'hm3d_longrange_rear_alignment_formal_overlay_bundle_v2_20260904',
 'local_git_head_context':sys.argv[2],
 'scientific_scope':'consumed rear-controller-support mechanism diagnostic',
 'protocol_sha256':sys.argv[3],
 'formal_runtime_parent_source_receipt_sha256':sys.argv[4],
 'runtime_derivation':'sealed formal route-tangent runtime plus proof-bound yaw only',
 'formal_visual_route_estimator_byte_identical':True,
 'formal_parent_result_dependent':True,
 'selected_histories':9,
 'parent_stuck_histories':5,
 'parent_success_histories':4,
 'arms':['route_tangent_unaligned','route_tangent_rear_aligned'],
 'navigation_claim_allowed':False,
 'fresh_confirmation_required':True,
 'runtime_rgb_buffer_storage':'SLURM_TMPDIR',
 'files':files,
}
(root/'source_bundle_manifest.json').write_text(
    json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${stage}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
receipt_sha=$(sha256sum "${stage}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
target=${PARENT}/hm3d_longrange_rear_alignment_formal_overlay_${receipt_sha:0:16}
[[ ! -e "${target}" ]] || fail "bundle already exists: ${target}"
chmod -R a-w "${stage}"
mv "${stage}" "${target}"
stage=
trap - EXIT
printf 'BUNDLE=%s\nRECEIPT_SHA256=%s\n' "${target}" "${receipt_sha}"
