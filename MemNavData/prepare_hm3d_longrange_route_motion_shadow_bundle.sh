#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${LOCAL_HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
PARENT=${LOCAL_BUNDLE_PARENT:-${ROOT}/.diagnostics/source_bundles}
PROTOCOL_SHA=589af0882063bca63b47892938887c8a93e74477ccb68f2f9cfb02778da99943
fail() { echo "ABORT: $*" >&2; exit 2; }

files=(
  MemNavData/hm3d_longrange_route_motion_shadow_protocol_20260903.json
  MemNavData/run_hm3d_longrange_route_motion_shadow_history.py
  MemNavData/analyze_hm3d_longrange_route_motion_shadow.py
  MemNavData/independent_verify_hm3d_longrange_route_motion_shadow.py
  MemNavData/slurm_hm3d_longrange_route_motion_shadow.sbatch
  MemNavData/slurm_hm3d_longrange_route_motion_shadow_result.sbatch
  MemNavData/submit_hm3d_longrange_route_motion_shadow_remote.sh
  MemNavData/test_hm3d_longrange_route_motion_shadow.py
  MemNavData/hm3d_longrange_route_tangent_experiment.py
  MemNavData/run_hm3d_longrange_route_tangent_history.py
  MemNavData/monocular_adjacent_motion.py
  MemNavData/monocular_route_tangent_runtime.py
  MemNavData/path_budgeted_route_compass.py
  MemNavData/se2_projected_route_compass.py
  MemNavData/hm3d_table3_causal_survey_contract.py
  MemNavData/hm3d_table3_length_contract.py
  MemNavData/audit_hm3d_table3_length_role_pairs.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/shared_online_double_revisit_runtime.py
  MemNavData/final14_mono_factorial.py
  MemNavData/run_final14_mono_factorial_episode.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/generate_twoleg.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/arrival_shadow.py
  MemNavData/bearing_diagnostics.py
  MemNavData/cec_bearing_alignment.py
  MemNavData/cec_authority_receipt.py
  MemNavData/cec_handoff_contract.py
  MemNavData/controller_portability_contract.py
  MemNavData/navdp_goal_switch.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/revisit_action_shadow.py
  MemNavData/xnavdp_revisit_contract.py
  MemNavData/certified_relocalization_contract.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/lingbot_pnp_localization.py
  MemNavData/lingbot_colored_registration.py
  MemNavData/monocular_depth_runtime.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_port_pair.sh
  MemNavData/slurm_safe_submit.sh
  MemNavData/test_monocular_adjacent_motion.py
  MemNavData/test_monocular_route_tangent_runtime.py
  MemNavData/test_path_budgeted_route_compass.py
  MemNavData/test_policy_agent_monocular_route_tangent.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/memnav/router_candidates.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/navdp/navdp_server.py
)
for path in "${files[@]}"; do
  [[ -f "${ROOT}/${path}" && ! -L "${ROOT}/${path}" ]] || \
    fail "missing physical source ${path}"
done
[[ -x "${PY}" && -x "${HAB_PY}" ]] || fail "local interpreter missing"
cd "${ROOT}"
[[ "$(sha256sum MemNavData/hm3d_longrange_route_motion_shadow_protocol_20260903.json | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "frozen diagnostic protocol changed"

env PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${PY}" -m pytest -q \
  MemNavData/test_hm3d_longrange_route_motion_shadow.py \
  MemNavData/test_monocular_adjacent_motion.py \
  MemNavData/test_monocular_route_tangent_runtime.py \
  MemNavData/test_policy_agent_monocular_route_tangent.py
"${PY}" -m py_compile \
  MemNavData/run_hm3d_longrange_route_motion_shadow_history.py \
  MemNavData/analyze_hm3d_longrange_route_motion_shadow.py \
  MemNavData/independent_verify_hm3d_longrange_route_motion_shadow.py \
  MemNavData/monocular_adjacent_motion.py \
  NavDP/baselines/memnav/policy_agent.py \
  NavDP/baselines/memnav/memnav_server.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_longrange_route_motion_shadow.sbatch \
  MemNavData/slurm_hm3d_longrange_route_motion_shadow_result.sbatch \
  MemNavData/submit_hm3d_longrange_route_motion_shadow_remote.sh \
  MemNavData/prepare_hm3d_longrange_route_motion_shadow_bundle.sh
source MemNavData/slurm_safe_submit.sh
for template in \
    MemNavData/slurm_hm3d_longrange_route_motion_shadow.sbatch \
    MemNavData/slurm_hm3d_longrange_route_motion_shadow_result.sbatch; do
  lint_sbatch_template "${template}" || fail "Slurm template lint failed"
done

mkdir -p "${PARENT}"
stage=$(mktemp -d "${PARENT}/hm3d_route_motion_shadow.partial.XXXXXX")
trap 'if [[ -n "${stage:-}" && -d "${stage}" ]]; then chmod -R u+w "${stage}"; rm -r -- "${stage}"; fi' EXIT
for path in "${files[@]}"; do
  mkdir -p "${stage}/$(dirname "${path}")"
  cp --preserve=mode,timestamps "${ROOT}/${path}" "${stage}/${path}"
done
(
  cd "${stage}"
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}:${stage}/MemNavData" "${PY}" -m pytest -q \
      MemNavData/test_hm3d_longrange_route_motion_shadow.py \
      MemNavData/test_monocular_adjacent_motion.py \
      MemNavData/test_monocular_route_tangent_runtime.py \
      MemNavData/test_policy_agent_monocular_route_tangent.py
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}:${stage}/MemNavData" "${HAB_PY}" \
      MemNavData/eval_shared_online_role_pairs.py --help >/dev/null
  if find . -type d -name __pycache__ -print -quit | grep -q .; then
    fail "bundle-only self-test leaked Python bytecode"
  fi
)

head=$(git -C "${ROOT}" rev-parse HEAD)
"${PY}" - "${stage}" "${head}" "${PROTOCOL_SHA}" <<'PY'
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
 'schema_version':'hm3d_longrange_route_motion_shadow_bundle_v1_20260903',
 'local_git_head_context':sys.argv[2],
 'scientific_scope':'consumed same-edge route-motion mechanism diagnostic',
 'protocol_sha256':sys.argv[3],
 'formal_parent_result_dependent':True,
 'selected_geometry_stop_histories':14,
 'primary_estimator':'fundamental_magsac_then_depth_pnp_ransac',
 'shadow_estimator':'direct_depth_pnp_ransac',
 'shadow_control_authority':False,
 'navigation_claim_allowed':False,
 'threshold_search_allowed':False,
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
target=${PARENT}/hm3d_longrange_route_motion_shadow_${receipt_sha:0:16}
[[ ! -e "${target}" ]] || fail "bundle already exists: ${target}"
chmod -R a-w "${stage}"
mv "${stage}" "${target}"
stage=
trap - EXIT
printf 'BUNDLE=%s\nRECEIPT_SHA256=%s\n' "${target}" "${receipt_sha}"
