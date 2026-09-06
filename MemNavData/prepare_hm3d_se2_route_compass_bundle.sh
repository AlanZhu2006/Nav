#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
PARENT=${LOCAL_BUNDLE_PARENT:-${ROOT}/.diagnostics/source_bundles}
fail() { echo "ABORT: $*" >&2; exit 2; }

files=(
  .diagnostics/se2_projection_replay_20260902/se2_local_odometry_counterfactual_audit.json
  MemNavData/LONG_RANGE_SE2_PROGRESS_PROJECTION_DEV_20260902.md
  MemNavData/LONG_RANGE_SE2_PROGRESS_PROJECTION_DEV_20260902.md.sha256
  MemNavData/LONG_RANGE_ACTION_COORDINATE_COMPASS_PROTOCOL_20260902.md
  MemNavData/LONG_RANGE_ACTION_COORDINATE_COMPASS_PROTOCOL_20260902.md.sha256
  MemNavData/LONG_RANGE_ACTION_COORDINATE_COMPASS_RESULT_20260902.md
  MemNavData/se2_projected_route_compass.py
  MemNavData/episodic_route_filter.py
  MemNavData/audit_se2_progress_projection.py
  MemNavData/bearing_diagnostics.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/final14_mono_factorial.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/run_final14_mono_factorial_episode.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_action_coordinate_compass_gate.py
  MemNavData/run_hm3d_se2_route_compass_gate.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/shared_online_double_revisit_runtime.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/lingbot_pnp_localization.py
  MemNavData/lingbot_colored_registration.py
  MemNavData/cdec_pairwise_runtime.py
  MemNavData/patch_temporal_router.py
  MemNavData/goat_terminal_alignment.py
  MemNavData/pi3x_online_relocalizer.py
  MemNavData/pi3x_spatial_proof_runtime.py
  MemNavData/pi3x_spatial_reliability_model.py
  MemNavData/slurm_hm3d_se2_route_compass_gate.sbatch
  MemNavData/submit_hm3d_se2_route_compass_gate_remote.sh
  MemNavData/slurm_safe_submit.sh
  MemNavData/test_episodic_route_filter.py
  MemNavData/test_se2_projected_route_compass.py
  MemNavData/test_policy_agent_action_coordinate.py
  MemNavData/test_policy_agent_se2_route.py
  MemNavData/test_hm3d_action_coordinate_compass_gate.py
  MemNavData/test_hm3d_se2_route_compass_gate.py
  MemNavData/test_shared_online_double_revisit_runtime.py
  MemNavData/test_revisit_bearing_adapter.py
  MemNavData/test_policy_agent_graph.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/memnav/router_candidates.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/navdp/navdp_server.py
)
for path in "${files[@]}"; do
  [[ -f "${ROOT}/${path}" && ! -L "${ROOT}/${path}" ]] || \
    fail "missing physical source ${path}"
done
[[ -x "${PY}" ]] || fail "missing MemNav interpreter"

cd "${ROOT}"
(cd MemNavData && sha256sum -c --quiet \
  LONG_RANGE_SE2_PROGRESS_PROJECTION_DEV_20260902.md.sha256)
[[ "$(sha256sum .diagnostics/se2_projection_replay_20260902/se2_local_odometry_counterfactual_audit.json | awk '{print $1}')" == \
  9a8f55728f4a10f5bc091005ed183ceb20dc3042bbe713b6f6a0c9034a216615 ]] || \
  fail "counterfactual audit changed"
env PYTHONPATH="${ROOT}/MemNavData:${ROOT}" "${PY}" -m unittest -q \
  MemNavData.test_episodic_route_filter \
  MemNavData.test_se2_projected_route_compass \
  MemNavData.test_policy_agent_action_coordinate \
  MemNavData.test_policy_agent_se2_route \
  MemNavData.test_hm3d_action_coordinate_compass_gate \
  MemNavData.test_hm3d_se2_route_compass_gate \
  MemNavData.test_shared_online_double_revisit_runtime \
  MemNavData.test_revisit_bearing_adapter \
  MemNavData.test_policy_agent_graph
"${PY}" -m py_compile \
  MemNavData/se2_projected_route_compass.py \
  MemNavData/audit_se2_progress_projection.py \
  MemNavData/run_hm3d_action_coordinate_compass_gate.py \
  MemNavData/run_hm3d_se2_route_compass_gate.py \
  MemNavData/eval_2leg_habitat.py \
  MemNavData/eval_shared_online_role_pairs.py \
  MemNavData/shared_online_double_revisit_runtime.py \
  NavDP/baselines/memnav/policy_agent.py \
  NavDP/baselines/memnav/memnav_server.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_se2_route_compass_gate.sbatch \
  MemNavData/submit_hm3d_se2_route_compass_gate_remote.sh \
  MemNavData/prepare_hm3d_se2_route_compass_bundle.sh \
  MemNavData/slurm_safe_submit.sh

mkdir -p "${PARENT}"
stage=$(mktemp -d "${PARENT}/hm3d_se2_route_compass.partial.XXXXXX")
for path in "${files[@]}"; do
  mkdir -p "${stage}/$(dirname "${path}")"
  cp --preserve=mode,timestamps "${ROOT}/${path}" "${stage}/${path}"
done
(
  cd "${stage}"
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}/MemNavData:${stage}" "${PY}" -m unittest -q \
      MemNavData.test_episodic_route_filter \
      MemNavData.test_se2_projected_route_compass \
      MemNavData.test_policy_agent_action_coordinate \
      MemNavData.test_policy_agent_se2_route \
      MemNavData.test_hm3d_action_coordinate_compass_gate \
      MemNavData.test_hm3d_se2_route_compass_gate \
      MemNavData.test_shared_online_double_revisit_runtime \
      MemNavData.test_revisit_bearing_adapter \
      MemNavData.test_policy_agent_graph
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}/MemNavData:${stage}" "${PY}" -c \
      'import MemNavData.se2_projected_route_compass; import MemNavData.run_hm3d_se2_route_compass_gate; import NavDP.baselines.memnav.policy_agent'
  if find . -type d -name __pycache__ -print -quit | grep -q .; then
    fail "bundle-only self-test leaked Python bytecode"
  fi
)
head=$(git -C "${ROOT}" rev-parse HEAD)
"${PY}" - "${stage}" "${head}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob('*')):
    if path.is_symlink(): raise SystemExit('bundle symlink: '+str(path))
    if path.is_file() and path.name not in {
            'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(
            path.read_bytes()).hexdigest()
payload={
 'schema_version':'hm3d_se2_route_compass_bundle_v1_20260902',
 'local_git_head_context':sys.argv[2],
 'scientific_scope':'SR-hidden index-40 runtime gate only',
 'protocol_sha256':'9c8d5b2da5300d34f2b042a53cc6968ec3e990dddc3cea10c2614d4d83e0d8f7',
 'counterfactual_audit_sha256':'9a8f55728f4a10f5bc091005ed183ceb20dc3042bbe713b6f6a0c9034a216615',
 'prospective_history_index':40,
 'gate_query_step_guard':80,
 'historical_route_coordinate':'frame-bound local executor SE(2)',
 'query_progress_coordinate':'frame-bound local executor SE(2)',
 'route_progress_estimator':'monotone orthogonal projection',
 'route_control_horizon_m':2.5,
 'motion_receipt_contract':'frame_bound_local_se2_v1',
 'habitat_odometry_source':'habitat_pose_difference_odometry_proxy_v1',
 'pure_rgb_claim_authorized':False,
 'navigation_success_read_by_gate':False,
 'navigation_final_distance_read_by_gate':False,
 'metric_scale_consumed_by_route':False,
 'visual_gate_after_authorization':False,
 'distance_regime_present':False,
 'endpoint_or_native_fallback_after_authorization':False,
 'canonical_cec_default_changed':False,
 'scalar_action_coordinate_mode_changed':False,
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
target=${PARENT}/hm3d_se2_route_compass_${receipt_sha:0:16}
[[ ! -e "${target}" ]] || fail "bundle already exists: ${target}"
chmod -R a-w "${stage}"
mv "${stage}" "${target}"
printf 'BUNDLE=%s\nRECEIPT_SHA256=%s\n' "${target}" "${receipt_sha}"
