#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
NAVDP_PY=${LOCAL_NAVDP_PY:-/home/asus/miniconda3/envs/navdp/bin/python}
PARENT=${LOCAL_BUNDLE_PARENT:-${ROOT}/.diagnostics/source_bundles}
PROTOCOL_SHA=9606f129c1bfc80d57d070151dcb97b960a28f72d9875d0a39779052395e2640
fail() { echo "ABORT: $*" >&2; exit 2; }

files=(
  MemNavData/LONG_RANGE_ORACLE_ATTRIBUTION_PROTOCOL_20260903.md
  MemNavData/LONG_RANGE_ORACLE_ATTRIBUTION_PROTOCOL_20260903.md.sha256
  MemNavData/longrange_oracle_attribution.py
  MemNavData/eval_hm3d_longrange_oracle_attribution.py
  MemNavData/run_hm3d_longrange_oracle_attribution.py
  MemNavData/analyze_hm3d_longrange_oracle_attribution.py
  MemNavData/independent_verify_hm3d_longrange_oracle_attribution.py
  MemNavData/slurm_hm3d_longrange_oracle_attribution.sbatch
  MemNavData/slurm_hm3d_longrange_oracle_attribution_analysis.sbatch
  MemNavData/submit_hm3d_longrange_oracle_attribution_remote.sh
  MemNavData/run_hm3d_action_coordinate_compass_gate.py
  MemNavData/run_hm3d_action_coordinate_compass_pair.py
  MemNavData/analyze_hm3d_action_coordinate_compass_pair.py
  MemNavData/episodic_route_filter.py
  MemNavData/bearing_diagnostics.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/final14_mono_factorial.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/run_final14_mono_factorial_episode.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/audit_hm3d_table3_length_role_pairs.py
  MemNavData/hm3d_table3_length_contract.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/shared_online_double_revisit_runtime.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/xnavdp_revisit_contract.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/slurm_safe_submit.sh
  MemNavData/test_longrange_oracle_attribution.py
  MemNavData/test_hm3d_longrange_oracle_attribution.py
  MemNavData/test_navdp_memory_replay.py
  MemNavData/test_hm3d_action_coordinate_compass_gate.py
  MemNavData/test_hm3d_action_coordinate_compass_pair.py
  MemNavData/test_shared_online_double_revisit_runtime.py
  MemNavData/test_revisit_bearing_adapter.py
  MemNavData/test_policy_agent_action_coordinate.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/memnav/router_candidates.py
)
for path in "${files[@]}"; do
  [[ -f "${ROOT}/${path}" && ! -L "${ROOT}/${path}" ]] || \
    fail "missing physical source ${path}"
done
[[ -x "${PY}" && -x "${NAVDP_PY}" ]] || fail "local interpreter missing"

cd "${ROOT}"
(cd MemNavData && sha256sum -c --quiet \
  LONG_RANGE_ORACLE_ATTRIBUTION_PROTOCOL_20260903.md.sha256)
[[ "$(sha256sum MemNavData/LONG_RANGE_ORACLE_ATTRIBUTION_PROTOCOL_20260903.md | awk '{print $1}')" == "${PROTOCOL_SHA}" ]] || \
  fail "oracle protocol changed"
env PYTHONPATH="${ROOT}/MemNavData:${ROOT}" "${PY}" -m unittest -q \
  MemNavData.test_longrange_oracle_attribution \
  MemNavData.test_navdp_memory_replay \
  MemNavData.test_hm3d_action_coordinate_compass_gate \
  MemNavData.test_hm3d_action_coordinate_compass_pair \
  MemNavData.test_shared_online_double_revisit_runtime \
  MemNavData.test_revisit_bearing_adapter \
  MemNavData.test_policy_agent_action_coordinate
env PYTHONPATH="${ROOT}/MemNavData:${ROOT}" "${PY}" - <<'PY'
from MemNavData.test_hm3d_longrange_oracle_attribution import (
    test_four_arm_rotation_is_balanced,
    test_oracle_receipt_audits_mixed_and_pure_controllers,
)
test_four_arm_rotation_is_balanced()
test_oracle_receipt_audits_mixed_and_pure_controllers()
PY
"${PY}" -m py_compile \
  MemNavData/longrange_oracle_attribution.py \
  MemNavData/eval_hm3d_longrange_oracle_attribution.py \
  MemNavData/run_hm3d_longrange_oracle_attribution.py \
  MemNavData/analyze_hm3d_longrange_oracle_attribution.py \
  MemNavData/independent_verify_hm3d_longrange_oracle_attribution.py \
  MemNavData/eval_2leg_habitat.py \
  MemNavData/eval_shared_online_role_pairs.py \
  NavDP/baselines/navdp/policy_agent.py \
  NavDP/baselines/navdp/navdp_server.py
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_longrange_oracle_attribution.sbatch \
  MemNavData/slurm_hm3d_longrange_oracle_attribution_analysis.sbatch \
  MemNavData/submit_hm3d_longrange_oracle_attribution_remote.sh \
  MemNavData/prepare_hm3d_longrange_oracle_attribution_bundle.sh

mkdir -p "${PARENT}"
stage=$(mktemp -d "${PARENT}/hm3d_longrange_oracle.partial.XXXXXX")
for path in "${files[@]}"; do
  mkdir -p "${stage}/$(dirname "${path}")"
  cp --preserve=mode,timestamps "${ROOT}/${path}" "${stage}/${path}"
done
(
  cd "${stage}"
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}/MemNavData:${stage}" "${PY}" -m unittest -q \
      MemNavData.test_longrange_oracle_attribution \
      MemNavData.test_hm3d_action_coordinate_compass_gate \
      MemNavData.test_hm3d_action_coordinate_compass_pair \
      MemNavData.test_shared_online_double_revisit_runtime \
      MemNavData.test_revisit_bearing_adapter \
      MemNavData.test_policy_agent_action_coordinate
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}/NavDP/baselines/navdp:${ROOT}/NavDP/baselines/navdp:${stage}/MemNavData:${stage}" \
    "${NAVDP_PY}" -m unittest -q \
      MemNavData.test_navdp_memory_replay
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}/NavDP/baselines/navdp:${ROOT}/NavDP/baselines/navdp:${stage}/MemNavData:${stage}" \
    "${NAVDP_PY}" - "${stage}" <<'PY'
from pathlib import Path
import policy_agent
stage=Path(__import__('sys').argv[1]).resolve()
assert Path(policy_agent.__file__).resolve() == (
    stage/'NavDP/baselines/navdp/policy_agent.py')
assert hasattr(policy_agent.NavDP_Agent, 'resample_pointgoal')
PY
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}/MemNavData:${stage}" "${PY}" - <<'PY'
from MemNavData.test_hm3d_longrange_oracle_attribution import (
    test_four_arm_rotation_is_balanced,
    test_oracle_receipt_audits_mixed_and_pure_controllers,
)
test_four_arm_rotation_is_balanced()
test_oracle_receipt_audits_mixed_and_pure_controllers()
PY
  if find . -type d -name __pycache__ -print -quit | grep -q .; then
    fail "bundle-only self-test leaked Python bytecode"
  fi
)

head=$(git -C "${ROOT}" rev-parse HEAD)
"${PY}" - "${stage}" "${head}" "${PROTOCOL_SHA}" <<'PY'
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
 'schema_version':'hm3d_longrange_oracle_attribution_bundle_v1_20260903',
 'local_git_head_context':sys.argv[2],
 'scientific_scope':'consumed 16-history privileged attribution; not paper result',
 'protocol_sha256':sys.argv[3],
 'population_indices':list(range(32,48)),
 'distance_bin':'30_to_50_m',
 'query_role':'revisit',
 'arms':['action_coordinate_mixed','oracle_route_mixed',
         'oracle_geodesic_mixed','oracle_geodesic_point'],
 'privileged_evaluator_pose':True,
 'read_only_controller_resampling':True,
 'canonical_cec_default_changed':False,
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
target=${PARENT}/hm3d_longrange_oracle_attribution_${receipt_sha:0:16}
[[ ! -e "${target}" ]] || fail "bundle already exists: ${target}"
chmod -R a-w "${stage}"
mv "${stage}" "${target}"
printf 'BUNDLE=%s\nRECEIPT_SHA256=%s\n' "${target}" "${receipt_sha}"
