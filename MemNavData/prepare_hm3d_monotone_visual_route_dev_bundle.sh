#!/usr/bin/env bash
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
PARENT=${LOCAL_BUNDLE_PARENT:-${ROOT}/.diagnostics/source_bundles}
fail() { echo "ABORT: $*" >&2; exit 2; }

files=(
  MemNavData/LONG_RANGE_EPISODIC_PATH_FIELD_PROTOCOL_20260902.md
  MemNavData/LONG_RANGE_MONOTONE_VISUAL_ROUTE_DEV_PROTOCOL_20260902.md
  MemNavData/episodic_path_field.py
  MemNavData/audit_episodic_path_field_information.py
  MemNavData/bearing_diagnostics.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/final14_mono_factorial.py
  MemNavData/mdtec_raw_depth_gate_d.py
  MemNavData/hm3d_fullmono_mixed_role.py
  MemNavData/run_final14_mono_factorial_episode.py
  MemNavData/run_hm3d_fullmono_query_history.py
  MemNavData/run_hm3d_episodic_path_field_gate.py
  MemNavData/run_hm3d_monotone_visual_route_dev.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/lingbot_pnp_localization.py
  MemNavData/lingbot_colored_registration.py
  MemNavData/slurm_hm3d_monotone_visual_route_dev.sbatch
  MemNavData/submit_hm3d_monotone_visual_route_dev_remote.sh
  MemNavData/slurm_safe_submit.sh
  MemNavData/test_episodic_path_field.py
  MemNavData/test_audit_episodic_path_field_information.py
  MemNavData/test_policy_agent_path_field.py
  MemNavData/test_hm3d_episodic_path_field_gate.py
  MemNavData/test_hm3d_monotone_visual_route_dev.py
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
"${PY}" -m unittest -q \
  MemNavData.test_episodic_path_field \
  MemNavData.test_audit_episodic_path_field_information \
  MemNavData.test_policy_agent_path_field \
  MemNavData.test_hm3d_episodic_path_field_gate \
  MemNavData.test_hm3d_monotone_visual_route_dev \
  MemNavData.test_policy_agent_graph
"${PY}" -m py_compile \
  MemNavData/episodic_path_field.py \
  MemNavData/audit_episodic_path_field_information.py \
  MemNavData/run_hm3d_episodic_path_field_gate.py \
  MemNavData/run_hm3d_monotone_visual_route_dev.py \
  MemNavData/eval_2leg_habitat.py \
  NavDP/baselines/memnav/policy_agent.py \
  NavDP/baselines/memnav/memnav_server.py
bash -n MemNavData/slurm_hm3d_monotone_visual_route_dev.sbatch \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/prepare_hm3d_monotone_visual_route_dev_bundle.sh \
  MemNavData/submit_hm3d_monotone_visual_route_dev_remote.sh

mkdir -p "${PARENT}"
stage=$(mktemp -d "${PARENT}/hm3d_visual_route_dev.partial.XXXXXX")
for path in "${files[@]}"; do
  mkdir -p "${stage}/$(dirname "${path}")"
  cp --preserve=mode,timestamps "${ROOT}/${path}" "${stage}/${path}"
done
(
  cd "${stage}"
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}" "${PY}" -m unittest -q \
      MemNavData.test_episodic_path_field \
      MemNavData.test_audit_episodic_path_field_information \
      MemNavData.test_policy_agent_path_field \
      MemNavData.test_hm3d_episodic_path_field_gate \
      MemNavData.test_hm3d_monotone_visual_route_dev
  env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
    PYTHONPATH="${stage}" "${PY}" -c \
      'import MemNavData.lingbot_pnp_localization; import MemNavData.run_hm3d_monotone_visual_route_dev; import NavDP.baselines.memnav.policy_agent'
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
    if path.is_file() and path.name not in {'SOURCE_BUNDLE.sha256','source_bundle_manifest.json'}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 'schema_version':'hm3d_monotone_visual_route_dev_bundle_v1_20260902',
 'local_git_head_context':sys.argv[2],
 'scientific_scope':'consumed 48-history native-endpoint-route paired development',
 'arms':['mono_native','mono_cec_endpoint','mono_cec_visual_route'],
 'population_size':48,
 'canonical_cec_default_changed':False,
 'route_depth_cache_stride':8,
 'route_localization_horizon':'all_remaining_authorized_route',
 'route_control_horizon_m':2.5,
 'files':files,
}
(root/'source_bundle_manifest.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(
  cd "${stage}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
receipt_sha=$(sha256sum "${stage}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
target=${PARENT}/hm3d_monotone_visual_route_dev_${receipt_sha:0:16}
[[ ! -e "${target}" ]] || fail "bundle already exists: ${target}"
chmod -R a-w "${stage}"
mv "${stage}" "${target}"
printf 'BUNDLE=%s\nRECEIPT_SHA256=%s\n' "${target}" "${receipt_sha}"
