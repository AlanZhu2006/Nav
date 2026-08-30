#!/usr/bin/env bash
# Consumed-scene integration smoke for the all-CEC controller comparison.
#
# Every controller receives the same role-free CEC proof stream.  A rejected
# action falls back to the same monocular NavDP ImageGoal controller.  An
# accepted action is projected through the selected controller's audited CEC
# adapter.  Both temporal controllers receive observation-only shadow updates,
# so a later per-action accept/reject transition remains causal.

set -euo pipefail
umask 0022

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
CODE_OVERLAY_ROOT=${CODE_OVERLAY_ROOT:-}
HUB_SCRIPT=${ROOT}/MemNavData/cec_controller_portability_hub.py
HUB_CLI_COMPAT_ROOT=${CODE_OVERLAY_ROOT:-${ROOT}}
HUB_CLI_COMPAT=${HUB_CLI_COMPAT_ROOT}/MemNavData/cec_hub_cli_compat.py
CONTROLLER=${CONTROLLER:-navdp}
EVAL_KIND=${EVAL_KIND:-nnr_revisit}
HM3D_LIFELONG_PAIRED_SCOPES=${HM3D_LIFELONG_PAIRED_SCOPES:-}
LIFELONG_SHARED_C_ROOT=${LIFELONG_SHARED_C_ROOT:-}
SCENE=${SCENE:-dhjEzFoUFzH}
EPISODE=${EPISODE:-episode_0005}
MAX_STEPS=${MAX_STEPS:-600}
EVAL_SEED=${EVAL_SEED:-5}

MEMNAV_PORT=${MEMNAV_PORT:-21840}
FALLBACK_PORT=${FALLBACK_PORT:-21841}
UPSTREAM_PORT=${UPSTREAM_PORT:-21842}
PROXY_PORT=${PROXY_PORT:-21843}
HUB_PORT=${HUB_PORT:-21844}
FORCED_HUB_PORT=${FORCED_HUB_PORT:-21845}

# A controller-portability query can run the granted and forced-reject arms
# against the same already-loaded MemNav/NavDP/controller processes.  This is
# the paired unit used by the proof-locked Fresh-HM3D experiment; it avoids the
# cross-machine CUDA confound that invalidated older controller tables.
PORTABILITY_AUTHORITY_PAIR=${PORTABILITY_AUTHORITY_PAIR:-0}
PORTABILITY_AUTHORITY_ORDER=${PORTABILITY_AUTHORITY_ORDER:-grant,forced_reject_native}
# The default preserves every prior authority-pair run.  A separately frozen
# held-out ViNT protocol may opt into the physical proof-bearing executor; the
# forced-reject arm remains byte-for-byte native and therefore always receives
# ``off`` below.
PORTABILITY_CEC_ACCEPT_ALIGNMENT=${PORTABILITY_CEC_ACCEPT_ALIGNMENT:-off}
PORTABILITY_DIRECTION_TRIPLE=${PORTABILITY_DIRECTION_TRIPLE:-0}
PORTABILITY_DIRECTION_ORDER=${PORTABILITY_DIRECTION_ORDER:-anchor_unaligned,native_bearing_aligned,anchor_bearing_aligned}
ROLE_PAIR_QUERY_ROLE=${ROLE_PAIR_QUERY_ROLE:-all}
ROLE_PAIR_QUERY_MANIFEST=${ROLE_PAIR_QUERY_MANIFEST:-}
CEC_REJECT_POLICY=${CEC_REJECT_POLICY:-shared_native_exact}
ROLE_PAIR_SCOPE=${ROLE_PAIR_SCOPE:-consumed_integration}

MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
VINT_PY=${VINT_PY:-${ROOT}/.diagnostics/controller_portability_20260821/envs/vint/bin/python}
# GNM and NoMaD are RGB-only members of the same visualnav-transformer family
# as ViNT.  GNM needs nothing beyond the plain navdp env; NoMaD additionally
# needs the diffusion_policy submodule on PYTHONPATH (see NOMAD_PYTHONPATH_EXTRA
# below).  Both run cleanly in the already-sealed vint venv, so no new venv is
# built for them.
GNM_PY=${GNM_PY:-${VINT_PY}}
NOMAD_PY=${NOMAD_PY:-${VINT_PY}}
VIPLANNER_PY=${VIPLANNER_PY:-${ROOT}/.diagnostics/controller_portability_20260821/envs/viplanner-py310-cu118/bin/python}

MEMNAV_CKPT=${MEMNAV_CKPT:-/home/asus/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/home/asus/Research/Nav/NavDP/baselines/navdp/checkpoints/navdp_checkpoint.ckpt}
VINT_CKPT=${VINT_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/vint.pth}
GNM_CKPT=${GNM_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/gnm.pth}
NOMAD_CKPT=${NOMAD_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/nomad.pth}
IPLANNER_CKPT=${IPLANNER_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/iplanner.pth}
VIPLANNER_CKPT=${VIPLANNER_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/viplanner.pt}
MASK2FORMER_CKPT=${MASK2FORMER_CKPT:-${ROOT}/.diagnostics/controller_portability_20260821/checkpoints/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth}
MASK2FORMER_CONFIG=${MASK2FORMER_CONFIG:-${ROOT}/.diagnostics/controller_portability_20260821/envs/viplanner-py310-cu118/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py}

LINGBOT_REPO=${LINGBOT_REPO:-/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
LIGHTGLUE_REPO=${LIGHTGLUE_REPO:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_ROOT=${DEPENDENCY_ROOT:-${ROOT}/.diagnostics/dependencies/python}
INTERNNAV_ROOT=${INTERNNAV_ROOT:-${ROOT}/InternNav}

ASSET_ROOT=${ASSET_ROOT:-/home/asus/Research/datasets/mp3d_20scene/assets}
SCENE_FILE=${SCENE_FILE:-${ASSET_ROOT}/${SCENE}/${SCENE}.glb}
BENCHMARK_ROOT=${BENCHMARK_ROOT:-${ROOT}/.diagnostics/shared_online_nnr_smoke_20260813/benchmark_qw_native/${SCENE}}
TRACE_ROOT=${TRACE_ROOT:-${ROOT}/.diagnostics/shared_online_nnr_smoke_20260813/qw_native_shared_traces/${SCENE}}
RUN_ROOT=${RUN_ROOT:-${ROOT}/.diagnostics/controller_portability_20260821/local_cec_${CONTROLLER}_${SCENE}_${EPISODE}}

fail() { echo "ABORT: $*" >&2; exit 2; }

case "${CONTROLLER}" in
  navdp|vint|gnm|nomad|iplanner|viplanner) ;;
  *) fail "CONTROLLER must be navdp, vint, gnm, nomad, iplanner, or viplanner" ;;
esac
case "${EVAL_KIND}" in
  nnr_revisit|role_pair_mixed|lifelong_5leg|lifelong_nnr|hm3d_lifelong|lifelong_shared_c_collect|lifelong_shared_c_b2|hm3d_shared_c_collect|hm3d_shared_c_b2) ;;
  *) fail "unsupported EVAL_KIND=${EVAL_KIND}" ;;
esac
case "${ROLE_PAIR_QUERY_ROLE}" in
  all|novel|revisit) ;;
  *) fail "ROLE_PAIR_QUERY_ROLE must be all, novel, or revisit" ;;
esac
case "${ROLE_PAIR_SCOPE}" in
  consumed_integration|paper_heldout|paper_replication) ;;
  *) fail "ROLE_PAIR_SCOPE must be consumed_integration, paper_heldout, or paper_replication" ;;
esac
case "${PORTABILITY_AUTHORITY_PAIR}" in
  0|1) ;;
  *) fail "PORTABILITY_AUTHORITY_PAIR must be 0 or 1" ;;
esac
case "${PORTABILITY_CEC_ACCEPT_ALIGNMENT}" in
  off|first_certified_bounded) ;;
  *) fail "PORTABILITY_CEC_ACCEPT_ALIGNMENT must be off or first_certified_bounded" ;;
esac
case "${PORTABILITY_DIRECTION_TRIPLE}" in
  0|1) ;;
  *) fail "PORTABILITY_DIRECTION_TRIPLE must be 0 or 1" ;;
esac
if [[ "${PORTABILITY_DIRECTION_TRIPLE}" == 1 ]]; then
  [[ "${PORTABILITY_AUTHORITY_PAIR}" == 0 ]] || \
    fail "direction triple and authority pair are mutually exclusive"
  [[ "${CONTROLLER}" == vint && "${EVAL_KIND}" == role_pair_mixed ]] || \
    fail "direction triple is restricted to the ViNT role-pair mechanism test"
  [[ "${CEC_REJECT_POLICY}" == controller_native_exact ]] || \
    fail "direction triple requires ViNT-native exact fallback"
  [[ "${ROLE_PAIR_SCOPE}" == consumed_integration ]] || \
    fail "direction triple is consumed development only"
  [[ -n "${ROLE_PAIR_QUERY_MANIFEST}" ]] || \
    fail "direction triple requires a frozen query manifest"
  [[ "${PORTABILITY_CEC_ACCEPT_ALIGNMENT}" == off ]] || \
    fail "direction triple owns its alignment treatments"
  case "${PORTABILITY_DIRECTION_ORDER}" in
    anchor_unaligned,native_bearing_aligned,anchor_bearing_aligned|\
    native_bearing_aligned,anchor_bearing_aligned,anchor_unaligned|\
    anchor_bearing_aligned,anchor_unaligned,native_bearing_aligned) ;;
    *) fail "invalid balanced PORTABILITY_DIRECTION_ORDER" ;;
  esac
fi
case "${CEC_REJECT_POLICY}" in
  shared_native_exact) ;;
  controller_native_exact)
    [[ "${CONTROLLER}" == vint || "${CONTROLLER}" == gnm \
       || "${CONTROLLER}" == nomad ]] || \
      fail "controller_native_exact requires an RGB ImageGoal controller"
    ;;
  *) fail "invalid CEC_REJECT_POLICY=${CEC_REJECT_POLICY}" ;;
esac
if [[ "${PORTABILITY_AUTHORITY_PAIR}" == 1 ]]; then
  [[ "${EVAL_KIND}" == role_pair_mixed ]] || \
    fail "authority pairing is currently restricted to role_pair_mixed"
  [[ "${PORTABILITY_AUTHORITY_ORDER}" == grant,forced_reject_native \
     || "${PORTABILITY_AUTHORITY_ORDER}" == forced_reject_native,grant ]] || \
    fail "invalid PORTABILITY_AUTHORITY_ORDER"
fi
if [[ "${PORTABILITY_CEC_ACCEPT_ALIGNMENT}" == first_certified_bounded ]]; then
  [[ "${PORTABILITY_AUTHORITY_PAIR}" == 1 \
     && "${CONTROLLER}" == vint \
     && "${EVAL_KIND}" == role_pair_mixed ]] || \
    fail "bounded CEC alignment requires a paired ViNT role-pair run"
  [[ "${CEC_REJECT_POLICY}" == controller_native_exact ]] || \
    fail "bounded CEC alignment requires exact native ViNT rejection"
  [[ "${ROLE_PAIR_SCOPE}" == paper_heldout \
     || "${ROLE_PAIR_SCOPE}" == paper_replication ]] || \
    fail "bounded CEC alignment requires a frozen complete population"
  [[ -z "${ROLE_PAIR_QUERY_MANIFEST}" ]] || \
    fail "bounded held-out formal must run the complete population"
fi
if [[ -n "${ROLE_PAIR_QUERY_MANIFEST}" ]]; then
  [[ "${EVAL_KIND}" == role_pair_mixed ]] || \
    fail "ROLE_PAIR_QUERY_MANIFEST requires role_pair_mixed"
  [[ "${ROLE_PAIR_QUERY_ROLE}" == all ]] || \
    fail "query manifest cannot be combined with role filtering"
  [[ -r "${ROLE_PAIR_QUERY_MANIFEST}" ]] || \
    fail "missing ROLE_PAIR_QUERY_MANIFEST=${ROLE_PAIR_QUERY_MANIFEST}"
fi
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || fail "MAX_STEPS must be positive"
if [[ -n "${HM3D_LIFELONG_PAIRED_SCOPES}" ]]; then
  [[ "${EVAL_KIND}" == hm3d_lifelong \
     || "${EVAL_KIND}" == lifelong_shared_c_b2 \
     || "${EVAL_KIND}" == hm3d_shared_c_b2 ]] || \
    fail "paired scopes require a lifelong B2 evaluator"
  [[ "${LIFELONG_HISTORY_SCOPE:-}" != forced_reject_native ]] || \
    fail "forced-reject requires its own hub process"
fi

required=(
  "${MEMNAV_PY}"
  "${HAB_PY}"
  "${MEMNAV_CKPT}"
  "${NAVDP_CKPT}"
  "${LINGBOT_WEIGHTS}"
  "${SCENE_FILE}"
  "${HUB_SCRIPT}"
  "${HUB_CLI_COMPAT}"
  "${ROOT}/MemNavData/controller_portability_proxy.py"
)
if [[ "${EVAL_KIND}" == nnr_revisit \
   || "${EVAL_KIND}" == lifelong_nnr \
   || "${EVAL_KIND}" == lifelong_shared_c_collect \
   || "${EVAL_KIND}" == lifelong_shared_c_b2 ]]; then
  required+=(
    "${BENCHMARK_ROOT}/manifest.json"
    "${BENCHMARK_ROOT}/${EPISODE}/benchmark.json"
    "${TRACE_ROOT}/${EPISODE}_leg1_trace.json"
    "${TRACE_ROOT}/${EPISODE}_legB_trace.json"
    "${ROOT}/MemNavData/eval_shared_online_novel_revisit.py"
  )
  if [[ "${EVAL_KIND}" == lifelong_shared_c_b2 ]]; then
    required+=(
      "${LIFELONG_SHARED_C_ROOT}/population.json"
      "${LIFELONG_SHARED_C_ROOT}/SEALED"
      "${ROOT}/MemNavData/eval_lifelong_shared_c_b2.py"
    )
  elif [[ "${EVAL_KIND}" == lifelong_shared_c_collect ]]; then
    required+=("${ROOT}/MemNavData/collect_lifelong_shared_c.py")
  fi
elif [[ "${EVAL_KIND}" == hm3d_lifelong \
     || "${EVAL_KIND}" == hm3d_shared_c_collect \
     || "${EVAL_KIND}" == hm3d_shared_c_b2 ]]; then
  required+=(
    "${BENCHMARK_ROOT}/${EPISODE}/benchmark.json"
    "${BENCHMARK_ROOT}/${EPISODE}/${EPISODE}_legB_trace.json"
    "${ROOT}/MemNavData/eval_hm3d_fullmono_lifelong.py"
  )
  if [[ "${EVAL_KIND}" == hm3d_shared_c_collect ]]; then
    required+=("${ROOT}/MemNavData/collect_hm3d_lifelong_shared_c.py")
  elif [[ "${EVAL_KIND}" == hm3d_shared_c_b2 ]]; then
    required+=(
      "${ROOT}/MemNavData/eval_hm3d_lifelong_shared_c_b2.py"
      "${LIFELONG_SHARED_C_ROOT}/population.json"
      "${LIFELONG_SHARED_C_ROOT}/SEALED"
    )
  fi
elif [[ "${EVAL_KIND}" == role_pair_mixed ]]; then
  required+=(
    "${BENCHMARK_ROOT}/../manifest.json"
    "${BENCHMARK_ROOT}/${EPISODE}/role_pairs.json"
    "${ROOT}/MemNavData/eval_shared_online_role_pairs.py"
  )
else
  required+=(
    "${BENCHMARK_ROOT}/${EPISODE}/meta/gen_meta.json"
    "${BENCHMARK_ROOT}/${EPISODE}/goal_1.jpg"
    "${BENCHMARK_ROOT}/${EPISODE}/goal_2.jpg"
    "${ROOT}/MemNavData/eval_lifelong_5leg_habitat.py"
  )
fi
case "${CONTROLLER}" in
  vint) required+=("${VINT_PY}" "${VINT_CKPT}") ;;
  gnm) required+=("${GNM_PY}" "${GNM_CKPT}") ;;
  nomad) required+=("${NOMAD_PY}" "${NOMAD_CKPT}") ;;
  iplanner) required+=("${IPLANNER_CKPT}") ;;
  viplanner)
    required+=("${VIPLANNER_PY}" "${VIPLANNER_CKPT}"
               "${MASK2FORMER_CKPT}" "${MASK2FORMER_CONFIG}")
    ;;
esac
for item in "${required[@]}"; do
  [[ -r "${item}" ]] || fail "missing input ${item}"
done
[[ ! -e "${RUN_ROOT}" ]] || fail "output already exists: ${RUN_ROOT}"

ports=("${MEMNAV_PORT}" "${FALLBACK_PORT}" "${HUB_PORT}")
if [[ "${PORTABILITY_AUTHORITY_PAIR}" == 1 \
   || "${PORTABILITY_DIRECTION_TRIPLE}" == 1 ]]; then
  ports+=("${FORCED_HUB_PORT}")
fi
if [[ "${CONTROLLER}" != navdp ]]; then
  ports+=("${UPSTREAM_PORT}" "${PROXY_PORT}")
fi
for port in "${ports[@]}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    fail "port ${port} is already in use"
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
if [[ "${PORTABILITY_AUTHORITY_PAIR}" == 1 ]]; then
  "${MEMNAV_PY}" - "${RUN_ROOT}/authority_pair_contract.json" \
    "${CONTROLLER}" "${SCENE}" "${EPISODE}" "${MAX_STEPS}" \
    "${PORTABILITY_AUTHORITY_ORDER}" "${ROLE_PAIR_QUERY_MANIFEST}" \
    "${BENCHMARK_ROOT}/../manifest.json" "${CEC_REJECT_POLICY}" \
    "${ROLE_PAIR_SCOPE}" "${PORTABILITY_CEC_ACCEPT_ALIGNMENT}" <<'PY'
import hashlib,json,sys
(path,controller,scene,episode,max_steps,order,query_manifest,
 benchmark_manifest,reject_policy,role_pair_scope,accept_alignment)=sys.argv[1:]
def digest(path):
 return hashlib.sha256(open(path,"rb").read()).hexdigest()
payload={
 "schema_version":(
   "cec_authority_pair_contract_v3_20260829"
   if accept_alignment == "first_certified_bounded"
   else "cec_authority_pair_contract_v2_20260828"),
 "controller":controller,"scene":scene,"episode":episode,
 "max_steps":int(max_steps),"authority_order":order.split(","),
 "same_loaded_processes":True,"runtime_role_visibility":"none",
 "handoff_packets_required":True,
 "reject_policy":reject_policy,"role_pair_scope":role_pair_scope,
 "query_manifest_path":query_manifest or None,
 "query_manifest_sha256":digest(query_manifest) if query_manifest else None,
 "benchmark_manifest_path":benchmark_manifest,
 "benchmark_manifest_sha256":digest(benchmark_manifest),
}
if accept_alignment == "first_certified_bounded":
 payload.update({
   "grant_bearing_alignment":accept_alignment,
   "forced_reject_bearing_alignment":"off",
 })
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
fi
if [[ "${PORTABILITY_DIRECTION_TRIPLE}" == 1 ]]; then
  "${MEMNAV_PY}" - "${RUN_ROOT}/direction_triple_contract.json" \
    "${CONTROLLER}" "${SCENE}" "${EPISODE}" "${MAX_STEPS}" \
    "${PORTABILITY_DIRECTION_ORDER}" "${ROLE_PAIR_QUERY_MANIFEST}" \
    "${BENCHMARK_ROOT}/../manifest.json" "${CEC_REJECT_POLICY}" <<'PY'
import hashlib,json,sys
(path,controller,scene,episode,max_steps,order,query_manifest,
 benchmark_manifest,reject_policy)=sys.argv[1:]
def digest(value):
 return hashlib.sha256(open(value,"rb").read()).hexdigest()
payload={
 "schema_version":"vint_cec_direction_triple_contract_v1_20260828",
 "scope":"outcome-aware consumed mechanism test; not a paper SR result",
 "controller":controller,"scene":scene,"episode":episode,
 "max_steps":int(max_steps),"arm_order":order.split(","),
 "same_loaded_processes":True,"runtime_role_visibility":"none",
 "handoff_packets_required":True,"reject_policy":reject_policy,
 "query_manifest_path":query_manifest,
 "query_manifest_sha256":digest(query_manifest),
 "benchmark_manifest_path":benchmark_manifest,
 "benchmark_manifest_sha256":digest(benchmark_manifest),
 "alignment_contract":(
   "first certified robot-local bearing; idealized zero-translation yaw; "
   "then unchanged controller-local trajectory"),
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
fi
runtime_root=$(mktemp -d /tmp/cec_controller_portability.XXXXXX)
mkdir -p "${runtime_root}/memnav" "${runtime_root}/fallback" \
         "${runtime_root}/upstream" "${runtime_root}/proxy" \
         "${runtime_root}/hub" "${runtime_root}/forced_hub"

memnav_pid=
fallback_pid=
upstream_pid=
proxy_pid=
hub_pid=
forced_hub_pid=
cleanup() {
  for process_id in "${forced_hub_pid}" "${hub_pid}" "${proxy_pid}" "${upstream_pid}" \
                    "${fallback_pid}" "${memnav_pid}"; do
    if [[ -n "${process_id}" ]] && kill -0 "${process_id}" 2>/dev/null; then
      kill "${process_id}" 2>/dev/null || true
      wait "${process_id}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

wait_for_port() {
  local label=$1 pid=$2 port=$3 log=$4 ready=0
  for _attempt in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 160 "${log}" >&2 || true
      fail "${label} exited during startup"
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || fail "${label} did not bind port ${port}"
}

hab_site_packages=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
hab_requests_vendor=${hab_site_packages}/pip/_vendor
source_pythonpath=${ROOT}
if [[ -n "${CODE_OVERLAY_ROOT}" ]]; then
  [[ -d "${CODE_OVERLAY_ROOT}/MemNavData" ]] || \
    fail "missing CODE_OVERLAY_ROOT=${CODE_OVERLAY_ROOT}"
  for overlay_file in cec_hub_cli_compat.py navdp_replay_contract.py \
    eval_3leg_habitat.py \
    collect_hm3d_lifelong_shared_c.py eval_hm3d_lifelong_shared_c_b2.py; do
    [[ -r "${CODE_OVERLAY_ROOT}/MemNavData/${overlay_file}" ]] || \
      fail "missing overlay module ${overlay_file}"
  done
  source_pythonpath=${CODE_OVERLAY_ROOT}/MemNavData:${CODE_OVERLAY_ROOT}:${ROOT}/MemNavData:${ROOT}
fi
hab_pythonpath=${source_pythonpath}:${hab_requests_vendor}
requests_init=${hab_requests_vendor}/requests/__init__.py
requests_version=${hab_requests_vendor}/requests/__version__.py
[[ -r "${requests_init}" && -r "${requests_version}" ]] || \
  fail "missing Habitat vendored requests dependency"
if [[ -n "${EXPECTED_HAB_REQUESTS_VERSION:-}" ]]; then
  : "${EXPECTED_HAB_REQUESTS_INIT_BYTES:?}" \
    "${EXPECTED_HAB_REQUESTS_INIT_SHA:?}" \
    "${EXPECTED_HAB_REQUESTS_VERSION_BYTES:?}" \
    "${EXPECTED_HAB_REQUESTS_VERSION_SHA:?}"
  [[ "$(stat -c '%s' "${requests_init}")" == \
     "${EXPECTED_HAB_REQUESTS_INIT_BYTES}" ]] || \
    fail "Habitat vendored requests __init__ size changed"
  [[ "$(sha256sum "${requests_init}" | awk '{print $1}')" == \
     "${EXPECTED_HAB_REQUESTS_INIT_SHA}" ]] || \
    fail "Habitat vendored requests __init__ hash changed"
  [[ "$(stat -c '%s' "${requests_version}")" == \
     "${EXPECTED_HAB_REQUESTS_VERSION_BYTES}" ]] || \
    fail "Habitat vendored requests version size changed"
  [[ "$(sha256sum "${requests_version}" | awk '{print $1}')" == \
     "${EXPECTED_HAB_REQUESTS_VERSION_SHA}" ]] || \
    fail "Habitat vendored requests version hash changed"
  env PYTHONPATH="${hab_requests_vendor}" "${HAB_PY}" -c \
    'import requests,sys; assert requests.__version__ == sys.argv[1]; assert "/pip/_vendor/requests/" in requests.__file__' \
    "${EXPECTED_HAB_REQUESTS_VERSION}" || \
    fail "Habitat vendored requests version/import mismatch"
else
  env PYTHONPATH="${hab_requests_vendor}" "${HAB_PY}" -c \
    'import requests; assert "/pip/_vendor/requests/" in requests.__file__' || \
    fail "Habitat vendored requests import failed"
fi

"${MEMNAV_PY}" -m py_compile \
  "${ROOT}/MemNavData/controller_portability_contract.py" \
  "${ROOT}/MemNavData/controller_portability_proxy.py" \
  "${HUB_SCRIPT}" "${HUB_CLI_COMPAT}"
"${HAB_PY}" -m py_compile \
  "${ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${ROOT}/MemNavData/eval_3leg_habitat.py" \
  "${ROOT}/MemNavData/eval_lifelong_5leg_habitat.py" \
  "${ROOT}/MemNavData/eval_shared_online_lifelong_nnr.py" \
  "${ROOT}/MemNavData/collect_lifelong_shared_c.py" \
  "${ROOT}/MemNavData/eval_lifelong_shared_c_b2.py" \
  "${ROOT}/MemNavData/eval_hm3d_fullmono_lifelong.py" \
  "${ROOT}/MemNavData/collect_hm3d_lifelong_shared_c.py" \
  "${ROOT}/MemNavData/eval_hm3d_lifelong_shared_c_b2.py" \
  "${ROOT}/MemNavData/eval_shared_online_novel_revisit.py" \
  "${ROOT}/MemNavData/eval_shared_online_role_pairs.py"
if [[ -n "${CODE_OVERLAY_ROOT}" ]]; then
  "${HAB_PY}" -m py_compile \
    "${CODE_OVERLAY_ROOT}/MemNavData/navdp_replay_contract.py" \
    "${CODE_OVERLAY_ROOT}/MemNavData/eval_3leg_habitat.py" \
    "${CODE_OVERLAY_ROOT}/MemNavData/collect_hm3d_lifelong_shared_c.py" \
    "${CODE_OVERLAY_ROOT}/MemNavData/eval_hm3d_lifelong_shared_c_b2.py"
  env PYTHONPATH="${hab_pythonpath}" "${HAB_PY}" - \
    "${CODE_OVERLAY_ROOT}" <<'PY'
import importlib.util
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
for name in (
    "navdp_replay_contract", "eval_3leg_habitat",
    "collect_hm3d_lifelong_shared_c", "eval_hm3d_lifelong_shared_c_b2",
):
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.origin is not None, name
    actual = pathlib.Path(spec.origin).resolve()
    expected = (root / "MemNavData" / f"{name}.py").resolve()
    assert actual == expected, (name, actual, expected)
PY
fi

hub_cli_mode=$("${MEMNAV_PY}" "${HUB_CLI_COMPAT}" \
  --hub-script "${HUB_SCRIPT}" --reject-policy "${CEC_REJECT_POLICY}") || \
  fail "hub reject-policy CLI contract could not be proved"
hub_reject_policy_args=()
case "${hub_cli_mode}" in
  explicit_cli)
    hub_reject_policy_args=(--reject-policy "${CEC_REJECT_POLICY}")
    ;;
  legacy_shared_native_exact)
    [[ "${PORTABILITY_AUTHORITY_PAIR}" == 0 \
       && "${PORTABILITY_DIRECTION_TRIPLE}" == 0 ]] || \
      fail "legacy hub cannot emit sealed authority/direction handoff packets"
    ;;
  *) fail "unknown hub CLI compatibility mode: ${hub_cli_mode}" ;;
esac
printf 'hub_cli_contract=%s reject_policy=%s\n' \
  "${hub_cli_mode}" "${CEC_REJECT_POLICY}"

server_pythonpath=${source_pythonpath}:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${INTERNNAV_ROOT}/src/diffusion-policy
(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${server_pythonpath}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u \
      "${ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${RUN_ROOT}/buffer" \
      --certified_relocalization --lightglue_repo "${LIGHTGLUE_REPO}" \
      --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
      --lightglue_max_keypoints 2048
) >"${RUN_ROOT}/logs/server_memnav.log" 2>&1 &
memnav_pid=$!

(
  cd "${runtime_root}/fallback"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${source_pythonpath}" \
    "${MEMNAV_PY}" -u \
      "${ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${FALLBACK_PORT}" --checkpoint "${NAVDP_CKPT}" \
      --depth_source monocular_sidecar \
      --monocular_depth_url \
        "http://127.0.0.1:${MEMNAV_PORT}/monocular_depth_query"
) >"${RUN_ROOT}/logs/server_fallback_navdp.log" 2>&1 &
fallback_pid=$!

wait_for_port memnav "${memnav_pid}" "${MEMNAV_PORT}" \
  "${RUN_ROOT}/logs/server_memnav.log"
wait_for_port fallback_navdp "${fallback_pid}" "${FALLBACK_PORT}" \
  "${RUN_ROOT}/logs/server_fallback_navdp.log"

controller_url=http://127.0.0.1:${FALLBACK_PORT}
if [[ "${CONTROLLER}" != navdp ]]; then
  case "${CONTROLLER}" in
    vint)
      (
        cd "${ROOT}/NavDP/baselines/vint"
        exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
          "${VINT_PY}" -u vint_server.py --port "${UPSTREAM_PORT}" \
          --robot_config configs/robot_config.yaml \
          --vint_config configs/vint.yaml --vint_checkpoint "${VINT_CKPT}"
      ) >"${RUN_ROOT}/logs/server_vint.log" 2>&1 &
      upstream_pid=$!
      proxy_depth=none
      checkpoint_args=(--checkpoint "vint=${VINT_CKPT}")
      ;;
    gnm)
      (
        cd "${ROOT}/NavDP/baselines/gnm"
        exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
          "${GNM_PY}" -u gnm_server.py --port "${UPSTREAM_PORT}" \
          --robot_config configs/robot_config.yaml \
          --gnm_config configs/gnm.yaml --gnm_checkpoint "${GNM_CKPT}"
      ) >"${RUN_ROOT}/logs/server_gnm.log" 2>&1 &
      upstream_pid=$!
      proxy_depth=none
      checkpoint_args=(--checkpoint "gnm=${GNM_CKPT}")
      ;;
    nomad)
      (
        cd "${ROOT}/NavDP/baselines/nomad"
        exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
          PYTHONPATH="${INTERNNAV_ROOT}/src/diffusion-policy" \
          "${NOMAD_PY}" -u nomad_server.py --port "${UPSTREAM_PORT}" \
          --robot_config configs/robot_config.yaml \
          --data_config configs/data_config.yaml \
          --nomad_config configs/nomad.yaml \
          --nomad_checkpoint "${NOMAD_CKPT}"
      ) >"${RUN_ROOT}/logs/server_nomad.log" 2>&1 &
      upstream_pid=$!
      proxy_depth=none
      checkpoint_args=(--checkpoint "nomad=${NOMAD_CKPT}")
      ;;
    iplanner)
      (
        cd "${ROOT}/NavDP/baselines/iplanner"
        exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
          "${MEMNAV_PY}" -u iplanner_server.py --port "${UPSTREAM_PORT}" \
          --config configs/iplanner.yaml --checkpoint "${IPLANNER_CKPT}"
      ) >"${RUN_ROOT}/logs/server_iplanner.log" 2>&1 &
      upstream_pid=$!
      proxy_depth=monocular_sidecar
      checkpoint_args=(--checkpoint "iplanner=${IPLANNER_CKPT}")
      ;;
    viplanner)
      (
        cd "${ROOT}/NavDP/baselines/viplanner"
        exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
          "${VIPLANNER_PY}" -u viplanner_server.py \
          --port "${UPSTREAM_PORT}" --config configs/viplanner.yaml \
          --checkpoint "${VIPLANNER_CKPT}" \
          --m2f_config "${MASK2FORMER_CONFIG}" \
          --m2f_checkpoint "${MASK2FORMER_CKPT}"
      ) >"${RUN_ROOT}/logs/server_viplanner.log" 2>&1 &
      upstream_pid=$!
      proxy_depth=monocular_sidecar
      checkpoint_args=(
        --checkpoint "planner=${VIPLANNER_CKPT}"
        --checkpoint "mask2former=${MASK2FORMER_CKPT}"
      )
      ;;
  esac
  wait_for_port "${CONTROLLER}" "${upstream_pid}" "${UPSTREAM_PORT}" \
    "${RUN_ROOT}/logs/server_${CONTROLLER}.log"
  (
    cd "${runtime_root}/proxy"
    exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH="${source_pythonpath}" "${MEMNAV_PY}" -u \
      "${ROOT}/MemNavData/controller_portability_proxy.py" \
        --controller "${CONTROLLER}" --protocol cec_proof_hybrid \
        --depth-source "${proxy_depth}" --query-population mixed_role \
        --reject-policy "${CEC_REJECT_POLICY}" \
        --fallback-controller "$([[ "${CEC_REJECT_POLICY}" == controller_native_exact ]] && printf '%s' "${CONTROLLER}" || printf navdp)" \
        --repo-root "${ROOT}" \
        --upstream-base "http://127.0.0.1:${UPSTREAM_PORT}" \
        "${checkpoint_args[@]}" --host 127.0.0.1 --port "${PROXY_PORT}"
  ) >"${RUN_ROOT}/logs/server_proxy.log" 2>&1 &
  proxy_pid=$!
  wait_for_port proxy "${proxy_pid}" "${PROXY_PORT}" \
    "${RUN_ROOT}/logs/server_proxy.log"
  controller_url=http://127.0.0.1:${PROXY_PORT}
fi

hub_extra=()
if [[ "${PORTABILITY_AUTHORITY_PAIR}" == 0 \
   && "${LIFELONG_HISTORY_SCOPE:-}" == forced_reject_native ]]; then
  # Shared-native system baseline: identical pipeline/receipts, no takeover.
  hub_extra+=(--force-reject-native)
fi
if [[ "${PORTABILITY_AUTHORITY_PAIR}" == 1 \
   || "${PORTABILITY_DIRECTION_TRIPLE}" == 1 ]]; then
  hub_extra+=(--emit-handoff-packets)
fi
(
  cd "${runtime_root}/hub"
  exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${source_pythonpath}" \
    "${MEMNAV_PY}" -u \
      "${HUB_SCRIPT}" \
      --host 127.0.0.1 --port "${HUB_PORT}" \
      --controller "${CONTROLLER}" \
      --memnav-url "http://127.0.0.1:${MEMNAV_PORT}" \
      --controller-url "${controller_url}" \
      --fallback-navdp-url "http://127.0.0.1:${FALLBACK_PORT}" \
      --camera-height-m 0.5 "${hub_reject_policy_args[@]}" \
      "${hub_extra[@]}"
) >"${RUN_ROOT}/logs/server_hub.log" 2>&1 &
hub_pid=$!
wait_for_port hub "${hub_pid}" "${HUB_PORT}" \
  "${RUN_ROOT}/logs/server_hub.log"

if [[ "${PORTABILITY_AUTHORITY_PAIR}" == 1 \
   || "${PORTABILITY_DIRECTION_TRIPLE}" == 1 ]]; then
  (
    cd "${runtime_root}/forced_hub"
    exec env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${source_pythonpath}" \
      "${MEMNAV_PY}" -u \
        "${HUB_SCRIPT}" \
        --host 127.0.0.1 --port "${FORCED_HUB_PORT}" \
        --controller "${CONTROLLER}" \
        --memnav-url "http://127.0.0.1:${MEMNAV_PORT}" \
        --controller-url "${controller_url}" \
        --fallback-navdp-url "http://127.0.0.1:${FALLBACK_PORT}" \
        --camera-height-m 0.5 "${hub_reject_policy_args[@]}" \
        --force-reject-native \
        --emit-handoff-packets
  ) >"${RUN_ROOT}/logs/server_forced_hub.log" 2>&1 &
  forced_hub_pid=$!
  wait_for_port forced_hub "${forced_hub_pid}" "${FORCED_HUB_PORT}" \
    "${RUN_ROOT}/logs/server_forced_hub.log"
fi

leg1_mode=shared_trace
if [[ "${EVAL_KIND}" == lifelong_5leg ]]; then
  leg1_mode=policy
fi

common_eval=(
  --episode_root "${BENCHMARK_ROOT}" --episode_ids "${EPISODE}"
  --scene "${SCENE_FILE}" --scene_identity "${SCENE}"
  --host 127.0.0.1 --port "${HUB_PORT}"
  --server_backend cec_portability --navdp_depth_source monocular_sidecar
  --success_dist 1.0
  --max_steps "${MAX_STEPS}" --exec_horizon 8
  --trajectory_selector server --trajectory_selector_scope all
  --leg1_mode "${leg1_mode}"
  --leg1_goal_source own --seed "${EVAL_SEED}"
  --terminal_uturn off --terminal_visual_refine off
  --deterministic_plan_seeds --retrieval_override off
  --certified_cdec_rescue off --certified_stagnation_graph off
  --hybrid_route phase --revisit_controller navdp_mixed
  --revisit_adapter legacy_metric
)
for remap_name in SHARED_PATH_REMAP_1 SHARED_PATH_REMAP_2; do
  remap_value=${!remap_name:-}
  if [[ -n "${remap_value}" ]]; then
    common_eval+=(--shared_path_remap "${remap_value}")
  fi
done
if [[ "${EVAL_KIND}" == nnr_revisit ]]; then
  evaluator=${ROOT}/MemNavData/eval_shared_online_novel_revisit.py
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${TRACE_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
  )
elif [[ "${EVAL_KIND}" == lifelong_nnr ]]; then
  evaluator=${ROOT}/MemNavData/eval_shared_online_lifelong_nnr.py
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${TRACE_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
    --lifelong_history_scope "${LIFELONG_HISTORY_SCOPE:-all_prior}"
  )
elif [[ "${EVAL_KIND}" == lifelong_shared_c_collect ]]; then
  evaluator=${ROOT}/MemNavData/collect_lifelong_shared_c.py
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${TRACE_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
    --lifelong_history_scope all_prior
  )
elif [[ "${EVAL_KIND}" == lifelong_shared_c_b2 ]]; then
  evaluator=${ROOT}/MemNavData/eval_lifelong_shared_c_b2.py
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${TRACE_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
    --lifelong_history_scope "${LIFELONG_HISTORY_SCOPE:-all_prior}"
    --lifelong_shared_c_trace_root "${LIFELONG_SHARED_C_ROOT}"
  )
elif [[ "${EVAL_KIND}" == hm3d_lifelong ]]; then
  evaluator=${ROOT}/MemNavData/eval_hm3d_fullmono_lifelong.py
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${BENCHMARK_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
  )
elif [[ "${EVAL_KIND}" == hm3d_shared_c_collect ]]; then
  evaluator=${ROOT}/MemNavData/collect_hm3d_lifelong_shared_c.py
  if [[ -n "${CODE_OVERLAY_ROOT}" ]]; then
    evaluator=${CODE_OVERLAY_ROOT}/MemNavData/collect_hm3d_lifelong_shared_c.py
  fi
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${BENCHMARK_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
    --lifelong_history_scope all_prior
  )
elif [[ "${EVAL_KIND}" == hm3d_shared_c_b2 ]]; then
  evaluator=${ROOT}/MemNavData/eval_hm3d_lifelong_shared_c_b2.py
  if [[ -n "${CODE_OVERLAY_ROOT}" ]]; then
    evaluator=${CODE_OVERLAY_ROOT}/MemNavData/eval_hm3d_lifelong_shared_c_b2.py
  fi
  eval_extra=(
    --navdp_goal_switch_reset before_c
    --shared_leg1_trace_root "${BENCHMARK_ROOT}"
    --double_revisit_c_history initial_leg_only
    --shared_online_nnr_arm cec_portability
    --lifelong_history_scope "${LIFELONG_HISTORY_SCOPE:-all_prior}"
    --lifelong_shared_c_trace_root "${LIFELONG_SHARED_C_ROOT}"
  )
elif [[ "${EVAL_KIND}" == role_pair_mixed ]]; then
  evaluator=${ROOT}/MemNavData/eval_shared_online_role_pairs.py
  eval_extra=(
    --role_pair_scope "${ROLE_PAIR_SCOPE}"
    --role_pair_query_role "${ROLE_PAIR_QUERY_ROLE}"
  )
  if [[ -n "${ROLE_PAIR_QUERY_MANIFEST}" ]]; then
    eval_extra+=(--role_pair_query_manifest "${ROLE_PAIR_QUERY_MANIFEST}")
  fi
else
  evaluator=${ROOT}/MemNavData/eval_lifelong_5leg_habitat.py
  eval_extra=(
    --navdp_goal_switch_reset carry
    --lifelong_sequence natural_abcbc
    --lifelong_history_scope "${LIFELONG_HISTORY_SCOPE:-all_prior}"
  )
fi
if [[ "${EVAL_KIND}" == nnr_revisit \
   || "${EVAL_KIND}" == lifelong_nnr \
   || "${EVAL_KIND}" == lifelong_shared_c_collect \
   || "${EVAL_KIND}" == lifelong_shared_c_b2 ]]; then
  receipt_inputs=(
    "${BENCHMARK_ROOT}/manifest.json"
    "${BENCHMARK_ROOT}/${EPISODE}/benchmark.json"
    "${TRACE_ROOT}/${EPISODE}_leg1_trace.json"
    "${TRACE_ROOT}/${EPISODE}_legB_trace.json"
  )
  if [[ "${EVAL_KIND}" == lifelong_shared_c_b2 ]]; then
    receipt_inputs+=(
      "${LIFELONG_SHARED_C_ROOT}/population.json"
      "${LIFELONG_SHARED_C_ROOT}/population.json.sha256"
    )
  fi
elif [[ "${EVAL_KIND}" == hm3d_lifelong \
     || "${EVAL_KIND}" == hm3d_shared_c_collect \
     || "${EVAL_KIND}" == hm3d_shared_c_b2 ]]; then
  receipt_inputs=(
    "${BENCHMARK_ROOT}/${EPISODE}/benchmark.json"
    "${BENCHMARK_ROOT}/${EPISODE}/${EPISODE}_legB_trace.json"
    "${BENCHMARK_ROOT}/${EPISODE}/factual_B_completion.json"
  )
  if [[ "${EVAL_KIND}" == hm3d_shared_c_b2 ]]; then
    receipt_inputs+=(
      "${LIFELONG_SHARED_C_ROOT}/population.json"
      "${LIFELONG_SHARED_C_ROOT}/population.json.sha256"
    )
  fi
elif [[ "${EVAL_KIND}" == role_pair_mixed ]]; then
  receipt_inputs=(
    "${BENCHMARK_ROOT}/../manifest.json"
    "${BENCHMARK_ROOT}/${EPISODE}/role_pairs.json"
  )
  if [[ -n "${ROLE_PAIR_QUERY_MANIFEST}" ]]; then
    receipt_inputs+=("${ROLE_PAIR_QUERY_MANIFEST}")
  fi
else
  receipt_inputs=(
    "${BENCHMARK_ROOT}/${EPISODE}/meta/gen_meta.json"
    "${BENCHMARK_ROOT}/${EPISODE}/goal_1.jpg"
    "${BENCHMARK_ROOT}/${EPISODE}/goal_2.jpg"
  )
fi

run_evaluator() {
  local arm_root=$1
  local runtime_scope=$2
  local runtime_hub_port=$3
  local runtime_hub_pid=$4
  shift 4
  if [[ "${arm_root}" == "${RUN_ROOT}" ]]; then
    [[ ! -e "${arm_root}/result" \
       && ! -e "${arm_root}/logs/evaluator.log" ]] || \
      fail "single-arm output already exists: ${arm_root}"
  else
    [[ ! -e "${arm_root}" ]] || fail "arm output already exists: ${arm_root}"
    mkdir -p "${arm_root}/logs"
  fi
  env PYTHONPATH="${hab_pythonpath}" PYTHONDONTWRITEBYTECODE=1 \
    "${HAB_PY}" -u "${evaluator}" "${common_eval[@]}" \
      --port "${runtime_hub_port}" \
      --out "${arm_root}/result" "${eval_extra[@]}" "$@" \
      >"${arm_root}/logs/evaluator.log" 2>&1
  curl --fail --silent "http://127.0.0.1:${runtime_hub_port}/healthz" \
    >"${arm_root}/hub_health.json"
  local gpu_uuid memnav_start fallback_start hub_start upstream_start proxy_start
  gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '1p')
  memnav_start=$(awk '{print $22}' "/proc/${memnav_pid}/stat")
  fallback_start=$(awk '{print $22}' "/proc/${fallback_pid}/stat")
  hub_start=$(awk '{print $22}' "/proc/${runtime_hub_pid}/stat")
  upstream_start=
  proxy_start=
  if [[ -n "${upstream_pid}" ]]; then
    upstream_start=$(awk '{print $22}' "/proc/${upstream_pid}/stat")
  fi
  if [[ -n "${proxy_pid}" ]]; then
    proxy_start=$(awk '{print $22}' "/proc/${proxy_pid}/stat")
  fi
  "${MEMNAV_PY}" - "${arm_root}/compute_identity.json" \
    "$(hostname)" "${gpu_uuid}" "${runtime_scope}" \
    "${memnav_pid}" "${memnav_start}" \
    "${fallback_pid}" "${fallback_start}" \
    "${runtime_hub_pid}" "${hub_start}" \
    "${upstream_pid}" "${upstream_start}" \
    "${proxy_pid}" "${proxy_start}" \
    "${HM3D_LIFELONG_PAIRED_SCOPES}" \
    "${CUDA_VISIBLE_DEVICES:-}" "${runtime_hub_port}" \
    "${hub_cli_mode}" "${CEC_REJECT_POLICY}" <<'PY'
import json,sys
(path,host,gpu,scope,memnav_pid,memnav_start,navdp_pid,navdp_start,
 hub_pid,hub_start,controller_pid,controller_start,proxy_pid,proxy_start,
 pair_order,cuda_visible,hub_port,hub_cli_mode,reject_policy)=sys.argv[1:]
def process(pid,start):
 return None if not pid else {"pid":int(pid),"process_start_ticks":int(start)}
payload={
 "schema_version":"cec_compute_identity_v1_20260824",
 "host":host,"gpu_uuid":gpu,"cuda_visible_devices":cuda_visible,
 "runtime_scope":scope,
 "memnav":{"pid":int(memnav_pid),"process_start_ticks":int(memnav_start)},
 "navdp":{"pid":int(navdp_pid),"process_start_ticks":int(navdp_start)},
 "cec_hub":{"pid":int(hub_pid),"process_start_ticks":int(hub_start),
            "port":int(hub_port),"cli_contract":hub_cli_mode,
            "reject_policy":reject_policy},
 "accepted_controller":process(controller_pid,controller_start),
 "controller_proxy":process(proxy_pid,proxy_start),
 "paired_scope_order":pair_order.split(",") if pair_order else [],
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
  mapfile -t result_files < <(find "${arm_root}/result" -maxdepth 1 \
    -type f -name '*.json' -print | sort)
  sha256sum "${receipt_inputs[@]}" "${result_files[@]}" \
    "${arm_root}/compute_identity.json" \
    >"${arm_root}/result_inputs.sha256"
  echo "DONE controller=${CONTROLLER} result=${arm_root}/result/summary.json"
}

if [[ "${PORTABILITY_DIRECTION_TRIPLE}" == 1 ]]; then
  IFS=',' read -r -a direction_order <<<"${PORTABILITY_DIRECTION_ORDER}"
  for treatment in "${direction_order[@]}"; do
    case "${treatment}" in
      anchor_unaligned)
        run_evaluator "${RUN_ROOT}/${treatment}" "${treatment}" \
          "${HUB_PORT}" "${hub_pid}" \
          --cec_initial_bearing_alignment off
        ;;
      native_bearing_aligned)
        run_evaluator "${RUN_ROOT}/${treatment}" "${treatment}" \
          "${FORCED_HUB_PORT}" "${forced_hub_pid}" \
          --cec_initial_bearing_alignment first_certified
        ;;
      anchor_bearing_aligned)
        run_evaluator "${RUN_ROOT}/${treatment}" "${treatment}" \
          "${HUB_PORT}" "${hub_pid}" \
          --cec_initial_bearing_alignment first_certified
        ;;
      *) fail "unexpected direction treatment ${treatment}" ;;
    esac
  done
elif [[ "${PORTABILITY_AUTHORITY_PAIR}" == 1 ]]; then
  IFS=',' read -r -a authority_order <<<"${PORTABILITY_AUTHORITY_ORDER}"
  for authority in "${authority_order[@]}"; do
    if [[ "${authority}" == grant ]]; then
      run_evaluator "${RUN_ROOT}/grant" grant "${HUB_PORT}" "${hub_pid}" \
        --cec_initial_bearing_alignment \
        "${PORTABILITY_CEC_ACCEPT_ALIGNMENT}"
    else
      run_evaluator "${RUN_ROOT}/forced_reject_native" \
        forced_reject_native "${FORCED_HUB_PORT}" "${forced_hub_pid}" \
        --cec_initial_bearing_alignment off
    fi
  done
elif [[ -n "${HM3D_LIFELONG_PAIRED_SCOPES}" ]]; then
  IFS=',' read -r -a paired_scopes <<<"${HM3D_LIFELONG_PAIRED_SCOPES}"
  [[ "${#paired_scopes[@]}" -eq 2 ]] || \
    fail "paired run requires exactly two scopes"
  [[ " ${paired_scopes[*]} " == *" all_prior "* \
     && " ${paired_scopes[*]} " == *" initial_leg_only "* ]] || \
    fail "paired scopes must be all_prior and initial_leg_only"
  for scope in "${paired_scopes[@]}"; do
    run_evaluator "${RUN_ROOT}/${scope}" "${scope}" \
      "${HUB_PORT}" "${hub_pid}" \
      --lifelong_history_scope "${scope}"
  done
else
  scope_args=()
  if [[ "${EVAL_KIND}" == hm3d_lifelong \
     || "${EVAL_KIND}" == lifelong_shared_c_b2 \
     || "${EVAL_KIND}" == hm3d_shared_c_b2 ]]; then
    scope_args=(--lifelong_history_scope \
      "${LIFELONG_HISTORY_SCOPE:-all_prior}")
  fi
  run_evaluator "${RUN_ROOT}" "${LIFELONG_HISTORY_SCOPE:-single}" \
    "${HUB_PORT}" "${hub_pid}" \
    "${scope_args[@]}"
fi
