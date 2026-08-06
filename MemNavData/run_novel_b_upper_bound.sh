#!/usr/bin/env bash
# Strict one-server, three-arm Novel-B upper-bound evaluation.

set -euo pipefail
umask 0022
export GIT_OPTIONAL_LOCKS=0

MODE=${1:-smoke}
case "${MODE}" in
  smoke|full) ;;
  *) echo "usage: $0 [smoke|full]" >&2; exit 2 ;;
esac

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
: "${RUN_ROOT:?set RUN_ROOT to a new output directory outside ROOT}"
: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the full benchmark commit SHA}"
: "${HAB_PY:?set HAB_PY to the Habitat Python executable}"
: "${NAVDP_PY:?set NAVDP_PY to the NavDP policy Python executable}"
: "${CONTAINER_IMAGE_PATH:?set the pinned container image path}"
: "${CONTAINER_IMAGE_BYTES:?set the pinned container image byte count}"
: "${CONTAINER_IMAGE_HEAD_SHA256:?set the pinned container image header SHA}"

MANIFEST=${MANIFEST:-${ROOT}/MemNavData/expanded_3leg_router_eval_20260805.json}
EXPECTED_MANIFEST_SHA=55096038d0493723d2280fe9abbcb2ea9ea7f2059c50b6eb08d02f560cdb281b
EVALUATOR=${ROOT}/MemNavData/eval_novel_b_habitat.py
SUMMARIZER=${ROOT}/MemNavData/summarize_novel_b_upper_bound.py
UNIT_TEST=${ROOT}/MemNavData/test_summarize_novel_b_upper_bound.py
VALIDATOR=${ROOT}/MemNavData/validate_expanded_3leg_router_eval.py
NAVDP_SERVER=${ROOT}/NavDP/baselines/navdp/navdp_server.py
NAVDP_CKPT=${NAVDP_CKPT:-/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
ASSET_ROOT_OVERRIDE=${ASSET_ROOT_OVERRIDE:-}
EPISODE_ROOT_OVERRIDE=${EPISODE_ROOT_OVERRIDE:-}
port_identity=${SLURM_JOB_ID:-$$}
[[ "${port_identity}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: SLURM_JOB_ID must be numeric" >&2
  exit 1
}
NAVDP_PORT=${NAVDP_PORT:-$((20000 + port_identity % 15000))}
SERVER_STARTUP_POLLS=${SERVER_STARTUP_POLLS:-240}

EXPECTED_HAB_REQUESTS_VERSION=${EXPECTED_HAB_REQUESTS_VERSION:-2.32.4}
EXPECTED_HAB_REQUESTS_INIT_BYTES=5057
EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f
EXPECTED_HAB_REQUESTS_VERSION_BYTES=435
EXPECTED_HAB_REQUESTS_VERSION_SHA=143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577

[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "ABORT: EXPECTED_COMMIT must be a full lowercase SHA" >&2
  exit 1
}
[[ "${NAVDP_PORT}" =~ ^[1-9][0-9]*$ ]] && (( NAVDP_PORT <= 65535 )) || {
  echo "ABORT: NAVDP_PORT must be in [1, 65535]" >&2
  exit 1
}
[[ "${SERVER_STARTUP_POLLS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: SERVER_STARTUP_POLLS must be positive" >&2
  exit 1
}

for command_name in git realpath sha256sum stat awk ss mktemp seq wc; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "ABORT: missing command dependency ${command_name}" >&2
    exit 1
  }
done

for executable in "${HAB_PY}" "${NAVDP_PY}"; do
  test -x "${executable}" || {
    echo "ABORT: missing executable ${executable}" >&2
    exit 1
  }
done
for required in "${MANIFEST}" "${EVALUATOR}" "${SUMMARIZER}" \
                "${UNIT_TEST}" "${VALIDATOR}" "${NAVDP_SERVER}" \
                "${NAVDP_CKPT}"; do
  test -r "${required}" || {
    echo "ABORT: missing dependency ${required}" >&2
    exit 1
  }
done

ROOT=$(realpath "${ROOT}")
RUN_ROOT=$(realpath -m "${RUN_ROOT}")
MANIFEST=$(realpath "${MANIFEST}")
case "${RUN_ROOT}/" in
  "${ROOT}/"*)
    echo "ABORT: RUN_ROOT must be outside the benchmark git checkout" >&2
    exit 1
    ;;
esac
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "ABORT: output already exists: ${RUN_ROOT}" >&2
  exit 1
}

TASK_FILES=(
  MemNavData/eval_novel_b_habitat.py
  MemNavData/summarize_novel_b_upper_bound.py
  MemNavData/test_summarize_novel_b_upper_bound.py
  MemNavData/run_novel_b_upper_bound.sh
  MemNavData/slurm_novel_b_upper_bound.sbatch
  MemNavData/eval_2leg_habitat.py
  MemNavData/generate_twoleg.py
  MemNavData/arrival_shadow.py
  MemNavData/conditional_c_protocol.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/global_subgoal_protocol.py
  MemNavData/navdp_goal_switch.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/validate_expanded_3leg_router_eval.py
  MemNavData/expanded_3leg_router_eval_20260805.json
  MemNavData/expanded_navdp_router_eval_20260805.json
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/deterministic_seed.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/navdp/policy_network.py
)
for dynamic_path in "${MANIFEST}" "${EVALUATOR}" "${SUMMARIZER}" \
                    "${UNIT_TEST}" "${VALIDATOR}" "${NAVDP_SERVER}"; do
  dynamic_relative=$(realpath --relative-to="${ROOT}" "${dynamic_path}")
  [[ "${dynamic_relative}" != ../* && "${dynamic_relative}" != ".." ]] || {
    echo "ABORT: benchmark source is outside ROOT: ${dynamic_path}" >&2
    exit 1
  }
  TASK_FILES+=("${dynamic_relative}")
done

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
git -C "${ROOT}" ls-files --error-unmatch -- "${TASK_FILES[@]}" \
  >/dev/null || {
  echo "ABORT: a benchmark source is not tracked by EXPECTED_COMMIT" >&2
  exit 1
}
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: a benchmark source differs from EXPECTED_COMMIT" >&2
  exit 1
}
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: a benchmark source has staged changes" >&2
  exit 1
}
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "ABORT: benchmark worktree is not completely clean" >&2
  git -C "${ROOT}" status --short >&2
  exit 1
}

actual_manifest_sha=$(sha256sum "${MANIFEST}" | awk '{print $1}')
[[ "${actual_manifest_sha}" == "${EXPECTED_MANIFEST_SHA}" ]] || {
  echo "ABORT: manifest SHA256 ${actual_manifest_sha} != ${EXPECTED_MANIFEST_SHA}" >&2
  exit 1
}

# The frozen Habitat environment obtains requests from pip's vendor directory.
# Keep that path scoped to Habitat subprocesses and verify the exact package.
HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_REQUESTS_VENDOR=${HAB_REQUESTS_VENDOR:-${HAB_SITE_PACKAGES}/pip/_vendor}
HAB_PYTHONPATH=${HAB_REQUESTS_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}
hab_python() {
  env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"
}
REQUESTS_INIT=${HAB_REQUESTS_VENDOR}/requests/__init__.py
REQUESTS_VERSION=${HAB_REQUESTS_VENDOR}/requests/__version__.py
for required in "${REQUESTS_INIT}" "${REQUESTS_VERSION}"; do
  test -r "${required}" || {
    echo "ABORT: missing frozen requests dependency ${required}" >&2
    exit 1
  }
done
[[ "$(stat -c '%s' "${REQUESTS_INIT}")" == \
    "${EXPECTED_HAB_REQUESTS_INIT_BYTES}" ]] || {
  echo "ABORT: vendored requests __init__ size mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${REQUESTS_INIT}" | awk '{print $1}')" == \
    "${EXPECTED_HAB_REQUESTS_INIT_SHA}" ]] || {
  echo "ABORT: vendored requests __init__ SHA256 mismatch" >&2
  exit 1
}
[[ "$(stat -c '%s' "${REQUESTS_VERSION}")" == \
    "${EXPECTED_HAB_REQUESTS_VERSION_BYTES}" ]] || {
  echo "ABORT: vendored requests version size mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${REQUESTS_VERSION}" | awk '{print $1}')" == \
    "${EXPECTED_HAB_REQUESTS_VERSION_SHA}" ]] || {
  echo "ABORT: vendored requests version SHA256 mismatch" >&2
  exit 1
}

hab_python -m py_compile "${EVALUATOR}" "${SUMMARIZER}" "${VALIDATOR}"
(
  cd "${ROOT}"
  hab_python -m unittest MemNavData.test_summarize_novel_b_upper_bound -v
)
hab_python -c \
  'import habitat_sim,numpy,pandas,pyarrow,PIL,requests,scipy,quaternion,sys; assert requests.__version__ == sys.argv[1]; print("Habitat dependencies OK", habitat_sim.__version__, "requests", requests.__version__)' \
  "${EXPECTED_HAB_REQUESTS_VERSION}"
NAVDP_ENV_JSON=$(
  cd "${ROOT}/NavDP/baselines/navdp"
  "${NAVDP_PY}" - "${NAVDP_CKPT}" <<'PY'
import contextlib
import importlib
import importlib.metadata
import io
import json
from pathlib import Path
from collections.abc import Mapping
import sys

import torch

with contextlib.redirect_stdout(sys.stderr):
    for module in (
        "torchvision", "transformers", "diffusers", "cv2", "flask",
        "imageio", "matplotlib", "policy_backbone", "policy_network",
        "policy_agent",
    ):
        importlib.import_module(module)
if not torch.cuda.is_available():
    raise SystemExit("NavDP preflight cannot see CUDA")
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
if not isinstance(checkpoint, Mapping) or not checkpoint:
    raise SystemExit("NavDP checkpoint is not a non-empty state mapping")
packages = {}
for distribution in (
    "torch", "torchvision", "transformers", "diffusers",
    "opencv-python", "Flask", "imageio", "matplotlib",
):
    packages[distribution] = importlib.metadata.version(distribution)
print(json.dumps({
    "python": sys.version.split()[0],
    "python_executable": str(Path(sys.executable).resolve()),
    "cuda_available": True,
    "checkpoint_top_level_keys": len(checkpoint),
    "packages": packages,
}, sort_keys=True, separators=(",", ":")))
PY
)
HAB_ENV_JSON=$(hab_python - <<'PY'
import contextlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

with contextlib.redirect_stdout(sys.stderr):
    for module in (
        "habitat_sim", "numpy", "pandas", "pyarrow", "PIL", "requests",
        "scipy", "quaternion",
    ):
        importlib.import_module(module)
packages = {}
for distribution in (
    "habitat-sim", "numpy", "pandas", "pyarrow", "Pillow", "requests",
    "scipy", "numpy-quaternion",
):
    try:
        packages[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        packages[distribution] = "not-recorded"
print(json.dumps({
    "python": sys.version.split()[0],
    "python_executable": str(Path(sys.executable).resolve()),
    "packages": packages,
}, sort_keys=True, separators=(",", ":")))
PY
)
"${NAVDP_PY}" -m py_compile \
  "${NAVDP_SERVER}" \
  "${ROOT}/NavDP/baselines/navdp/deterministic_seed.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_agent.py" \
  "${ROOT}/NavDP/baselines/navdp/policy_network.py"

# Validate all selected assets and episode inputs once.  Only NavDP's frozen
# checkpoint is required; unrelated MemNav/LingBot weights are deliberately
# excluded from this NavDP-only dependency boundary.
preflight_tmp=$(mktemp)
cleanup_preflight() {
  [[ -z "${preflight_tmp}" ]] || rm -f "${preflight_tmp}"
}
trap cleanup_preflight EXIT
hab_python - "${ROOT}" "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" \
  "${MODE}" "${ASSET_ROOT_OVERRIDE}" "${EPISODE_ROOT_OVERRIDE}" \
  "${NAVDP_CKPT}" "${NAVDP_ENV_JSON}" "${HAB_ENV_JSON}" \
  "${CONTAINER_IMAGE_PATH}" "${CONTAINER_IMAGE_BYTES}" \
  "${CONTAINER_IMAGE_HEAD_SHA256}" >"${preflight_tmp}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
expected_sha = sys.argv[3]
mode = sys.argv[4]
asset_override = sys.argv[5]
episode_override = sys.argv[6]
navdp_checkpoint = Path(sys.argv[7])
navdp_environment = json.loads(sys.argv[8])
habitat_environment = json.loads(sys.argv[9])
container_image = {
    "path": sys.argv[10],
    "bytes": int(sys.argv[11]),
    "head_sha256": sys.argv[12],
}
sys.path.insert(0, str(root / "MemNavData"))

from validate_expanded_3leg_router_eval import (  # noqa: E402
    require,
    sha256,
    validate_file,
    validate_image,
    validate_selection,
)

actual_sha = sha256(manifest_path)
require(actual_sha == expected_sha, "3-leg manifest SHA256 mismatch")
manifest = json.loads(manifest_path.read_text())
base_record = manifest["base_manifest"]
base_path = manifest_path.parent / base_record["file"]
require(sha256(base_path) == base_record["sha256"],
        "base manifest SHA256 mismatch")
base = json.loads(base_path.read_text())
scenes = validate_selection(manifest, base)
require(manifest["evaluation"] == {
    "episodes_per_scene": 1,
    "base_seed": 20260803,
    "success_distance_m": 1.0,
    "max_steps_per_leg": 1200,
    "execution_horizon": 8,
    "goal_roles": {"A": "novel", "B": "novel", "C": "revisit"},
}, "frozen evaluation protocol changed")
selected = scenes[:1] if mode == "smoke" else scenes
asset_root = Path(asset_override) if asset_override else Path(
    manifest["paths"]["asset_root"])
episode_root = Path(episode_override) if episode_override else Path(
    manifest["paths"]["episode_root"])

checked = []
for scene in selected:
    asset = asset_root / scene / f"{scene}.glb"
    asset_sha = validate_file(asset, base["assets"][scene], "scene asset")
    records = manifest["episodes"][scene]
    require(len(records) == 1, "each scene must have one episode")
    for record in records:
        episode = episode_root / scene / record["episode"]
        files = {
            "metadata": episode / "meta/gen_meta.json",
            "parquet": episode / "data/chunk-000/episode_000000.parquet",
            "goal_b": episode / "goal_1.jpg",
            "goal_c": episode / "goal_2.jpg",
        }
        file_shas = {
            label: validate_file(path, record["files"][label], label)
            for label, path in files.items()
        }
        metadata = json.loads(files["metadata"].read_text())
        require(metadata["scene"] == f"{scene}.glb", "episode scene mismatch")
        require(int(metadata["n_legs"]) == 3, "episode is not three-leg")
        require(int(metadata["n_frames"]) == record["n_frames"],
                "episode frame count changed")
        switches = [int(value) for value in metadata["switches"]]
        require(switches == record["switches"], "goal switches changed")
        require(0 < switches[0] < switches[1] < record["n_frames"],
                "goal switches are out of bounds")
        require(len(metadata["goals"]) == 2, "goal count changed")
        require(metadata["goals"][0]["kind"] == "novel",
                "Goal B is not Novel")
        require(metadata["goals"][1]["kind"] == "revisit",
                "Goal C is not Revisit")
        require(int(metadata["goals"][1]["recall_gap"])
                == record["c_recall_gap"], "Goal-C recall gap changed")
        rgb_root = episode / "videos/chunk-000/observation.images.rgb"
        expected_rgb = {
            f"{index}.jpg" for index in range(record["n_frames"])
        }
        actual_rgb = {path.name for path in rgb_root.glob("*.jpg")}
        require(actual_rgb == expected_rgb, "RGB frame set is incomplete")
        validate_image(rgb_root / f"{switches[0] - 1}.jpg")
        validate_image(files["goal_b"])
        validate_image(files["goal_c"])
        checked.append({
            "scene": scene,
            "episode": record["episode"],
            "asset_sha256": asset_sha,
            "files": file_shas,
        })

dependency = base["dependencies"]["navdp_checkpoint"]
checkpoint_sha = validate_file(
    navdp_checkpoint, dependency, "navdp_checkpoint")
print(json.dumps({
    "status": "ok",
    "mode": mode,
    "manifest_sha256": actual_sha,
    "base_manifest_sha256": base_record["sha256"],
    "asset_root": str(asset_root),
    "episode_root": str(episode_root),
    "navdp_checkpoint": str(navdp_checkpoint),
    "navdp_checkpoint_sha256": checkpoint_sha,
    "environment": {
        "container_image": container_image,
        "navdp": navdp_environment,
        "habitat": habitat_environment,
    },
    "episodes": checked,
    "policy_training_overlap": [],
}, indent=2, sort_keys=True))
PY

mkdir -p "$(dirname "${RUN_ROOT}")"
mkdir "${RUN_ROOT}"
mkdir "${RUN_ROOT}/preflight" "${RUN_ROOT}/logs" "${RUN_ROOT}/scenes"
mv "${preflight_tmp}" "${RUN_ROOT}/preflight/inputs.json"
preflight_tmp=

mapfile -t ALL_SCENES < <(hab_python - "${MANIFEST}" <<'PY'
import json, sys
print(*json.load(open(sys.argv[1]))["selection"]["selected_scenes"], sep="\n")
PY
)
if [[ "${MODE}" == smoke ]]; then
  SCENES=("${ALL_SCENES[0]}")
else
  SCENES=("${ALL_SCENES[@]}")
fi
(( ${#SCENES[@]} > 0 )) || {
  echo "ABORT: no scenes selected" >&2
  exit 1
}

MANIFEST_ASSET_ROOT=$(hab_python - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["paths"]["asset_root"])
PY
)
MANIFEST_EPISODE_ROOT=$(hab_python - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["paths"]["episode_root"])
PY
)
ASSET_ROOT=${ASSET_ROOT_OVERRIDE:-${MANIFEST_ASSET_ROOT}}
EPISODE_ROOT=${EPISODE_ROOT_OVERRIDE:-${MANIFEST_EPISODE_ROOT}}

if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$"; then
  echo "ABORT: port ${NAVDP_PORT} is already in use" >&2
  exit 1
fi

RUNTIME_ROOT=${SLURM_TMPDIR:-/tmp}/novel_b_upper_bound_${SLURM_JOB_ID:-local}_$$
mkdir -p "${RUNTIME_ROOT}/navdp"
SERVER_PID=
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  [[ -z "${preflight_tmp:-}" ]] || rm -f "${preflight_tmp}"
}
trap cleanup EXIT INT TERM

(
  cd "${RUNTIME_ROOT}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    "${NAVDP_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) >"${RUN_ROOT}/logs/server_navdp.log" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 "${SERVER_STARTUP_POLLS}"); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ABORT: NavDP server exited during startup" >&2
    tail -n 120 "${RUN_ROOT}/logs/server_navdp.log" >&2
    exit 1
  fi
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${NAVDP_PORT}$"; then
    ready=1
    break
  fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || {
  echo "ABORT: NavDP server did not bind port ${NAVDP_PORT}" >&2
  tail -n 120 "${RUN_ROOT}/logs/server_navdp.log" >&2
  exit 1
}

hab_python - "${RUN_ROOT}" "${MODE}" "${EXPECTED_MANIFEST_SHA}" \
  "${EXPECTED_COMMIT}" "${SERVER_PID}" "${NAVDP_PORT}" \
  "${SCENES[@]}" <<'PY'
import json
from pathlib import Path
import sys

run_root = Path(sys.argv[1])
mode, manifest_sha, commit = sys.argv[2:5]
server_pid, port = map(int, sys.argv[5:7])
scenes = sys.argv[7:]
payload = {
    "schema_version": 1,
    "protocol": "novel_b_upper_bound_v1",
    "mode": mode,
    "manifest_sha256": manifest_sha,
    "expected_commit": commit,
    "server_start_count": 1,
    "server_pid": server_pid,
    "server_port": port,
    "same_live_server_for_all_arms_and_scenes": True,
    "arms_order": [
        "native_imagegoal",
        "oracle_short_1p25m",
        "oracle_final_point",
    ],
    "scene_order": scenes,
}
(run_root / "protocol.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

ARMS=(native_imagegoal oracle_short_1p25m oracle_final_point)
for scene in "${SCENES[@]}"; do
  scene_index=-1
  for index in "${!ALL_SCENES[@]}"; do
    if [[ "${ALL_SCENES[index]}" == "${scene}" ]]; then
      scene_index=${index}
      break
    fi
  done
  (( scene_index >= 0 )) || {
    echo "ABORT: selected scene is absent from manifest order: ${scene}" >&2
    exit 1
  }
  episode=$(hab_python - "${MANIFEST}" "${scene}" <<'PY'
import json, sys
records = json.load(open(sys.argv[1]))["episodes"][sys.argv[2]]
if len(records) != 1:
    raise SystemExit("expected exactly one episode")
print(records[0]["episode"])
PY
  )
  scene_root=${RUN_ROOT}/scenes/$(printf '%02d' "${scene_index}")_${scene}
  mkdir "${scene_root}"
  scene_file=${ASSET_ROOT}/${scene}/${scene}.glb
  episode_scene_root=${EPISODE_ROOT}/${scene}

  COMMON_ARGS=(
    --episode_root "${episode_scene_root}"
    --scene "${scene_file}"
    --host 127.0.0.1
    --port "${NAVDP_PORT}"
    --server_backend navdp
    --navdp_stop_threshold -0.5
    --leg1_mode policy
    --leg1_goal_source own
    --success_dist 1.0
    --max_steps 1200
    --exec_horizon 8
    --trajectory_selector server
    --trajectory_selector_scope all
    --oracle_candidate_seed_count 1
    --navdp_goal_switch_reset carry
    --retrieval_override off
    --terminal_uturn off
    --terminal_visual_refine off
    --arrival_shadow off
    --seed 20260803
    --deterministic_plan_seeds
    --episode_ids "${episode}"
  )

  for arm in "${ARMS[@]}"; do
    kill -0 "${SERVER_PID}" 2>/dev/null || {
      echo "ABORT: the one live NavDP server died before ${scene}/${arm}" >&2
      tail -n 120 "${RUN_ROOT}/logs/server_navdp.log" >&2
      exit 1
    }
    out=${scene_root}/${arm}
    mkdir "${out}"
    echo "[eval] mode=${MODE} scene=${scene} episode=${episode} arm=${arm} server_pid=${SERVER_PID}"
    (
      cd "${RUNTIME_ROOT}"
      hab_python -u "${EVALUATOR}" \
        --novel-b-arm "${arm}" \
        "${COMMON_ARGS[@]}" \
        --out "${out}"
    ) >"${RUN_ROOT}/logs/eval_$(printf '%02d' "${scene_index}")_${scene}_${arm}.log" 2>&1
    test -s "${out}/metric.csv" || {
      echo "ABORT: ${scene}/${arm} did not produce metric.csv" >&2
      exit 1
    }
    test -s "${out}/summary.json" || {
      echo "ABORT: ${scene}/${arm} did not produce summary.json" >&2
      exit 1
    }
    test -s "${out}/${episode}_audit.json" || {
      echo "ABORT: ${scene}/${arm} did not produce its audit artifact" >&2
      exit 1
    }
    row_count=$(($(wc -l <"${out}/metric.csv") - 1))
    [[ "${row_count}" -eq 1 ]] || {
      echo "ABORT: ${scene}/${arm} produced ${row_count} metric rows" >&2
      exit 1
    }
  done

  # Fail immediately on any Goal-A nondeterminism.  Waiting until all ten
  # scenes finish would waste the rest of a GPU allocation after the first
  # invalid paired scene.
  hab_python - "${scene_root}" "${scene}" "${episode}" 20260803 \
    "${ARMS[@]}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

scene_root = Path(sys.argv[1])
scene, episode = sys.argv[2:4]
expected_seed = int(sys.argv[4])
arms = sys.argv[5:]
records = []
for arm in arms:
    path = scene_root / arm / f"{episode}_audit.json"
    record = json.loads(path.read_text())
    if record.get("scene") != scene or record.get("episode") != episode:
        raise SystemExit(f"identity mismatch in {path}")
    if record.get("arm") != arm or record.get("seed") != expected_seed:
        raise SystemExit(f"arm/seed mismatch in {path}")
    goal_a = record.get("goal_a")
    encoded = json.dumps(
        goal_a, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if record.get("goal_a_sha256") != digest:
        raise SystemExit(f"Goal-A digest mismatch in {path}")
    records.append((arm, goal_a, digest))
baseline_arm, baseline, baseline_sha = records[0]
for arm, candidate, candidate_sha in records[1:]:
    if candidate_sha != baseline_sha or candidate != baseline:
        raise SystemExit(
            f"Goal-A mismatch after {scene}/{episode}: {baseline_arm} vs {arm}"
        )
(scene_root / "goal_a_pairing.json").write_text(json.dumps({
    "status": "ok",
    "scene": scene,
    "episode": episode,
    "seed": expected_seed,
    "arms": arms,
    "goal_a_sha256": baseline_sha,
}, indent=2, sort_keys=True) + "\n")
print(f"[paired Goal-A] {scene}/{episode} sha={baseline_sha[:12]}")
PY
done

kill -0 "${SERVER_PID}" 2>/dev/null || {
  echo "ABORT: the one live NavDP server died before paired postflight" >&2
  exit 1
}
summary_tmp=${RUN_ROOT}/summary.json.tmp
hab_python "${SUMMARIZER}" \
  --manifest "${MANIFEST}" \
  --run-root "${RUN_ROOT}" \
  --mode "${MODE}" >"${summary_tmp}"
hab_python - "${summary_tmp}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
if summary.get("audit", {}).get("status") != "ok":
    raise SystemExit("paired summary audit did not pass")
if not summary["audit"].get("goal_A_full_record_field_match"):
    raise SystemExit("Goal-A field equality was not established")
PY
mv "${summary_tmp}" "${RUN_ROOT}/summary.json"
echo "[complete] mode=${MODE} scenes=${#SCENES[@]} output=${RUN_ROOT} server_pid=${SERVER_PID}"
