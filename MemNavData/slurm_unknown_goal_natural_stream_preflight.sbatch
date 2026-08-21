#!/usr/bin/env bash
# One-scene, two-episode train-overlay compatibility preflight.  This job only
# records shadow evidence and builds offline labels; it does not enable the
# router, compare SR arms, train a model, or authorize the 40-scene collection.
#SBATCH --job-name=ugstream_pre
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=l40s_public,h100_tandon,h200_public,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable SOURCE_ROOT}"
: "${RUN_ROOT:?set unique RUN_ROOT}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA256}"
: "${EXPECTED_LAUNCHER_SHA:?set spooled launcher SHA256}"

SCENE=${SCENE:-17DRP5sb8fy}
EPISODE_IDS=${EPISODE_IDS:-episode_0000,episode_0001}
[[ "${SCENE}" =~ ^[A-Za-z0-9]+$ ]] || {
  echo "ABORT: invalid scene" >&2; exit 2; }
[[ "${EPISODE_IDS}" =~ ^episode_[0-9]{4}(,episode_[0-9]{4})*$ ]] || {
  echo "ABORT: invalid episode list" >&2; exit 2; }

SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
EPISODE_ROOT=/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg/${SCENE}
SCENE_FILE=/scratch/lg154/Research/datasets/mp3d/mp3d/${SCENE}/${SCENE}.glb
MEMNAV_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
NAVDP_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
INTERNNAV_ROOT=${SOURCE_ROOT}/InternNav

EXPECTED_OVERLAY_BYTES=128854888448
EXPECTED_OVERLAY_HEAD_SHA=43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b
EXPECTED_MEMNAV_CKPT_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_CKPT_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_WEIGHTS_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409

fail() { echo "ABORT: $*" >&2; exit 2; }
for command_name in singularity sha256sum nvidia-smi ss; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
for path in "$0" "${SOURCE_RECEIPT}" "${BASE_SIF}" "${OVERLAY}" \
    "${SCENE_FILE}" "${MEMNAV_CKPT}" \
    "${NAVDP_CKPT}" "${LINGBOT_WEIGHTS}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing/symlink input: ${path}"
done
for interpreter in "${HAB_PY}" "${MEMNAV_PY}"; do
  resolved_interpreter=$(readlink -f "${interpreter}") || \
    fail "cannot resolve interpreter: ${interpreter}"
  [[ -x "${interpreter}" && -f "${resolved_interpreter}" && \
      ! -L "${resolved_interpreter}" ]] || \
    fail "missing/broken interpreter: ${interpreter}"
done
[[ "$(sha256sum "$0" | awk '{print $1}')" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "spooled launcher SHA mismatch"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
  "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt SHA mismatch"
(cd / && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(stat -c '%s' "${OVERLAY}")" == "${EXPECTED_OVERLAY_BYTES}" ]] || \
  fail "overlay size changed"
[[ "$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | \
  sha256sum | awk '{print $1}')" == "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || \
  fail "overlay header changed"
[[ "$(sha256sum "${MEMNAV_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_MEMNAV_CKPT_SHA}" ]] || fail "MemNav checkpoint changed"
[[ "$(sha256sum "${NAVDP_CKPT}" | awk '{print $1}')" == \
  "${EXPECTED_NAVDP_CKPT_SHA}" ]] || fail "NavDP checkpoint changed"
[[ "$(sha256sum "${LINGBOT_WEIGHTS}" | awk '{print $1}')" == \
  "${EXPECTED_LINGBOT_WEIGHTS_SHA}" ]] || fail "LingBot weights changed"
[[ ! -e "${RUN_ROOT}" ]] || fail "output already exists: ${RUN_ROOT}"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer" \
  "${RUN_ROOT}/plans" "${RUN_ROOT}/teacher"
cp "${SOURCE_RECEIPT}" "${RUN_ROOT}/source_bundle.sha256"
chmod a-w "${RUN_ROOT}/source_bundle.sha256"
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/ugstream_pycache_${SLURM_JOB_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

port_key=$(( SLURM_JOB_ID % 15000 ))
MEMNAV_PORT=$((21000 + port_key * 2))
NAVDP_PORT=$((MEMNAV_PORT + 1))
for port in "${MEMNAV_PORT}" "${NAVDP_PORT}"; do
  ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$" && \
    fail "port ${port} is in use"
done

runtime_root=${SLURM_TMPDIR:-/tmp}/ugstream_${SLURM_JOB_ID}
mkdir -p "${runtime_root}/memnav" "${runtime_root}/navdp"
MEMNAV_PID=; NAVDP_PID=
cleanup() {
  for pid in "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

(
  cd "${runtime_root}/memnav"
  exec env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${SOURCE_ROOT}:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
      --port "${MEMNAV_PORT}" --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" --num_samples 16 \
      --exclude_recent 32 --retrieval raw \
      --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
      --graph_subgoal_spacing_m 0.0 --graph_subgoal_arrival_m 0.60 \
      --flow_gate auto --buffer_root "${RUN_ROOT}/buffer"
) > "${RUN_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!

(
  cd "${runtime_root}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${MEMNAV_PY}" -u "${SOURCE_ROOT}/NavDP/baselines/navdp/navdp_server.py" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${RUN_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

for spec in \
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${RUN_ROOT}/logs/server_memnav.log" \
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${RUN_ROOT}/logs/server_navdp.log"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    kill -0 "${pid}" 2>/dev/null || {
      tail -n 160 "${log}" >&2; fail "${label} server exited"; }
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1; break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || {
    tail -n 160 "${log}" >&2; fail "${label} server timeout"; }
done

HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_PYTHONPATH="${SOURCE_ROOT}:${HAB_SITE_PACKAGES}/pip/_vendor${PYTHONPATH:+:${PYTHONPATH}}"
CONTAINER=(singularity exec --nv --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}")

echo "host=$(hostname) job=${SLURM_JOB_ID} scene=${SCENE} episodes=${EPISODE_IDS}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${CONTAINER[@]}" env PYTHONPATH="${HAB_PYTHONPATH}" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -m unittest -v \
    MemNavData.test_unknown_goal_natural_stream_teacher \
    MemNavData.test_summarize_unknown_goal_natural_stream_shadow \
    MemNavData.test_covisibility_teacher \
  > "${RUN_ROOT}/logs/tests.log" 2>&1

"${CONTAINER[@]}" env PYTHONPATH="${HAB_PYTHONPATH}" \
  "${HAB_PY}" - "${RUN_ROOT}/input_receipt.json" "${EPISODE_ROOT}" \
  "${SCENE_FILE}" "${EPISODE_IDS}" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, episode_root, scene, episode_csv = sys.argv[1:]
def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
files = {"scene": Path(scene)}
for episode in episode_csv.split(","):
    root = Path(episode_root) / episode
    files[f"{episode}/metadata"] = root / "meta" / "gen_meta.json"
    files[f"{episode}/parquet"] = root / "data" / "chunk-000" / "episode_000000.parquet"
    files[f"{episode}/goal_1"] = root / "goal_1.jpg"
    files[f"{episode}/goal_2"] = root / "goal_2.jpg"
for label, path in files.items():
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"input absent/symlink: {label} {path}")
payload = {label: {"path": str(path), "bytes": path.stat().st_size,
                   "sha256": sha(path)} for label, path in files.items()}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

"${CONTAINER[@]}" env PYTHONPATH="${HAB_PYTHONPATH}" \
  PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  "${HAB_PY}" -u "${SOURCE_ROOT}/MemNavData/eval_3leg_habitat.py" \
    --episode_root "${EPISODE_ROOT}" --scene "${SCENE_FILE}" \
    --port "${MEMNAV_PORT}" --novel_port "${NAVDP_PORT}" \
    --out "${RUN_ROOT}/plans" --server_backend hybrid_pose \
    --hybrid_route memory_geometry --revisit_adapter legacy_metric \
    --leg1_mode policy --navdp_goal_switch_reset carry \
    --router_visual_floor -1.0 --router_min_matches 1000000000 \
    --router_min_inliers 1000000000 --router_min_inlier_ratio 1.0 \
    --router_confirm_plans 100000 --router_verify_top_k 8 \
    --success_dist 1.0 --max_steps 500 --exec_horizon 8 \
    --trajectory_selector server --trajectory_selector_scope all \
    --seed 20260803 --deterministic_plan_seeds \
    --episode_ids "${EPISODE_IDS}" \
  > "${RUN_ROOT}/logs/evaluator.log" 2>&1

"${CONTAINER[@]}" env PYTHONPATH="${HAB_PYTHONPATH}" \
  "${HAB_PY}" -u -m MemNavData.summarize_unknown_goal_natural_stream_shadow \
    --run-root "${RUN_ROOT}/plans" --buffer-root "${RUN_ROOT}/buffer" \
    --allow-censored-legs --out "${RUN_ROOT}/shadow_report.json" \
  > "${RUN_ROOT}/logs/shadow_summarizer.log" 2>&1

while IFS= read -r plan_trace; do
  episode=$(basename "${plan_trace}" _plans.json)
  buffer_episode=$("${CONTAINER[@]}" env PYTHONPATH="${HAB_PYTHONPATH}" \
    "${HAB_PY}" - "${plan_trace}" "${RUN_ROOT}/buffer" <<'PY'
import hashlib, json, sys
from pathlib import Path
plan, root = map(Path, sys.argv[1:])
payload = json.loads(plan.read_text())
probe = None
for leg in ("legA", "legB", "legC"):
    traces = payload["memory_traces"][leg]
    rolls = {row["step"]: row for row in payload["rollout_traces"][leg]}
    if traces:
        row = traces[0]
        probe = row["frame_idx"], rolls[row["step"]]["jpg_sha256"]
        break
if probe is None:
    raise SystemExit("episode contains no natural frames")
frame, expected = probe
matches = []
for directory in sorted(root.glob("ep_*")):
    path = directory / f"{frame}.jpg"
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        matches.append(directory)
if len(matches) != 1:
    raise SystemExit(f"expected one RGB buffer match, got {len(matches)}")
print(matches[0])
PY
  )
  "${CONTAINER[@]}" env PYTHONPATH="${HAB_PYTHONPATH}" \
    "${HAB_PY}" -u \
      "${SOURCE_ROOT}/MemNavData/build_unknown_goal_natural_stream_teacher.py" \
      --plan-trace "${plan_trace}" \
      --episode-root "${EPISODE_ROOT}/${episode}" \
      --scene "${SCENE_FILE}" --buffer-episode-root "${buffer_episode}" \
      --negative-threshold 0.2 --positive-threshold 0.5 \
      --allow-censored-legs --out "${RUN_ROOT}/teacher/${episode}.json" \
    > "${RUN_ROOT}/logs/teacher_${episode}.log" 2>&1
done < <(find "${RUN_ROOT}/plans" -maxdepth 1 -name 'episode_*_plans.json' \
  -type f | sort)

"${CONTAINER[@]}" env PYTHONPATH="${HAB_PYTHONPATH}" \
  "${HAB_PY}" - "${RUN_ROOT}" <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
teachers = [json.loads(path.read_text())
            for path in sorted((root / "teacher").glob("episode_*.json"))]
if not teachers:
    raise SystemExit("no teacher artifacts")
labels = {"positive": 0, "negative": 0, "ambiguous": 0}
candidate_labels = dict(labels)
plans = trials = frames = 0
by_episode = {}
for teacher in teachers:
    summary = teacher["summary"]
    plans += summary["plans"]
    trials += summary["candidate_trials"]
    frames += summary["natural_frames"]
    for key in labels:
        labels[key] += summary["topk_label_counts"][key]
        candidate_labels[key] += summary["candidate_label_counts"][key]
    episode = Path(teacher["inputs"]["episode_root"]).name
    by_episode[episode] = summary
with (root / "plans" / "metric.csv").open(newline="") as handle:
    metrics = list(csv.DictReader(handle))
report = {
    "schema_version": "unknown_goal_natural_stream_overlay_preflight_v1",
    "status": "complete",
    "scope": "one-scene data compatibility only; no model/SR claim",
    "episodes": len(teachers),
    "natural_frames": frames,
    "plans": plans,
    "candidate_trials": trials,
    "topk_label_counts": labels,
    "candidate_label_counts": candidate_labels,
    "label_non_degenerate": labels["positive"] > 0 and labels["negative"] > 0,
    "metrics": metrics,
    "by_episode": by_episode,
}
(root / "preflight_report.json").write_text(
    json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
for name in ("input_receipt.json", "shadow_report.json", "preflight_report.json"):
    path = root / name
    print(hashlib.sha256(path.read_bytes()).hexdigest(), path)
PY

sha256sum "${RUN_ROOT}/input_receipt.json" \
  "${RUN_ROOT}/shadow_report.json" "${RUN_ROOT}/preflight_report.json" \
  "${RUN_ROOT}"/teacher/episode_*.json > "${RUN_ROOT}/FINAL_ARTIFACTS.sha256"
echo "COMPLETE ${RUN_ROOT}"
