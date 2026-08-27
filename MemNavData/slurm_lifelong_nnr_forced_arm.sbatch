#!/usr/bin/env bash
#SBATCH --job-name=lifeNNRforced
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

# Shared-native baseline addendum: run ONLY the forced_reject_native arm on
# the same frozen population as the completed two-arm expansion.  The
# completed arms are never rerun or written to; this job only checks that
# its pair directory exists (provenance pairing, no outcome reads).
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?submit as array 0-18}"
: "${SOURCE_ROOT:?set immutable task source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${BASE_SOURCE_ROOT:?set immutable dependency source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set base source receipt SHA}"
: "${DEPENDENCY_RECEIPT:?set model dependency receipt}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set dependency receipt SHA}"
: "${NNR_ROOT:?set sealed NNR run root}"
: "${ORIGINAL_RUN_ROOT:?set the completed two-arm run root (read-only)}"
: "${RUN_ROOT:?set unique forced-arm addendum run root}"

MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
SUPPORT_ROOT=${ORIGINAL_RUN_ROOT}/factual_b_support
POPULATION=${SUPPORT_ROOT}/population.json
BENCHMARK_ALL=${NNR_ROOT}/prepared/benchmark
TRACE_ALL=${NNR_ROOT}/prepared/traces
EVALUATION_ROOT=${RUN_ROOT}/evaluation
ASSET_ROOT=/scratch/lg154/Research/datasets/mp3d/mp3d
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
RUNNER=${SOURCE_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
(cd "${SUPPORT_ROOT}" && sha256sum -c --quiet population.json.sha256) || \
  fail "support population changed"
[[ -f "${SUPPORT_ROOT}/SEALED" ]] || fail "support population is not sealed"
[[ -f "${ORIGINAL_RUN_ROOT}/VERIFIED" ]] || \
  fail "two-arm run is not independently verified"
(cd "${BENCHMARK_ALL}" && sha256sum -c --quiet manifest.json.sha256) || \
  fail "NNR benchmark changed"

readarray -t identity < <("${MEMNAV_PY}" - "${POPULATION}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); i=int(sys.argv[2]); rows=p["accepted"]
if i >= len(rows): raise SystemExit(0)
r=rows[i]
print(r["scene"]); print(r["episode"]); print(r["benchmark_sha256"])
PY
)
if [[ "${#identity[@]}" -eq 0 ]]; then
  mkdir -p "${RUN_ROOT}/skipped"
  printf 'index=%s\nreason=outside_supported_population\n' \
    "${SLURM_ARRAY_TASK_ID}" > \
    "${RUN_ROOT}/skipped/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").txt"
  echo "SKIP index=${SLURM_ARRAY_TASK_ID}"
  exit 0
fi
[[ "${#identity[@]}" -eq 3 ]] || fail "population identity is incomplete"
SCENE=${identity[0]}; EPISODE=${identity[1]}; BENCHMARK_SHA=${identity[2]}
BENCHMARK_ROOT=${BENCHMARK_ALL}/${SCENE}
TRACE_ROOT=${TRACE_ALL}/${SCENE}
SCENE_FILE=${ASSET_ROOT}/${SCENE}/${SCENE}.glb
[[ "$(sha256sum "${BENCHMARK_ROOT}/${EPISODE}/benchmark.json" | awk '{print $1}')" == \
   "${BENCHMARK_SHA}" ]] || fail "selected benchmark changed"

label=$(printf '%03d_%s_%s' "${SLURM_ARRAY_TASK_ID}" "${SCENE}" "${EPISODE}")
ORIGINAL_PAIR=${ORIGINAL_RUN_ROOT}/evaluation/${label}
[[ -d "${ORIGINAL_PAIR}/all_prior" && -d "${ORIGINAL_PAIR}/initial_leg_only" ]] \
  || fail "verified pair directory missing for ${label}"

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
    row=r["dependencies"][name]
    if os.stat(row["path"]).st_size != int(row["bytes"]):
        raise SystemExit(f"dependency size changed: {name}")
    print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "dependency receipt invalid"
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}

ARM_ROOT=${EVALUATION_ROOT}/${label}/forced_reject_native
[[ ! -e "${ARM_ROOT}" ]] || fail "forced-arm output exists"
mkdir -p "${ARM_ROOT%/*}"

port_slot=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 59) % 7000 ))
base_port=$((20000 + port_slot * 5))
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/life_forced_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
echo "host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} scene=${SCENE} episode=${EPISODE} arm=forced_reject_native"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B /scratch/qw2440 "${BASE_SIF}" env \
  ROOT="${SOURCE_ROOT}" CONTROLLER=navdp EVAL_KIND=lifelong_nnr \
  LIFELONG_HISTORY_SCOPE=forced_reject_native SCENE="${SCENE}" \
  EPISODE="${EPISODE}" MAX_STEPS=600 \
  MEMNAV_PORT="${base_port}" FALLBACK_PORT="$((base_port + 1))" \
  UPSTREAM_PORT="$((base_port + 2))" PROXY_PORT="$((base_port + 3))" \
  HUB_PORT="$((base_port + 4))" MEMNAV_PY="${MEMNAV_PY}" \
  HAB_PY="${HAB_PY}" MEMNAV_CKPT="${MEMNAV_CKPT}" \
  NAVDP_CKPT="${NAVDP_CKPT}" LINGBOT_REPO="${LINGBOT_REPO}" \
  LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" LIGHTGLUE_REPO="${LIGHTGLUE_REPO}" \
  DEPENDENCY_ROOT="${DEPENDENCY_ROOT}" INTERNNAV_ROOT="${INTERNNAV_ROOT}" \
  ASSET_ROOT="${ASSET_ROOT}" SCENE_FILE="${SCENE_FILE}" \
  BENCHMARK_ROOT="${BENCHMARK_ROOT}" TRACE_ROOT="${TRACE_ROOT}" \
  RUN_ROOT="${ARM_ROOT}" PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  bash "${RUNNER}"
chmod -R a-w "${ARM_ROOT}"
echo "COMPLETE forced_arm=${ARM_ROOT}"
