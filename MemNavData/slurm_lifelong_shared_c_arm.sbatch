#!/usr/bin/env bash
#SBATCH --job-name=shared-c-b2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --account=torch_pr_769_tandon_advanced
# Factual MP3D traces were generated on gh001.  Cross-node Habitat rendering
# has produced byte-level RGB drift, so exact replay must stay on that node.
#SBATCH --nodelist=gh001
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm-%x-%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm-%x-%A_%a.err

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${NNR_ROOT:?}" "${ORIGINAL_RUN_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${BASE_SOURCE_RECEIPT_SHA:?}" "${DEPENDENCY_RECEIPT:?}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?}" "${PORTABILITY_ENV_ROOT:?}"
: "${PORTABILITY_CHECKPOINT_ROOT:?}" "${RUN_ROOT:?}" "${CONTROLLER:?}"
: "${STAGE:?set STAGE=collect|evaluate}"

case "${STAGE}" in collect|evaluate) ;; *) echo "ABORT: bad STAGE" >&2; exit 2;; esac
case "${CONTROLLER}" in vint|gnm|nomad|iplanner|viplanner) ;;
  *) echo "ABORT: bad CONTROLLER" >&2; exit 2;; esac

MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
SUPPORT_ROOT=${ORIGINAL_RUN_ROOT}/factual_b_support
SOURCE_POPULATION=${SUPPORT_ROOT}/population.json
SHARED_C_ROOT=${RUN_ROOT}/shared_c_population
BENCHMARK_ALL=${NNR_ROOT}/prepared/benchmark
TRACE_ALL=${NNR_ROOT}/prepared/traces
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
  fail "source population changed"
[[ -f "${SUPPORT_ROOT}/SEALED" ]] || fail "source population is not sealed"
(cd "${BENCHMARK_ALL}" && sha256sum -c --quiet manifest.json.sha256) || \
  fail "NNR benchmark changed"
(cd "${PORTABILITY_CHECKPOINT_ROOT}" && sha256sum -c --quiet CHECKPOINTS.sha256) || \
  fail "checkpoint receipt failed"

population=${SOURCE_POPULATION}
if [[ "${STAGE}" == evaluate ]]; then
  population=${SHARED_C_ROOT}/population.json
  [[ -f "${SHARED_C_ROOT}/SEALED" ]] || fail "shared-C population is not sealed"
  (cd "${SHARED_C_ROOT}" && sha256sum -c --quiet population.json.sha256) || \
    fail "shared-C population changed"
fi
readarray -t identity < <("${MEMNAV_PY}" - "${population}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); i=int(sys.argv[2]); rows=p["accepted"]
if i >= len(rows): raise SystemExit(0)
r=rows[i]
print(r["scene"]); print(r["episode"]); print(r["benchmark_sha256"])
PY
)
if [[ "${#identity[@]}" -eq 0 ]]; then
  mkdir -p "${RUN_ROOT}/skipped/${STAGE}"
  printf 'index=%s\nreason=outside_population\n' "${SLURM_ARRAY_TASK_ID}" > \
    "${RUN_ROOT}/skipped/${STAGE}/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").txt"
  exit 0
fi
[[ "${#identity[@]}" -eq 3 ]] || fail "population identity is incomplete"
SCENE=${identity[0]}; EPISODE=${identity[1]}; BENCHMARK_SHA=${identity[2]}
BENCHMARK_ROOT=${BENCHMARK_ALL}/${SCENE}
TRACE_ROOT=${TRACE_ALL}/${SCENE}
SCENE_FILE=${ASSET_ROOT}/${SCENE}/${SCENE}.glb
[[ "$(sha256sum "${BENCHMARK_ROOT}/${EPISODE}/benchmark.json" | awk '{print $1}')" == \
   "${BENCHMARK_SHA}" ]] || fail "benchmark changed"

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
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}
label=$(printf '%03d_%s_%s' "${SLURM_ARRAY_TASK_ID}" "${SCENE}" "${EPISODE}")
port_slot=$(( $$ % 7000 )); base_port=$((20000 + port_slot * 5))
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/shared_c_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"

run_one() {
  local kind=$1 output=$2 scope=$3 paired=${4:-}
  [[ ! -e "${output}" ]] || fail "output exists: ${output}"
  local pair_env=()
  [[ -z "${paired}" ]] || pair_env+=(HM3D_LIFELONG_PAIRED_SCOPES="${paired}")
  singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
    -B /scratch/qw2440 "${BASE_SIF}" env \
    ROOT="${SOURCE_ROOT}" CONTROLLER="${CONTROLLER}" EVAL_KIND="${kind}" \
    LIFELONG_HISTORY_SCOPE="${scope}" LIFELONG_SHARED_C_ROOT="${SHARED_C_ROOT}" \
    SCENE="${SCENE}" EPISODE="${EPISODE}" MAX_STEPS=600 \
    MEMNAV_PORT="${base_port}" FALLBACK_PORT="$((base_port + 1))" \
    UPSTREAM_PORT="$((base_port + 2))" PROXY_PORT="$((base_port + 3))" \
    HUB_PORT="$((base_port + 4))" MEMNAV_PY="${MEMNAV_PY}" HAB_PY="${HAB_PY}" \
    MEMNAV_CKPT="${MEMNAV_CKPT}" NAVDP_CKPT="${NAVDP_CKPT}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    LIGHTGLUE_REPO="${LIGHTGLUE_REPO}" DEPENDENCY_ROOT="${DEPENDENCY_ROOT}" \
    INTERNNAV_ROOT="${INTERNNAV_ROOT}" ASSET_ROOT="${ASSET_ROOT}" \
    SCENE_FILE="${SCENE_FILE}" BENCHMARK_ROOT="${BENCHMARK_ROOT}" \
    TRACE_ROOT="${TRACE_ROOT}" \
    VINT_PY="${PORTABILITY_ENV_ROOT}/vint/bin/python" \
    GNM_PY="${PORTABILITY_ENV_ROOT}/vint/bin/python" \
    NOMAD_PY="${PORTABILITY_ENV_ROOT}/vint/bin/python" \
    VIPLANNER_PY="${PORTABILITY_ENV_ROOT}/viplanner/bin/python" \
    VINT_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/vint.pth" \
    GNM_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/gnm.pth" \
    NOMAD_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/nomad.pth" \
    IPLANNER_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/iplanner.pth" \
    VIPLANNER_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/viplanner.pt" \
    MASK2FORMER_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth" \
    MASK2FORMER_CONFIG="${PORTABILITY_ENV_ROOT}/viplanner/lib/python3.10/site-packages/mmdet/.mim/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py" \
    RUN_ROOT="${output}" PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    "${pair_env[@]}" bash "${RUNNER}"
}

echo "host=$(hostname) stage=${STAGE} controller=${CONTROLLER} scene=${SCENE} episode=${EPISODE}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
if [[ "${STAGE}" == collect ]]; then
  run_one lifelong_shared_c_collect "${RUN_ROOT}/collection/${label}" all_prior
else
  # Primary causal comparison shares the exact server processes and GPU.
  run_one lifelong_shared_c_b2 "${RUN_ROOT}/evaluation/${label}" all_prior \
    all_prior,initial_leg_only
  # Supporting native baseline uses the same allocation/GPU and exact C
  # replay, but requires a force-reject hub process by contract.
  run_one lifelong_shared_c_b2 \
    "${RUN_ROOT}/evaluation/${label}/forced_reject_native" \
    forced_reject_native
fi
echo "COMPLETE stage=${STAGE} controller=${CONTROLLER} label=${label}"
