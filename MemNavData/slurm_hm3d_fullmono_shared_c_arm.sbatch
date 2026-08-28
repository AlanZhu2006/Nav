#!/usr/bin/env bash
#SBATCH --job-name=h3sharedC
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1
: "${SLURM_ARRAY_TASK_ID:?}" "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${AMENDMENT_ROOT:?}" "${AMENDMENT_RECEIPT:?}"
: "${EXPECTED_AMENDMENT_RECEIPT_SHA:?}"
: "${RUN_ROOT:?}" "${DEPENDENCY_RECEIPT:?}" "${STAGE:?collect|evaluate}"
case "${STAGE}" in collect|evaluate) ;; *) echo "bad STAGE" >&2; exit 2;; esac
[[ "$(sha256sum "${AMENDMENT_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_AMENDMENT_RECEIPT_SHA}" ]]
(cd "${AMENDMENT_ROOT}" && sha256sum -c --quiet "${AMENDMENT_RECEIPT}")
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]]
if [[ -n "${EXPECTED_REPLAY_NODE:-}" ]]; then
  [[ "$(hostname -s)" == "${EXPECTED_REPLAY_NODE}" ]] || {
    echo "replay node mismatch: expected=${EXPECTED_REPLAY_NODE} actual=$(hostname -s)" >&2
    exit 2
  }
fi

BASE_POP_ROOT=${RUN_ROOT}/population
SHARED_C_ROOT=${RUN_ROOT}/shared_c_population
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}}
population=${BASE_POP_ROOT}/population.json
if [[ "${STAGE}" == evaluate ]]; then population=${SHARED_C_ROOT}/population.json; fi
[[ -f "${population%/*}/SEALED" ]]
(cd "${population%/*}" && sha256sum -c --quiet population.json.sha256)
readarray -t identity < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${population}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); i=int(sys.argv[2]); rows=p["accepted"]
if i >= len(rows): raise SystemExit(0)
r=rows[i]
for value in (i,r["scene"],r["episode"],r["benchmark_sha256"]): print(value)
PY
)
if [[ "${#identity[@]}" -eq 0 ]]; then
  mkdir -p "${OUTPUT_ROOT}/skipped/${STAGE}"
  printf 'outside_sealed_population\n' > \
    "${OUTPUT_ROOT}/skipped/${STAGE}/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").txt"
  exit 0
fi
INDEX=${identity[0]}; SCENE=${identity[1]}; EPISODE=${identity[2]}
BENCHMARK_SHA=${identity[3]}
BENCHMARK_ROOT=${BASE_POP_ROOT}/benchmark/${SCENE}
BENCHMARK=${BENCHMARK_ROOT}/${EPISODE}/benchmark.json
[[ "$(sha256sum "${BENCHMARK}" | awk '{print $1}')" == "${BENCHMARK_SHA}" ]]
SCENE_FILE=$(/scratch/lg154/conda-envs/memnav/bin/python - "${BENCHMARK}" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["source_scene_asset"])
PY
)
readarray -t dependencies < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
 row=r["dependencies"][name]
 if os.stat(row["path"]).st_size != int(row["bytes"]): raise SystemExit(name)
 print(row["path"])
PY
)
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}
label=$(printf '%03d_%s_%s' "${INDEX}" "${SCENE}" "${EPISODE}")
source "${AMENDMENT_ROOT}/MemNavData/slurm_port_pair.sh"
claim_slurm_tcp_port_block hm3d_lifelong 5 12000 2400
base_port=${CEC_PORT_BLOCK_BASE}
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
RUNNER=${AMENDMENT_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh
pycache=${SLURM_TMPDIR:-/tmp}/h3shared_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"

run_one() {
  local kind=$1 output=$2 scope=$3 paired=${4:-}
  [[ ! -e "${output}" ]] || { echo "output exists: ${output}" >&2; exit 2; }
  local pair_env=(); [[ -z "${paired}" ]] || \
    pair_env+=(HM3D_LIFELONG_PAIRED_SCOPES="${paired}")
  singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
    "${BASE_SIF}" env ROOT="${TASK_ROOT}" CONTROLLER=navdp \
    CODE_OVERLAY_ROOT="${AMENDMENT_ROOT}" \
    EVAL_KIND="${kind}" LIFELONG_HISTORY_SCOPE="${scope}" \
    LIFELONG_SHARED_C_ROOT="${SHARED_C_ROOT}" \
    SCENE="${SCENE}" EPISODE="${EPISODE}" MAX_STEPS="${MAX_STEPS:-600}" \
    MEMNAV_PORT="${base_port}" FALLBACK_PORT="$((base_port + 1))" \
    UPSTREAM_PORT="$((base_port + 2))" PROXY_PORT="$((base_port + 3))" \
    HUB_PORT="$((base_port + 4))" MEMNAV_PY="${MEMNAV_PY}" HAB_PY="${HAB_PY}" \
    MEMNAV_CKPT="${MEMNAV_CKPT}" NAVDP_CKPT="${NAVDP_CKPT}" \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    LIGHTGLUE_REPO="${LIGHTGLUE_REPO}" DEPENDENCY_ROOT="${DEPENDENCY_ROOT}" \
    INTERNNAV_ROOT="${INTERNNAV_ROOT}" SCENE_FILE="${SCENE_FILE}" \
    BENCHMARK_ROOT="${BENCHMARK_ROOT}" TRACE_ROOT="${BENCHMARK_ROOT}" \
    RUN_ROOT="${output}" PYTHONPYCACHEPREFIX="${pycache}" \
    "${pair_env[@]}" bash "${RUNNER}"
}

echo "host=$(hostname) expected_replay_node=${EXPECTED_REPLAY_NODE:-unbound} stage=${STAGE} index=${INDEX} scene=${SCENE} episode=${EPISODE}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
if [[ "${STAGE}" == collect ]]; then
  [[ "${MAX_STEPS:-600}" -eq 600 ]] || { echo "C collection requires 600 steps" >&2; exit 2; }
  run_one hm3d_shared_c_collect "${OUTPUT_ROOT}/shared_c_collection/${label}" all_prior
else
  # C is already byte/pose frozen; balance only the B2 treatment order.
  if (( INDEX % 2 == 0 )); then pair=all_prior,initial_leg_only;
  else pair=initial_leg_only,all_prior; fi
  run_one hm3d_shared_c_b2 "${OUTPUT_ROOT}/shared_c_evaluation/${label}" \
    all_prior "${pair}"
  run_one hm3d_shared_c_b2 \
    "${OUTPUT_ROOT}/shared_c_evaluation/${label}/forced_reject_native" \
    forced_reject_native
fi
echo "COMPLETE stage=${STAGE} ${label}"
