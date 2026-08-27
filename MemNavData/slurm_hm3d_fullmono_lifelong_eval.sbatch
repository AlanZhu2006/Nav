#!/usr/bin/env bash
#SBATCH --job-name=h3lifeEval
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
# One allocation runs both primary CEC arms in one persistent model process,
# then the forced-native arm on the same GPU.  This is the paired-compute
# contract, not three independent array elements.
#SBATCH --time=04:00:00
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
: "${SLURM_ARRAY_TASK_ID:?submit one task per sealed history}"
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}"
: "${PROTOCOL:?}" "${DEPENDENCY_RECEIPT:?}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?}"

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_BASE_RECEIPT_SHA}" ]]
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c --quiet "${BASE_RECEIPT}")
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]]

POP_ROOT=${RUN_ROOT}/population
OUTPUT_ROOT=${OUTPUT_ROOT:-${RUN_ROOT}}
POPULATION=${POP_ROOT}/population.json
[[ -f "${POP_ROOT}/SEALED" ]]
(cd "${POP_ROOT}" && sha256sum -c --quiet population.json.sha256)
readarray -t identity < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${POPULATION}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); index=int(sys.argv[2]); rows=p["accepted"]
if index >= len(rows): raise SystemExit(0)
r=rows[index]
for value in (index,r["scene"],r["episode"],r["benchmark_sha256"]):
    print(value)
PY
)
if [[ "${#identity[@]}" -eq 0 ]]; then
  mkdir -p "${OUTPUT_ROOT}/evaluation/skipped"
  printf 'outside_sealed_population\n' > \
    "${OUTPUT_ROOT}/evaluation/skipped/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").txt"
  exit 0
fi
[[ "${#identity[@]}" -eq 4 ]] || { echo "bad population identity" >&2; exit 2; }
INDEX=${identity[0]}; SCENE=${identity[1]}; EPISODE=${identity[2]}
BENCHMARK_SHA=${identity[3]}
BENCHMARK_ROOT=${POP_ROOT}/benchmark/${SCENE}
BENCHMARK=${BENCHMARK_ROOT}/${EPISODE}/benchmark.json
[[ "$(sha256sum "${BENCHMARK}" | awk '{print $1}')" == "${BENCHMARK_SHA}" ]]
SCENE_FILE=$(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${BENCHMARK}" <<'PY'
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
 if os.stat(row["path"]).st_size != int(row["bytes"]):
  raise SystemExit(f"dependency size changed: {name}")
 print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]]
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}
label=$(printf '%03d_%s_%s' "${INDEX}" "${SCENE}" "${EPISODE}")
EVAL_ROOT=${OUTPUT_ROOT}/evaluation/${label}
[[ ! -e "${EVAL_ROOT}" ]] || { echo "history output exists" >&2; exit 2; }
mkdir -p "${EVAL_ROOT%/*}"

port_slot=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 59) % 7000 ))
base_port=$((20000 + port_slot * 5))
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
RUNNER=${TASK_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh
pycache=${SLURM_TMPDIR:-/tmp}/h3life_eval_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"
if (( INDEX % 2 == 0 )); then
  PAIRED_SCOPES=all_prior,initial_leg_only
else
  PAIRED_SCOPES=initial_leg_only,all_prior
fi
echo "host=$(hostname) task=${SLURM_ARRAY_TASK_ID} index=${INDEX} scene=${SCENE} episode=${EPISODE} paired_scopes=${PAIRED_SCOPES}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
# The primary all-prior/initial-only contrast shares the exact same loaded
# MemNav, NavDP, and CEC hub processes.  Each evaluator call starts with a
# full /reset, while the process/GPU identity remains fixed.
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env ROOT="${TASK_ROOT}" CONTROLLER=navdp \
  EVAL_KIND=hm3d_lifelong LIFELONG_HISTORY_SCOPE=all_prior \
  HM3D_LIFELONG_PAIRED_SCOPES="${PAIRED_SCOPES}" \
  SCENE="${SCENE}" EPISODE="${EPISODE}" \
  MAX_STEPS="${MAX_STEPS:-600}" \
  MEMNAV_PORT="${base_port}" FALLBACK_PORT="$((base_port + 1))" \
  UPSTREAM_PORT="$((base_port + 2))" PROXY_PORT="$((base_port + 3))" \
  HUB_PORT="$((base_port + 4))" MEMNAV_PY="${MEMNAV_PY}" \
  HAB_PY="${HAB_PY}" MEMNAV_CKPT="${MEMNAV_CKPT}" \
  NAVDP_CKPT="${NAVDP_CKPT}" LINGBOT_REPO="${LINGBOT_REPO}" \
  LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" LIGHTGLUE_REPO="${LIGHTGLUE_REPO}" \
  DEPENDENCY_ROOT="${DEPENDENCY_ROOT}" INTERNNAV_ROOT="${INTERNNAV_ROOT}" \
  SCENE_FILE="${SCENE_FILE}" BENCHMARK_ROOT="${BENCHMARK_ROOT}" \
  TRACE_ROOT="${BENCHMARK_ROOT}" RUN_ROOT="${EVAL_ROOT}" \
  PYTHONPYCACHEPREFIX="${pycache}/paired" bash "${RUNNER}"

# The exact-fallback system baseline uses a fresh hub process (authority is a
# startup-time fail-closed flag), but remains on this same allocated GPU/node.
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env ROOT="${TASK_ROOT}" CONTROLLER=navdp \
  EVAL_KIND=hm3d_lifelong LIFELONG_HISTORY_SCOPE=forced_reject_native \
  SCENE="${SCENE}" EPISODE="${EPISODE}" \
  MAX_STEPS="${MAX_STEPS:-600}" \
  MEMNAV_PORT="${base_port}" FALLBACK_PORT="$((base_port + 1))" \
  UPSTREAM_PORT="$((base_port + 2))" PROXY_PORT="$((base_port + 3))" \
  HUB_PORT="$((base_port + 4))" MEMNAV_PY="${MEMNAV_PY}" \
  HAB_PY="${HAB_PY}" MEMNAV_CKPT="${MEMNAV_CKPT}" \
  NAVDP_CKPT="${NAVDP_CKPT}" LINGBOT_REPO="${LINGBOT_REPO}" \
  LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" LIGHTGLUE_REPO="${LIGHTGLUE_REPO}" \
  DEPENDENCY_ROOT="${DEPENDENCY_ROOT}" INTERNNAV_ROOT="${INTERNNAV_ROOT}" \
  SCENE_FILE="${SCENE_FILE}" BENCHMARK_ROOT="${BENCHMARK_ROOT}" \
  TRACE_ROOT="${BENCHMARK_ROOT}" \
  RUN_ROOT="${EVAL_ROOT}/forced_reject_native" \
  PYTHONPYCACHEPREFIX="${pycache}/forced" bash "${RUNNER}"
chmod -R a-w "${EVAL_ROOT}"
echo "COMPLETE ${EVAL_ROOT}"
