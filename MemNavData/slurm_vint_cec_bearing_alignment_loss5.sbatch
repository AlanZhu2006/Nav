#!/usr/bin/env bash
#SBATCH --job-name=vintCECdir5
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

set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SLURM_ARRAY_TASK_ID:?submit as an array}"
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${FRESH_ROOT:?}" "${RUN_ROOT:?}" "${BASE_SOURCE_ROOT:?}"
: "${BASE_SOURCE_RECEIPT_SHA:?}" "${DEPENDENCY_RECEIPT:?}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?}" "${PORTABILITY_ENV_ROOT:?}"
: "${PORTABILITY_CHECKPOINT_ROOT:?}" "${SELECTION_MANIFEST_SHA:?}"

MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
BASE_SOURCE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
BENCH_ROOT=${FRESH_ROOT}/benchmarks/natural_direction
BENCHMARK_MANIFEST=${BENCH_ROOT}/manifest.json
SELECTION_MANIFEST=${SOURCE_ROOT}/MemNavData/vint_cec_direction_loss5_manifest_20260828.json
RUNNER=${SOURCE_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh
fail() { echo "ABORT: $*" >&2; exit 2; }

(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < 5 )) || \
  fail "array index outside frozen Loss-5 population"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ "$(sha256sum "${SELECTION_MANIFEST}" | awk '{print $1}')" == \
   "${SELECTION_MANIFEST_SHA}" ]] || fail "selection manifest changed"
[[ "$(sha256sum "${BENCHMARK_MANIFEST}" | awk '{print $1}')" == \
   "aada40d25d01e9385df3ffdcaf37847f471b63c7be785a704eade961346a50b0" ]] || \
  fail "benchmark manifest changed"
[[ -r "${PORTABILITY_ENV_ROOT}/environment_receipt.json" ]] || \
  fail "portability environment receipt missing"
(cd "${PORTABILITY_CHECKPOINT_ROOT}" && sha256sum -c --quiet CHECKPOINTS.sha256) || \
  fail "controller checkpoint receipt failed"

readarray -t identity < <("${MEMNAV_PY}" - "${SELECTION_MANIFEST}" \
  "${BENCHMARK_MANIFEST}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
selection=json.load(open(sys.argv[1])); benchmark=json.load(open(sys.argv[2]))
entry=selection["queries"][int(sys.argv[3])]
matches=[row for row in benchmark["episodes"]
         if row["scene"]==entry["scene"] and row["episode"]==entry["episode"]]
if len(matches)!=1: raise SystemExit("selection did not resolve one history")
source=matches[0]; receipt=json.load(open(source["online_a_episode"]+"/receipt.json"))
print(source["scene"]); print(source["episode"]); print(entry["query_id"])
print(receipt["source_asset"]); print(receipt["source_asset_sha256"])
PY
)
[[ "${#identity[@]}" -eq 5 ]] || fail "history identity is incomplete"
SCENE=${identity[0]}; EPISODE=${identity[1]}; QUERY_ID=${identity[2]}
SCENE_FILE=${identity[3]}; EXPECTED_SCENE_SHA=${identity[4]}
[[ "${QUERY_ID}" == pair_00_revisit ]] || fail "selected query changed"
[[ "$(sha256sum "${SCENE_FILE}" | awk '{print $1}')" == \
   "${EXPECTED_SCENE_SHA}" ]] || fail "HM3D scene asset changed"

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
receipt=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
 row=receipt["dependencies"][name]
 if os.stat(row["path"]).st_size != int(row["bytes"]):
  raise SystemExit("dependency size changed: "+name)
 print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "model dependencies incomplete"
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}

orders=(
  anchor_unaligned,native_bearing_aligned,anchor_bearing_aligned
  native_bearing_aligned,anchor_bearing_aligned,anchor_unaligned
  anchor_bearing_aligned,anchor_unaligned,native_bearing_aligned
)
DIRECTION_ORDER=${orders[SLURM_ARRAY_TASK_ID % 3]}
label=$(printf '%03d_%s_%s' "${SLURM_ARRAY_TASK_ID}" "${SCENE}" "${EPISODE}")
CELL_ROOT=${RUN_ROOT}/evaluation/${label}/vint
[[ ! -e "${CELL_ROOT}" ]] || fail "cell output already exists"
mkdir -p "${CELL_ROOT%/*}"

port_slot=$(( $$ % 6000 ))
base_port=$((20000 + port_slot * 6))
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/vint_cec_dir5_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
echo "index=${SLURM_ARRAY_TASK_ID} scene=${SCENE} episode=${EPISODE} order=${DIRECTION_ORDER} host=$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B /scratch/qw2440 "${BASE_SIF}" env \
  ROOT="${SOURCE_ROOT}" CONTROLLER=vint EVAL_KIND=role_pair_mixed \
  SCENE="${SCENE}" EPISODE="${EPISODE}" MAX_STEPS=600 \
  PORTABILITY_DIRECTION_TRIPLE=1 \
  PORTABILITY_DIRECTION_ORDER="${DIRECTION_ORDER}" \
  ROLE_PAIR_QUERY_ROLE=all ROLE_PAIR_SCOPE=consumed_integration \
  ROLE_PAIR_QUERY_MANIFEST="${SELECTION_MANIFEST}" \
  CEC_REJECT_POLICY=controller_native_exact \
  MEMNAV_PORT="${base_port}" FALLBACK_PORT="$((base_port + 1))" \
  UPSTREAM_PORT="$((base_port + 2))" PROXY_PORT="$((base_port + 3))" \
  HUB_PORT="$((base_port + 4))" FORCED_HUB_PORT="$((base_port + 5))" \
  MEMNAV_PY="${MEMNAV_PY}" HAB_PY="${HAB_PY}" \
  MEMNAV_CKPT="${MEMNAV_CKPT}" NAVDP_CKPT="${NAVDP_CKPT}" \
  LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map \
  LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
  LIGHTGLUE_REPO="${BASE_SOURCE_ROOT}/third_party/LightGlue" \
  DEPENDENCY_ROOT="${BASE_SOURCE_ROOT}/third_party/python" \
  INTERNNAV_ROOT="${BASE_SOURCE_ROOT}/InternNav" \
  SCENE_FILE="${SCENE_FILE}" BENCHMARK_ROOT="${BENCH_ROOT}/${SCENE}" \
  VINT_PY="${PORTABILITY_ENV_ROOT}/vint/bin/python" \
  VINT_CKPT="${PORTABILITY_CHECKPOINT_ROOT}/vint.pth" \
  RUN_ROOT="${CELL_ROOT}" PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  bash "${RUNNER}"

env PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" -u \
  "${SOURCE_ROOT}/MemNavData/audit_vint_cec_bearing_alignment_cell.py" \
  --root "${CELL_ROOT}" --out "${CELL_ROOT}/direction_triple_audit.json"
sha256sum "${CELL_ROOT}/direction_triple_audit.json" \
  >"${CELL_ROOT}/direction_triple_audit.json.sha256"
chmod -R a-w "${CELL_ROOT}"
echo "COMPLETE cell=${CELL_ROOT}"
