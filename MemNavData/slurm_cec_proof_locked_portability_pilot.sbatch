#!/usr/bin/env bash
#SBATCH --job-name=cecPortPilot
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

# Four frozen Fresh-HM3D accepted histories x three primary executors.  Each
# cell runs grant and forced-reject-native sequentially against the same loaded
# processes and emits an independent authority-pair audit.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SLURM_ARRAY_TASK_ID:?submit as array 0-11}"
: "${SOURCE_ROOT:?}" "${SOURCE_RECEIPT:?}" "${EXPECTED_SOURCE_RECEIPT_SHA:?}"
: "${FRESH_ROOT:?}" "${RUN_ROOT:?}" "${QUERY_MANIFEST:?}"
: "${EXPECTED_QUERY_MANIFEST_SHA:?}" "${BASE_SOURCE_ROOT:?}"
: "${BASE_SOURCE_RECEIPT_SHA:?}" "${DEPENDENCY_RECEIPT:?}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?}" "${PORTABILITY_ENV_ROOT:?}"
: "${PORTABILITY_CHECKPOINT_ROOT:?}"

MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
BASE_SOURCE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
BENCH_ROOT=${FRESH_ROOT}/benchmarks/natural_direction
RUNNER=${SOURCE_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c --quiet "${SOURCE_RECEIPT}") || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ "$(sha256sum "${QUERY_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_QUERY_MANIFEST_SHA}" ]] || fail "query manifest changed"
[[ -r "${PORTABILITY_ENV_ROOT}/environment_receipt.json" ]] || \
  fail "portability environment receipt missing"
(cd "${PORTABILITY_CHECKPOINT_ROOT}" && sha256sum -c --quiet CHECKPOINTS.sha256) || \
  fail "controller checkpoint receipt failed"

controllers=(navdp vint iplanner)
pilot_histories=(0 7 14 21)
controller_index=$((SLURM_ARRAY_TASK_ID / 4))
pilot_index=$((SLURM_ARRAY_TASK_ID % 4))
(( controller_index < ${#controllers[@]} )) || fail "array index outside pilot"
CONTROLLER=${controllers[controller_index]}
HISTORY_INDEX=${pilot_histories[pilot_index]}

readarray -t identity < <("${MEMNAV_PY}" - "${QUERY_MANIFEST}" \
  "${BENCH_ROOT}/manifest.json" "${HISTORY_INDEX}" <<'PY'
import json,sys
query=json.load(open(sys.argv[1])); benchmark=json.load(open(sys.argv[2]))
index=int(sys.argv[3])
rows=[row for row in query["queries"] if int(row["history_index"]) == index]
if len(rows) != 1: raise SystemExit("accepted history is not unique")
row=rows[0]; source=benchmark["episodes"][index]
if (row["scene"],row["episode"]) != (source["scene"],source["episode"]):
 raise SystemExit("accepted/benchmark identity mismatch")
receipt=json.load(open(source["online_a_episode"]+"/receipt.json"))
print(row["scene"]); print(row["episode"]); print(row["query_id"])
print(receipt["source_asset"]); print(receipt["source_asset_sha256"])
PY
)
[[ "${#identity[@]}" -eq 5 ]] || fail "pilot identity is incomplete"
SCENE=${identity[0]}; EPISODE=${identity[1]}; QUERY_ID=${identity[2]}
SCENE_FILE=${identity[3]}; EXPECTED_SCENE_SHA=${identity[4]}
[[ "$(sha256sum "${SCENE_FILE}" | awk '{print $1}')" == \
   "${EXPECTED_SCENE_SHA}" ]] || fail "HM3D scene asset changed"

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
r=json.load(open(sys.argv[1]))
for name in ("gatecurr600","navdp_checkpoint","lingbot_map_long"):
 row=r["dependencies"][name]
 if os.stat(row["path"]).st_size != int(row["bytes"]):
  raise SystemExit("dependency size changed: "+name)
 print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 3 ]] || fail "model dependencies are incomplete"
MEMNAV_CKPT=${dependencies[0]}; NAVDP_CKPT=${dependencies[1]}
LINGBOT_WEIGHTS=${dependencies[2]}

label=$(printf '%03d_%s_%s_%s' "${HISTORY_INDEX}" "${SCENE}" \
  "${EPISODE}" "${QUERY_ID}")
CELL_ROOT=${RUN_ROOT}/evaluation/${label}/${CONTROLLER}
[[ ! -e "${CELL_ROOT}" ]] || fail "pilot cell already exists"
mkdir -p "${CELL_ROOT%/*}"

if (( (HISTORY_INDEX + controller_index) % 2 == 0 )); then
  AUTHORITY_ORDER=grant,forced_reject_native
else
  AUTHORITY_ORDER=forced_reject_native,grant
fi
port_slot=$(( $$ % 6000 ))
base_port=$((20000 + port_slot * 6))
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/cec_port_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
echo "host=$(hostname) gpu_task=${SLURM_ARRAY_TASK_ID} history=${HISTORY_INDEX} controller=${CONTROLLER} order=${AUTHORITY_ORDER}"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B /scratch/qw2440 "${BASE_SIF}" env \
  ROOT="${SOURCE_ROOT}" CONTROLLER="${CONTROLLER}" \
  EVAL_KIND=role_pair_mixed SCENE="${SCENE}" EPISODE="${EPISODE}" \
  MAX_STEPS=600 PORTABILITY_AUTHORITY_PAIR=1 \
  PORTABILITY_AUTHORITY_ORDER="${AUTHORITY_ORDER}" \
  ROLE_PAIR_QUERY_ROLE=all ROLE_PAIR_QUERY_MANIFEST="${QUERY_MANIFEST}" \
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
  SCENE_FILE="${SCENE_FILE}" \
  BENCHMARK_ROOT="${BENCH_ROOT}/${SCENE}" \
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
  RUN_ROOT="${CELL_ROOT}" PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  bash "${RUNNER}"

env PYTHONPATH="${SOURCE_ROOT}" "${MEMNAV_PY}" -u \
  "${SOURCE_ROOT}/MemNavData/audit_cec_authority_pair.py" \
  --root "${CELL_ROOT}" --controller "${CONTROLLER}" \
  --query-manifest "${QUERY_MANIFEST}" \
  --out "${CELL_ROOT}/authority_pair_audit.json"
sha256sum "${CELL_ROOT}/authority_pair_audit.json" \
  >"${CELL_ROOT}/authority_pair_audit.json.sha256"
chmod -R a-w "${CELL_ROOT}"
echo "COMPLETE cell=${CELL_ROOT}"
