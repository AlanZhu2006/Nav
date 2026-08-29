#!/usr/bin/env bash
#SBATCH --job-name=h3T1ViNT
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

# Same-process ViNT native versus ViNT+CEC with bounded physical bearing turns.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

: "${SLURM_ARRAY_TASK_ID:?submit as a history array}"
: "${PHASE:?smoke or formal}"
: "${TASK_ROOT:?}" "${TASK_RECEIPT:?}" "${EXPECTED_TASK_RECEIPT_SHA:?}"
: "${FORMAL_RUN_ROOT:?}" "${BENCH_ROOT:?}"
: "${CONSTRUCTION_VERIFICATION:?}"
: "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA:?}"
: "${BASE_SOURCE_ROOT:?}" "${BASE_SOURCE_RECEIPT_SHA:?}"
: "${DEPENDENCY_RECEIPT:?}" "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?}"
: "${PORTABILITY_ENV_ROOT:?}" "${PORTABILITY_CHECKPOINT_ROOT:?}"
ROLE_PAIR_SCOPE=${ROLE_PAIR_SCOPE:-paper_heldout}

MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
BASE_SOURCE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
BENCHMARK_MANIFEST=${BENCH_ROOT}/manifest.json
RUNNER=${TASK_ROOT}/MemNavData/run_cec_controller_portability_smoke_local.sh
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "${PHASE}" == smoke || "${PHASE}" == formal ]] || fail "invalid phase"
[[ "${ROLE_PAIR_SCOPE}" == paper_heldout \
   || "${ROLE_PAIR_SCOPE}" == paper_replication ]] || \
  fail "invalid formal role-pair scope"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}") || \
  fail "task bundle changed"
[[ "$(sha256sum "${CONSTRUCTION_VERIFICATION}" | awk '{print $1}')" == \
   "${EXPECTED_CONSTRUCTION_VERIFICATION_SHA}" ]] || \
  fail "construction verification changed"
[[ "$(sha256sum "${BASE_SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ -r "${PORTABILITY_ENV_ROOT}/environment_receipt.json" ]] || \
  fail "portability environment receipt missing"
(cd "${PORTABILITY_CHECKPOINT_ROOT}" && \
  sha256sum -c --quiet CHECKPOINTS.sha256) || fail "checkpoint receipt failed"

readarray -t gate < <("${MEMNAV_PY}" - "${CONSTRUCTION_VERIFICATION}" \
  "${BENCHMARK_MANIFEST}" <<'PY'
import hashlib,json,sys
verification=json.load(open(sys.argv[1])); manifest=sys.argv[2]
print(str(verification.get("verified") is True).lower())
print(str(verification.get("formal_policy_evaluation_authorized") is True).lower())
print(verification.get("benchmark_manifest_sha256"))
print(hashlib.sha256(open(manifest,"rb").read()).hexdigest())
print(verification.get("histories"))
PY
)
[[ "${#gate[@]}" -eq 5 && "${gate[0]}" == true \
   && "${gate[1]}" == true && "${gate[2]}" == "${gate[3]}" ]] || \
  fail "construction power gate or manifest binding failed"
[[ "${gate[4]}" =~ ^[1-9][0-9]*$ ]] || fail "invalid history count"

if [[ "${PHASE}" == smoke ]]; then
  HISTORY_INDEX=0
  PHASE_ROOT=${FORMAL_RUN_ROOT}/smoke/vint
  MAX_STEPS=80
else
  (( SLURM_ARRAY_TASK_ID < gate[4] )) || fail "history index outside gate"
  HISTORY_INDEX=${SLURM_ARRAY_TASK_ID}
  PHASE_ROOT=${FORMAL_RUN_ROOT}/formal/vint
  MAX_STEPS=600
fi

readarray -t identity < <("${MEMNAV_PY}" - "${BENCHMARK_MANIFEST}" \
  "${HISTORY_INDEX}" <<'PY'
import json,sys
manifest=json.load(open(sys.argv[1])); index=int(sys.argv[2])
source=manifest["episodes"][index]
receipt=json.load(open(source["online_a_episode"]+"/receipt.json"))
print(source["scene"]); print(source["episode"])
print(receipt["source_asset"]); print(receipt["source_asset_sha256"])
PY
)
[[ "${#identity[@]}" -eq 4 ]] || fail "history identity is incomplete"
SCENE=${identity[0]}; EPISODE=${identity[1]}
SCENE_FILE=${identity[2]}; EXPECTED_SCENE_SHA=${identity[3]}
[[ "$(sha256sum "${SCENE_FILE}" | awk '{print $1}')" == \
   "${EXPECTED_SCENE_SHA}" ]] || fail "scene asset changed"

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

label=$(printf '%03d_%s_%s' "${HISTORY_INDEX}" "${SCENE}" "${EPISODE}")
CELL_ROOT=${PHASE_ROOT}/evaluation/${label}/vint
[[ ! -e "${CELL_ROOT}" ]] || fail "cell output already exists"
mkdir -p "${CELL_ROOT%/*}"
if (( HISTORY_INDEX % 2 == 0 )); then
  AUTHORITY_ORDER=grant,forced_reject_native
else
  AUTHORITY_ORDER=forced_reject_native,grant
fi

port_slot=$(( $$ % 6000 ))
base_port=$((20000 + port_slot * 6))
export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/vint_cec_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${PYTHONPYCACHEPREFIX}"
echo "phase=${PHASE} history=${HISTORY_INDEX} scene=${SCENE} episode=${EPISODE} order=${AUTHORITY_ORDER} host=$(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader

singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  -B /scratch/qw2440 "${SIF}" env \
  ROOT="${TASK_ROOT}" CONTROLLER=vint EVAL_KIND=role_pair_mixed \
  SCENE="${SCENE}" EPISODE="${EPISODE}" MAX_STEPS="${MAX_STEPS}" \
  PORTABILITY_AUTHORITY_PAIR=1 \
  PORTABILITY_AUTHORITY_ORDER="${AUTHORITY_ORDER}" \
  PORTABILITY_CEC_ACCEPT_ALIGNMENT=first_certified_bounded \
  ROLE_PAIR_QUERY_ROLE=all ROLE_PAIR_SCOPE="${ROLE_PAIR_SCOPE}" \
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

env PYTHONPATH="${TASK_ROOT}" "${MEMNAV_PY}" -u \
  "${TASK_ROOT}/MemNavData/audit_vint_controller_native_pair.py" \
  --root "${CELL_ROOT}" \
  --out "${CELL_ROOT}/controller_native_pair_audit.json"
sha256sum "${CELL_ROOT}/controller_native_pair_audit.json" \
  >"${CELL_ROOT}/controller_native_pair_audit.json.sha256"
chmod -R a-w "${CELL_ROOT}"
echo "COMPLETE phase=${PHASE} cell=${CELL_ROOT}"
