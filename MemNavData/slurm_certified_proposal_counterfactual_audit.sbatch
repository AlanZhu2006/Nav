#!/usr/bin/env bash
# One consumed held-out Revisit history per task; no closed-loop query rollout.
#SBATCH --job-name=cec_prop_audit
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --partition=h100_tandon,h200_public,a100_tandon
#SBATCH --qos=gpu48
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022

: "${SLURM_ARRAY_TASK_ID:?submit over the frozen audit manifest}"
: "${SOURCE_ROOT:?set immutable audit source root}"
: "${SOURCE_RECEIPT:?set immutable source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${BASE_SOURCE_ROOT:?set frozen base source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set base source receipt SHA}"
: "${DEPENDENCY_RECEIPT:?set model dependency receipt}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set dependency receipt SHA}"
: "${AUDIT_MANIFEST:?set frozen audit population manifest}"
: "${EXPECTED_AUDIT_MANIFEST_SHA:?set audit manifest SHA}"
: "${RUN_ROOT:?set diagnostic output root}"

MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
MEMNAV_SERVER=${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py
RUNNER=${SOURCE_ROOT}/MemNavData/run_certified_proposal_counterfactual_episode.py
SERVER_LAUNCHER=${SOURCE_ROOT}/MemNavData/retrying_server_launcher.py
INTERNNAV_ROOT=${BASE_SOURCE_ROOT}/InternNav
LIGHTGLUE_REPO=${BASE_SOURCE_ROOT}/third_party/LightGlue
DEPENDENCY_ROOT=${BASE_SOURCE_ROOT}/third_party/python
LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
TORCH_HOME=${BASE_SOURCE_ROOT}/torch_home
BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${SLURM_ARRAY_TASK_ID}" =~ ^[0-9]+$ ]] || fail "bad array index"
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ "$(sha256sum "${AUDIT_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_AUDIT_MANIFEST_SHA}" ]] || fail "audit manifest changed"
[[ -x "${MEMNAV_PY}" && -x "${HAB_PY}" && -f "${BASE_SIF}" ]] || \
  fail "runtime dependency missing"

readarray -t spec < <("${MEMNAV_PY}" - "${AUDIT_MANIFEST}" \
  "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
manifest=json.load(open(sys.argv[1])); index=int(sys.argv[2])
rows=manifest["episodes"]
if not 0 <= index < len(rows):
    raise SystemExit("array index outside frozen population")
row=rows[index]
print(row["cohort"])
print(row["benchmark_root"])
print(row["benchmark_manifest_sha256"])
print(row["population_index"])
PY
)
[[ "${#spec[@]}" -eq 4 ]] || fail "audit row is malformed"
COHORT=${spec[0]}; BENCHMARK_ROOT=${spec[1]}
BENCHMARK_MANIFEST_SHA=${spec[2]}; POPULATION_INDEX=${spec[3]}
[[ "${COHORT}" =~ ^[a-z0-9_]+$ ]] || fail "bad cohort"
[[ "${BENCHMARK_MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "bad manifest SHA"
[[ "${POPULATION_INDEX}" =~ ^[0-9]+$ ]] || fail "bad population index"
[[ "$(sha256sum "${BENCHMARK_ROOT}/manifest.json" | awk '{print $1}')" == \
   "${BENCHMARK_MANIFEST_SHA}" ]] || fail "benchmark input changed"

readarray -t dependencies < <("${MEMNAV_PY}" - "${DEPENDENCY_RECEIPT}" <<'PY'
import json,os,sys
receipt=json.load(open(sys.argv[1]))
for name in ("gatecurr600","lingbot_map_long"):
    row=receipt["dependencies"][name]
    if os.stat(row["path"]).st_size != int(row["bytes"]):
        raise SystemExit(f"dependency size changed: {name}")
    print(row["path"])
PY
)
[[ "${#dependencies[@]}" -eq 2 ]] || fail "dependency receipt invalid"
MEMNAV_CKPT=${dependencies[0]}; LINGBOT_WEIGHTS=${dependencies[1]}

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/cec_prop_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}:${BASE_SOURCE_ROOT}/NavDP/baselines/navdp:${BASE_SOURCE_ROOT}/NavDP/baselines/memnav:${INTERNNAV_ROOT}/src/diffusion-policy${PYTHONPATH:+:${PYTHONPATH}}
export TORCH_HOME
runtime_root=${SLURM_TMPDIR:-/tmp}/cec_prop_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${runtime_root}/memnav" "${RUN_ROOT}/records" \
  "${RUN_ROOT}/logs" "${RUN_ROOT}/buffer"
label=$(printf '%03d' "${SLURM_ARRAY_TASK_ID}")
port_key=$(( (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 67) % 15000 ))
base_port=$((20000 + port_key))
port_file=${runtime_root}/memnav.port
launcher_log=${RUN_ROOT}/logs/server_${label}.log
launcher_pid=
cleanup() {
  if [[ -n "${launcher_pid}" ]] && kill -0 "${launcher_pid}" 2>/dev/null; then
    kill "${launcher_pid}" 2>/dev/null || true
    wait "${launcher_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  exec env PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
    PYTHONPATH="${PYTHONPATH}" TORCH_HOME="${TORCH_HOME}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    LINGBOT_REPO="${LINGBOT_REPO}" LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 MEMNAV_NUM_SCALE=8 MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${SERVER_LAUNCHER}" \
      --base-port "${base_port}" --range-start 20000 --range-end 49999 \
      --stride 1994 --max-attempts 12 --ready-timeout 720 \
      --port-file "${port_file}" \
      --receipt-file "${RUN_ROOT}/logs/server_${label}.launcher.json" \
      --log-prefix "${launcher_log}.child" \
      --cwd "${runtime_root}/memnav" -- \
      "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" --port '{port}' \
        --checkpoint "${MEMNAV_CKPT}" --internnav_root "${INTERNNAV_ROOT}" \
        --num_samples 16 --exclude_recent 32 --retrieval raw \
        --retrieval_candidate_top_k 32 --retrieval_candidate_min_gap 16 \
        --flow_gate auto --buffer_root "${RUN_ROOT}/buffer/${label}" \
        --certified_relocalization --certified_counterfactual_audit \
        --lightglue_repo "${LIGHTGLUE_REPO}" \
        --lightglue_dependency_root "${DEPENDENCY_ROOT}" \
        --lightglue_max_keypoints 2048
) >"${launcher_log}" 2>&1 &
launcher_pid=$!

ready=0
for _ in $(seq 1 420); do
  kill -0 "${launcher_pid}" 2>/dev/null || {
    tail -n 200 "${launcher_log}" >&2
    fail "MemNav launcher died"
  }
  if [[ -s "${port_file}" ]]; then ready=1; break; fi
  sleep 2
done
[[ "${ready}" -eq 1 ]] || fail "MemNav launcher timed out"
port=$(<"${port_file}")
[[ "${port}" =~ ^[0-9]+$ ]] || fail "bad MemNav port"

output=${RUN_ROOT}/records/${label}_${COHORT}.json
[[ ! -e "${output}" ]] || fail "audit output already exists"
singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX}" \
  PYTHONPATH="${PYTHONPATH}" "${HAB_PY}" -u "${RUNNER}" \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --expected-manifest-sha256 "${BENCHMARK_MANIFEST_SHA}" \
    --index "${POPULATION_INDEX}" \
    --memnav-url "http://127.0.0.1:${port}" --out "${output}"

echo "[complete] ${output}"
