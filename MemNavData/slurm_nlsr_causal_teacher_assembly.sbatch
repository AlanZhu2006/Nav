#!/usr/bin/env bash
# Stage B: select candidates and assemble causal geometry labels from the signed
# Stage-A embeddings.  Torch/LingBot are not imported.  A GPU is still required
# because this Habitat-Sim 0.3.1 build uses a headless EGL render context.
#SBATCH --job-name=nlsr_teacher_b
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to the exact clean Stage-B checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed launcher SHA256}"
: "${EXPECTED_EMBEDDING_RECEIPT_SHA:?externally pin the completed Stage-A receipt}"
: "${EXPECTED_STAGE_A_JOB_ID:?record the successful Stage-A Slurm job ID}"
: "${RUN_TAG:?export the audited NLSR run tag}"

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
MANIFEST="${RUN_ROOT}/manifest_multistage/candidate_manifest_multistage.json"
ACCEPTANCE_ROOT="${RUN_ROOT}/causal_ground_scale_acceptance_c8f499dc9c4f_f6f28e20b9f2"
ACCEPTANCE="${ACCEPTANCE_ROOT}/causal_ground_scale_acceptance.json"
OUTPUT_ROOT="${RUN_ROOT}/causal_covisibility_teacher_${EXPECTED_COMMIT:0:12}_bc6bf58536f6"
EMBEDDING_BUNDLE="${OUTPUT_ROOT}/dino_embedding_bundle"
PROGRESS_ROOT="${OUTPUT_ROOT}/stage_b_progress"
TEACHER_BUNDLE="${OUTPUT_ROOT}/teacher_bundle"
STDOUT_TMP="${OUTPUT_ROOT}/.stage_b_stdout.${SLURM_JOB_ID:-manual}.tmp"
STDOUT_FINAL="${OUTPUT_ROOT}/stage_b_stdout.json"

ENTRYPOINT_REL="MemNavData/assemble_manifest_causal_covisibility_teacher.py"
PRODUCER_REL="MemNavData/build_manifest_causal_covisibility_teacher.py"
GEOMETRY_REL="MemNavData/covisibility_teacher.py"
LAUNCHER_REL="MemNavData/slurm_nlsr_causal_teacher_assembly.sbatch"
HABITAT_PY="/scratch/lg154/conda-envs/habitat/bin/python"
EXPECTED_PYTHON_TARGET="/scratch/lg154/conda-envs/habitat/bin/python3.9"
HABITAT_BINDINGS="/scratch/lg154/conda-envs/habitat/lib/python3.9/site-packages/habitat_sim-0.3.1-py3.9-linux-x86_64.egg/habitat_sim/_ext/habitat_sim_bindings.cpython-39-x86_64-linux-gnu.so"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
RAW_SCENE="${RUN_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"

EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_ACCEPTANCE_SHA="0ced1b3a56fe0e089bad35c5807c219da27daad48e6bbd4bb29436ef2b8eb3ac"
EXPECTED_PRODUCER_SHA="5bcba961491d2d2b6750d477ac9499b45c3b0eae90265b6ab1062d658b686aa1"
EXPECTED_GEOMETRY_SHA="1b0b16dbff6a306c8371fbef43816ebd146117ca963402a4d69f785f99d107d0"
EXPECTED_ENTRYPOINT_SHA="2d77ca52f93b5f1cfe9e4b3a250acdbcee5f20c803c75bb6d4ab53e84dea97c4"
EXPECTED_PYTHON_SHA="8e9776d5b77f7d22be61bfebc82a544313837abb7d6ca6b110f4547f73259d89"
EXPECTED_HABITAT_BINDINGS_SHA="5f0a6a4036d10c6b396193001883cf08ddd9b20f5a90551ab5cfb5be705bd1a8"
EXPECTED_SIF_BYTES="10035580928"
EXPECTED_SIF_HEAD_SHA="990c1377491fbf58c1d7d37bcf8043249826bc7a094eacc4c7a443b9905126c7"
EXPECTED_OVERLAY_BYTES="128854888448"
EXPECTED_OVERLAY_HEAD_SHA="43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b"

fail() {
  echo "ABORT: $*" >&2
  exit 2
}

full_sha_is() {
  local path="$1" expected="$2" label="$3" actual
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical ${label}: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || \
    fail "${label} SHA mismatch: ${actual} != ${expected}"
}

prefix_identity_is() {
  local path="$1" bytes="$2" head_sha="$3" label="$4" actual
  [[ -f "${path}" && ! -L "${path}" && \
    "$(stat -c '%s' "${path}")" == "${bytes}" ]] || fail "${label} byte identity changed"
  actual="$(dd if="${path}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${head_sha}" ]] || fail "${label} header SHA changed"
}

pinned_python_is() {
  local resolved actual
  [[ -L "${HABITAT_PY}" && "$(readlink "${HABITAT_PY}")" == "python3.9" ]] || \
    fail "Habitat Python symlink changed"
  resolved="$(readlink -f "${HABITAT_PY}")"
  [[ "${resolved}" == "${EXPECTED_PYTHON_TARGET}" ]] || \
    fail "resolved Habitat Python path changed: ${resolved}"
  actual="$(sha256sum "${resolved}" | awk '{print $1}')"
  [[ "${actual}" == "${EXPECTED_PYTHON_SHA}" ]] || fail "Habitat Python SHA changed"
}

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid commit pin"
[[ "${EXPECTED_LAUNCHER_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "invalid launcher pin"
[[ "${EXPECTED_EMBEDDING_RECEIPT_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "invalid Stage-A receipt pin"
[[ "${EXPECTED_STAGE_A_JOB_ID}" =~ ^[0-9]+$ ]] || fail "invalid Stage-A job ID"
for command_name in git sha256sum singularity flock nvidia-smi; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
[[ "${gpu_name}" == NVIDIA\ A100* ]] || fail "Stage B requires the pinned A100 class: ${gpu_name}"
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || fail "not a checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "Stage-B checkout commit changed"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "Stage-B checkout is not clean"
fi
full_sha_is "$0" "${EXPECTED_LAUNCHER_SHA}" "spooled Stage-B launcher"
full_sha_is "${REPO_ROOT}/${LAUNCHER_REL}" "${EXPECTED_LAUNCHER_SHA}" \
  "repository Stage-B launcher"
full_sha_is "${REPO_ROOT}/${PRODUCER_REL}" "${EXPECTED_PRODUCER_SHA}" \
  "causal teacher producer"
full_sha_is "${REPO_ROOT}/${ENTRYPOINT_REL}" "${EXPECTED_ENTRYPOINT_SHA}" \
  "Stage-B entrypoint"
full_sha_is "${REPO_ROOT}/${GEOMETRY_REL}" "${EXPECTED_GEOMETRY_SHA}" \
  "co-visibility geometry implementation"
full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "formal causal manifest"
full_sha_is "${ACCEPTANCE}" "${EXPECTED_ACCEPTANCE_SHA}" "causal-scale acceptance"
full_sha_is "${EMBEDDING_BUNDLE}/embedding_receipt.json" \
  "${EXPECTED_EMBEDDING_RECEIPT_SHA}" "signed Stage-A receipt"
full_sha_is "${HABITAT_BINDINGS}" "${EXPECTED_HABITAT_BINDINGS_SHA}" \
  "Habitat-Sim bindings"
pinned_python_is
prefix_identity_is "${BASE_SIF}" "${EXPECTED_SIF_BYTES}" "${EXPECTED_SIF_HEAD_SHA}" \
  "base SIF"
prefix_identity_is "${OVERLAY}" "${EXPECTED_OVERLAY_BYTES}" \
  "${EXPECTED_OVERLAY_HEAD_SHA}" "dataset overlay"
[[ -d "${RAW_SCENE}" ]] || fail "audited YmJ raw scene is absent"

exec 9>"${RUN_ROOT}/.causal_teacher_stage_b.lock"
flock -n 9 || fail "another causal teacher Stage B is active"
mkdir -p "${OUTPUT_ROOT}"
resume_args=()
if [[ -e "${PROGRESS_ROOT}" || -e "${TEACHER_BUNDLE}" ]]; then
  resume_args+=(--resume)
fi

CONTAINER=(singularity exec --nv
  --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export APPTAINERENV_PYTHONPATH="${PYTHONPATH}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"

"${CONTAINER[@]}" "${HABITAT_PY}" - "${MANIFEST}" "${HABITAT_BINDINGS}" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys
import habitat_sim

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("summary", {}).get("sample_count") != 600:
    raise SystemExit("formal manifest sample count changed")
if habitat_sim.__version__ != "0.3.1":
    raise SystemExit("Habitat-Sim version changed")
if Path(habitat_sim._ext.habitat_sim_bindings.__file__).resolve() != Path(sys.argv[2]):
    raise SystemExit("Habitat-Sim bindings origin changed")
if importlib.util.find_spec("torch") is not None:
    raise SystemExit("Torch leaked into the isolated Stage-B runtime")
required = ("habitat_sim", "magnum", "quaternion", "numpy", "pyarrow")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Stage-B dependencies are missing: {missing}")
print(json.dumps({"stage_b_preflight": "passed", "sample_count": 600},
                 sort_keys=True))
PY

echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} stage=causal_teacher_b stage_a_job=${EXPECTED_STAGE_A_JOB_ID}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
rm -f "${STDOUT_TMP}"
trap 'rm -f "${STDOUT_TMP}"' EXIT
"${CONTAINER[@]}" "${HABITAT_PY}" "${REPO_ROOT}/${ENTRYPOINT_REL}" \
  --manifest "${MANIFEST}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
  --expected-sample-count 600 \
  --embedding-bundle "${EMBEDDING_BUNDLE}" \
  --expected-embedding-receipt-sha256 "${EXPECTED_EMBEDDING_RECEIPT_SHA}" \
  --expected-habitat-version 0.3.1 \
  --expected-python-sha256 "${EXPECTED_PYTHON_SHA}" \
  --habitat-bindings-file "${HABITAT_BINDINGS}" \
  --expected-habitat-bindings-sha256 "${EXPECTED_HABITAT_BINDINGS_SHA}" \
  --expected-producer-sha256 "${EXPECTED_PRODUCER_SHA}" \
  --expected-geometry-sha256 "${EXPECTED_GEOMETRY_SHA}" \
  --expected-entrypoint-sha256 "${EXPECTED_ENTRYPOINT_SHA}" \
  --top-k 32 \
  --temporal-nms-radius 4 \
  --backprojection-stride 6 \
  --depth-tolerance-m 0.3 \
  --positive-threshold 0.5 \
  --negative-threshold 0.1 \
  --progress-dir "${PROGRESS_ROOT}" \
  --out-dir "${TEACHER_BUNDLE}" \
  "${resume_args[@]}" | tee "${STDOUT_TMP}"
mv -f "${STDOUT_TMP}" "${STDOUT_FINAL}"
trap - EXIT

"${CONTAINER[@]}" "${HABITAT_PY}" - "${TEACHER_BUNDLE}" \
  "${EXPECTED_MANIFEST_SHA}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = {
    "teacher.json", "teacher.json.sha256", "teacher.csv", "teacher.csv.sha256",
    "audit.json", "audit.json.sha256",
}
if {path.name for path in root.iterdir()} != expected:
    raise SystemExit("Stage-B bundle file set changed")
audit_path = root / "audit.json"
audit_raw = audit_path.read_bytes()
audit_sha = hashlib.sha256(audit_raw).hexdigest()
if (root / "audit.json.sha256").read_text(encoding="ascii") != \
        f"{audit_sha}  audit.json\n":
    raise SystemExit("Stage-B audit sidecar changed")
audit = json.loads(audit_raw)
if audit.get("status") != "audited_not_deployment_approved":
    raise SystemExit("Stage-B audit status changed")
if audit.get("manifest", {}).get("content_sha256") != sys.argv[2]:
    raise SystemExit("Stage-B bundle is bound to another manifest")
if not audit.get("invariants", {}).get("exact_manifest_sample_cover"):
    raise SystemExit("Stage-B exact sample cover failed")
print(json.dumps({"stage_b": "passed", "audit_sha256": audit_sha,
                  "counts": audit.get("counts")}, sort_keys=True))
PY

echo "NLSR causal teacher Stage B passed: ${TEACHER_BUNDLE}"
