#!/usr/bin/env bash
# Independently accept the immutable causal-scale artifact by reopening every
# routed cache pair and re-hashing all 100 causal pose/RGB prefixes. CPU-only.
#SBATCH --job-name=nlsr_scale_audit
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REPO_ROOT:?export REPO_ROOT to the exact clean acceptance checkout}"
: "${EXPECTED_COMMIT:?export its full commit SHA}"
: "${EXPECTED_LAUNCHER_SHA:?export the committed launcher SHA256}"
: "${RUN_TAG:?export the audited NLSR run tag}"

RESULT_BASE="${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807}"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"
MANIFEST="${RUN_ROOT}/manifest_multistage/candidate_manifest_multistage.json"
SCALE_ARTIFACT="${RUN_ROOT}/causal_ground_scale/causal_ground_scale.json"
AUDITOR_REL="MemNavData/audit_causal_ground_scale_artifact.py"
LAUNCHER_REL="MemNavData/slurm_nlsr_causal_ground_scale_acceptance.sbatch"
OUTPUT_ROOT="${RUN_ROOT}/causal_ground_scale_acceptance_${EXPECTED_COMMIT:0:12}_f6f28e20b9f2"
OUTPUT="${OUTPUT_ROOT}/causal_ground_scale_acceptance.json"

MEMNAV_PY="/scratch/lg154/conda-envs/memnav/bin/python"
EXPECTED_PYTHON_TARGET="/scratch/lg154/conda-envs/memnav/bin/python3.10"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
LINGBOT_REPO="/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map"
WEIGHTS="${LINGBOT_REPO}/weights/lingbot-map-long.pt"
RAW_SCENE="${RUN_ROOT}/raw/mp3d_3leg/YmJkqBEsHnH"
EPISODE_ROOT="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"

EXPECTED_MANIFEST_SHA="bc6bf58536f6c159d1898ac03abe365eadba65c22cd246e39401821962abb34c"
EXPECTED_MANIFEST_SIDECAR_SHA="64de7deaba664264aaa74fb566ec9ae866b277b49930b404903f0ea737374d0c"
EXPECTED_ARTIFACT_SHA="f6f28e20b9f21a764132af04b425f23b65e1dc779798d831a571a7df26bd78db"
EXPECTED_ARTIFACT_SIDECAR_SHA="9f8ec2451566c45d735ccdfa3abb0ff48749f65821671cd63be37b3cdb592093"
EXPECTED_AUDITOR_SHA="ca1d73398400823763030c31931853cbadda7d0bf5dbc709976e467b075072c8"
EXPECTED_CONTRACT_SHA="0eb883ba9b1e1c9ff7cb61bd395417b908f14a8a107b625a87413c9abdbf4a96"
EXPECTED_PRODUCER_SOURCE_SHA="523126bb1f56f4754d0b8a0a34b327e3d334fbc01bd0ae51f12c27c50ffebf20"
EXPECTED_FRONTIER_SOURCE_SHA="d2e5889736488f21449fd5d9a054d945cd17bdb509bc311eb23dd58c4de23aa5"
EXPECTED_FLOW_ROUTER_SHA="4896a77fa35871588f4add5c83c10e48fbb9bf45a15845f0a5d4332bc63a0035"
EXPECTED_CACHE_SCHEMA_SHA="412fdc9ba11947406297ba639790b9d3acf7d9c3ab6125dea8e1a1246e38f227"
EXPECTED_PRODUCER_BUNDLE_SHA="8da2a4c42de010e699c5ab65a2f6bbc9600c29acc96b71ef82d89126d8b4ca26"
EXPECTED_CONFIGURATION_SHA="f1b01a8b6cd07a8cfb0ee306083e36d6399d21f22f242b3e2eb0f5ee7624945d"
EXPECTED_STREAM_SHA="b5142b76f9062634db826a0e7f2c3da19e4b07e21c2325ecbc60a44cc52370b0"
EXPECTED_LINGBOT_COMMIT="7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2"
EXPECTED_WEIGHTS_SHA="832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"
EXPECTED_PYTHON_SHA="cf18c572f59ed93caec89258837eb1cbc5b4a6893e272888c3abebb90ae5ff86"
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
  [[ -f "${path}" && ! -L "${path}" ]] || \
    fail "missing physical ${label}: ${path}"
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || \
    fail "${label} SHA mismatch: ${actual} != ${expected}"
}

prefix_identity_is() {
  local path="$1" bytes="$2" head_sha="$3" label="$4" actual
  [[ -f "${path}" && ! -L "${path}" && \
    "$(stat -c '%s' "${path}")" == "${bytes}" ]] || \
    fail "${label} byte identity changed"
  actual="$(dd if="${path}" bs=1048576 count=1 status=none | \
    sha256sum | awk '{print $1}')"
  [[ "${actual}" == "${head_sha}" ]] || fail "${label} header SHA changed"
}

pinned_python_is() {
  local resolved actual
  [[ -L "${MEMNAV_PY}" && "$(readlink "${MEMNAV_PY}")" == "python3.10" ]] || \
    fail "MemNav Python symlink changed"
  resolved="$(readlink -f "${MEMNAV_PY}")"
  [[ "${resolved}" == "${EXPECTED_PYTHON_TARGET}" ]] || \
    fail "resolved MemNav Python path changed: ${resolved}"
  [[ -f "${resolved}" && ! -L "${resolved}" ]] || \
    fail "resolved MemNav Python is not physical"
  actual="$(sha256sum "${resolved}" | awk '{print $1}')"
  [[ "${actual}" == "${EXPECTED_PYTHON_SHA}" ]] || \
    fail "MemNav Python SHA changed: ${actual}"
}

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${EXPECTED_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || fail "invalid commit pin"
[[ "${EXPECTED_LAUNCHER_SHA}" =~ ^[0-9a-f]{64}$ ]] || \
  fail "invalid launcher pin"
for command_name in git sha256sum singularity flock; do
  command -v "${command_name}" >/dev/null || fail "${command_name} is unavailable"
done
[[ -d "${REPO_ROOT}/.git" || -f "${REPO_ROOT}/.git" ]] || fail "not a checkout"
[[ "$(git -C "${REPO_ROOT}" rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || \
  fail "acceptance checkout commit changed"
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain)" ]]; then
  git -C "${REPO_ROOT}" status --short >&2
  fail "acceptance checkout is not clean"
fi
launcher_sha="$(sha256sum "$0" | awk '{print $1}')"
repository_launcher_sha="$(sha256sum \
  "${REPO_ROOT}/${LAUNCHER_REL}" | awk '{print $1}')"
[[ "${launcher_sha}" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "spooled launcher changed"
[[ "${repository_launcher_sha}" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "repository launcher changed"

full_sha_is "${REPO_ROOT}/${AUDITOR_REL}" "${EXPECTED_AUDITOR_SHA}" \
  "causal-scale auditor"
full_sha_is "${REPO_ROOT}/MemNavData/external_causal_scale_contract.py" \
  "${EXPECTED_CONTRACT_SHA}" "external-scale contract"
full_sha_is "${REPO_ROOT}/MemNavData/build_causal_ground_scale.py" \
  "${EXPECTED_PRODUCER_SOURCE_SHA}" "causal-scale producer"
full_sha_is "${REPO_ROOT}/MemNavData/build_novel_frontier_candidates.py" \
  "${EXPECTED_FRONTIER_SOURCE_SHA}" "frontier/cache dependency"
full_sha_is "${REPO_ROOT}/MemNavData/flow_cache_routing.py" \
  "${EXPECTED_FLOW_ROUTER_SHA}" "flow-cache router"
full_sha_is \
  "${REPO_ROOT}/InternNav/internnav/model/basemodel/memnav/cache_schema.py" \
  "${EXPECTED_CACHE_SCHEMA_SHA}" "versioned cache schema"
full_sha_is \
  "${REPO_ROOT}/InternNav/internnav/model/basemodel/memnav/lingbot_stream.py" \
  "${EXPECTED_STREAM_SHA}" "LingBot stream source"
full_sha_is "${MANIFEST}" "${EXPECTED_MANIFEST_SHA}" "multistage manifest"
full_sha_is "${MANIFEST}.sha256" "${EXPECTED_MANIFEST_SIDECAR_SHA}" \
  "multistage manifest sidecar"
full_sha_is "${SCALE_ARTIFACT}" "${EXPECTED_ARTIFACT_SHA}" \
  "causal-scale artifact"
full_sha_is "${SCALE_ARTIFACT}.sha256" "${EXPECTED_ARTIFACT_SIDECAR_SHA}" \
  "causal-scale artifact sidecar"
pinned_python_is
prefix_identity_is "${BASE_SIF}" "${EXPECTED_SIF_BYTES}" \
  "${EXPECTED_SIF_HEAD_SHA}" "base SIF"
prefix_identity_is "${OVERLAY}" "${EXPECTED_OVERLAY_BYTES}" \
  "${EXPECTED_OVERLAY_HEAD_SHA}" "dataset overlay"
full_sha_is "${WEIGHTS}" "${EXPECTED_WEIGHTS_SHA}" "LingBot weights"
[[ -d "${RAW_SCENE}" ]] || fail "audited YmJ raw scene is absent"

lingbot_commit="$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)"
[[ "${lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" ]] || \
  fail "LingBot commit changed: ${lingbot_commit}"
if [[ -n "$(git -c safe.directory="${LINGBOT_REPO}" \
    -C "${LINGBOT_REPO}" status --porcelain --untracked-files=no)" ]]; then
  fail "LingBot checkout has changed tracked files"
fi

exec 9>"${RUN_ROOT}/.causal_ground_scale_acceptance.lock"
flock -n 9 || fail "another causal-scale acceptance stage is active"

CONTAINER=(singularity exec
  --overlay "${OVERLAY}:ro"
  -B /scratch/lg154
  -B /scratch/yz11502
  -B "${RAW_SCENE}:${EPISODE_ROOT}/YmJkqBEsHnH:ro"
  "${BASE_SIF}")
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SINGULARITYENV_PYTHONPATH="${PYTHONPATH}"

"${CONTAINER[@]}" "${MEMNAV_PY}" - \
  "${MANIFEST}" "${SCALE_ARTIFACT}" <<'PY'
import importlib.util
import json
from pathlib import Path
import sys

missing = [name for name in ("numpy",) if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing causal-scale acceptance dependencies: {missing}")
from MemNavData.audit_causal_ground_scale_artifact import AuditContract
from MemNavData.build_novel_frontier_candidates import (
    _validate_versioned_cache_pair,
)
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
artifact = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected = {"scene_count": 50, "episode_count": 100, "sample_count": 600}
for key, value in expected.items():
    if manifest.get("summary", {}).get(key) != value:
        raise SystemExit(f"manifest {key} changed")
summary = artifact.get("summary", {})
if (summary.get("episode_count"), summary.get("valid_episode_count"),
        summary.get("future_frames_consumed")) != (100, 100, 0):
    raise SystemExit("causal-scale artifact summary changed")
print(json.dumps({"causal_scale_acceptance_preflight": "passed", **expected},
                 sort_keys=True))
PY

stdout_tmp="${RUN_ROOT}/.causal_scale_acceptance_stdout.${SLURM_JOB_ID:-manual}.tmp"
rm -f "${stdout_tmp}"
trap 'rm -f "${stdout_tmp}"' EXIT
"${CONTAINER[@]}" "${MEMNAV_PY}" "${REPO_ROOT}/${AUDITOR_REL}" \
  --manifest "${MANIFEST}" \
  --artifact "${SCALE_ARTIFACT}" \
  --output "${OUTPUT}" \
  --repository-root "${REPO_ROOT}" \
  --auditor-path "${REPO_ROOT}/${AUDITOR_REL}" \
  --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}" \
  --expected-artifact-sha256 "${EXPECTED_ARTIFACT_SHA}" \
  --expected-producer-sha256 "${EXPECTED_PRODUCER_BUNDLE_SHA}" \
  --expected-configuration-sha256 "${EXPECTED_CONFIGURATION_SHA}" \
  --expected-lingbot-commit "${EXPECTED_LINGBOT_COMMIT}" \
  --expected-weights-sha256 "${EXPECTED_WEIGHTS_SHA}" \
  --expected-stream-source-sha256 "${EXPECTED_STREAM_SHA}" \
  --expected-auditor-sha256 "${EXPECTED_AUDITOR_SHA}" \
  --expected-acceptance-commit "${EXPECTED_COMMIT}" \
  --expected-scene-count 50 \
  --expected-episode-count 100 \
  --expected-sample-count 600 \
  | tee "${stdout_tmp}"
mv -f "${stdout_tmp}" "${OUTPUT_ROOT}/auditor_stdout.json"
trap - EXIT

"${CONTAINER[@]}" "${MEMNAV_PY}" - "${OUTPUT}" \
  "${EXPECTED_MANIFEST_SHA}" "${EXPECTED_ARTIFACT_SHA}" \
  "${EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
raw = path.read_bytes()
record = json.loads(raw)
canonical = (json.dumps(record, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False) + "\n").encode()
if raw != canonical:
    raise SystemExit("acceptance receipt is not canonical JSON")
digest = hashlib.sha256(raw).hexdigest()
if Path(f"{path}.sha256").read_text(encoding="ascii") != (
        f"{digest}  {path.name}\n"):
    raise SystemExit("acceptance receipt sidecar changed")
if record.get("schema_version") != "nlsr_causal_ground_scale_acceptance_v1":
    raise SystemExit("acceptance receipt schema changed")
if record.get("status") != "causal_prefixes_physically_rebound":
    raise SystemExit("causal-scale artifact was not accepted")
if record.get("acceptance_commit") != sys.argv[4]:
    raise SystemExit("acceptance commit binding changed")
inputs = record.get("inputs", {})
if (inputs.get("manifest_sha256"), inputs.get("scale_artifact_sha256")) != (
        sys.argv[2], sys.argv[3]):
    raise SystemExit("acceptance input binding changed")
coverage = record.get("coverage", {})
expected = {"scene_count": 50, "episode_count": 100, "sample_count": 600,
            "future_frames_consumed": 0,
            "all_episode_estimates_valid": True}
if coverage != expected:
    raise SystemExit(f"acceptance coverage changed: {coverage}")
print(json.dumps({"causal_scale_acceptance": "passed", "sha256": digest,
                  **coverage}, sort_keys=True))
PY

echo "NLSR causal-scale acceptance passed: ${OUTPUT}"
