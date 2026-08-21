#!/usr/bin/env bash
# Full train-only RANSAC-expert qualification audit.  The wall time is spent
# computing exact SIFT/essential evidence for every causal candidate; no GPU or
# development/blind sample is used.
#SBATCH --job-name=rev_geo_xpt
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=cs
#SBATCH --qos=cpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err

set -euo pipefail
umask 0022

: "${REV_GEO_SOURCE_ROOT:?export the immutable staged source root}"
: "${EXPECTED_LAUNCHER_SHA:?export the spooled launcher SHA256}"
: "${RUN_TAG:?export a unique alphanumeric run tag}"

[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG=${RUN_TAG}" >&2; exit 2; }

BUILDER="${REV_GEO_SOURCE_ROOT}/MemNavData/build_revisit_geometry_expert_table.py"
ANALYZER="${REV_GEO_SOURCE_ROOT}/MemNavData/analyze_revisit_geometry_expert.py"
TEST="${REV_GEO_SOURCE_ROOT}/MemNavData/test_revisit_geometry_expert.py"
PROTOCOL="${REV_GEO_SOURCE_ROOT}/MemNavData/REVISIT_GEOMETRY_EXPERT_PROTOCOL_20260811.md"

# These identities are patched only after tests freeze the source bundle.
EXPECTED_BUILDER_SHA="6825b5c72cc39641bc70422dd9789ae2429c3e225b91e75a0a012ac3d02a8322"
EXPECTED_ANALYZER_SHA="0918d3b6aa42bbbe6d97084651a1621cbff4620eb0a56dcaef741c90939a6404"
EXPECTED_TEST_SHA="9226930fc07feb81b0c5b5dd30ff855925c50fd19af1d271cce422d764685324"
EXPECTED_PROTOCOL_SHA="0942a39e978b637bc447afe712cb491d8b37b2308a0d11cff82e77e309ea16e1"

UPSTREAM_ROOT="/scratch/yz11502/Research/Nav-axis-uturn-results/nlsr_gapfill_20260807/nlsr_routes_41c1773_20260807_v1"
TEACHER="${UPSTREAM_ROOT}/causal_covisibility_teacher_a55ae5d40ba3_bc6bf58536f6/teacher_bundle/teacher.csv"
EXPECTED_TEACHER_SHA="fd52dcfcb7e8d1bdac0703f7ba0b6d10f04067d4b8793386adf65ede6bc8e313"
BASE_SIF="/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif"
OVERLAY="/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf"
NAVDP_PY="/scratch/lg154/conda-envs/navdp/bin/python"
FALLBACK_OLD="/mp3d_revisit_v0/vln_n1/traj_data/mp3d_3leg"
FALLBACK_NEW="${UPSTREAM_ROOT}/raw/mp3d_3leg"
RESULT_BASE="/scratch/yz11502/Research/Nav-axis-uturn-results/revisit_geometry_expert_20260811"
RUN_ROOT="${RESULT_BASE}/${RUN_TAG}"

EXPECTED_OVERLAY_BYTES="128854888448"
EXPECTED_OVERLAY_HEAD_SHA="43d803c2b39735e764f0c6df8f7c40a7035a1c23789e22fcc9831d29928c801b"

fail() { echo "ABORT: $*" >&2; exit 2; }
for command_name in singularity sha256sum; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
for required_path in "$0" "${BUILDER}" "${ANALYZER}" "${TEST}" \
    "${PROTOCOL}" "${TEACHER}" "${BASE_SIF}" "${OVERLAY}"; do
  [[ -f "${required_path}" && ! -L "${required_path}" ]] || \
    fail "missing or symbolic-link input: ${required_path}"
done

[[ "$(sha256sum "$0" | awk '{print $1}')" == "${EXPECTED_LAUNCHER_SHA}" ]] || \
  fail "spooled launcher SHA mismatch"
[[ "$(sha256sum "${BUILDER}" | awk '{print $1}')" == "${EXPECTED_BUILDER_SHA}" ]] || \
  fail "builder SHA mismatch"
[[ "$(sha256sum "${ANALYZER}" | awk '{print $1}')" == "${EXPECTED_ANALYZER_SHA}" ]] || \
  fail "analyzer SHA mismatch"
[[ "$(sha256sum "${TEST}" | awk '{print $1}')" == "${EXPECTED_TEST_SHA}" ]] || \
  fail "test SHA mismatch"
[[ "$(sha256sum "${PROTOCOL}" | awk '{print $1}')" == "${EXPECTED_PROTOCOL_SHA}" ]] || \
  fail "protocol SHA mismatch"
[[ "$(sha256sum "${TEACHER}" | awk '{print $1}')" == "${EXPECTED_TEACHER_SHA}" ]] || \
  fail "causal teacher SHA mismatch"
[[ "$(stat -c '%s' "${OVERLAY}")" == "${EXPECTED_OVERLAY_BYTES}" ]] || \
  fail "overlay byte identity changed"
[[ "$(dd if="${OVERLAY}" bs=1048576 count=1 status=none | sha256sum | awk '{print $1}')" == \
  "${EXPECTED_OVERLAY_HEAD_SHA}" ]] || fail "overlay header SHA changed"

mkdir -p "${RUN_ROOT}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
export PYTHONPATH="${REV_GEO_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

CONTAINER=(singularity exec --overlay "${OVERLAY}:ro"
  -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}")

echo "host=$(hostname) job=${SLURM_JOB_ID:-manual} run_root=${RUN_ROOT}"
echo "scope=train_only scenes=40 sessions=480 candidates=14172 ransac_repeats=5"

"${CONTAINER[@]}" "${NAVDP_PY}" -m py_compile "${BUILDER}" "${ANALYZER}" "${TEST}"
"${CONTAINER[@]}" "${NAVDP_PY}" -m unittest -v MemNavData.test_revisit_geometry_expert

if [[ ! -f "${RUN_ROOT}/evidence/report.json" ]]; then
  "${CONTAINER[@]}" "${NAVDP_PY}" -u "${BUILDER}" \
    --teacher-csv "${TEACHER}" \
    --expected-teacher-sha256 "${EXPECTED_TEACHER_SHA}" \
    --output-dir "${RUN_ROOT}/evidence" \
    --split-role train \
    --fallback-map "${FALLBACK_OLD}=${FALLBACK_NEW}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --ransac-repeats 5 \
    --min-matches 20 \
    --min-inliers 12 \
    --min-inlier-ratio 0.50 \
    | tee "${RUN_ROOT}/evidence_stdout.jsonl"
else
  echo "evidence report exists; verifying and continuing"
fi

EVIDENCE="${RUN_ROOT}/evidence/geometry_evidence.csv"
[[ -f "${EVIDENCE}" ]] || fail "evidence CSV absent after builder"
EVIDENCE_SHA="$(sha256sum "${EVIDENCE}" | awk '{print $1}')"

if [[ ! -f "${RUN_ROOT}/analysis/report.json" ]]; then
  "${CONTAINER[@]}" "${NAVDP_PY}" -u "${ANALYZER}" \
    --evidence-csv "${EVIDENCE}" \
    --expected-evidence-sha256 "${EVIDENCE_SHA}" \
    --output-dir "${RUN_ROOT}/analysis" \
    --folds 5 \
    --seed 20260811 \
    --bootstrap-samples 5000 \
    | tee "${RUN_ROOT}/analysis_stdout.jsonl"
else
  echo "analysis report already exists"
fi

sha256sum \
  "${RUN_ROOT}/evidence/geometry_evidence.csv" \
  "${RUN_ROOT}/evidence/report.json" \
  "${RUN_ROOT}/analysis/scene_oof_scores.csv" \
  "${RUN_ROOT}/analysis/report.json" \
  > "${RUN_ROOT}/FINAL_SHA256SUMS"
echo "COMPLETE ${RUN_ROOT}"
