#!/usr/bin/env bash
# Freeze and submit the pre-registered consumed-pool CDEC causal comparison.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
RESULT_BASE=${RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/cdec_consumed_closed_loop_20260813}
RUN_TAG=${RUN_TAG:-cdec_consumed_cl_$(date -u +%Y%m%dT%H%M%SZ)}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REMOTE_MEMNAV_PY=${REMOTE_MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}

BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
TRACE_RUN_ROOT=${TRACE_RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/certified_relocalization_closed_loop_20260812/certrel_bearing_v1_20260812T1050}
EXPECTED_MANIFEST_SHA=8013fa2a768d84638a9f9ecc50df46dda67ebb79250d14ad0a8087ac52fd33e5
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
EXPECTED_TRACE_REPORT_SHA=0e41a6d9b339d143229ba405b04802654d2053b5d641a03ed2d09aefc1a589f4

GATE_RUN_ROOT=${GATE_RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/cdec_dual_proposal_certificate_20260813/cdec_dual_sameprocess_nonarrayfix_20260813}
GATE_VERIFICATION=${GATE_VERIFICATION:-${GATE_RUN_ROOT}/independent_verification_v1.json}
CDEC_LOCAL=${CDEC_LOCAL:-${LOCAL_ROOT}/.diagnostics/certificate_distilled_compass_20260813/factorized_pairwise_oof_fixedbatch_v2/cdec_pairwise_runtime_unapproved_v3.json}
EXPECTED_CDEC_ARTIFACT_SHA=eea77098531c6e5865516d49540374edca7bb87a419613ef40d56cc2c85add31

remote() { ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"; }
fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid RUN_TAG"
[[ "${ARRAY_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "invalid concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ -f "${CDEC_LOCAL}" && ! -L "${CDEC_LOCAL}" ]] || fail "missing CDEC artifact"
[[ "$(sha256sum "${CDEC_LOCAL}" | awk '{print $1}')" == \
  "${EXPECTED_CDEC_ARTIFACT_SHA}" ]] || fail "CDEC artifact changed"

# The expensive closed loop has no authority until the independent raw-CSV
# verifier approves the train-only proposal gate.
if [[ "${DRY_RUN}" == 0 ]]; then
  remote "test -f '${GATE_VERIFICATION}' && test ! -w '${GATE_VERIFICATION}'" || \
    fail "independent proposal gate is not complete/immutable"
  remote "'${REMOTE_MEMNAV_PY}' - '${GATE_VERIFICATION}'" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
scope=x.get("scope") or {}
gate=((x.get("reconstructed") or {}).get("method_gate") or {})
checks=gate.get("requirements") or {}
if x.get("verified") is not True:
    raise SystemExit("independent verification failed")
if (scope.get("train40_only") is not True
        or scope.get("development_or_blind_read") is not False
        or scope.get("closed_loop") is not False):
    raise SystemExit("proposal-gate scope changed")
if gate.get("pass") is not True or not checks or not all(checks.values()):
    raise SystemExit("proposal gate did not authorize closed loop")
PY
  GATE_VERIFICATION_SHA=$(remote \
    "sha256sum '${GATE_VERIFICATION}' | awk '{print \$1}'")
  [[ "${GATE_VERIFICATION_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "bad gate SHA"
else
  # A dry run validates/stages code without pretending that a pending gate
  # authorized navigation.
  GATE_VERIFICATION_SHA=pending_independent_gate
fi

required=(
  MemNavData/CDEC_CONSUMED_CLOSED_LOOP_PROTOCOL_20260813.md
  MemNavData/prepare_cdec_consumed_closed_loop.py
  MemNavData/test_prepare_cdec_consumed_closed_loop.py
  MemNavData/run_cdec_consumed_closed_loop_scene.sh
  MemNavData/slurm_cdec_consumed_closed_loop.sbatch
  MemNavData/summarize_cdec_consumed_closed_loop.py
  MemNavData/test_summarize_cdec_consumed_closed_loop.py
  MemNavData/slurm_cdec_consumed_closed_loop_summary.sbatch
  MemNavData/independent_verify_cdec_consumed_closed_loop.py
  MemNavData/test_independent_verify_cdec_consumed_closed_loop.py
  MemNavData/slurm_independent_verify_cdec_consumed_closed_loop.sbatch
)
for relative in "${required[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || \
    fail "missing physical source ${relative}"
done

export PYTHONPATH=${LOCAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/prepare_cdec_consumed_closed_loop.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_cdec_consumed_closed_loop.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_cdec_consumed_closed_loop.py" \
  "${LOCAL_ROOT}/MemNavData/cdec_pairwise_runtime.py" \
  "${LOCAL_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py"
"${MEMNAV_PY}" -m pytest -q \
  MemNavData/test_prepare_cdec_consumed_closed_loop.py \
  MemNavData/test_summarize_cdec_consumed_closed_loop.py \
  MemNavData/test_independent_verify_cdec_consumed_closed_loop.py \
  MemNavData/test_cdec_pairwise_runtime.py \
  MemNavData/test_certified_relocalization_runtime.py \
  MemNavData/test_revisit_bearing_adapter.py \
  MemNavData/test_policy_agent_graph.py
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_cdec_consumed_closed_loop_scene.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_cdec_consumed_closed_loop.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_cdec_consumed_closed_loop_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_independent_verify_cdec_consumed_closed_loop.sbatch"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/memnav" "${STAGING}/artifacts"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for relative in \
  MemNavData/CDEC_CONSUMED_CLOSED_LOOP_PROTOCOL_20260813.md \
  MemNavData/run_cdec_consumed_closed_loop_scene.sh \
  MemNavData/slurm_cdec_consumed_closed_loop.sbatch \
  MemNavData/slurm_cdec_consumed_closed_loop_summary.sbatch \
  MemNavData/slurm_independent_verify_cdec_consumed_closed_loop.sbatch; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" \
    "${STAGING}/${relative}"
done
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/NavDP/baselines/memnav/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/NavDP/baselines/memnav" \
  -maxdepth 1 -type f -name '*.py' -print0)
cp --preserve=mode,timestamps "${CDEC_LOCAL}" \
  "${STAGING}/artifacts/cdec_pairwise_runtime_unapproved_v3.json"

# Exercise the exact staged modules.  The bundle is an overlay over the pinned
# certified base, but every module modified by CDEC is physically present here.
PYTHONPATH=${STAGING} "${MEMNAV_PY}" -m pytest -q -p no:cacheprovider \
  "${STAGING}/MemNavData/test_prepare_cdec_consumed_closed_loop.py" \
  "${STAGING}/MemNavData/test_summarize_cdec_consumed_closed_loop.py" \
  "${STAGING}/MemNavData/test_independent_verify_cdec_consumed_closed_loop.py" \
  "${STAGING}/MemNavData/test_cdec_pairwise_runtime.py" \
  "${STAGING}/MemNavData/test_certified_relocalization_runtime.py" \
  "${STAGING}/MemNavData/test_revisit_bearing_adapter.py" \
  "${STAGING}/MemNavData/test_policy_agent_graph.py"

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" \
  "${GATE_VERIFICATION_SHA}" "${TRACE_RUN_ROOT}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"cdec_consumed_closed_loop_bundle_v1_20260813",
 "local_git_head_context":sys.argv[2],
 "scope":"consumed 20-scene causal comparison; no development/blind",
 "proposal_gate_verification_sha256":sys.argv[3],
 "trace_source_run_root":sys.argv[4],
 "arms":["geometry_certificate","cdec_cascade"],
 "stagnation_graph":"off",
 "learned_authority":"rank shortlist only",
 "activation_authority":"unchanged atomic PnP certificate",
 "files":files,
}
(root/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
SOURCE_RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/cdec_consumed_closed_loop_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-${RUN_TAG}
RUN_ROOT=${RESULT_BASE}/${RUN_TAG}

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "DRY_RUN_BUNDLE_MANIFEST_SHA=${BUNDLE_MANIFEST_SHA}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_GATE_STATE=${GATE_VERIFICATION_SHA}"
  exit 0
fi

if remote "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified immutable overlay ${REMOTE_BUNDLE}"
else
  remote "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi

REMOTE_SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
BASE_SOURCE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
REMOTE_ARTIFACT=${REMOTE_BUNDLE}/artifacts/cdec_pairwise_runtime_unapproved_v3.json
remote "test \"\$(sha256sum '${BASE_SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}' && cd '${BASE_SOURCE_ROOT}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
remote "test \"\$(sha256sum '${REMOTE_ARTIFACT}' | awk '{print \$1}')\" = '${EXPECTED_CDEC_ARTIFACT_SHA}'"

SOURCE_MANIFEST=${TRACE_RUN_ROOT}/data_manifest.json
SOURCE_DEPENDENCY=${TRACE_RUN_ROOT}/dependency_receipt.json
TRACE_REPORT=${TRACE_RUN_ROOT}/report.json
remote "'${REMOTE_MEMNAV_PY}' '${REMOTE_BUNDLE}/MemNavData/prepare_cdec_consumed_closed_loop.py' \
  --source-manifest '${SOURCE_MANIFEST}' \
  --expected-manifest-sha256 '${EXPECTED_MANIFEST_SHA}' \
  --source-dependency-receipt '${SOURCE_DEPENDENCY}' \
  --expected-dependency-receipt-sha256 '${EXPECTED_DEPENDENCY_RECEIPT_SHA}' \
  --trace-run-root '${TRACE_RUN_ROOT}' \
  --trace-run-report '${TRACE_REPORT}' \
  --expected-trace-run-report-sha256 '${EXPECTED_TRACE_REPORT_SHA}' \
  --run-root '${RUN_ROOT}'"

exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},TRACE_RUN_ROOT=${TRACE_RUN_ROOT},SOURCE_RECEIPT=${REMOTE_SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},BASE_SOURCE_RECEIPT=${BASE_SOURCE_RECEIPT},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},GATE_VERIFICATION=${GATE_VERIFICATION},EXPECTED_GATE_VERIFICATION_SHA=${GATE_VERIFICATION_SHA},CDEC_ARTIFACT=${REMOTE_ARTIFACT},EXPECTED_CDEC_ARTIFACT_SHA=${EXPECTED_CDEC_ARTIFACT_SHA}"
EVAL=${REMOTE_BUNDLE}/MemNavData/slurm_cdec_consumed_closed_loop.sbatch
SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_cdec_consumed_closed_loop_summary.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_independent_verify_cdec_consumed_closed_loop.sbatch
remote "mkdir -p /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs && sbatch --test-only --array=0 --export='${exports}' '${EVAL}' >/dev/null"
eval_raw=$(remote "sbatch --parsable --array=0-19%${ARRAY_CONCURRENCY} --export='${exports}' '${EVAL}'")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[1-9][0-9]*$ ]] || fail "bad evaluation id"
remote "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}' >/dev/null"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[1-9][0-9]*$ ]] || fail "bad summary id"
remote "sbatch --test-only --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}' >/dev/null"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[1-9][0-9]*$ ]] || fail "bad verifier id"

remote "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/submission.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${GATE_VERIFICATION}' '${GATE_VERIFICATION_SHA}' '${eval_id}' '${summary_id}' '${verify_id}'" <<'PY'
import json,sys
path,bundle,source_sha,gate,gate_sha,evaluation,summary,verification=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"cdec_consumed_closed_loop_submission_v1_20260813",
      "scope":"consumed 20-scene causal comparison; not paper confirmation",
      "development_read":False,"blind_read":False,
      "source_bundle":bundle,"source_receipt_sha256":source_sha,
      "proposal_gate_verification":gate,
      "proposal_gate_verification_sha256":gate_sha,
      "jobs":{
        "evaluation_array":int(evaluation),"summary":int(summary),
        "independent_verification":int(verification),
      },
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "evaluation=${eval_id} summary=${summary_id} verification=${verify_id}"
