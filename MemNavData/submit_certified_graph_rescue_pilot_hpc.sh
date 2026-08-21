#!/usr/bin/env bash
# Build an immutable bundle and submit the six-episode three-arm mechanism pilot.
set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/certified_stagnation_graph_pilot_20260813}
RUN_TAG=${RUN_TAG:-cgraph_pilot_$(date -u +%Y%m%dT%H%M%SZ)}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
FROZEN_RUN_ROOT=${FROZEN_RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
REMOTE_MEMNAV_PY=${REMOTE_MEMNAV_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
ARRAY_CONCURRENCY=${ARRAY_CONCURRENCY:-3}
DRY_RUN=${DRY_RUN:-0}
SUMMARY_ONLY_AFTER_JOB_ID=${SUMMARY_ONLY_AFTER_JOB_ID:-}
SUMMARY_REPAIR_REASON=${SUMMARY_REPAIR_REASON:-exclude non-causal wall-clock *_ms fields from exact plan-prefix equality}
SUMMARY_ONLY_NO_DEPENDENCY=${SUMMARY_ONLY_NO_DEPENDENCY:-0}
RETRY_INDICES=${RETRY_INDICES:-}
RETRY_ARCHIVE_RECEIPT=${RETRY_ARCHIVE_RECEIPT:-}
RETRY_ARCHIVE_RECEIPT_SHA=${RETRY_ARCHIVE_RECEIPT_SHA:-}
RETRY_SUPERSEDED_JOB_ID=${RETRY_SUPERSEDED_JOB_ID:-}
EXPANSION_INDICES=${EXPANSION_INDICES:-}
PILOT_REPORT=${PILOT_REPORT:-}
EXPECTED_PILOT_REPORT_SHA=${EXPECTED_PILOT_REPORT_SHA:-}
EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID=${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID:-}
EXPANSION_SUMMARY_REPAIR_REASON=${EXPANSION_SUMMARY_REPAIR_REASON:-}
SUPERSEDED_EXPANSION_SUMMARY_ID=${SUPERSEDED_EXPANSION_SUMMARY_ID:-}
EXPANSION_SUMMARY_REPAIR_RECEIPT_BASENAME=${EXPANSION_SUMMARY_REPAIR_RECEIPT_BASENAME:-expansion_summary_repair_submission.json}
EXPANSION_SUMMARY_NO_DEPENDENCY=${EXPANSION_SUMMARY_NO_DEPENDENCY:-0}

remote() { ssh -o BatchMode=yes "${REMOTE_HOST}" "$@"; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "ABORT: invalid RUN_TAG" >&2; exit 2; }
[[ "${ARRAY_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: invalid ARRAY_CONCURRENCY" >&2; exit 2; }
if [[ -n "${SUMMARY_ONLY_AFTER_JOB_ID}" \
      && ! "${SUMMARY_ONLY_AFTER_JOB_ID}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ABORT: invalid SUMMARY_ONLY_AFTER_JOB_ID" >&2; exit 2
fi
[[ "${SUMMARY_ONLY_NO_DEPENDENCY}" =~ ^[01]$ ]] || {
  echo "ABORT: SUMMARY_ONLY_NO_DEPENDENCY must be 0 or 1" >&2; exit 2; }
if [[ "${SUMMARY_ONLY_NO_DEPENDENCY}" == 1 \
      && -z "${SUMMARY_ONLY_AFTER_JOB_ID}" ]]; then
  echo "ABORT: dependency bypass is only valid in summary-only mode" >&2
  exit 2
fi
if [[ -n "${RETRY_INDICES}" ]]; then
  [[ -z "${SUMMARY_ONLY_AFTER_JOB_ID}" && -z "${EXPANSION_INDICES}" ]] || {
    echo "ABORT: retry, expansion, and summary-only modes are exclusive" >&2
    exit 2
  }
  [[ "${RETRY_INDICES}" =~ ^(2|7|14)(,(2|7|14))*$ ]] || {
    echo "ABORT: retries are restricted to known-failure indices 2,7,14" >&2
    exit 2
  }
  [[ -n "${RETRY_ARCHIVE_RECEIPT}" \
        && "${RETRY_ARCHIVE_RECEIPT_SHA}" =~ ^[0-9a-f]{64}$ \
        && "${RETRY_SUPERSEDED_JOB_ID}" =~ ^[1-9][0-9]*$ ]] || {
    echo "ABORT: retry requires archive receipt, SHA, and superseded job id" >&2
    exit 2
  }
fi
if [[ -n "${EXPANSION_INDICES}" ]]; then
  [[ -z "${SUMMARY_ONLY_AFTER_JOB_ID}" && -z "${RETRY_INDICES}" ]] || {
    echo "ABORT: expansion, retry, and summary-only modes are exclusive" >&2
    exit 2
  }
  [[ "${EXPANSION_INDICES}" == \
    "4,5,6,8,9,10,11,12,13,15,16,17,18,19" ]] || {
    echo "ABORT: expansion must contain the frozen 14 unselected indices" >&2
    exit 2
  }
  [[ -n "${PILOT_REPORT}" \
        && "${EXPECTED_PILOT_REPORT_SHA}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ABORT: expansion requires the pinned authorizing pilot report" >&2
    exit 2
  }
fi
if [[ -n "${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID}" ]]; then
  [[ "${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID}" =~ ^[1-9][0-9]*$ \
        && "${SUPERSEDED_EXPANSION_SUMMARY_ID}" =~ ^[1-9][0-9]*$ \
        && -n "${EXPANSION_SUMMARY_REPAIR_REASON}" \
        && -n "${PILOT_REPORT}" \
        && "${EXPECTED_PILOT_REPORT_SHA}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ABORT: incomplete expansion summary repair contract" >&2
    exit 2
  }
  [[ -z "${SUMMARY_ONLY_AFTER_JOB_ID}" && -z "${RETRY_INDICES}" \
        && -z "${EXPANSION_INDICES}" ]] || {
    echo "ABORT: expansion summary repair is mutually exclusive" >&2
    exit 2
  }
  [[ "${EXPANSION_SUMMARY_REPAIR_RECEIPT_BASENAME}" =~ \
    ^[A-Za-z0-9][A-Za-z0-9_.-]*[.]json$ ]] || {
    echo "ABORT: invalid expansion summary repair receipt basename" >&2
    exit 2
  }
  [[ "${EXPANSION_SUMMARY_NO_DEPENDENCY}" =~ ^[01]$ ]] || {
    echo "ABORT: EXPANSION_SUMMARY_NO_DEPENDENCY must be 0 or 1" >&2
    exit 2
  }
fi

required=(
  MemNavData/run_certified_graph_rescue_pilot_episode.sh
  MemNavData/slurm_certified_graph_rescue_pilot.sbatch
  MemNavData/slurm_certified_graph_rescue_pilot_summary.sbatch
  MemNavData/slurm_certified_graph_rescue_expansion_summary.sbatch
  MemNavData/summarize_certified_graph_rescue_pilot.py
  MemNavData/summarize_certified_graph_rescue_expansion.py
  MemNavData/CERTIFIED_STAGNATION_GRAPH_PILOT_20260813.md
  MemNavData/CERTIFIED_STAGNATION_GRAPH_EXPANSION_PROTOCOL_20260813.md
  MemNavData/eval_shared_online_double_revisit.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/shared_online_double_revisit_runtime.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/policy_agent.py
)
for relative in "${required[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || {
    echo "ABORT: missing physical input ${relative}" >&2; exit 2; }
done

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/shared_online_double_revisit_runtime.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_certified_graph_rescue_pilot.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_certified_graph_rescue_expansion.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/memnav_server.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py"
PYTHONPATH="${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData" "${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${LOCAL_ROOT}/MemNavData/eval_shared_online_double_revisit.py"
"${MEMNAV_PY}" -m unittest \
  MemNavData.test_policy_agent_graph \
  MemNavData.test_shared_online_double_revisit_runtime \
  MemNavData.test_summarize_certified_graph_rescue_pilot \
  MemNavData.test_summarize_certified_graph_rescue_expansion
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_certified_graph_rescue_pilot_episode.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_certified_graph_rescue_pilot.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_certified_graph_rescue_pilot_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_certified_graph_rescue_expansion_summary.sbatch"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/memnav" "${STAGING}/NavDP/baselines/navdp"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for relative in \
  MemNavData/run_certified_graph_rescue_pilot_episode.sh \
  MemNavData/slurm_certified_graph_rescue_pilot.sbatch \
  MemNavData/slurm_certified_graph_rescue_pilot_summary.sbatch \
  MemNavData/slurm_certified_graph_rescue_expansion_summary.sbatch \
  MemNavData/CERTIFIED_STAGNATION_GRAPH_PILOT_20260813.md \
  MemNavData/CERTIFIED_STAGNATION_GRAPH_EXPANSION_PROTOCOL_20260813.md; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${STAGING}/${relative}"
done
for component in memnav navdp; do
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${STAGING}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" \
    -maxdepth 1 -type f -name '*.py' -print0)
done

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" "${FROZEN_RUN_ROOT}" \
  "${BASE_SOURCE_ROOT}" "${BASE_SOURCE_RECEIPT_SHA}" \
  "${EXPANSION_INDICES}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
payload={
 "schema_version":"certified_stagnation_graph_bundle_v3_full20_capable",
 "local_git_head_context":sys.argv[2],"frozen_run_root":sys.argv[3],
 "base_source_root":sys.argv[4],"base_source_receipt_sha256":sys.argv[5],
 "scope":"post-hoc failure mechanism pilot; not an SR estimate",
 "pilot_indices":[0,1,2,3,7,14],"known_failure_indices":[2,7,14],
 "control_indices":[0,1,3],
 "arms":["direct","budget_control","rescue"],"files":files,
}
if sys.argv[6]:
    payload["authorized_expansion_indices"]=[
        int(value) for value in sys.argv[6].split(",")]
(root/"source_bundle_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
(
  cd "${STAGING}"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum > SOURCE_BUNDLE.sha256
  sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null
)
SOURCE_RECEIPT_SHA=$(sha256sum "${STAGING}/SOURCE_BUNDLE.sha256" | awk '{print $1}')
BUNDLE_MANIFEST_SHA=$(sha256sum "${STAGING}/source_bundle_manifest.json" | awk '{print $1}')
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/certified_stagnation_graph_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  exit 0
fi

remote "test -f '${FROZEN_RUN_ROOT}/prepared/SEALED' && test -f '${FROZEN_RUN_ROOT}/report.json'"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
if remote "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified task bundle ${REMOTE_BUNDLE}"
else
  remote "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r "${STAGING}/" \
    "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi

if [[ -n "${SUMMARY_ONLY_AFTER_JOB_ID}" || -n "${RETRY_INDICES}" \
      || -n "${EXPANSION_INDICES}" \
      || -n "${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID}" ]]; then
  remote "test -d '${RUN_ROOT}'"
else
  remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}/logs'"
fi
SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},FROZEN_RUN_ROOT=${FROZEN_RUN_ROOT},RUN_ROOT=${RUN_ROOT}"
EVAL_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_certified_graph_rescue_pilot.sbatch
SUMMARY_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_certified_graph_rescue_pilot_summary.sbatch
EXPANSION_SUMMARY_SBATCH=${REMOTE_BUNDLE}/MemNavData/slurm_certified_graph_rescue_expansion_summary.sbatch
if [[ -n "${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID}" ]]; then
  repair_receipt=${RUN_ROOT}/${EXPANSION_SUMMARY_REPAIR_RECEIPT_BASENAME}
  remote "test -f '${RUN_ROOT}/expansion_submission.json' && test -f '${PILOT_REPORT}' && test ! -e '${repair_receipt}' && test ! -e '${RUN_ROOT}/full20_report.json'"
  remote "test \"\$(sha256sum '${PILOT_REPORT}' | awk '{print \$1}')\" = '${EXPECTED_PILOT_REPORT_SHA}'"
  expansion_exports="${exports},EXPERIMENT_KIND=fresh20_expansion,PILOT_REPORT=${PILOT_REPORT},EXPECTED_PILOT_REPORT_SHA=${EXPECTED_PILOT_REPORT_SHA}"
  dependency_mode=afterok
  if [[ "${EXPANSION_SUMMARY_NO_DEPENDENCY}" == 1 ]]; then
    remote "states=\$(sacct -X -j '${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID}' --noheader --parsable2 --format=State | sed '/^\$/d'); test \"\$(printf '%s\\n' \"\${states}\" | wc -l)\" -eq 15; test -z \"\$(printf '%s\\n' \"\${states}\" | grep -Ev '^COMPLETED[|]?$' || true)\""
    dependency_mode=verified_all_array_children_completed_no_dependency
    remote "sbatch --test-only --export='${expansion_exports}' '${EXPANSION_SUMMARY_SBATCH}' >/dev/null"
    summary_raw=$(remote "sbatch --parsable --export='${expansion_exports}' '${EXPANSION_SUMMARY_SBATCH}'")
  else
    remote "sbatch --test-only --dependency=afterok:${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID} --kill-on-invalid-dep=yes --export='${expansion_exports}' '${EXPANSION_SUMMARY_SBATCH}' >/dev/null"
    summary_raw=$(remote "sbatch --parsable --dependency=afterok:${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID} --kill-on-invalid-dep=yes --export='${expansion_exports}' '${EXPANSION_SUMMARY_SBATCH}'")
  fi
  summary_id=${summary_raw%%;*}
  [[ "${summary_id}" =~ ^[0-9]+$ ]] || {
    echo "ABORT: bad expansion repair summary id" >&2; exit 2; }
  remote "'${REMOTE_MEMNAV_PY}' - '${repair_receipt}' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${EXPANSION_SUMMARY_ONLY_AFTER_JOB_ID}' '${SUPERSEDED_EXPANSION_SUMMARY_ID}' '${summary_id}' '${EXPANSION_SUMMARY_REPAIR_REASON}' '${dependency_mode}'" <<'PY'
import json,sys
path,bundle,source_sha,evaluation,superseded,summary,reason,dependency_mode=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"certified_stagnation_graph_expansion_summary_repair_v1",
      "reason":reason,"navigation_outcomes_reused":True,
      "source_bundle":bundle,"source_receipt_sha256":source_sha,
      "evaluation_array":int(evaluation),
      "superseded_summary":int(superseded),
      "replacement_summary":int(summary),
      "dependency_mode":dependency_mode,
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
  echo "expansion_repair_summary=${summary_id}"
  exit 0
fi
if [[ -n "${RETRY_INDICES}" ]]; then
  remote "test -f '${RUN_ROOT}/submission.json' && test ! -e '${RUN_ROOT}/report.json' && test ! -e '${RUN_ROOT}/retry_submission_stream_restore.json'"
  remote "test -f '${RETRY_ARCHIVE_RECEIPT}' && test \"\$(sha256sum '${RETRY_ARCHIVE_RECEIPT}' | awk '{print \$1}')\" = '${RETRY_ARCHIVE_RECEIPT_SHA}'"
  IFS=, read -r -a retry_index_array <<<"${RETRY_INDICES}"
  for retry_index in "${retry_index_array[@]}"; do
    retry_label=$(printf '%03d' "${retry_index}")
    remote "test -z \"\$(find '${RUN_ROOT}/scenes' -mindepth 1 -maxdepth 1 -type d -name '${retry_label}_*' -print -quit)\""
  done
  remote "sbatch --test-only --array='${RETRY_INDICES}' --export='${exports}' '${EVAL_SBATCH}' >/dev/null"
  eval_raw=$(remote "sbatch --parsable --array='${RETRY_INDICES}%${ARRAY_CONCURRENCY}' --export='${exports}' '${EVAL_SBATCH}'")
  eval_id=${eval_raw%%;*}
  [[ "${eval_id}" =~ ^[0-9]+$ ]] || {
    echo "ABORT: bad retry eval id" >&2; exit 2; }
  remote "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}' >/dev/null"
  summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}'")
  summary_id=${summary_raw%%;*}
  [[ "${summary_id}" =~ ^[0-9]+$ ]] || {
    echo "ABORT: bad retry summary id" >&2; exit 2; }
  remote "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/retry_submission_stream_restore.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${RETRY_INDICES}' '${RETRY_ARCHIVE_RECEIPT}' '${RETRY_ARCHIVE_RECEIPT_SHA}' '${RETRY_SUPERSEDED_JOB_ID}' '${eval_id}' '${summary_id}'" <<'PY'
import json,sys
(path,bundle,source_sha,indices,archive,archive_sha,superseded,
 evaluation,summary)=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"certified_stagnation_graph_pilot_retry_v1",
      "reason":"preserve live NavDP KV stream across late metric-scale estimation",
      "navigation_outcomes_reused":False,
      "retry_indices":[int(value) for value in indices.split(",")],
      "source_bundle":bundle,"source_receipt_sha256":source_sha,
      "archive_receipt":archive,"archive_receipt_sha256":archive_sha,
      "superseded_evaluation_array":int(superseded),
      "jobs":{"evaluation_array":int(evaluation),"summary":int(summary)},
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
  echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "retry_evaluation=${eval_id} summary=${summary_id}"
  exit 0
fi
if [[ -n "${EXPANSION_INDICES}" ]]; then
  remote "test -f '${RUN_ROOT}/submission.json' && test -f '${PILOT_REPORT}' && test ! -e '${RUN_ROOT}/full20_report.json' && test ! -e '${RUN_ROOT}/expansion_submission.json'"
  remote "test \"\$(sha256sum '${PILOT_REPORT}' | awk '{print \$1}')\" = '${EXPECTED_PILOT_REPORT_SHA}'"
  remote "'${REMOTE_MEMNAV_PY}' - '${PILOT_REPORT}'" <<'PY'
import json,sys
report=json.load(open(sys.argv[1]))
if report.get("gate",{}).get("decision") != "expand_to_unselected_fresh20":
    raise SystemExit("pilot report did not authorize expansion")
PY
  remote "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/scenes' '${EXPANSION_INDICES}'" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
for raw in sys.argv[2].split(","):
    matches=list(root.glob(f"{int(raw):03d}_*"))
    if matches:
        raise SystemExit(f"expansion output already exists: {matches[0]}")
PY
  expansion_exports="${exports},EXPERIMENT_KIND=fresh20_expansion,PILOT_REPORT=${PILOT_REPORT},EXPECTED_PILOT_REPORT_SHA=${EXPECTED_PILOT_REPORT_SHA}"
  remote "sbatch --test-only --job-name=cgraph20 --array='${EXPANSION_INDICES}' --export='${expansion_exports}' '${EVAL_SBATCH}' >/dev/null"
  eval_raw=$(remote "sbatch --parsable --job-name=cgraph20 --array='${EXPANSION_INDICES}%${ARRAY_CONCURRENCY}' --export='${expansion_exports}' '${EVAL_SBATCH}'")
  eval_id=${eval_raw%%;*}
  [[ "${eval_id}" =~ ^[0-9]+$ ]] || {
    echo "ABORT: bad expansion eval id" >&2; exit 2; }
  remote "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${expansion_exports}' '${EXPANSION_SUMMARY_SBATCH}' >/dev/null"
  summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${expansion_exports}' '${EXPANSION_SUMMARY_SBATCH}'")
  summary_id=${summary_raw%%;*}
  [[ "${summary_id}" =~ ^[0-9]+$ ]] || {
    echo "ABORT: bad expansion summary id" >&2; exit 2; }
  remote "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/expansion_submission.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${EXPANSION_INDICES}' '${PILOT_REPORT}' '${EXPECTED_PILOT_REPORT_SHA}' '${eval_id}' '${summary_id}'" <<'PY'
import json,sys
(path,bundle,source_sha,indices,pilot,pilot_sha,evaluation,summary)=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"certified_stagnation_graph_fresh20_expansion_submission_v1",
      "scope":"internal full-fresh20 expansion; not paper confirmation",
      "expansion_indices":[int(value) for value in indices.split(",")],
      "source_bundle":bundle,"source_receipt_sha256":source_sha,
      "authorizing_pilot_report":pilot,
      "authorizing_pilot_report_sha256":pilot_sha,
      "jobs":{"evaluation_array":int(evaluation),"summary":int(summary)},
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
  echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  echo "expansion_evaluation=${eval_id} summary=${summary_id}"
  exit 0
fi
if [[ -n "${SUMMARY_ONLY_AFTER_JOB_ID}" ]]; then
  repair_receipt=${RUN_ROOT}/summary_repair_submission.json
  remote "test -f '${RUN_ROOT}/submission.json' && test ! -e '${repair_receipt}' && test ! -e '${RUN_ROOT}/report.json'"
  dependency_mode=afterok
  if [[ "${SUMMARY_ONLY_NO_DEPENDENCY}" == 1 ]]; then
    remote "states=\$(sacct -X -j '${SUMMARY_ONLY_AFTER_JOB_ID}' --noheader --parsable2 --format=State | sed '/^\$/d'); test -n \"\${states}\"; test -z \"\$(printf '%s\\n' \"\${states}\" | grep -Ev '^COMPLETED[|]?$' || true)\""
    dependency_mode=verified_completed_no_dependency
    remote "sbatch --test-only --export='${exports}' '${SUMMARY_SBATCH}' >/dev/null"
    summary_raw=$(remote "sbatch --parsable --export='${exports}' '${SUMMARY_SBATCH}'")
  else
    remote "sbatch --test-only --dependency=afterok:${SUMMARY_ONLY_AFTER_JOB_ID} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}' >/dev/null"
    summary_raw=$(remote "sbatch --parsable --dependency=afterok:${SUMMARY_ONLY_AFTER_JOB_ID} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}'")
  fi
  summary_id=${summary_raw%%;*}
  [[ "${summary_id}" =~ ^[0-9]+$ ]] || {
    echo "ABORT: bad repair summary id" >&2; exit 2; }
  remote "'${REMOTE_MEMNAV_PY}' - '${repair_receipt}' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${SUMMARY_ONLY_AFTER_JOB_ID}' '${summary_id}' '${SUMMARY_REPAIR_REASON}' '${dependency_mode}'" <<'PY'
import json,sys
path,bundle,receipt,evaluation,summary,reason,dependency_mode=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"certified_stagnation_graph_pilot_summary_repair_v1",
      "reason":reason,
      "navigation_outcomes_reused":True,"source_bundle":bundle,
      "source_receipt_sha256":receipt,"evaluation_array":int(evaluation),
      "replacement_summary":int(summary),"dependency_mode":dependency_mode,
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
  echo "RUN_ROOT=${RUN_ROOT}"
  echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
  echo "repair_summary=${summary_id}"
  exit 0
fi
remote "sbatch --test-only --array=0 --export='${exports}' '${EVAL_SBATCH}' >/dev/null"
eval_raw=$(remote "sbatch --parsable --array=0-3,7,14%${ARRAY_CONCURRENCY} --export='${exports}' '${EVAL_SBATCH}'")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad eval id" >&2; exit 2; }
remote "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}' >/dev/null"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY_SBATCH}'")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || { echo "ABORT: bad summary id" >&2; exit 2; }

remote "'${REMOTE_MEMNAV_PY}' - '${RUN_ROOT}/submission.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${eval_id}' '${summary_id}'" <<'PY'
import json,sys
path,bundle,receipt,evaluation,summary=sys.argv[1:]
with open(path,"x",encoding="utf-8") as handle:
    json.dump({
      "schema_version":"certified_stagnation_graph_pilot_submission_v2_budget_control",
      "scope":"post-hoc failure mechanism pilot; not an SR estimate",
      "source_bundle":bundle,"source_receipt_sha256":receipt,
      "pilot_indices":[0,1,2,3,7,14],
      "jobs":{"evaluation_array":int(evaluation),"summary":int(summary)},
    },handle,indent=2,sort_keys=True); handle.write("\n")
PY
echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "evaluation=${eval_id} summary=${summary_id}"
