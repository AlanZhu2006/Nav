#!/usr/bin/env bash
# Submit the sealed MP3D construction and five-arm evaluation pipeline.

set -euo pipefail
umask 0022

LOCAL_ROOT=$(git rev-parse --show-toplevel)
REMOTE_HOST=${REMOTE_HOST:-alantorch}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-}
CONTROLLED_GATE_AUDIT=${CONTROLLED_GATE_AUDIT:?set controlled gate audit JSON}
NATURAL_GATE_AUDIT=${NATURAL_GATE_AUDIT:?set natural gate audit JSON}
CONSTRUCTION_GATE_AUDIT=${CONSTRUCTION_GATE_AUDIT:?set corrected single-Revisit integration audit JSON}
CONSTRUCTION_GATE_INCIDENT_RECEIPT=${CONSTRUCTION_GATE_INCIDENT_RECEIPT:?set corrected integration orchestration incident JSON}
RUN_TAG=${RUN_TAG:-paper_online_a_$(date -u +%Y%m%dT%H%M%SZ)}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814}
RUN_ROOT=${RUN_ROOT:-${REMOTE_RESULT_BASE}/${RUN_TAG}}
CONCURRENCY=${CONCURRENCY:-8}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
EXPECTED_CONSTRUCTION_GATE_BENCH_SHA=0d7643dcb2b8484f5f1e872144da6b9c064ca44b5948827da5413301cbd2ff51
MEMNAV_PY=${MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
HAB_PY=${HAB_PY:-/home/asus/miniconda3/envs/habitat/bin/python}
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
SSH_ARGS=(-o BatchMode=yes)
RSYNC_RSH="ssh -o BatchMode=yes"
if [[ -n "${SSH_CONTROL_PATH}" ]]; then
  [[ "${SSH_CONTROL_PATH}" =~ ^/[A-Za-z0-9._/-]+$ \
     && -S "${SSH_CONTROL_PATH}" ]] || fail "invalid SSH control socket"
  SSH_ARGS+=(-S "${SSH_CONTROL_PATH}" -o ControlMaster=no)
  RSYNC_RSH="ssh -o BatchMode=yes -S ${SSH_CONTROL_PATH} -o ControlMaster=no"
fi
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }
[[ "${RUN_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "bad RUN_TAG"
[[ "${EXPECTED_REMOTE_USER}" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || \
  fail "bad expected remote user"
[[ "${CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad concurrency"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad eval concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"

for gate in "${CONTROLLED_GATE_AUDIT}" "${NATURAL_GATE_AUDIT}"; do
  [[ -r "${gate}" ]] || fail "missing gate audit ${gate}"
  "${MEMNAV_PY}" - "${gate}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert p["passed"] is True
assert p["paper_final_unlock_authorized"] is True
assert p["integration_audit"]["runtime_role_visibility"] == "none"
assert p["integration_audit"]["runtime_failure_plans"] == 0
PY
done
[[ -r "${CONSTRUCTION_GATE_AUDIT}" ]] || \
  fail "missing construction-amendment gate audit ${CONSTRUCTION_GATE_AUDIT}"
"${MEMNAV_PY}" - "${CONSTRUCTION_GATE_AUDIT}" \
  "${EXPECTED_CONSTRUCTION_GATE_BENCH_SHA}" \
  "${LOCAL_ROOT}/MemNavData/audit_shared_online_role_pair_smoke.py" <<'PY'
import hashlib,json,sys
p=json.load(open(sys.argv[1]))
assert p["ok"] is True
assert p["scenes"] == 3
assert p["benchmark_manifest_sha256"] == sys.argv[2]
assert p["max_steps"] == 120
assert p["arms"] == ["native","raw_direct","raw_fixed_bearing","certified"]
assert p["runtime_role_visibility"] == "none"
assert p["runtime_failure_plans"] == 0
assert p["novel_certified_accept_plans"] == 0
assert p["novel_certified_exact_fallback_scenes"] == p["scenes"]
assert p["revisit_certified_accept_plans"] > 0
assert p["auditor_sha256"] == hashlib.sha256(open(sys.argv[3],"rb").read()).hexdigest()
PY
[[ -r "${CONSTRUCTION_GATE_INCIDENT_RECEIPT}" ]] || \
  fail "missing construction gate incident receipt"
"${MEMNAV_PY}" - "${CONSTRUCTION_GATE_INCIDENT_RECEIPT}" \
  "${CONSTRUCTION_GATE_AUDIT}" <<'PY'
import hashlib,json,sys
incident=json.load(open(sys.argv[1]))
audit_sha=hashlib.sha256(open(sys.argv[2],"rb").read()).hexdigest()
assert incident["status"] == (
    "complete_outputs_with_post_evaluation_orchestration_error")
assert incident["orchestrator_exit_code"] == 2
assert incident["arm_metric_file_count"] == 12
assert incident["arm_summary_file_count"] == 12
assert incident["post_run_server_processes_remaining"] == 0
assert incident["independent_audit_ok"] is True
assert incident["independent_audit_sha256"] == audit_sha
PY

required=(
  MemNavData/PAPER_EVALUATION_PROTOCOL_20260814.md
  MemNavData/PAPER_CONSTRUCTION_AMENDMENT_20260814.md
  MemNavData/PAPER_PORT_RACE_AMENDMENT_20260814.md
  MemNavData/PAPER_ATTEMPT6_PORT_RACE_INCIDENT_20260814.json
  MemNavData/SHARED_ONLINE_ROLE_PAIR_CONSUMED_GATE_20260814.md
  MemNavData/SHARED_ONLINE_ROLE_PAIR_NATURAL_GATE_20260814.md
  MemNavData/strict_graph_blind_20260806.json
  MemNavData/deterministic_eval_protocol.py
  MemNavData/eval_2leg_habitat.py
  MemNavData/eval_shared_online_role_pairs.py
  MemNavData/generate_twoleg.py
  MemNavData/build_paper_role_pair_scene.py
  MemNavData/build_single_revisit_source.py
  MemNavData/finalize_paper_role_pairs.py
  MemNavData/build_shared_online_double_revisit.py
  MemNavData/build_shared_online_role_pairs.py
  MemNavData/audit_shared_online_role_pairs.py
  MemNavData/shared_online_role_pair_contract.py
  MemNavData/materialize_online_a_traces.py
  MemNavData/materialize_paper_online_a_scene.py
  MemNavData/run_paper_online_a_scene.sh
  MemNavData/run_paper_role_pair_episode.sh
  MemNavData/retrying_server_launcher.py
  MemNavData/test_retrying_server_launcher.py
  MemNavData/slurm_paper_online_a_collect.sbatch
  MemNavData/slurm_paper_online_a_summary.sbatch
  MemNavData/slurm_paper_role_pair_eval.sbatch
  MemNavData/slurm_paper_role_pair_summary.sbatch
  MemNavData/slurm_paper_role_pair_verify.sbatch
  MemNavData/summarize_paper_online_a.py
  MemNavData/summarize_paper_role_pair_eval.py
  MemNavData/independent_verify_paper_role_pair_eval.py
  MemNavData/validate_expanded_navdp_router_eval.py
  MemNavData/validate_paper_online_a_scene.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/navdp/policy_backbone.py
  NavDP/baselines/navdp/policy_network.py
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/reverse_memory_graph.py
)
for relative in "${required[@]}"; do
  [[ -f "${LOCAL_ROOT}/${relative}" && ! -L "${LOCAL_ROOT}/${relative}" ]] || \
    fail "missing physical input ${relative}"
done
[[ "$(sha256sum "${LOCAL_ROOT}/MemNavData/strict_graph_blind_20260806.json" | awk '{print $1}')" == \
  b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9 ]] || \
  fail "frozen paper source manifest changed"

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${HAB_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/eval_2leg_habitat.py" \
  "${LOCAL_ROOT}/MemNavData/validate_paper_online_a_scene.py" \
  "${LOCAL_ROOT}/MemNavData/materialize_online_a_traces.py" \
  "${LOCAL_ROOT}/MemNavData/materialize_paper_online_a_scene.py" \
  "${LOCAL_ROOT}/MemNavData/build_single_revisit_source.py" \
  "${LOCAL_ROOT}/MemNavData/build_shared_online_role_pairs.py" \
  "${LOCAL_ROOT}/MemNavData/build_paper_role_pair_scene.py"
"${MEMNAV_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/summarize_paper_online_a.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_paper_role_pair_eval.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_paper_role_pair_eval.py" \
  "${LOCAL_ROOT}/MemNavData/retrying_server_launcher.py" \
  "${LOCAL_ROOT}/MemNavData/test_retrying_server_launcher.py" \
  "${LOCAL_ROOT}/MemNavData/finalize_paper_role_pairs.py" \
  "${LOCAL_ROOT}/MemNavData/audit_shared_online_role_pairs.py" \
  "${LOCAL_ROOT}/MemNavData/shared_online_role_pair_contract.py" \
  "${LOCAL_ROOT}/NavDP/baselines/navdp/navdp_server.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/memnav_server.py"
"${MEMNAV_PY}" "${LOCAL_ROOT}/MemNavData/test_retrying_server_launcher.py" -q
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_paper_online_a_scene.sh" \
  "${LOCAL_ROOT}/MemNavData/run_paper_role_pair_episode.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_paper_online_a_collect.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_paper_online_a_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_paper_role_pair_eval.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_paper_role_pair_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_paper_role_pair_verify.sbatch"

STAGING=$(mktemp -d)
trap 'rm -rf -- "${STAGING}"' EXIT
mkdir -p "${STAGING}/MemNavData" \
  "${STAGING}/NavDP/baselines/navdp" \
  "${STAGING}/NavDP/baselines/memnav" "${STAGING}/receipts"
while IFS= read -r -d '' path; do
  cp --preserve=mode,timestamps "${path}" \
    "${STAGING}/MemNavData/$(basename "${path}")"
done < <(find "${LOCAL_ROOT}/MemNavData" -maxdepth 1 -type f -name '*.py' -print0)
for relative in \
  MemNavData/PAPER_EVALUATION_PROTOCOL_20260814.md \
  MemNavData/PAPER_CONSTRUCTION_AMENDMENT_20260814.md \
  MemNavData/PAPER_PORT_RACE_AMENDMENT_20260814.md \
  MemNavData/PAPER_ATTEMPT6_PORT_RACE_INCIDENT_20260814.json \
  MemNavData/SHARED_ONLINE_ROLE_PAIR_CONSUMED_GATE_20260814.md \
  MemNavData/SHARED_ONLINE_ROLE_PAIR_NATURAL_GATE_20260814.md \
  MemNavData/strict_graph_blind_20260806.json \
  MemNavData/run_paper_online_a_scene.sh \
  MemNavData/run_paper_role_pair_episode.sh \
  MemNavData/slurm_paper_online_a_collect.sbatch \
  MemNavData/slurm_paper_online_a_summary.sbatch \
  MemNavData/slurm_paper_role_pair_eval.sbatch \
  MemNavData/slurm_paper_role_pair_summary.sbatch \
  MemNavData/slurm_paper_role_pair_verify.sbatch; do
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${STAGING}/${relative}"
done
for component in memnav navdp; do
  while IFS= read -r -d '' path; do
    cp --preserve=mode,timestamps "${path}" \
      "${STAGING}/NavDP/baselines/${component}/$(basename "${path}")"
  done < <(find "${LOCAL_ROOT}/NavDP/baselines/${component}" -maxdepth 1 \
    -type f -name '*.py' -print0)
done
cp "${CONTROLLED_GATE_AUDIT}" "${STAGING}/receipts/controlled_gate.json"
cp "${NATURAL_GATE_AUDIT}" "${STAGING}/receipts/natural_gate.json"
cp "${CONSTRUCTION_GATE_AUDIT}" \
  "${STAGING}/receipts/construction_amendment_gate.json"
cp "${CONSTRUCTION_GATE_INCIDENT_RECEIPT}" \
  "${STAGING}/receipts/construction_amendment_incident.json"

LOCAL_HEAD=$(git -C "${LOCAL_ROOT}" rev-parse HEAD)
"${MEMNAV_PY}" - "${STAGING}" "${LOCAL_HEAD}" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); files={}
for path in sorted(root.rglob("*")):
    if path.is_symlink(): raise SystemExit(f"bundle symlink: {path}")
    if path.is_file() and path.name not in {"source_bundle_manifest.json","SOURCE_BUNDLE.sha256"}:
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
gates={name:json.load(open(root/"receipts"/f"{name}_gate.json"))
       for name in ("controlled","natural")}
construction=json.load(open(
    root/"receipts"/"construction_amendment_gate.json"))
incident=json.load(open(
    root/"receipts"/"construction_amendment_incident.json"))
payload={
 "schema_version":"paper_online_a_task_bundle_v2_20260814",
 "local_git_head_context":sys.argv[2],
 "scope":"native Goal-A collection, outcome-blind single-Revisit role-pair construction amendment, then frozen five-arm query evaluation",
 "paper_source_manifest_sha256":"b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9",
 "readiness_gates":{name:{
    "passed":row["passed"],
    "run_contract_sha256":row["source_run_contract_sha256"],
 } for name,row in gates.items()},
 "construction_amendment_gate":{
    "ok":construction["ok"],
    "scenes":construction["scenes"],
    "novel_certificate_accept_plans":construction[
        "novel_certified_accept_plans"],
    "revisit_certificate_accept_plans":construction[
        "revisit_certified_accept_plans"],
 },
 "construction_amendment_orchestration":{
    "status":incident["status"],
    "orchestrator_exit_code":incident["orchestrator_exit_code"],
    "independent_audit_ok":incident["independent_audit_ok"],
 },
 "files":files,
}
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
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/paper_online_a_${BUNDLE_MANIFEST_SHA:0:16}
REMOTE_STAGING=${REMOTE_BUNDLE}.partial-$$

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_RUN_ROOT=${RUN_ROOT}"
  echo "DRY_RUN_REMOTE_BUNDLE=${REMOTE_BUNDLE}"
  echo "DRY_RUN_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
  exit 0
fi

# A shared workstation can hold multiple live SSH control sockets.  Fail before
# creating a remote staging directory if the selected socket belongs to another
# collaborator rather than the account that owns RUN_ROOT and REMOTE_BUNDLE.
actual_remote_user=$(remote "id -un")
[[ "${actual_remote_user}" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity mismatch: expected ${EXPECTED_REMOTE_USER}, got ${actual_remote_user}"

if remote "test -d '${REMOTE_BUNDLE}' && test \"\$(sha256sum '${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${SOURCE_RECEIPT_SHA}' && cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"; then
  echo "Reusing verified bundle ${REMOTE_BUNDLE}"
else
  remote "test ! -e '${REMOTE_BUNDLE}' && mkdir -p '${REMOTE_STAGING}'"
  rsync -e "${RSYNC_RSH}" -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    "${STAGING}/" "${REMOTE_HOST}:${REMOTE_STAGING}/"
  remote "test ! -e '${REMOTE_BUNDLE}' && cd '${REMOTE_STAGING}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null && chmod -R a-w '${REMOTE_STAGING}' && mv '${REMOTE_STAGING}' '${REMOTE_BUNDLE}'"
fi
remote "test ! -e '${RUN_ROOT}' && mkdir -p '${RUN_ROOT}/logs'"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"
remote "'/scratch/lg154/conda-envs/memnav/bin/python' - '${DEPENDENCY_RECEIPT}'" <<'PY'
import hashlib,json,sys
receipt=json.load(open(sys.argv[1]))
for name,row in receipt["dependencies"].items():
    digest=hashlib.sha256()
    with open(row["path"],"rb") as handle:
        for chunk in iter(lambda:handle.read(32<<20),b""):
            digest.update(chunk)
    if digest.hexdigest()!=row["sha256"]:
        raise SystemExit(f"dependency hash changed: {name}")
print("dependency content hashes verified")
PY

SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
MANIFEST=${REMOTE_BUNDLE}/MemNavData/strict_graph_blind_20260806.json
EXPECTED_MANIFEST_SHA=b90a03cd6c3456f7741c09c2d8aa4d8f15da1512b9fa6329d31f42b7a03c5fc9
exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},RUN_ROOT=${RUN_ROOT},MANIFEST=${MANIFEST},EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA}"
COLLECT=${REMOTE_BUNDLE}/MemNavData/slurm_paper_online_a_collect.sbatch
SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_online_a_summary.sbatch
EVAL=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_eval.sbatch
PAIR_SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_summary.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_paper_role_pair_verify.sbatch

remote "sbatch --test-only --array=0 --export='${exports}' '${COLLECT}' >/dev/null"
collect_raw=$(remote "sbatch --parsable --array=0-15%${CONCURRENCY} --export='${exports}' '${COLLECT}'")
collect_id=${collect_raw%%;*}
[[ "${collect_id}" =~ ^[0-9]+$ ]] || fail "bad collection job id"
remote "sbatch --test-only --dependency=afterok:${collect_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}' >/dev/null"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${collect_id} --kill-on-invalid-dep=yes --export='${exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job id"
remote "sbatch --test-only --array=0 --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}' >/dev/null"
eval_raw=$(remote "sbatch --parsable --array=0-63%${EVAL_CONCURRENCY} --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}'")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad evaluation job id"
remote "sbatch --test-only --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${PAIR_SUMMARY}' >/dev/null"
pair_summary_raw=$(remote "sbatch --parsable --dependency=afterok:${eval_id} --kill-on-invalid-dep=yes --export='${exports}' '${PAIR_SUMMARY}'")
pair_summary_id=${pair_summary_raw%%;*}
[[ "${pair_summary_id}" =~ ^[0-9]+$ ]] || fail "bad pair summary job id"
remote "sbatch --test-only --dependency=afterok:${pair_summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}' >/dev/null"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${pair_summary_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verification job id"

remote "'/scratch/lg154/conda-envs/memnav/bin/python' - '${RUN_ROOT}/submission.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${collect_id}' '${summary_id}' '${eval_id}' '${pair_summary_id}' '${verify_id}' '${CONCURRENCY}' '${EVAL_CONCURRENCY}'" <<'PY'
import json,sys
(path,bundle,receipt,collect,construction_summary,evaluation,
 policy_summary,verification,collection_concurrency,eval_concurrency)=sys.argv[1:]
with open(path,"x") as f:
 json.dump({
  "schema_version":"paper_online_a_submission_v2_20260814",
  "scope":"pre-query-amended single-Revisit MP3D construction and one-shot five-arm query evaluation",
  "source_bundle":bundle,"source_receipt_sha256":receipt,
  "arrays":{"collection_concurrency":int(collection_concurrency),
            "evaluation_concurrency":int(eval_concurrency)},
  "jobs":{"collect_array":int(collect),
          "construction_summary":int(construction_summary),
          "evaluation_array":int(evaluation),
          "policy_summary":int(policy_summary),
          "independent_verification":int(verification)},
 },f,indent=2,sort_keys=True); f.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA}"
echo "collect=${collect_id} construction_summary=${summary_id} eval=${eval_id} policy_summary=${pair_summary_id} verify=${verify_id}"
