#!/usr/bin/env bash
# Freeze an additive source bundle, run the first formal episode as a staged
# contract gate, then release the remaining formal array without code changes.

set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-yz11502@login.torch.hpc.nyu.edu}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/home/asus/.ssh/cm-novel-yz11502}
PARENT_BUNDLE=${PARENT_BUNDLE:-/scratch/yz11502/Research/source_bundles/paper_power_expansion_repair_c18ef4e2021ef3b5}
EXPECTED_PARENT_RECEIPT_SHA=88d5983ebfa58c9970572a112483d0a65010c13b243d1b3a0e8f67ec66958a9d
EXPECTED_PARENT_MANIFEST_SHA=c18ef4e2021ef3b5fdd10794f74fc8054ba5e4ed1d82e56364f6a9e1de810482
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
CONTROL_FREEZE_ROOT=${CONTROL_FREEZE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/novel_memory_direction_control_20260816/freeze_preoutcome}
CONTROL_MANIFEST=${CONTROL_MANIFEST:-${CONTROL_FREEZE_ROOT}/control_manifest.json}
BENCH_ROOT=${BENCH_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/paper_certified_compass_20260814/paper_power_expansion_20260814_pre_result/benchmarks/natural_direction}
RUN_ROOT=${RUN_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-results/novel_memory_direction_control_20260816/consumed_mechanism_attempt1}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT:-/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json}
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
REMOTE_PY=${REMOTE_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
LOCAL_PY=${LOCAL_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad eval concurrency"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ -S "${SSH_CONTROL_PATH}" ]] || fail "shared SSH control socket is missing"
SSH_ARGS=(-o BatchMode=yes -S "${SSH_CONTROL_PATH}" -o ControlMaster=no)
RSYNC_RSH="ssh -o BatchMode=yes -S ${SSH_CONTROL_PATH} -o ControlMaster=no"
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }

FILES=(
  MemNavData/NOVEL_MEMORY_DIRECTION_CAUSAL_CONTROL_PROTOCOL_20260815.md
  MemNavData/NOVEL_MEMORY_DIRECTION_CAUSAL_CONTROL_AMENDMENT_20260816.md
  MemNavData/NOVEL_MEMORY_DIRECTION_CAUSAL_CONTROL_PREFLIGHT_20260816.md
  MemNavData/novel_memory_direction_control.py
  MemNavData/eval_novel_memory_direction_control.py
  MemNavData/run_novel_memory_direction_control_episode.sh
  MemNavData/slurm_novel_memory_direction_control.sbatch
  MemNavData/summarize_novel_memory_direction_control.py
  MemNavData/independent_verify_novel_memory_direction_control.py
  MemNavData/slurm_novel_memory_direction_summary.sbatch
  MemNavData/slurm_novel_memory_direction_verify.sbatch
  MemNavData/freeze_mp3d_scene_budget.py
  MemNavData/freeze_novel_memory_direction_control.py
  MemNavData/test_novel_memory_direction_control.py
)
for relative in "${FILES[@]}"; do
  path=${LOCAL_ROOT}/${relative}
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input: ${relative}"
done

export PYTHONPATH=${LOCAL_ROOT}:${LOCAL_ROOT}/MemNavData${PYTHONPATH:+:${PYTHONPATH}}
"${LOCAL_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/novel_memory_direction_control.py" \
  "${LOCAL_ROOT}/MemNavData/eval_novel_memory_direction_control.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_novel_memory_direction_control.py" \
  "${LOCAL_ROOT}/MemNavData/independent_verify_novel_memory_direction_control.py"
"${LOCAL_PY}" -m unittest -q MemNavData/test_novel_memory_direction_control.py
bash -n \
  "${LOCAL_ROOT}/MemNavData/run_novel_memory_direction_control_episode.sh" \
  "${LOCAL_ROOT}/MemNavData/slurm_novel_memory_direction_control.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_novel_memory_direction_summary.sbatch" \
  "${LOCAL_ROOT}/MemNavData/slurm_novel_memory_direction_verify.sbatch"

[[ "$(remote 'id -un')" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity mismatch"
remote "test -d '${PARENT_BUNDLE}'"
PARENT_RECEIPT_SHA=$(remote \
  "sha256sum '${PARENT_BUNDLE}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")
PARENT_MANIFEST_SHA=$(remote \
  "sha256sum '${PARENT_BUNDLE}/source_bundle_manifest.json' | awk '{print \$1}'")
[[ "${PARENT_RECEIPT_SHA}" == "${EXPECTED_PARENT_RECEIPT_SHA}" ]] || \
  fail "unexpected parent source receipt"
[[ "${PARENT_MANIFEST_SHA}" == "${EXPECTED_PARENT_MANIFEST_SHA}" ]] || \
  fail "unexpected parent source manifest"
remote "cd '${PARENT_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'"
remote "test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'"
remote "test -f '${CONTROL_MANIFEST}' && test -f '${BENCH_ROOT}/manifest.json'"
CONTROL_MANIFEST_SHA=$(remote "sha256sum '${CONTROL_MANIFEST}' | awk '{print \$1}'")
readarray -t control_info < <(remote \
  "PYTHONPATH='${CONTROL_FREEZE_ROOT}' '${REMOTE_PY}' - '${CONTROL_MANIFEST}' '${BENCH_ROOT}/manifest.json'" <<'PY'
import hashlib,json,sys
from pathlib import Path
def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(8<<20),b""):
            digest.update(chunk)
    return digest.hexdigest()
control=json.load(open(sys.argv[1]))
if control.get("evaluation_stage") != "consumed_development_mechanism_only":
    raise SystemExit("wrong evaluation stage")
if control.get("confirmation_claim_allowed") is not False:
    raise SystemExit("control manifest claims confirmation")
if control.get("method_or_threshold_selection_allowed") is not False:
    raise SystemExit("control manifest grants selection authority")
if sha(sys.argv[2]) != control["benchmark_manifest_sha256"]:
    raise SystemExit("benchmark manifest changed")
used={str(row["scene"]) for row in control["episodes"]}
untouched=set(control["untouched_final_scenes_remain_unread"])
if used & untouched:
    raise SystemExit("untouched final scene leaked into control")
print(len(control["episodes"]))
print(len(used))
print(len(untouched))
PY
)
[[ "${#control_info[@]}" -eq 3 ]] || fail "control manifest audit failed"
POPULATION=${control_info[0]}; SCENES=${control_info[1]}; FINAL_SCENES=${control_info[2]}
[[ "${POPULATION}" =~ ^[1-9][0-9]*$ ]] || fail "empty control population"
[[ "${SCENES}" -ge 1 && "${FINAL_SCENES}" -eq 14 ]] || fail "scene budget changed"
remote "test ! -e '${RUN_ROOT}'"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_CONTROL_MANIFEST_SHA=${CONTROL_MANIFEST_SHA}"
  echo "DRY_RUN_POPULATION=${POPULATION}"
  echo "DRY_RUN_SCENES=${SCENES}"
  echo "DRY_RUN_FINAL_SCENES_REMAINING=${FINAL_SCENES}"
  exit 0
fi

PATCH_STAGE=$(mktemp -d /tmp/novel_memory_direction_bundle.XXXXXX)
REMOTE_PARTIAL=${REMOTE_BUNDLE_BASE}/novel_memory_direction_control.partial.$$
cleanup() {
  rm -rf -- "${PATCH_STAGE}"
}
trap cleanup EXIT
for relative in "${FILES[@]}"; do
  mkdir -p "${PATCH_STAGE}/$(dirname "${relative}")"
  cp "${LOCAL_ROOT}/${relative}" "${PATCH_STAGE}/${relative}"
done

remote "test ! -e '${REMOTE_PARTIAL}' && cp -a '${PARENT_BUNDLE}' '${REMOTE_PARTIAL}' && chmod -R u+w '${REMOTE_PARTIAL}'"
rsync -e "${RSYNC_RSH}" -a --chmod=Fu=rw,Fgo=r \
  "${PATCH_STAGE}/" "${REMOTE_HOST}:${REMOTE_PARTIAL}/"

allowed_json=$(printf '%s\n' "${FILES[@]}" | "${LOCAL_PY}" -c \
  'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')
remote "'${REMOTE_PY}' - '${PARENT_BUNDLE}' '${REMOTE_PARTIAL}' '${PARENT_RECEIPT_SHA}' '${PARENT_MANIFEST_SHA}' '${CONTROL_MANIFEST_SHA}' '${allowed_json}'" <<'PY'
import hashlib,json,sys
from pathlib import Path
parent=Path(sys.argv[1]); child=Path(sys.argv[2])
parent_receipt_sha=sys.argv[3]; parent_manifest_sha=sys.argv[4]
control_manifest_sha=sys.argv[5]; allowed=set(json.loads(sys.argv[6]))
ignored={"SOURCE_BUNDLE.sha256","source_bundle_manifest.json"}
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def inventory(root):
    return {p.relative_to(root).as_posix():digest(p)
            for p in sorted(root.rglob("*"))
            if p.is_file() and p.relative_to(root).as_posix() not in ignored}
before=inventory(parent); after=inventory(child)
missing=set(before)-set(after)
changed={name for name in set(before)&set(after) if before[name]!=after[name]}
added=set(after)-set(before)
if missing:
    raise SystemExit(f"parent files missing: {sorted(missing)}")
actual_delta=changed|added
if not actual_delta <= allowed:
    raise SystemExit(
        f"unexpected source delta: changed={sorted(changed)} added={sorted(added)}")
payload={
  "schema_version":"novel_memory_direction_source_bundle_v1_20260816",
  "scope":"consumed-development Novel causal mechanism control",
  "parent_bundle":str(parent),
  "parent_source_receipt_sha256":parent_receipt_sha,
  "parent_bundle_manifest_sha256":parent_manifest_sha,
  "control_manifest_sha256":control_manifest_sha,
  "causal_outcomes_read_before_source_freeze":False,
  "confirmation_claim_allowed":False,
  "method_or_threshold_selection_allowed":False,
  "production_policy_or_checkpoint_changed":False,
  "overlay_inputs":sorted(allowed),
  "actual_source_delta":sorted(actual_delta),
  "files":after,
}
(child/"source_bundle_manifest.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
remote "cd '${REMOTE_PARTIAL}' && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum > SOURCE_BUNDLE.sha256 && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
SOURCE_RECEIPT_SHA=$(remote \
  "sha256sum '${REMOTE_PARTIAL}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")
BUNDLE_MANIFEST_SHA=$(remote \
  "sha256sum '${REMOTE_PARTIAL}/source_bundle_manifest.json' | awk '{print \$1}'")
[[ "${SOURCE_RECEIPT_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "bad source receipt SHA"
[[ "${BUNDLE_MANIFEST_SHA}" =~ ^[0-9a-f]{64}$ ]] || fail "bad bundle manifest SHA"
REMOTE_BUNDLE=${REMOTE_BUNDLE_BASE}/novel_memory_direction_control_${BUNDLE_MANIFEST_SHA:0:16}
remote "test ! -e '${REMOTE_BUNDLE}' && chmod -R a-w '${REMOTE_PARTIAL}' && mv '${REMOTE_PARTIAL}' '${REMOTE_BUNDLE}'"
remote "cd '${REMOTE_BUNDLE}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"

SOURCE_RECEIPT=${REMOTE_BUNDLE}/SOURCE_BUNDLE.sha256
EVAL=${REMOTE_BUNDLE}/MemNavData/slurm_novel_memory_direction_control.sbatch
SUMMARY=${REMOTE_BUNDLE}/MemNavData/slurm_novel_memory_direction_summary.sbatch
VERIFY=${REMOTE_BUNDLE}/MemNavData/slurm_novel_memory_direction_verify.sbatch
common_exports="ALL,SOURCE_ROOT=${REMOTE_BUNDLE},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${SOURCE_RECEIPT_SHA},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA},BENCH_ROOT=${BENCH_ROOT},CONTROL_MANIFEST=${CONTROL_MANIFEST},EXPECTED_CONTROL_MANIFEST_SHA=${CONTROL_MANIFEST_SHA}"
formal_exports="${common_exports},RUN_ROOT=${RUN_ROOT}"

remote "mkdir '${RUN_ROOT}'"
remote "sbatch --test-only --array=0 --export='${formal_exports}' '${EVAL}' >/dev/null"
head_raw=$(remote \
  "sbatch --parsable --array=0 --export='${formal_exports}' '${EVAL}'")
head_id=${head_raw%%;*}; [[ "${head_id}" =~ ^[0-9]+$ ]] || fail "bad formal-head id"
if (( POPULATION > 1 )); then
  remote "sbatch --test-only --array=1 --dependency=afterok:${head_id} --kill-on-invalid-dep=yes --export='${formal_exports}' '${EVAL}' >/dev/null"
  eval_raw=$(remote \
    "sbatch --parsable --array=1-$((POPULATION-1))%${EVAL_CONCURRENCY} --dependency=afterok:${head_id} --kill-on-invalid-dep=yes --export='${formal_exports}' '${EVAL}'")
  eval_id=${eval_raw%%;*}; [[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad eval-tail id"
  summary_dependency=${eval_id}
else
  eval_id=none
  summary_dependency=${head_id}
fi
summary_raw=$(remote \
  "sbatch --parsable --dependency=afterok:${summary_dependency} --kill-on-invalid-dep=yes --export='${formal_exports}' '${SUMMARY}'")
summary_id=${summary_raw%%;*}; [[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary id"
verify_raw=$(remote \
  "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${formal_exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}; [[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verify id"

remote "'${REMOTE_PY}' - '${RUN_ROOT}/submission.json' '${REMOTE_BUNDLE}' '${SOURCE_RECEIPT_SHA}' '${BUNDLE_MANIFEST_SHA}' '${CONTROL_MANIFEST}' '${CONTROL_MANIFEST_SHA}' '${POPULATION}' '${SCENES}' '${head_id}' '${eval_id}' '${summary_id}' '${verify_id}'" <<'PY'
import json,sys,time
(formal_path,bundle,receipt,bundle_manifest,control,control_sha,population,
 scenes,formal_head,formal_tail,summary,verification)=sys.argv[1:]
payload={
  "schema_version":"novel_memory_direction_submission_v1_20260816",
  "evaluation_stage":"consumed_development_mechanism_only",
  "confirmation_claim_allowed":False,
  "method_or_threshold_selection_allowed":False,
  "source_bundle":bundle,
  "source_receipt_sha256":receipt,
  "bundle_manifest_sha256":bundle_manifest,
  "control_manifest":control,
  "control_manifest_sha256":control_sha,
  "population":{"episodes":int(population),"scenes":int(scenes)},
  "untouched_final_scenes_consumed":False,
  "runtime_smoke_run":False,
  "staged_first_episode_is_formal":True,
  "source_or_protocol_changes_allowed_after_first_episode":False,
  "submission_unix_time":time.time(),
  "scope":"consumed causal mechanism formal",
  "jobs":{"formal_head":int(formal_head),
          "formal_tail":(None if formal_tail=="none" else int(formal_tail)),
          "summary":int(summary),
          "independent_verification":int(verification)},
}
open(formal_path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "SOURCE_BUNDLE=${REMOTE_BUNDLE}"
echo "CONTROL_MANIFEST_SHA=${CONTROL_MANIFEST_SHA}"
echo "population=${POPULATION} scenes=${SCENES} final_scenes_untouched=${FINAL_SCENES}"
echo "formal_head=${head_id} formal_tail=${eval_id} summary=${summary_id} verify=${verify_id}"
