#!/usr/bin/env bash
# Additively repair the one Slurm-cancelled lifelong NNR pair, then aggregate.
set -euo pipefail
umask 0022

ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${REMOTE_HOST}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}
SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/lifelong_nnr_15b2e9c4d2ab9ed6
EXPECTED_SOURCE_RECEIPT_SHA=3be9f426dd7d34cc7cd3a51959ec2dc4436b81a323645fe51ea3468fd3c21266
NNR_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_nnr_20260814/shared_online_nnr_strict_v2_20260814
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/lifelong_nnr_20260821/lifelong_nnr_runtime_repair_20260821T083000Z
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80
BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY_RECEIPT=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
EXPECTED_DEPENDENCY_RECEIPT_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
ORIGINAL_ARRAY=16126310
ORIGINAL_MISSING_TASK=5
SUPERSEDED_AGGREGATE=16126345
SUPERSEDED_VERIFY=16126347
EXPECTED_PAIR_LABEL=005_e9zR4mvMWw7_episode_0007
RECEIPT=${ROOT}/MemNavData/LIFELONG_NNR_TASK5_REPAIR_SUBMISSION_RECEIPT_20260821.json
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  ssh -o BatchMode=yes -o ControlMaster=no -S "${SSH_CONTROL_PATH}" \
    "${REMOTE_HOST}" "$@"
}

[[ -S "${SSH_CONTROL_PATH}" ]] || fail "authoritative SSH master missing"
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ ! -e "${RECEIPT}" ]] || fail "local repair receipt already exists"

SOURCE_RECEIPT=${SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EVAL=${SOURCE_ROOT}/MemNavData/slurm_lifelong_nnr_paired_eval.sbatch
AGGREGATE=${SOURCE_ROOT}/MemNavData/slurm_lifelong_nnr_aggregate.sbatch
VERIFY=${SOURCE_ROOT}/MemNavData/slurm_lifelong_nnr_verify.sbatch

remote "set -euo pipefail
test \"\$(sha256sum '${SOURCE_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_SOURCE_RECEIPT_SHA}'
cd '${SOURCE_ROOT}' && sha256sum -c --quiet '${SOURCE_RECEIPT}'
test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${BASE_SOURCE_RECEIPT_SHA}'
test \"\$(sha256sum '${DEPENDENCY_RECEIPT}' | awk '{print \$1}')\" = '${EXPECTED_DEPENDENCY_RECEIPT_SHA}'
cd '${RUN_ROOT}/factual_b_support' && sha256sum -c --quiet population.json.sha256
test -f '${RUN_ROOT}/factual_b_support/SEALED'
test ! -e '${RUN_ROOT}/evaluation/${EXPECTED_PAIR_LABEL}'
test ! -e '${RUN_ROOT}/aggregate'
test ! -e '${RUN_ROOT}/VERIFIED'
/scratch/lg154/conda-envs/memnav/bin/python -c \"import json; p=json.load(open('${RUN_ROOT}/factual_b_support/population.json')); r=p['accepted'][5]; assert r['scene']=='e9zR4mvMWw7' and r['episode']=='episode_0007'\"
state=\$(sacct -j '${ORIGINAL_ARRAY}_${ORIGINAL_MISSING_TASK}' -X -n -o State | xargs)
reason=\$(sacct -j '${ORIGINAL_ARRAY}_${ORIGINAL_MISSING_TASK}' -X -n -o Reason | xargs)
case \"\${state}\" in CANCELLED*) ;; *) echo \"unexpected task state \${state}\" >&2; exit 2;; esac
test \"\${reason}\" = QOSGrpGRES
mkdir -p '${RUN_ROOT}/task5_repair'"

exports="ALL,SOURCE_ROOT=${SOURCE_ROOT},SOURCE_RECEIPT=${SOURCE_RECEIPT},EXPECTED_SOURCE_RECEIPT_SHA=${EXPECTED_SOURCE_RECEIPT_SHA},NNR_ROOT=${NNR_ROOT},RUN_ROOT=${RUN_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},BASE_SOURCE_RECEIPT_SHA=${BASE_SOURCE_RECEIPT_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY_RECEIPT},EXPECTED_DEPENDENCY_RECEIPT_SHA=${EXPECTED_DEPENDENCY_RECEIPT_SHA}"

repair_dep=afterany:${ORIGINAL_ARRAY}
remote "sbatch --test-only --job-name=lifeNNRpairR5 --array=${ORIGINAL_MISSING_TASK} --dependency=${repair_dep} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}' >/dev/null"
if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN repair_dependency=${repair_dep}"
  exit 0
fi

repair_raw=$(remote "sbatch --parsable --job-name=lifeNNRpairR5 --array=${ORIGINAL_MISSING_TASK} --dependency=${repair_dep} --kill-on-invalid-dep=yes --export='${exports}' '${EVAL}'")
repair_id=${repair_raw%%;*}
[[ "${repair_id}" =~ ^[0-9]+$ ]] || fail "bad repair job ID"

remote "sbatch --test-only --job-name=lifeNNRsumR5 --dependency=afterok:${repair_id} --kill-on-invalid-dep=yes --export='${exports}' '${AGGREGATE}' >/dev/null"
aggregate_raw=$(remote "sbatch --parsable --job-name=lifeNNRsumR5 --dependency=afterok:${repair_id} --kill-on-invalid-dep=yes --export='${exports}' '${AGGREGATE}'")
aggregate_id=${aggregate_raw%%;*}
[[ "${aggregate_id}" =~ ^[0-9]+$ ]] || fail "bad aggregate job ID"

remote "sbatch --test-only --job-name=lifeNNRverifyR5 --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}' >/dev/null"
verify_raw=$(remote "sbatch --parsable --job-name=lifeNNRverifyR5 --dependency=afterok:${aggregate_id} --kill-on-invalid-dep=yes --export='${exports}' '${VERIFY}'")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verifier job ID"

python3 - "${RECEIPT}" "${repair_id}" "${aggregate_id}" "${verify_id}" <<'PY'
import json
import sys

path, repair, aggregate, verify = sys.argv[1:]
payload = {
    "schema_version": "lifelong_nnr_task5_repair_submission_v1_20260821",
    "run_root": "/scratch/yz11502/Research/Nav-axis-uturn-results/lifelong_nnr_20260821/lifelong_nnr_runtime_repair_20260821T083000Z",
    "source_bundle": "/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/lifelong_nnr_15b2e9c4d2ab9ed6",
    "source_receipt_sha256": "3be9f426dd7d34cc7cd3a51959ec2dc4436b81a323645fe51ea3468fd3c21266",
    "original_array": 16126310,
    "missing_task": 5,
    "missing_pair_label": "005_e9zR4mvMWw7_episode_0007",
    "observed_slurm_state": "CANCELLED by 0",
    "observed_slurm_reason": "QOSGrpGRES",
    "repair_dependency": "afterany:16126310",
    "jobs": {
        "task5_additive_repair": int(repair),
        "replacement_aggregate": int(aggregate),
        "replacement_independent_verification": int(verify),
    },
    "superseded_jobs": [16126345, 16126347],
    "scientific_guards": {
        "population_or_selection_changed": False,
        "method_threshold_checkpoint_or_seed_changed": False,
        "existing_pair_output_overwrite_allowed": False,
        "query_outcomes_used_to_select_repair": False,
        "aggregate_requires_all_19_frozen_pairs": True,
    },
}
with open(path, "x") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

scp -q -o BatchMode=yes -o ControlMaster=no -o ControlPath="${SSH_CONTROL_PATH}" \
  "${RECEIPT}" "${REMOTE_HOST}:${RUN_ROOT}/task5_repair/submission.json"
remote "sha256sum '${RUN_ROOT}/task5_repair/submission.json' >'${RUN_ROOT}/task5_repair/submission.json.sha256'"

echo "REPAIR=${repair_id} AGGREGATE=${aggregate_id} VERIFY=${verify_id}"
