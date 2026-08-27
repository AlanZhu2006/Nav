#!/usr/bin/env bash
# Submit exactly-once factual mono B rollouts from the independently verified
# v4 materialization.  Downstream prefix construction is intentionally not
# submitted here: gpu48 counts dependency-held array elements toward the user
# submission limit, so the 99-element prefix array must be resident only after
# this 59-shard array leaves the queue.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
CONCURRENCY=${CONCURRENCY:-4}
MAXIMUM_HISTORIES_PER_SHARD=${MAXIMUM_HISTORIES_PER_SHARD:-2}
LOCAL_MEMNAV_PY=${LOCAL_MEMNAV_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
REPAIR_OF_JOB=${REPAIR_OF_JOB:-}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d
PARENT_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
ORIGINAL_V4_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_lifelong_natural_v4_d85fc50df19b1384
ORIGINAL_V4_RECEIPT=${ORIGINAL_V4_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_ORIGINAL_V4_RECEIPT_SHA=d85fc50df19b138499e07bb9555d9f9a0088da0f5040e32f78f24b74a975c59a
PROTOCOL=${ORIGINAL_V4_ROOT}/MemNavData/hm3d_fullmono_lifelong_direct_natural_protocol_20260827.json
EXPECTED_PROTOCOL_SHA=2bfc62c08cbee1dffd3c5a3f627b1cb58a7c5076a9c9b2e2554e28843564492f
SERVER_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_lifelong_375f0b6879b2ff87
SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
EXPECTED_SERVER_SOURCE_RECEIPT_SHA=375f0b6879b2ff87b7019dae4727880d1b03fd3185a1862e6239942a76b5bcc8
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256
EXPECTED_BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
MATERIALIZATION_VERIFY=${RUN_ROOT}/independent_natural_v4_materialization_verification.json
EXPECTED_MATERIALIZATION_VERIFY_SHA=77b802f234bbbd6ad588a7bfa6069897699d4d41a3140b089b3a56498222df26
AB_POPULATION_RECEIPT=${RUN_ROOT}/ab_population/population_receipt.json
EXPECTED_AB_POPULATION_RECEIPT_SHA=9230a49d9947215cd8086a1e36bc3d0147c452331cc44c1f26d3877a0a8f68ea
AB_MANIFEST=${RUN_ROOT}/ab_population/role_pairs/manifest.json
EXPECTED_AB_MANIFEST_SHA=e5c97dad42b26c67032c5b42a8467d857ce57f4b627a3f4851616a159ecef978
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
REMOTE_MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python
REMOTE_HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-$(
  ssh -G "${SSH_ALIAS}" 2>/dev/null |
    awk '$1=="controlpath"{value=$2} END{print value}'
)}

cd "${ROOT}"
fail() { echo "ABORT: $*" >&2; exit 2; }
remote() {
  timeout 180 ssh -n -T -o BatchMode=yes -o ControlMaster=no \
    -o ServerAliveInterval=15 -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" "$@"
}
remote_query() {
  local attempt output
  for attempt in 1 2 3; do
    if output=$(remote "$@"); then
      printf '%s\n' "${output}"
      return 0
    fi
    sleep 2
  done
  return 1
}
job_id() {
  tr -d '\r' | awk -F';' '/^[0-9]+(;|$)/ {print $1; exit}'
}

[[ "${CONCURRENCY}" == 4 ]] || fail "formal concurrency is frozen at four"
[[ "${MAXIMUM_HISTORIES_PER_SHARD}" == 2 ]] || \
  fail "formal shard size is frozen at two histories"
[[ -x "${LOCAL_MEMNAV_PY}" ]] || fail "local MemNav Python missing"
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "authoritative shared SSH socket missing"
timeout 15 ssh -O check -S "${SSH_CONTROL_PATH}" "${SSH_ALIAS}" \
  >/dev/null 2>&1 || fail "shared SSH master is not responsive"

files=(
  MemNavData/hm3d_fullmono_lifelong.py
  MemNavData/hm3d_fullmono_lifelong_direct_natural_protocol_20260827.json
  MemNavData/build_hm3d_fullmono_lifelong_natural_v4_b_shards.py
  MemNavData/test_build_hm3d_fullmono_lifelong_natural_v4_b_shards.py
  MemNavData/test_hm3d_fullmono_lifelong.py
  MemNavData/run_hm3d_fullmono_server_scene.sh
  MemNavData/collect_hm3d_fullmono_lifelong_b.py
  MemNavData/construct_hm3d_fullmono_lifelong_prefix.py
  MemNavData/finalize_hm3d_fullmono_lifelong_population.py
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_collect_b_shard.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_construct_prefix.sbatch
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize_population.sbatch
  MemNavData/slurm_safe_submit.sh
  MemNavData/submit_hm3d_fullmono_lifelong_natural_v4_factual_b_hpc.sh
)
for path in "${files[@]}"; do
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing physical input ${path}"
done
# These runtime programs are copied only to change Python's script directory;
# their bytes must remain identical to the previously frozen server source.
[[ "$(sha256sum MemNavData/collect_hm3d_fullmono_lifelong_b.py | awk '{print $1}')" == \
  8214cdf74c9587b4bfb74bf56b45df915eab60a77c1d65fd10eb0e61637adbed ]]
[[ "$(sha256sum MemNavData/construct_hm3d_fullmono_lifelong_prefix.py | awk '{print $1}')" == \
  7ee63003ac5f9452e76ba2e26cce0e22de47d10aeaeeaeee2059b6708030bdde ]]
[[ "$(sha256sum MemNavData/finalize_hm3d_fullmono_lifelong_population.py | awk '{print $1}')" == \
  086d4b1bd2ed72671a8004b45ed85633ea13d8e05fb672b3e1529ca5bd28809e ]]
bash -n \
  MemNavData/run_hm3d_fullmono_server_scene.sh \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_collect_b_shard.sbatch \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_construct_prefix.sbatch \
  MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize_population.sbatch \
  MemNavData/slurm_safe_submit.sh \
  MemNavData/submit_hm3d_fullmono_lifelong_natural_v4_factual_b_hpc.sh
(
  source MemNavData/slurm_safe_submit.sh
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_collect_b_shard.sbatch
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_construct_prefix.sbatch
  lint_sbatch_template \
    MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_finalize_population.sbatch
)
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${ROOT}/MemNavData" \
  "${LOCAL_MEMNAV_PY}" -m unittest -q \
  test_build_hm3d_fullmono_lifelong_natural_v4_b_shards \
  test_hm3d_fullmono_lifelong.FullMonoLifelongContractTest.test_direct_natural_v4_protocol_freezes_five_leg_gate

scratch=$(mktemp -d /tmp/h3life_v4_b_submit.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT
mkdir -p "${scratch}/root/MemNavData"
for path in "${files[@]}"; do
  cp -p -- "${path}" "${scratch}/root/MemNavData/$(basename "${path}")"
done
(
  cd "${scratch}/root"
  find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | \
    xargs -0 sha256sum >SOURCE_BUNDLE.sha256
  sha256sum -c --quiet SOURCE_BUNDLE.sha256
)
task_receipt_sha=$(sha256sum "${scratch}/root/SOURCE_BUNDLE.sha256" | awk '{print $1}')
bundle_key=${task_receipt_sha:0:16}
task_root=${REMOTE_BUNDLES}/hm3d_lifelong_natural_v4_factual_b_${bundle_key}
task_stage=${task_root}.partial.$$

preflight=$(remote_query "set -euo pipefail; \
  echo IDENTITY=\$(id -un); \
  echo ORIGINAL_V4=\$(sha256sum '${ORIGINAL_V4_RECEIPT}' | awk '{print \$1}'); \
  echo PROTOCOL=\$(sha256sum '${PROTOCOL}' | awk '{print \$1}'); \
  echo SERVER=\$(sha256sum '${SERVER_SOURCE_RECEIPT}' | awk '{print \$1}'); \
  echo BASE=\$(sha256sum '${BASE_RECEIPT}' | awk '{print \$1}'); \
  echo VERIFY=\$(sha256sum '${MATERIALIZATION_VERIFY}' | awk '{print \$1}'); \
  echo POPULATION=\$(sha256sum '${AB_POPULATION_RECEIPT}' | awk '{print \$1}'); \
  echo MANIFEST=\$(sha256sum '${AB_MANIFEST}' | awk '{print \$1}'); \
  test -f '${RUN_ROOT}/ab_population/SEALED'; \
  test ! -e '${RUN_ROOT}/factual_b'; \
  test ! -e '${RUN_ROOT}/prefix_fragments'; \
  test ! -e '${RUN_ROOT}/population'; \
  test -z \"\$(squeue -h -u yz11502 -n h3lifeV4B -o '%i')\"; \
  '${REMOTE_MEMNAV_PY}' -c \"import json; p=json.load(open('${MATERIALIZATION_VERIFY}')); assert p['verified'] is True and p['factual_B_gate_verified'] is True and p['factual_B_executed'] is False and p['navigation_outcomes_read'] is False\"; \
  echo COMPLETE")
value_for() {
  local key=$1
  printf '%s\n' "${preflight}" | tr -d '\r' | \
    awk -F= -v key="${key}" '$1==key {print $2; exit}'
}
[[ "$(value_for IDENTITY)" == yz11502 ]] || fail "wrong remote identity"
[[ "$(value_for ORIGINAL_V4)" == "${EXPECTED_ORIGINAL_V4_RECEIPT_SHA}" ]] || fail "original v4 bundle changed"
[[ "$(value_for PROTOCOL)" == "${EXPECTED_PROTOCOL_SHA}" ]] || fail "protocol changed"
[[ "$(value_for SERVER)" == "${EXPECTED_SERVER_SOURCE_RECEIPT_SHA}" ]] || fail "server source changed"
[[ "$(value_for BASE)" == "${EXPECTED_BASE_RECEIPT_SHA}" ]] || fail "base source changed"
[[ "$(value_for VERIFY)" == "${EXPECTED_MATERIALIZATION_VERIFY_SHA}" ]] || fail "materialization verifier changed"
[[ "$(value_for POPULATION)" == "${EXPECTED_AB_POPULATION_RECEIPT_SHA}" ]] || fail "A/B population receipt changed"
[[ "$(value_for MANIFEST)" == "${EXPECTED_AB_MANIFEST_SHA}" ]] || fail "A/B manifest changed"
printf '%s\n' "${preflight}" | tr -d '\r' | grep -qx COMPLETE || fail "preflight incomplete"
if [[ -n "${REPAIR_OF_JOB}" ]]; then
  repair_state=$(remote_query "sacct -j '${REPAIR_OF_JOB}' -X -n -o State | awk 'NF{print \$1; exit}'" | tr -d '\r')
  [[ "${repair_state}" == CANCELLED ]] || fail "repair parent is not cancelled"
  completed_before_repair=$(remote_query "find '${RUN_ROOT}/factual_b' -mindepth 2 -maxdepth 2 -name completion.json -type f 2>/dev/null | wc -l" | tr -d '\r' | tr -d ' ')
  [[ "${completed_before_repair}" == 0 ]] || fail "repair would overwrite factual-B outcomes"
fi

if remote "test -d '${task_root}'"; then
  remote "test \"\$(sha256sum '${task_root}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${task_receipt_sha}' && cd '${task_root}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256"
else
  remote "test ! -e '${task_stage}' && mkdir -p '${task_stage}'"
  timeout 240 rsync -a --partial \
    --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
    -e "ssh -o BatchMode=yes -o ControlMaster=no -S ${SSH_CONTROL_PATH}" \
    "${scratch}/root/" "${SSH_ALIAS}:${task_stage}/"
  remote "cd '${task_stage}' && sha256sum -c --quiet SOURCE_BUNDLE.sha256 && chmod -R a-w '${task_stage}' && mv '${task_stage}' '${task_root}'"
fi

task_receipt=${task_root}/SOURCE_BUNDLE.sha256
overlay_path=${task_root}:${task_root}/MemNavData:${SERVER_SOURCE_ROOT}:${SERVER_SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 '${BASE_SIF}' env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${overlay_path}' '${REMOTE_MEMNAV_PY}' -m unittest -q test_build_hm3d_fullmono_lifelong_natural_v4_b_shards test_hm3d_fullmono_lifelong.FullMonoLifelongContractTest.test_direct_natural_v4_protocol_freezes_five_leg_gate"
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 '${BASE_SIF}' env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${overlay_path}' '${REMOTE_MEMNAV_PY}' -c \"from pathlib import Path; import collect_hm3d_fullmono_lifelong_b as c; import finalize_hm3d_fullmono_lifelong_population as f; from hm3d_fullmono_lifelong import load_protocol; load_protocol(Path('${PROTOCOL}')); assert Path(c.__file__).resolve().is_relative_to(Path('${task_root}')); assert Path(f.__file__).resolve().is_relative_to(Path('${task_root}'))\""
remote "singularity exec -B /scratch/lg154 -B /scratch/yz11502 '${BASE_SIF}' env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${overlay_path}' '${REMOTE_HAB_PY}' -c \"from pathlib import Path; import construct_hm3d_fullmono_lifelong_prefix as p; from hm3d_fullmono_lifelong import load_protocol; load_protocol(Path('${PROTOCOL}')); assert Path(p.__file__).resolve().is_relative_to(Path('${task_root}'))\""

schedule=${RUN_ROOT}/factual_b_schedule
schedule_stage=${schedule}.partial.$$
if remote "test -d '${schedule}'"; then
  remote "set -euo pipefail; cd '${schedule}'; sha256sum -c --quiet shards.json.sha256; \
    '${REMOTE_MEMNAV_PY}' -c \"import json; p=json.load(open('${schedule}/shards.json')); assert p['benchmark_manifest_sha256']=='${EXPECTED_AB_MANIFEST_SHA}'; assert p['candidate_histories']==99; assert p['source_recipient_histories']==61; assert p['scene_clusters']==35; assert p['maximum_histories_per_shard']==2; assert p['shard_count']==59; assert p['all_candidates_partitioned_once'] is True; assert p['query_policy_outcomes_read'] is False; assert p['navigation_outcomes_read'] is False\""
else
  remote "set -euo pipefail; test ! -e '${schedule_stage}' && mkdir -p '${schedule_stage}'; \
    env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='${overlay_path}' '${REMOTE_MEMNAV_PY}' -u '${task_root}/MemNavData/build_hm3d_fullmono_lifelong_natural_v4_b_shards.py' \
      --manifest '${AB_MANIFEST}' --maximum-histories-per-shard '${MAXIMUM_HISTORIES_PER_SHARD}' --out '${schedule_stage}/shards.json'; \
    '${REMOTE_MEMNAV_PY}' -c \"import json; p=json.load(open('${schedule_stage}/shards.json')); assert p['benchmark_manifest_sha256']=='${EXPECTED_AB_MANIFEST_SHA}'; assert p['candidate_histories']==99; assert p['source_recipient_histories']==61; assert p['scene_clusters']==35; assert p['maximum_histories_per_shard']==2; assert p['shard_count']==59; assert p['all_candidates_partitioned_once'] is True; assert p['query_policy_outcomes_read'] is False; assert p['navigation_outcomes_read'] is False\"; \
    cd '${schedule_stage}' && sha256sum -c --quiet shards.json.sha256; \
    chmod -R a-w '${schedule_stage}'; mv '${schedule_stage}' '${schedule}'"
fi

shard_manifest=${schedule}/shards.json
shard_manifest_sha=$(remote_query "sha256sum '${shard_manifest}' | awk '{print \$1}'" | tr -d '\r')
[[ "${shard_manifest_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "bad shard manifest digest"
shard_count=$(remote_query "'${REMOTE_MEMNAV_PY}' -c \"import json; print(json.load(open('${shard_manifest}'))['shard_count'])\"" | tr -d '\r')
[[ "${shard_count}" == 59 ]] || fail "sealed shard count changed"
array_spec=0-$((shard_count - 1))%${CONCURRENCY}

b_script=${task_root}/MemNavData/slurm_hm3d_fullmono_lifelong_natural_v4_collect_b_shard.sbatch
if [[ -n "${REPAIR_OF_JOB}" ]]; then
  [[ "${REPAIR_OF_JOB}" == 16471189 ]] || fail "unexpected repair parent"
  runtime_attempt_prefix=parserfix1_shard
  receipt=MemNavData/HM3D_FULLMONO_LIFELONG_NATURAL_V4_FACTUAL_B_PARSER_REPAIR_SUBMISSION_20260828.json
  remote_receipt=factual_b_parser_repair_submission.json
else
  runtime_attempt_prefix=shard
  receipt=MemNavData/HM3D_FULLMONO_LIFELONG_NATURAL_V4_FACTUAL_B_SUBMISSION_20260827.json
  remote_receipt=factual_b_submission.json
fi
common="ALL,TASK_ROOT=${task_root},SERVER_SOURCE_ROOT=${SERVER_SOURCE_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_ROOT=${PARENT_ROOT},PROTOCOL=${PROTOCOL},SHARD_MANIFEST=${shard_manifest},EXPECTED_SHARD_MANIFEST_SHA=${shard_manifest_sha},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${task_receipt_sha},SERVER_SOURCE_RECEIPT=${SERVER_SOURCE_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${EXPECTED_SERVER_SOURCE_RECEIPT_SHA},BASE_RECEIPT=${BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${EXPECTED_BASE_RECEIPT_SHA},RUNTIME_ATTEMPT_PREFIX=${runtime_attempt_prefix}"
remote "source '${task_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --test-only --qos=gpu48 --array=0 --export='${common}' '${b_script}' >/dev/null"
b_job=$(remote "source '${task_root}/MemNavData/slurm_safe_submit.sh'; safe_sbatch --lint-fatal --parsable --qos=gpu48 --array='${array_spec}' --export='${common}' '${b_script}'" | job_id)
[[ "${b_job}" =~ ^[0-9]+$ ]] || fail "bad factual-B array job id"

"${LOCAL_MEMNAV_PY}" - "${receipt}" "${task_root}" "${task_receipt_sha}" \
  "${shard_manifest}" "${shard_manifest_sha}" "${array_spec}" "${b_job}" \
  "${REPAIR_OF_JOB:-none}" "${runtime_attempt_prefix}" <<'PY'
import json, sys
path, task, task_sha, shards, shards_sha, array, job, repair_of, runtime_prefix = sys.argv[1:]
payload = {
    "schema_version": "hm3d_fullmono_lifelong_natural_v4_factual_b_submission_v1_20260827",
    "scope": "exactly-once factual mono B collection only",
    "run_root": "/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fullmono_lifelong_natural_v4_20260827/formal_materialize_20260827T133704Z_d85fc50d",
    "task_bundle": task,
    "task_bundle_receipt_sha256": task_sha,
    "materialization_independent_verification_sha256": "77b802f234bbbd6ad588a7bfa6069897699d4d41a3140b089b3a56498222df26",
    "shard_manifest": shards,
    "shard_manifest_sha256": shards_sha,
    "candidate_histories": 99,
    "source_recipient_histories": 61,
    "scene_clusters": 35,
    "maximum_histories_per_shard": 2,
    "shard_count": 59,
    "array": array,
    "concurrency": 4,
    "factual_B_array_job": int(job),
    "repair_of_factual_B_array_job": (
        None if repair_of == "none" else int(repair_of)
    ),
    "repair_reason": (
        None if repair_of == "none" else
        "direct server-bundle script path shadowed the v4 protocol parser"
    ),
    "superseded_factual_B_completions": 0,
    "runtime_attempt_prefix": runtime_prefix,
    "controller": "frozen NavDP native_sidecar",
    "navdp_depth_source": "monocular_sidecar",
    "maximum_steps": 600,
    "query_policy_outcomes_read_at_submission": False,
    "navigation_outcomes_read_at_submission": False,
    "prefix_or_C_B2_C2_jobs_submitted": False,
    "downstream_submission_policy": "resident-sequential after factual-B leaves gpu48 queue",
}
open(path, "x").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
timeout 120 scp -q -o BatchMode=yes -o ControlMaster=no \
  -o ControlPath="${SSH_CONTROL_PATH}" "${receipt}" \
  "${SSH_ALIAS}:${RUN_ROOT}/${remote_receipt}"
remote "sha256sum '${RUN_ROOT}/${remote_receipt}' >'${RUN_ROOT}/${remote_receipt}.sha256'; chmod a-w '${RUN_ROOT}/${remote_receipt}' '${RUN_ROOT}/${remote_receipt}.sha256'; squeue -j '${b_job}' -o '%.18i %.24j %.2t %.10M %.40R'"
printf 'TASK_ROOT=%s\nSHARD_MANIFEST=%s\nFACTUAL_B_JOB=%s\nARRAY=%s\n' \
  "${task_root}" "${shard_manifest}" "${b_job}" "${array_spec}"
