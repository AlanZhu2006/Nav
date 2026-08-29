#!/usr/bin/env bash
# Run inside the already authenticated yz11502 HPC PTY after publishing this
# script's immutable wrapper bundle. It never reads navigation outcomes.
set -euo pipefail
umask 0022
export PYTHONDONTWRITEBYTECODE=1

WRAPPER_ROOT=${WRAPPER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WRAPPER_RECEIPT=${WRAPPER_ROOT}/SOURCE_BUNDLE.sha256
WRAPPER_RECEIPT_SHA=$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')
TASK_NAV=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/mp3d_table1_controller_portability_eb7cdf82477f6aa1
TASK_NAV_SHA=eb7cdf82477f6aa192b5becf42c85e93d490de7bfe7564132bbff285910a32c4
TASK_VINT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/mp3d_table1_controller_portability_7f1713660eee6f5c
TASK_VINT_SHA=7f1713660eee6f5c4893c46b8705767a5e64c117c0ac0aa55fa2f80ef504169e
RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_controller_portability_20260829/formal_20260829T085025Z
BENCH=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5/population/natural_direction
CONSTRUCTION=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5/mp3d_table1_new_query_verification.json
CONSTRUCTION_SHA=618c409f7c7c62ad739687935cdd6f2e564e96aed6ccf6059d887d795c3e953e
NAV_BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
NAV_BASE_RECEIPT=${NAV_BASE}/source_inputs.sha256
NAV_BASE_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
NAV_SERVER=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_table1_navdp_cache_repair_2ae34ad0c1503958
NAV_SERVER_RECEIPT=${NAV_SERVER}/SOURCE_BUNDLE.sha256
NAV_SERVER_SHA=2ae34ad0c150395849d4461913fc086f3b6ea7acf7249c763fe3e8808356ed6d
SOURCE_RUN=/scratch/yz11502/Research/Nav-axis-uturn-results/mdtec_monocular_cec_composition_20260819/formal_20260819T055600Z_624f9fa9
PARENT=/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_fullmono_source_expansion_20260829/source_expansion_20260829T060541Z_f3e7c3e5/population/parent_manifest.json
PROTOCOL=${TASK_NAV}/MemNavData/mp3d_table1_new_query_protocol_20260829.json
VINT_BASE=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80
VINT_BASE_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
DEPENDENCY=/scratch/yz11502/Research/Nav-axis-uturn-results/shared_online_double_revisit_fresh_20260813/double_revisit_fresh40_20260813T200121Z/dependency_receipt.json
DEPENDENCY_SHA=4eb0ca6479a26f8e04f85a31d906cee4e68b1785f66cfd3ac23bf65424d36e5e
PORT_ENV=/scratch/yz11502/Research/Nav-axis-uturn-envs/controller_portability_a9ec7146bce7_v1
PORT_CKPT=/scratch/yz11502/Research/Nav-axis-uturn-checkpoints/controller_portability_50387aa89be8
REPAIR_TAG=mp3d_table1_controller_exact_retry1_20260829
REPAIR_ROOT=${RUN}/repairs/${REPAIR_TAG}
ARCHIVE=${REPAIR_ROOT}/failed_attempts
NAV_PARTIAL=${RUN}/formal/navdp/evaluation/natural_direction/029_kEZ7cmS4wCh_episode_0004
VINT_PARTIAL=${RUN}/formal/vint/evaluation/024_pRbA3pwrgk9_episode_0004/vint
NAV_RUNTIME=${RUN}/formal/navdp/runtime/eval_27
NAV_WRAP=${WRAPPER_ROOT}/MemNavData/slurm_hm3d_table1_navdp_pair.sbatch
VINT_WRAP=${WRAPPER_ROOT}/MemNavData/slurm_hm3d_table1_vint_pair.sbatch
NAV_ANALYSIS=${TASK_NAV}/MemNavData/slurm_hm3d_table1_navdp_analysis.sbatch
VINT_ANALYSIS=${TASK_VINT}/MemNavData/slurm_hm3d_table1_vint_analysis.sbatch
SEAL=${TASK_VINT}/MemNavData/slurm_hm3d_table1_controller_seal.sbatch
PY=/scratch/lg154/conda-envs/memnav/bin/python
DRY_RUN=${DRY_RUN:-0}
fail() { echo "ABORT: $*" >&2; exit 2; }
job_id() { awk -F';' 'NR==1{print $1}'; }
assert_job() {
  local id=$1 state=$2 exit_code=$3
  sacct -X -j "${id}" -n -o State,ExitCode | \
    awk -v s="${state}" -v e="${exit_code}" \
      '$1==s && $2==e{ok=1} END{exit !ok}' || \
    fail "job ${id} is not ${state} ${exit_code}"
}

[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "$(id -un)" == yz11502 ]] || fail "wrong HPC identity"
[[ "$(sha256sum "${WRAPPER_RECEIPT}" | awk '{print $1}')" == \
   "${WRAPPER_RECEIPT_SHA}" ]] || fail "wrapper receipt changed"
(cd "${WRAPPER_ROOT}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "wrapper bundle changed"
[[ "$(sha256sum "${TASK_NAV}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${TASK_NAV_SHA}" ]] || fail "NavDP task receipt changed"
(cd "${TASK_NAV}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "NavDP task bundle changed"
[[ "$(sha256sum "${TASK_VINT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${TASK_VINT_SHA}" ]] || fail "ViNT task receipt changed"
(cd "${TASK_VINT}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "ViNT task bundle changed"
[[ "$(sha256sum "${NAV_SERVER_RECEIPT}" | awk '{print $1}')" == \
   "${NAV_SERVER_SHA}" ]] || fail "cache-repair overlay receipt changed"
(cd "${NAV_SERVER}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "cache-repair overlay changed"
[[ "$(sha256sum "${NAV_BASE_RECEIPT}" | awk '{print $1}')" == \
   "${NAV_BASE_SHA}" ]] || fail "NavDP base receipt changed"
(cd "${NAV_BASE}" && sha256sum -c --quiet "${NAV_BASE_RECEIPT}") || \
  fail "NavDP base bundle changed"
[[ "$(sha256sum "${VINT_BASE}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${VINT_BASE_SHA}" ]] || fail "ViNT base receipt changed"
(cd "${VINT_BASE}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256) || \
  fail "ViNT base bundle changed"
[[ "$(sha256sum "${DEPENDENCY}" | awk '{print $1}')" == \
   "${DEPENDENCY_SHA}" ]] || fail "dependency receipt changed"
(cd "${PORT_CKPT}" && sha256sum -c --quiet CHECKPOINTS.sha256) || \
  fail "portability checkpoints changed"
[[ -r "${PORT_ENV}/environment_receipt.json" ]] || fail "portability env missing"
[[ "$(sha256sum "${CONSTRUCTION}" | awk '{print $1}')" == \
   "${CONSTRUCTION_SHA}" ]] || fail "construction verifier changed"

assert_job 16548405_27 FAILED 1:0
assert_job 16548592_24 FAILED 2:0
for id in 16548433 16548444 16548600 16548605 16548606; do
  assert_job "${id}" CANCELLED 0:0
done
grep -q 'cached monocular depth belongs to a different transaction' \
  "${NAV_RUNTIME}/logs/server_navdp.log" || fail "NavDP failure signature changed"
grep -q 'Address already in use' \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/h3T1ViNT_16548592_24.err || \
  fail "ViNT failure signature changed"
grep -q 'FORMAL_INDICES_OVERRIDE' \
  "${TASK_NAV}/MemNavData/run_hm3d_fullmono_server_scene.sh" || \
  fail "frozen NavDP runner lacks exact override"
grep -q 'The cache is keyed by image digest' \
  "${NAV_SERVER}/NavDP/baselines/navdp/navdp_server.py" || \
  fail "verified identical-frame cache fix missing"

"${PY}" - "${RUN}" "${BENCH}/manifest.json" <<'PY'
import json,pathlib,sys
run=pathlib.Path(sys.argv[1]); manifest=json.load(open(sys.argv[2]))
nav=[]; vint=[]
for i,row in enumerate(manifest['episodes']):
    label=f"{i:03d}_{row['scene']}_{row['episode']}"
    n=run/'formal/navdp/evaluation/natural_direction'/label/'completion.json'
    v=run/'formal/vint/evaluation'/label/'vint/controller_native_pair_audit.json'
    if not n.is_file(): nav.append(i)
    if not v.is_file(): vint.append(i)
if nav != [29,30] or vint != [24]:
    raise SystemExit(f'missing receipt set changed: nav={nav} vint={vint}')
for path in (
    run/'formal/navdp/navdp_table1_summary.json',
    run/'formal/navdp/navdp_table1_independent_verification.json',
    run/'formal/vint/vint_table1_summary.json',
    run/'formal/vint/vint_table1_independent_verification.json',
    run/'mp3d_table1_controller_portability_receipt.json'):
    if path.exists(): raise SystemExit('downstream output already exists: '+str(path))
PY
[[ -d "${NAV_PARTIAL}" && -d "${VINT_PARTIAL}" ]] || \
  fail "expected partial directory missing"
[[ ! -e "${REPAIR_ROOT}" ]] || fail "repair root already exists"

source "${WRAPPER_ROOT}/MemNavData/slurm_safe_submit.sh"
lint_sbatch_template "${NAV_WRAP}" || fail "NavDP wrapper lint failed"
lint_sbatch_template "${VINT_WRAP}" || fail "ViNT wrapper lint failed"

wrapper="WRAPPER_ROOT=${WRAPPER_ROOT},WRAPPER_RECEIPT=${WRAPPER_RECEIPT},EXPECTED_WRAPPER_RECEIPT_SHA=${WRAPPER_RECEIPT_SHA}"
nav_common="ALL,TASK_ROOT=${TASK_NAV},TASK_RECEIPT=${TASK_NAV}/SOURCE_BUNDLE.sha256,EXPECTED_TASK_RECEIPT_SHA=${TASK_NAV_SHA},FORMAL_RUN_ROOT=${RUN},BENCH_ROOT=${BENCH},CONSTRUCTION_VERIFICATION=${CONSTRUCTION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${CONSTRUCTION_SHA},DATASET=MP3D,CLAIM_SCOPE=conference_table_mp3d_reused_scene_history_new_query_replication,BASE_SOURCE_ROOT=${NAV_BASE},BASE_RECEIPT=${NAV_BASE_RECEIPT},EXPECTED_BASE_RECEIPT_SHA=${NAV_BASE_SHA},SERVER_SOURCE_ROOT=${NAV_SERVER},SERVER_SOURCE_RECEIPT=${NAV_SERVER_RECEIPT},EXPECTED_SERVER_SOURCE_RECEIPT_SHA=${NAV_SERVER_SHA},SOURCE_RUN_ROOT=${SOURCE_RUN},PARENT_MANIFEST=${PARENT},PROTOCOL=${PROTOCOL},ROLE_PAIR_SCOPE=paper_replication,${wrapper}"
vint_common="ALL,TASK_ROOT=${TASK_VINT},TASK_RECEIPT=${TASK_VINT}/SOURCE_BUNDLE.sha256,EXPECTED_TASK_RECEIPT_SHA=${TASK_VINT_SHA},FORMAL_RUN_ROOT=${RUN},BENCH_ROOT=${BENCH},CONSTRUCTION_VERIFICATION=${CONSTRUCTION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${CONSTRUCTION_SHA},DATASET=MP3D,CLAIM_SCOPE=conference_table_mp3d_reused_scene_history_new_query_replication,BASE_SOURCE_ROOT=${VINT_BASE},BASE_SOURCE_RECEIPT_SHA=${VINT_BASE_SHA},DEPENDENCY_RECEIPT=${DEPENDENCY},EXPECTED_DEPENDENCY_RECEIPT_SHA=${DEPENDENCY_SHA},PORTABILITY_ENV_ROOT=${PORT_ENV},PORTABILITY_CHECKPOINT_ROOT=${PORT_CKPT},ROLE_PAIR_SCOPE=paper_replication,${wrapper}"
nav_analysis_common="ALL,TASK_ROOT=${TASK_NAV},TASK_RECEIPT=${TASK_NAV}/SOURCE_BUNDLE.sha256,EXPECTED_TASK_RECEIPT_SHA=${TASK_NAV_SHA},FORMAL_RUN_ROOT=${RUN},BENCH_ROOT=${BENCH},CONSTRUCTION_VERIFICATION=${CONSTRUCTION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${CONSTRUCTION_SHA},DATASET=MP3D,CLAIM_SCOPE=conference_table_mp3d_reused_scene_history_new_query_replication"
vint_analysis_common="ALL,TASK_ROOT=${TASK_VINT},TASK_RECEIPT=${TASK_VINT}/SOURCE_BUNDLE.sha256,EXPECTED_TASK_RECEIPT_SHA=${TASK_VINT_SHA},FORMAL_RUN_ROOT=${RUN},BENCH_ROOT=${BENCH},CONSTRUCTION_VERIFICATION=${CONSTRUCTION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${CONSTRUCTION_SHA},DATASET=MP3D,CLAIM_SCOPE=conference_table_mp3d_reused_scene_history_new_query_replication"
seal_common="ALL,FORMAL_RUN_ROOT=${RUN},CONSTRUCTION_VERIFICATION=${CONSTRUCTION},EXPECTED_CONSTRUCTION_VERIFICATION_SHA=${CONSTRUCTION_SHA},DATASET=MP3D,CLAIM_SCOPE=conference_table_mp3d_reused_scene_history_new_query_replication"

sbatch --test-only --array=27 \
  --export="${nav_common},PHASE=formal,EXACT_REPAIR=1,RUNTIME_ATTEMPT=mp3d_t1_navdp_exact_retry1_20260829,FORMAL_INDICES_SPEC=29:30" \
  "${NAV_WRAP}" >/dev/null
sbatch --test-only --array=24 --export="${vint_common},PHASE=formal" \
  "${VINT_WRAP}" >/dev/null
sbatch --test-only --export="${nav_analysis_common},MODE=aggregate" \
  "${NAV_ANALYSIS}" >/dev/null
sbatch --test-only --export="${vint_analysis_common},MODE=aggregate" \
  "${VINT_ANALYSIS}" >/dev/null
sbatch --test-only --export="${seal_common}" "${SEAL}" >/dev/null

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_OK navdp=29,30 vint=24 wrapper=${WRAPPER_ROOT}"
  exit 0
fi

mkdir -p "${ARCHIVE}/navdp" "${ARCHIVE}/vint"
mv "${NAV_PARTIAL}" "${ARCHIVE}/navdp/029_kEZ7cmS4wCh_episode_0004"
mv "${VINT_PARTIAL}" "${ARCHIVE}/vint/024_pRbA3pwrgk9_episode_0004_vint"
"${PY}" - "${ARCHIVE}" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); files=[]
for path in sorted(root.rglob('*')):
    if path.is_file():
        files.append({'path':path.relative_to(root).as_posix(),
                      'bytes':path.stat().st_size,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
payload={'schema_version':'mp3d_table1_controller_failed_partials_v1_20260829',
         'files':files,'deleted':False,
         'navigation_outcomes_used_for_repair_selection':False}
path=root/'archive_manifest.json'
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${ARCHIVE}/archive_manifest.json" >"${ARCHIVE}/archive_manifest.json.sha256"
chmod -R a-w "${ARCHIVE}"

nav_job=$(safe_sbatch --lint-fatal --parsable --job-name=m3T1NavR1 --array=27 \
  --export="${nav_common},PHASE=formal,EXACT_REPAIR=1,RUNTIME_ATTEMPT=mp3d_t1_navdp_exact_retry1_20260829,FORMAL_INDICES_SPEC=29:30" \
  "${NAV_WRAP}" | job_id)
vint_job=$(safe_sbatch --lint-fatal --parsable --job-name=m3T1ViNR1 --array=24 \
  --export="${vint_common},PHASE=formal" "${VINT_WRAP}" | job_id)
[[ "${nav_job}" =~ ^[0-9]+$ && "${vint_job}" =~ ^[0-9]+$ ]] || \
  fail "invalid exact-retry job id"
nav_agg=$(sbatch --parsable --dependency=afterok:${nav_job} \
  --kill-on-invalid-dep=yes --export="${nav_analysis_common},MODE=aggregate" \
  "${NAV_ANALYSIS}" | job_id)
nav_verify=$(sbatch --parsable --dependency=afterok:${nav_agg} \
  --kill-on-invalid-dep=yes --export="${nav_analysis_common},MODE=verify" \
  "${NAV_ANALYSIS}" | job_id)
vint_agg=$(sbatch --parsable --dependency=afterok:${vint_job} \
  --kill-on-invalid-dep=yes --export="${vint_analysis_common},MODE=aggregate" \
  "${VINT_ANALYSIS}" | job_id)
vint_verify=$(sbatch --parsable --dependency=afterok:${vint_agg} \
  --kill-on-invalid-dep=yes --export="${vint_analysis_common},MODE=verify" \
  "${VINT_ANALYSIS}" | job_id)
seal_job=$(sbatch --parsable --dependency=afterok:${nav_verify}:${vint_verify} \
  --kill-on-invalid-dep=yes --export="${seal_common}" "${SEAL}" | job_id)
for id in "${nav_agg}" "${nav_verify}" "${vint_agg}" "${vint_verify}" \
          "${seal_job}"; do
  [[ "${id}" =~ ^[0-9]+$ ]] || fail "invalid downstream job id: ${id}"
done

"${PY}" - "${REPAIR_ROOT}/repair_submission.json" \
  "${WRAPPER_ROOT}" "${WRAPPER_RECEIPT_SHA}" "${nav_job}" "${vint_job}" \
  "${nav_agg}" "${nav_verify}" "${vint_agg}" "${vint_verify}" \
  "${seal_job}" "${ARCHIVE}" <<'PY'
import hashlib,json,os,sys
(path,bundle,bundle_sha,nav_job,vint_job,nav_agg,nav_verify,vint_agg,
 vint_verify,seal,archive)=sys.argv[1:]
archive_manifest=os.path.join(archive,'archive_manifest.json')
payload={
 'schema_version':'mp3d_table1_controller_exact_repair_submission_v1_20260829',
 'run_root':'/scratch/yz11502/Research/Nav-axis-uturn-results/mp3d_table1_controller_portability_20260829/formal_20260829T085025Z',
 'original_jobs':{'navdp_array':16548405,'vint_array':16548592},
 'superseded_jobs':{'navdp_aggregate':16548433,'navdp_verify':16548444,
                    'vint_aggregate':16548600,'vint_verify':16548605,
                    'joint_seal':16548606},
 'exact_repair':{'navdp_scene_rank':27,'navdp_histories':[29,30],
                 'vint_histories':[24]},
 'jobs':{'navdp_exact_retry':int(nav_job),'vint_exact_retry':int(vint_job),
         'navdp_aggregate':int(nav_agg),'navdp_verify':int(nav_verify),
         'vint_aggregate':int(vint_agg),'vint_verify':int(vint_verify),
         'joint_seal':int(seal)},
 'wrapper_bundle':bundle,'wrapper_receipt_sha256':bundle_sha,
 'archive_root':archive,
 'archive_manifest_sha256':hashlib.sha256(open(archive_manifest,'rb').read()).hexdigest(),
 'method_or_population_changed':False,
 'partial_outcome_visibility_incident':True,
 'incident_scope':'one already-completed NavDP history',
 'repair_selection_influenced_by_incident':False,
 'partial_aggregate_or_sr_computed':False,
}
fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(fd,'w') as handle:
    json.dump(payload,handle,indent=2,sort_keys=True); handle.write('\n')
PY
sha256sum "${REPAIR_ROOT}/repair_submission.json" \
  >"${REPAIR_ROOT}/repair_submission.json.sha256"
chmod a-w "${REPAIR_ROOT}/repair_submission.json" \
  "${REPAIR_ROOT}/repair_submission.json.sha256"
printf 'NAV_REPAIR=%s VINT_REPAIR=%s NAV_AGG=%s NAV_VERIFY=%s VINT_AGG=%s VINT_VERIFY=%s SEAL=%s\n' \
  "${nav_job}" "${vint_job}" "${nav_agg}" "${nav_verify}" \
  "${vint_agg}" "${vint_verify}" "${seal_job}"
