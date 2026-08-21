#!/usr/bin/env bash
# Run on an authenticated HPC login node after the immutable bundle is uploaded.
set -euo pipefail
umask 0022

: "${TASK_ROOT:?set immutable task bundle root}"
: "${BASE_SOURCE_ROOT:?set immutable CEC base source root}"
: "${RUN_ROOT:?set new run root}"
: "${DATA_ROOT:?set HM3D held-out-val10 data root}"
: "${EXPECTED_TASK_RECEIPT_SHA:?set task receipt SHA}"
: "${EXPECTED_BASE_SOURCE_RECEIPT_SHA:?set base receipt SHA}"

GEN_CONCURRENCY=${GEN_CONCURRENCY:-6}
EVAL_CONCURRENCY=${EVAL_CONCURRENCY:-4}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "$(id -un)" == "${EXPECTED_REMOTE_USER}" ]] || fail "remote user differs"
[[ "${GEN_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad generation concurrency"
[[ "${EVAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || fail "bad evaluation concurrency"

readonly TASK_RECEIPT=${TASK_ROOT}/SOURCE_BUNDLE.sha256
readonly BASE_RECEIPT=${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256
readonly ARCHIVE=/scratch/yz11502/Research/datasets/goat_bench_20260814/downloads/hm3d-val-habitat-v0.2.tar
readonly PREP=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_prepare.sbatch
readonly GENERATE=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_generate.sbatch
readonly MANIFEST=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_manifest.sbatch
readonly EVAL=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_eval.sbatch
readonly SUMMARY=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_summary.sbatch
readonly VERIFY=${TASK_ROOT}/MemNavData/slurm_hm3d_heldout_val10_revisit_verify.sbatch
readonly PY=/scratch/lg154/conda-envs/memnav/bin/python

[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_TASK_RECEIPT_SHA}" ]] || fail "task receipt changed"
(cd "${TASK_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) ||
  fail "task bundle validation failed"
[[ "$(sha256sum "${BASE_RECEIPT}" | awk '{print $1}')" == \
    "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base receipt changed"
(cd "${BASE_SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null) ||
  fail "base bundle validation failed"
for path in "${ARCHIVE}" "${PREP}" "${GENERATE}" "${MANIFEST}" \
  "${EVAL}" "${SUMMARY}" "${VERIFY}" "${PY}" \
  /scratch/lg154/conda-envs/habitat/bin/python \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt \
  /scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt \
  /scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt; do
  test -r "${path}" || fail "missing runtime input ${path}"
done
[[ ! -e "${RUN_ROOT}" ]] || fail "run root already exists"
scontrol ping | grep -q 'is UP' || fail "Slurm controller is not UP"

exports="ALL,TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},DATA_ROOT=${DATA_ROOT},EXPECTED_TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"

# Validate every resource request before creating the run root or any job.
sbatch --test-only --export="${exports}" "${PREP}" >/dev/null
sbatch --test-only --array=0 --export="${exports}" "${GENERATE}" >/dev/null
sbatch --test-only --export="${exports}" "${MANIFEST}" >/dev/null
sbatch --test-only --array=0 --export="${exports}" "${EVAL}" >/dev/null
sbatch --test-only --export="${exports}" "${SUMMARY}" >/dev/null
sbatch --test-only --export="${exports}" "${VERIFY}" >/dev/null

mkdir -p "${RUN_ROOT}/logs" \
  /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs

prep_raw=$(sbatch --parsable --export="${exports}" "${PREP}")
prep_id=${prep_raw%%;*}
[[ "${prep_id}" =~ ^[0-9]+$ ]] || fail "bad prepare job ID"

generate_raw=$(sbatch --parsable --array="0-9%${GEN_CONCURRENCY}" \
  --dependency="afterok:${prep_id}" --kill-on-invalid-dep=yes \
  --export="${exports}" "${GENERATE}")
generate_id=${generate_raw%%;*}
[[ "${generate_id}" =~ ^[0-9]+$ ]] || fail "bad generation job ID"

manifest_raw=$(sbatch --parsable --dependency="afterok:${generate_id}" \
  --kill-on-invalid-dep=yes --export="${exports}" "${MANIFEST}")
manifest_id=${manifest_raw%%;*}
[[ "${manifest_id}" =~ ^[0-9]+$ ]] || fail "bad manifest job ID"

eval_raw=$(sbatch --parsable --array="0-9%${EVAL_CONCURRENCY}" \
  --dependency="afterok:${manifest_id}" --kill-on-invalid-dep=yes \
  --export="${exports}" "${EVAL}")
eval_id=${eval_raw%%;*}
[[ "${eval_id}" =~ ^[0-9]+$ ]] || fail "bad evaluation job ID"

summary_raw=$(sbatch --parsable --dependency="afterok:${eval_id}" \
  --kill-on-invalid-dep=yes --export="${exports}" "${SUMMARY}")
summary_id=${summary_raw%%;*}
[[ "${summary_id}" =~ ^[0-9]+$ ]] || fail "bad summary job ID"

verify_raw=$(sbatch --parsable --dependency="afterok:${summary_id}" \
  --kill-on-invalid-dep=yes --export="${exports}" "${VERIFY}")
verify_id=${verify_raw%%;*}
[[ "${verify_id}" =~ ^[0-9]+$ ]] || fail "bad verification job ID"

"${PY}" - "${RUN_ROOT}/submission.json" "${TASK_ROOT}" \
  "${EXPECTED_TASK_RECEIPT_SHA}" "${DATA_ROOT}" "${prep_id}" \
  "${generate_id}" "${manifest_id}" "${eval_id}" "${summary_id}" \
  "${verify_id}" "${GEN_CONCURRENCY}" "${EVAL_CONCURRENCY}" <<'PY'
import hashlib,json,os,pathlib,sys
(path,bundle,receipt,data_root,prepare,generation,manifest,evaluation,
 summary,verification,gen_concurrency,eval_concurrency)=sys.argv[1:]
protocol=pathlib.Path(bundle)/"MemNavData/hm3d_heldout_val10_revisit_protocol_20260816.json"
payload={
 "schema_version":"hm3d_heldout_val10_revisit_submission_v1_20260816",
 "objective":"non-MP3D external causal-Revisit transfer",
 "dataset":"HM3D v0.2 outcome-disjoint val10",
 "scene_count":10,"episode_count":40,"source_bundle":bundle,
 "task_receipt_sha256":receipt,
 "protocol_sha256":hashlib.sha256(protocol.read_bytes()).hexdigest(),
 "data_root":data_root,
 "guards":{"no_mp3d_evaluation":True,"intention_to_treat":True,
           "no_outcome_filtering":True,"no_hm3d_heldout_val10_tuning":True},
 "concurrency":{"generation":int(gen_concurrency),
                "evaluation":int(eval_concurrency)},
 "jobs":{"prepare":int(prepare),"generation_array":int(generation),
         "manifest":int(manifest),"evaluation_array":int(evaluation),
         "summary":int(summary),
         "independent_verification":int(verification)},
}
descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o444)
with os.fdopen(descriptor,"w",encoding="utf-8") as handle:
 json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n")
PY

echo "RUN_ROOT=${RUN_ROOT}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "TASK_BUNDLE=${TASK_ROOT}"
echo "TASK_RECEIPT_SHA=${EXPECTED_TASK_RECEIPT_SHA}"
echo "prepare=${prep_id} generation=${generate_id} manifest=${manifest_id} eval=${eval_id} summary=${summary_id} verify=${verify_id}"
squeue -j "${prep_id},${generate_id},${manifest_id},${eval_id},${summary_id},${verify_id}" \
  -o '%.18i %.14P %.24j %.2t %.10M %.6D %R'
