#!/usr/bin/env bash
# Exact-index repair for the fresh HM3D Full-Mono QUERY evaluation array
# (16131090) plus afterok-chained summary/verify resubmission.
#
# Why: eval task 36 FAILED (one transaction fail-fast on history 019's second
# query -- treated as transient per the transaction-repair convention unless
# it reproduces) and task 6 was QoS-cancelled.  The original summary/verify
# (16131095/16131098) depend afterok on the full array and can never run;
# with --kill-on-invalid-dep they self-cancel at array terminal.
#
# Pattern follows the project's additive-repair precedent: identical immutable
# task bundle, new runtime attempt tag, resume-incomplete collector, never
# overwriting completed outputs.  The only new artifact is a thin sbatch
# wrapper that forwards RESUME_INCOMPLETE/RUNTIME_ATTEMPT (the bundled eval
# sbatch's singularity env list drops them) and then executes the unchanged
# immutable runner.
set -euo pipefail

SSH=${SSH:-alantorch}
RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6
TASK_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/hm3d_fullmono_transaction_repair_27132081e26acfb9
TASK_RECEIPT_SHA=deea747eb7c8aad79a3dd76ab6fab6542ad987d1b768d4b7453e30032200da2e
BASE_SOURCE_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_mono_factorial_5690569a4373f2d2
BASE_RECEIPT_SHA=5690569a4373f2d2768671418f0c604c4a03aa4b0ffe01baf70b288af03ba216
ORIG_ARRAY=16131090
OLD_SUMMARY=16131095
OLD_VERIFY=16131098
ATTEMPT=${ATTEMPT:-evalrepair1}
CONCURRENCY=${CONCURRENCY:-2}

remote() { ssh "$SSH" "$1"; }

# 1. The original array must be fully terminal.
still=$(remote "squeue -j ${ORIG_ARRAY} -h -o '%T' 2>/dev/null || true" | wc -l)
[[ "$still" -eq 0 ]] || { echo "array ${ORIG_ARRAY} still has $still live elements; wait" >&2; exit 2; }

# 2. Repair set = indices whose newest sacct state is not COMPLETED.
indices=$(remote "sacct -j ${ORIG_ARRAY} --format=JobID%20,State -X -n" \
  | awk '{split($1, a, "_"); print a[2], $2}' \
  | awk '$1 ~ /^[0-9]+$/ && $2 != "COMPLETED" && $2 != "RUNNING" && $2 != "PENDING" {print $1}' \
  | sort -n | uniq | paste -sd,)
[[ -n "$indices" ]] || { echo "no failed/missing indices; nothing to repair" >&2; exit 0; }
echo "repair indices: $indices"

# 3. Stage the thin wrapper sbatch next to the run root (writable area).
remote "mkdir -p '${RUN_ROOT}/eval_repair'"
remote "cat > '${RUN_ROOT}/eval_repair/eval_repair_wrapper.sbatch' <<'WRAP'
#!/usr/bin/env bash
#SBATCH --job-name=h3monoEvalR
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=72G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced
set -euo pipefail
: \"\${TASK_ROOT:?}\" \"\${BASE_SOURCE_ROOT:?}\" \"\${RUN_ROOT:?}\"
: \"\${PARENT_MANIFEST:?}\" \"\${PROTOCOL:?}\" \"\${BENCH_ROOT:?}\"
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \\
  \"\${BASE_SIF}\" env MODE=eval TASK_ROOT=\"\${TASK_ROOT}\" \\
  BASE_SOURCE_ROOT=\"\${BASE_SOURCE_ROOT}\" RUN_ROOT=\"\${RUN_ROOT}\" \\
  PARENT_MANIFEST=\"\${PARENT_MANIFEST}\" PROTOCOL=\"\${PROTOCOL}\" \\
  BENCH_ROOT=\"\${BENCH_ROOT}\" SCENE_INDEX=\"\${SLURM_ARRAY_TASK_ID}\" \\
  MAX_STEPS=600 \\
  RESUME_INCOMPLETE=1 RUNTIME_ATTEMPT=\"\${RUNTIME_ATTEMPT}\" \\
  HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python \\
  MEMNAV_PY=/scratch/lg154/conda-envs/memnav/bin/python \\
  TASK_RECEIPT=\"\${TASK_RECEIPT}\" \\
  EXPECTED_TASK_RECEIPT_SHA=\"\${EXPECTED_TASK_RECEIPT_SHA}\" \\
  BASE_RECEIPT=\"\${BASE_RECEIPT}\" \\
  EXPECTED_BASE_RECEIPT_SHA=\"\${EXPECTED_BASE_RECEIPT_SHA}\" \\
  bash \"\${TASK_ROOT}/MemNavData/run_hm3d_fullmono_server_scene.sh\"
WRAP"

parent_manifest=${RUN_ROOT}/sealed_inputs/parent_manifest.json
protocol=${TASK_ROOT}/MemNavData/hm3d_fresh_fullmono_mixed_role_protocol_20260820.json
task_receipt=${TASK_ROOT}/SOURCE_BUNDLE.sha256
bench_root=${RUN_ROOT}/benchmarks/natural_direction
common="ALL,TASK_ROOT=${TASK_ROOT},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},RUN_ROOT=${RUN_ROOT},PARENT_MANIFEST=${parent_manifest},PROTOCOL=${protocol},TASK_RECEIPT=${task_receipt},EXPECTED_TASK_RECEIPT_SHA=${TASK_RECEIPT_SHA},BASE_RECEIPT=${BASE_SOURCE_ROOT}/source_inputs.sha256,EXPECTED_BASE_RECEIPT_SHA=${BASE_RECEIPT_SHA}"

# 4. Clear the doomed summary/verify if still queued, then submit the chain.
remote "scancel ${OLD_SUMMARY} ${OLD_VERIFY} 2>/dev/null || true"
repair_id=$(remote "sbatch --parsable --qos=gpu48 --array=${indices}%${CONCURRENCY} --export='${common},BENCH_ROOT=${bench_root},RUNTIME_ATTEMPT=${ATTEMPT}' '${RUN_ROOT}/eval_repair/eval_repair_wrapper.sbatch'" | tr -d '\r')
summary_id=$(remote "sbatch --parsable --dependency=afterok:${repair_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=summary' '${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_analysis.sbatch'" | tr -d '\r')
verify_id=$(remote "sbatch --parsable --dependency=afterok:${summary_id} --kill-on-invalid-dep=yes --export='${common},BENCH_ROOT=${bench_root},MODE=verify' '${TASK_ROOT}/MemNavData/slurm_hm3d_fullmono_analysis.sbatch'" | tr -d '\r')

remote "cat > '${RUN_ROOT}/eval_repair/submission.json' <<EOF
{
  \"schema_version\": \"hm3d_fullmono_eval_repair_submission_v1_20260822\",
  \"repair_indices\": \"${indices}\",
  \"runtime_attempt\": \"${ATTEMPT}\",
  \"task_bundle\": \"${TASK_ROOT}\",
  \"superseded_summary_jobs\": [${OLD_SUMMARY}, ${OLD_VERIFY}],
  \"jobs\": {\"eval_repair\": ${repair_id}, \"summary\": ${summary_id}, \"verify\": ${verify_id}},
  \"query_outcomes_read_before_repair\": false,
  \"completed_episode_overwrite_allowed\": false
}
EOF"
echo "eval_repair=${repair_id} summary=${summary_id} verify=${verify_id} indices=${indices}"
