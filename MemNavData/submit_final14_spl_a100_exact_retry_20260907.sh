#!/usr/bin/env bash
# Preserve SIGABRT attempts and replay exactly those whole paired histories.
set -euo pipefail
[[ "$(id -un)" == yz11502 ]]
export REPLAY_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/final14_spl_replay_50eb1204020a4d7c
export RUN_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_table3_exact_spl_20260907
source "${REPLAY_ROOT}/MemNavData/final14_spl_replay_env.sh"
source "${REPLAY_ROOT}/MemNavData/slurm_safe_submit.sh"
repair=${RUN_ROOT}/repairs/a100_exact_retry1_20260907
indices=(6 7 8 11 12 14)
# Index 18 was already cancelled with zero runtime at 23:11:16 EDT; its
# controller record was inspected before expiry. It never created an output.
pending_indices=(19 20)
array_spec=6,7,8,11,12,14,18,19,20%2
gpu=${REPLAY_ROOT}/MemNavData/slurm_final14_spl_replay.sbatch
cpu=${REPLAY_ROOT}/MemNavData/slurm_final14_spl_replay_cpu.sbatch
[[ ! -e "${repair}/submission.jobs" && ! -e "${repair}/failed_attempts.sha256" ]]
[[ ! -e "${RUN_ROOT}/tasks/18" ]]
[[ "$(sha256sum "${REPLAY_ROOT}/source_inputs.sha256" | awk '{print $1}')" == \
   50eb1204020a4d7c38e34269cfb024989a1fee0fac6e9848792619c2472e36d5 ]]
(cd "${REPLAY_ROOT}" && sha256sum -c --quiet source_inputs.sha256)

declare -A labels
for index in "${indices[@]}"; do
  job=17057432_${index}
  state=$(sacct -j "${job}" -X -n -P --format=JobID,State |
    awk -F '|' -v wanted="${job}" '$1==wanted {print $2}')
  [[ "${state}" == FAILED ]]
  suffix=$(jq -r --argjson i "${index}" '.episodes[$i] | .scene+"_"+.episode' \
    "${BENCH_ROOT}/manifest.json")
  labels[$index]=$(printf '%03d_%s' "${index}" "${suffix}")
  [[ -d "${RUN_ROOT}/tasks/${index}" ]]
  [[ ! -e "${RUN_ROOT}/tasks/${index}/independent_verification.json" ]]
  [[ -d "${RUN_ROOT}/evaluation/natural_direction/${labels[$index]}" ]]
  [[ ! -e "${RUN_ROOT}/evaluation/natural_direction/${labels[$index]}/completion.json" ]]
  grep -qF 'evaluator failed (-6)' "${RUN_ROOT}/tasks/${index}/logs/evaluator.log"
done
for index in "${pending_indices[@]}"; do
  state=$(scontrol show job -o "17057432_${index}" | tr ' ' '\n' | sed -n 's/^JobState=//p')
  [[ ( "${state}" == PENDING || "${state}" == CANCELLED ) && ! -e "${RUN_ROOT}/tasks/${index}" ]]
done
safe_sbatch --lint-fatal --test-only --partition=a100_tandon --array="${array_spec}" "${gpu}"
safe_sbatch --lint-fatal --test-only --partition=cpu_short --export=ALL,MODE=summary "${cpu}"

mkdir -p "${repair}/failed_attempts/tasks" "${repair}/failed_attempts/evaluation"
cp -- "${BASH_SOURCE[0]}" "${repair}/submission_script_final.sh"
cp -- "$(dirname "${BASH_SOURCE[0]}")/FINAL14_SPL_A100_EXACT_RETRY_PROTOCOL_20260907.md" \
  "${repair}/protocol.md"
for index in "${pending_indices[@]}"; do
  state=$(scontrol show job -o "17057432_${index}" | tr ' ' '\n' | sed -n 's/^JobState=//p')
  if [[ "${state}" == PENDING ]]; then scancel --state=PENDING "17057432_${index}"; fi
  # Controller state is current; sacct may not yet contain a never-started task.
  state=$(scontrol show job -o "17057432_${index}" | tr ' ' '\n' | sed -n 's/^JobState=//p')
  [[ "${state}" == CANCELLED && ! -e "${RUN_ROOT}/tasks/${index}" ]]
done
for index in "${indices[@]}"; do
  mv -- "${RUN_ROOT}/tasks/${index}" "${repair}/failed_attempts/tasks/${index}"
  mv -- "${RUN_ROOT}/evaluation/natural_direction/${labels[$index]}" \
    "${repair}/failed_attempts/evaluation/${labels[$index]}"
done
(cd "${repair}" && find failed_attempts -type f -print0 | sort -z | xargs -0 sha256sum \
  > failed_attempts.sha256)

retry=$(safe_sbatch --lint-fatal --parsable --partition=a100_tandon --time=01:00:00 \
  --array="${array_spec}" --export=ALL "${gpu}")
retry=${retry%%;*}
printf 'retry=%s\n' "${retry}" | tee "${repair}/submission.jobs"
summary=$(safe_sbatch --lint-fatal --parsable --partition=cpu_short --export=ALL,MODE=summary \
  --dependency="afterany:17057431:17057432,afterok:${retry}" "${cpu}")
summary=${summary%%;*}
printf 'summary=%s\n' "${summary}" | tee -a "${repair}/submission.jobs"
old_state=$(sacct -j 17057433 -X -n -P --format=JobID,State |
  awk -F '|' '$1=="17057433" {print $2}')
if [[ "${old_state}" == PENDING ]]; then
  scancel 17057433
  printf 'superseded_pending_summary=17057433\n' | tee -a "${repair}/submission.jobs"
fi
squeue -u yz11502 -o '%.30i %.14P %.22j %.10T %.10M %.10l %R'
