#!/usr/bin/env bash
# Run only through the established yz11502 shared SSH session.
set -euo pipefail
: "${REPLAY_ROOT:?}" "${RUN_ROOT:?}"
export REPLAY_ROOT RUN_ROOT
source "${REPLAY_ROOT}/MemNavData/slurm_safe_submit.sh"
[[ ! -e "${RUN_ROOT}" ]] || { echo "submission/output root already exists" >&2; exit 1; }
(cd "${REPLAY_ROOT}" && sha256sum -c --quiet source_inputs.sha256)
gpu=${REPLAY_ROOT}/MemNavData/slurm_final14_spl_replay.sbatch
cpu=${REPLAY_ROOT}/MemNavData/slurm_final14_spl_replay_cpu.sbatch
lint_sbatch_template "${gpu}"
lint_sbatch_template "${cpu}"
safe_sbatch --lint-fatal --test-only --array=17 "${gpu}"
safe_sbatch --lint-fatal --partition=cpu_short --test-only --export=ALL,MODE=preflight "${cpu}"
mkdir -p "${RUN_ROOT}" /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs
preflight=$(safe_sbatch --lint-fatal --partition=cpu_short --parsable --export=ALL,MODE=preflight "${cpu}")
preflight=${preflight%%;*}
printf 'preflight=%s\n' "${preflight}" | tee -a "${RUN_ROOT}/submission.jobs"
gate=$(safe_sbatch --lint-fatal --parsable --array=17 --dependency="afterok:${preflight}" --kill-on-invalid-dep=yes "${gpu}")
gate=${gate%%;*}
printf 'gate=%s\n' "${gate}" | tee -a "${RUN_ROOT}/submission.jobs"
full=$(safe_sbatch --lint-fatal --parsable --array=0-16,18-20%4 --dependency="afterok:${gate}" --kill-on-invalid-dep=yes "${gpu}")
full=${full%%;*}
printf 'remaining=%s\n' "${full}" | tee -a "${RUN_ROOT}/submission.jobs"
summary=$(safe_sbatch --lint-fatal --partition=cpu_short --parsable --export=ALL,MODE=summary \
  --dependency="afterany:${gate}:${full}" "${cpu}")
summary=${summary%%;*}
printf 'summary=%s\n' "${summary}" | tee -a "${RUN_ROOT}/submission.jobs"
cp "${REPLAY_ROOT}/MemNavData/FINAL14_TABLE3_EXACT_SPL_REPLAY_PROTOCOL_20260907.md" "${RUN_ROOT}/protocol.md"
cp "${REPLAY_ROOT}/source_inputs.sha256" "${RUN_ROOT}/replay_source_inputs.sha256"
squeue -u yz11502 -o '%.18i %.18j %.10P %.2t %.10M %.35R'
