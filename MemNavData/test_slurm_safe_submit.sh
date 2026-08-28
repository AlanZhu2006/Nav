#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source "${ROOT}/MemNavData/slurm_safe_submit.sh"

scratch=$(mktemp -d /tmp/test_slurm_safe_submit.XXXXXX)
cleanup() { rm -rf -- "${scratch}"; }
trap cleanup EXIT

template=${scratch}/job.sbatch
args_file=${scratch}/args.txt
{
  printf '%s\n' '#!/usr/bin/env bash'
  printf '%s\n' '#SBATCH --account=torch_pr_769_tandon_advanced'
  printf '%s\n' '#SBATCH --partition=cpu_short'
  printf '%s\n' 'export PYTHONDONTWRITEBYTECODE=1'
} >"${template}"

sbatch() { printf '%s\n' "$@" >"${args_file}"; }

safe_sbatch --lint-fatal --parsable "${template}"
[[ "$(sed -n '1p' "${args_file}")" == \
   "--partition=h100_tandon,a100_tandon" ]]

safe_sbatch --lint-fatal --parsable --partition=cpu_short "${template}"
[[ "$(grep -cE '^--partition(=|$)' "${args_file}")" == 1 ]]
grep -Fx -- '--partition=cpu_short' "${args_file}" >/dev/null
! grep -F -- 'h100_tandon,a100_tandon' "${args_file}" >/dev/null

safe_sbatch --lint-fatal --parsable --partition cpu_short "${template}"
[[ "$(grep -cE '^--partition$' "${args_file}")" == 1 ]]
grep -Fx -- 'cpu_short' "${args_file}" >/dev/null

safe_sbatch --lint-fatal --parsable -pa100_tandon "${template}"
grep -Fx -- '-pa100_tandon' "${args_file}" >/dev/null
! grep -F -- 'h100_tandon,a100_tandon' "${args_file}" >/dev/null

if safe_sbatch --lint-fatal --partition >/dev/null 2>&1; then
  echo "missing partition value unexpectedly accepted" >&2
  exit 1
fi

printf 'slurm_safe_submit tests passed\n'
