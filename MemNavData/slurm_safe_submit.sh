#!/usr/bin/env bash
# Shared Slurm conventions for all Nav experiment submissions.
#
# Source this from a submit script, then use safe_sbatch in place of sbatch.
# It enforces the two lessons this project keeps re-learning per experiment:
#
# 1. Partition safety: this account's jobs on h200_public / l40s_public get
#    root-cancelled (QOSGrpGRES) or hit the gh-node LingBot runtime stall
#    (Gate D task 11: ~19 s/frame then SIGABRT).  Every stable formal run
#    used h100_tandon,a100_tandon.  safe_sbatch passes SAFE_PARTITIONS on
#    the command line, which OVERRIDES any stale #SBATCH partition line in
#    the template, and lint_sbatch_template flags contaminated templates.
#
# 2. Dependency safety: afterok on a job array never releases if any array
#    element was cancelled.  Use afterany for summary jobs plus an explicit
#    output-completeness check inside the summary itself.
set -uo pipefail

SAFE_PARTITIONS=${SAFE_PARTITIONS:-h100_tandon,a100_tandon}
FORBIDDEN_PARTITIONS=${FORBIDDEN_PARTITIONS:-h200_public l40s_public h200_plus l40s_plus}

lint_sbatch_template() {
  local template=$1 bad=0 part
  [[ -r "$template" ]] || { echo "[lint] missing template: $template" >&2; return 1; }
  for part in $FORBIDDEN_PARTITIONS; do
    if grep -E "^#SBATCH[[:space:]]+--partition=.*\b${part}\b" "$template" >/dev/null; then
      echo "[lint] $template requests forbidden partition '$part'" \
           "(will be overridden to $SAFE_PARTITIONS, but fix the template)" >&2
      bad=1
    fi
  done
  if ! grep -q "PYTHONDONTWRITEBYTECODE" "$template"; then
    echo "[lint] $template does not set PYTHONDONTWRITEBYTECODE=1" \
         "(read-only bundle __pycache__ hazard)" >&2
    bad=1
  fi
  if ! grep -Eq '^#SBATCH[[:space:]]+--account=torch_pr_[A-Za-z0-9_]+' \
      "$template"; then
    echo "[lint] $template has no explicit Torch project account" >&2
    bad=1
  fi
  return "$bad"
}

# safe_sbatch [--lint-fatal] <sbatch args...> <template>
# Lints the final argument (the template), then submits with SAFE_PARTITIONS
# on the command line, which overrides any in-file #SBATCH partition line.
# A caller-supplied --partition in the args still wins over the default,
# so deliberate overrides remain possible.
safe_sbatch() {
  local lint_fatal=0
  if [[ "${1:-}" == "--lint-fatal" ]]; then lint_fatal=1; shift; fi
  local template=${*: -1}
  if ! lint_sbatch_template "$template"; then
    if [[ "$lint_fatal" -eq 1 ]]; then
      echo "[lint] refusing to submit $template" >&2
      return 1
    fi
  fi
  sbatch --partition="$SAFE_PARTITIONS" "$@"
}
