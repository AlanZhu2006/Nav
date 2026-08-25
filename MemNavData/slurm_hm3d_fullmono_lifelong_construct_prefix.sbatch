#!/usr/bin/env bash
#SBATCH --job-name=h3lifePrefix
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --partition=h100_tandon,a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?history array index required}"
: "${TASK_ROOT:?}" "${BASE_SOURCE_ROOT:?}" "${RUN_ROOT:?}" "${PROTOCOL:?}"
[[ "$(sha256sum "${TASK_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_TASK_RECEIPT_SHA}" ]]
(cd "${TASK_ROOT}" && sha256sum -c --quiet "${TASK_RECEIPT}")
AB_ROOT=${RUN_ROOT}/benchmarks_ab
manifest=${AB_ROOT}/role_pairs/manifest.json
readarray -t identity < <(/scratch/lg154/conda-envs/memnav/bin/python - \
  "${manifest}" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); i=int(sys.argv[2]); rows=p["episodes"]
if i >= len(rows): raise SystemExit(0)
print(rows[i]["scene"]); print(rows[i]["episode"])
PY
)
if [[ "${#identity[@]}" -eq 0 ]]; then
  mkdir -p "${RUN_ROOT}/prefix_fragments/skipped"
  printf 'outside_AB_population\n' > \
    "${RUN_ROOT}/prefix_fragments/skipped/$(printf '%03d' "${SLURM_ARRAY_TASK_ID}").txt"
  exit 0
fi
[[ "${#identity[@]}" -eq 2 ]] || { echo "bad history identity" >&2; exit 2; }
SCENE=${identity[0]}; EPISODE=${identity[1]}
label=$(printf '%03d_%s_%s' "${SLURM_ARRAY_TASK_ID}" "${SCENE}" "${EPISODE}")
OUT=${RUN_ROOT}/prefix_fragments/${label}
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
pycache=${SLURM_TMPDIR:-/tmp}/h3life_prefix_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${pycache}"
exec singularity exec --nv -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX="${pycache}" \
  PYTHONPATH="${TASK_ROOT}:${TASK_ROOT}/MemNavData:${BASE_SOURCE_ROOT}:${BASE_SOURCE_ROOT}/MemNavData" \
  /scratch/lg154/conda-envs/habitat/bin/python -u \
    "${TASK_ROOT}/MemNavData/construct_hm3d_fullmono_lifelong_prefix.py" \
      --protocol "${PROTOCOL}" --ab-root "${AB_ROOT}" \
      --b-root "${RUN_ROOT}/factual_b" \
      --history-index "${SLURM_ARRAY_TASK_ID}" --out "${OUT}"
