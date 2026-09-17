#!/usr/bin/env bash
#SBATCH --job-name=gem_source_prefix_seal
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --qos=cpu48
set -euo pipefail
: "${FULLMONO_QUERY_ADDON:?}" "${FULLMONO_QUERY_ADDON_SHA:?}"
: "${CORE_PLAN:?}" "${CORE_PLAN_SHA:?}" "${FULLMONO_CONSTRUCTION_ROOT:?}"
: "${GEM_SOURCE_PREFIX:?}" "${GEM_PREFIX_SEAL_OUT:?}"
: "${GEM_PREFIX_SEAL_SCRIPT:?}" "${GEM_PREFIX_SEAL_SCRIPT_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
[[ "$(sha256sum "${GEM_PREFIX_SEAL_SCRIPT}" | awk '{print $1}')" == "${GEM_PREFIX_SEAL_SCRIPT_SHA}" ]]
[[ "$(sha256sum "${FULLMONO_QUERY_ADDON}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == "${FULLMONO_QUERY_ADDON_SHA}" ]]
(cd "${FULLMONO_QUERY_ADDON}" && sha256sum -c --quiet SOURCE_BUNDLE.sha256)
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
GEM_SEAL_TMP=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/gem_prefix_seal_${SLURM_JOB_ID}_XXXXXX")
export TMPDIR=${GEM_SEAL_TMP} LIBFFI_TMPDIR=${GEM_SEAL_TMP} PYTHONPATH=${FULLMONO_QUERY_ADDON}
exec singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
  /scratch/lg154/conda-envs/habitat/bin/python - <<'PY'
import argparse
import json
import os
from pathlib import Path
from construct_repaired_fullmono_queries import read, require, save
from probe_repaired_fullmono_construction import sha
from seal_repaired_fullmono_queries import seal

plan_path = Path(os.environ['CORE_PLAN'])
require(sha(plan_path) == os.environ['CORE_PLAN_SHA'], 'Source plan changed')
plan = read(plan_path)
prefix = int(os.environ['GEM_SOURCE_PREFIX'])
require(prefix in [s['scene_prefix'] for s in plan['stages']], 'Unplanned prefix')
ranks = sorted({s['scene_rank'] for s in plan['sources'] if s['scene_rank'] < prefix})
roots = [Path(os.environ['FULLMONO_CONSTRUCTION_ROOT']) / f'scene_{rank}' for rank in ranks]
out = Path(os.environ['GEM_PREFIX_SEAL_OUT'])
missing = [str(r) for r in roots if not (r / 'independent_verification.json').is_file()]
if missing:
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'construction_status.json', dict(complete=False, requested_prefix=prefix,
        missing_scene_roots=missing, query_navigation_submitted=False))
    raise SystemExit('Incomplete source prefix; missing scenes recorded, no query population')
base = Path('/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/repaired_fullmono_c8cf8c60e7efd55f')
result = seal(argparse.Namespace(plan=plan_path, plan_sha256=os.environ['CORE_PLAN_SHA'],
    scenes=roots, runtime_receipt=base / 'SOURCE_BUNDLE.sha256',
    evaluation_receipt=Path(os.environ['FULLMONO_QUERY_ADDON']) / 'SOURCE_BUNDLE.sha256', out=out))
save(out / 'seal_receipt.json', dict(requested_prefix=prefix, result=result,
    query_navigation_submitted=False, script_sha256=os.environ['GEM_PREFIX_SEAL_SCRIPT_SHA']))
print(json.dumps(result, indent=2))
PY
