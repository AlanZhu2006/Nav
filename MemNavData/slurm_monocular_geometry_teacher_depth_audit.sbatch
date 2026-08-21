#!/usr/bin/env bash
#SBATCH --job-name=mdtec_depth_audit
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=cpu_short
#SBATCH --qos=cpu48
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
: "${SOURCE_ROOT:?set immutable diagnostic source root}"
: "${SOURCE_RECEIPT_SHA:?set SOURCE_BUNDLE.sha256 file hash}"
: "${OUTPUT:?set new output JSON}"

readonly OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
readonly BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
readonly FEATURE_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat_flowgate/vln_n1/traj_data
readonly SCENE_SPLIT=${SOURCE_ROOT}/MemNavData/router_multiscene_split_20260805.json

[[ ! -e "${OUTPUT}" ]] || { echo "output exists: ${OUTPUT}" >&2; exit 2; }
[[ "$(sha256sum "${SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${SOURCE_RECEIPT_SHA}" ]] || { echo "source receipt changed" >&2; exit 2; }
(cd "${SOURCE_ROOT}" && sha256sum -c SOURCE_BUNDLE.sha256 --quiet) || {
  echo "source bundle changed" >&2; exit 2; }
mkdir -p "$(dirname "${OUTPUT}")"

apptainer exec \
  --overlay "${OVERLAY}:ro" \
  -B /scratch/lg154 -B /scratch/yz11502 -B "${SOURCE_ROOT}" \
  "${BASE_SIF}" \
  bash -c '
    set -euo pipefail
    source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
    conda activate /scratch/lg154/conda-envs/memnav
    cd "'"${SOURCE_ROOT}"'"
    python MemNavData/audit_monocular_geometry_teacher_depth_population.py \
      --data-root /mp3d_revisit_v0/vln_n1/traj_data \
      --feature-root "'"${FEATURE_ROOT}"'" \
      --scene-split "'"${SCENE_SPLIT}"'" \
      --scene-field train \
      --max-episodes-per-scene 4 \
      --states-per-episode 4 \
      --output "'"${OUTPUT}"'"
  '
