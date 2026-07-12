#!/bin/bash
#SBATCH --job-name=mp3d_gen
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# Generate MP3D revisit trajectories with MemNavData/generate_twoleg.py.
# Habitat headless renders RGB via EGL -> needs a GPU. Produces the
# InternData-N1 layout under ${OUT}.
#
# Overridable env vars (defaults shown):
#   SCENE=.../mp3d/17DRP5sb8fy/17DRP5sb8fy.glb
#   OUT=.../mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg
#   N=5 N_LEGS=2 SEED=0
#
# Submit examples:
#   sbatch MemNavData/data_download/generate_mp3d.sbatch                       # 5 x 2-leg smoke
#   N=50 N_LEGS=2 OUT=.../mp3d_2leg sbatch ...
#   N=50 N_LEGS=3 OUT=.../mp3d_3leg sbatch ...

set -euo pipefail
mkdir -p /scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen

REPO=/scratch/lg154/Research/Nav/MemNavData
cd "${REPO}"

source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/habitat

MP3D_ROOT=/scratch/lg154/Research/datasets/mp3d/mp3d
SCENE=${SCENE:-${MP3D_ROOT}/17DRP5sb8fy/17DRP5sb8fy.glb}
NAVMESH=${NAVMESH:-${SCENE%.glb}.navmesh}
OUT=${OUT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg}
N=${N:-5}
N_LEGS=${N_LEGS:-2}
SEED=${SEED:-0}

# headless EGL rendering knobs
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER}/mp3dgen_${SLURM_JOB_ID:-$$}"; mkdir -p "${TMPDIR}"

[[ -f "${SCENE}" ]] || { echo "Missing scene: ${SCENE}" >&2; exit 1; }
mkdir -p "${OUT}"

echo "===== MP3D GEN ====="
echo "host:   $(hostname)"; nvidia-smi -L || true
echo "scene:  ${SCENE}"
echo "out:    ${OUT}"
echo "n=${N} n_legs=${N_LEGS} seed=${SEED}"
python -c 'import habitat_sim,quaternion,scipy,pandas; print("env OK", habitat_sim.__version__)'

exec python generate_twoleg.py \
    --scene "${SCENE}" \
    --navmesh "${NAVMESH}" \
    --out "${OUT}" \
    --n "${N}" \
    --n_legs "${N_LEGS}" \
    --seed "${SEED}"
