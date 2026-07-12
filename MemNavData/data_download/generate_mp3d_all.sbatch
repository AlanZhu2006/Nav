#!/bin/bash
#SBATCH --job-name=mp3d_gen_all
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/all-%A_%a.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/all-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --array=0-89%4
#SBATCH --export=ALL
#
# Full MP3D revisit dataset: one array task per scene (90 scenes). Each task loads the
# scene ONCE and generates both leg types (2-leg + 3-leg), reusing the sim + per-floor
# ESDF cache. Output nests as:
#   ${OUT_ROOT}/mp3d_2leg/<scene_id>/episode_XXXX
#   ${OUT_ROOT}/mp3d_3leg/<scene_id>/episode_XXXX
#
# %4 => up to 4 scenes (GPUs) concurrently. The a100_tandon QOS pool is 60 GPUs with no
# per-user cap, so 4-way is fine; each task writes its own scene dir (no conflicts). Resumable:
# run_legs() skips a (scene,leg) that already has N complete episodes, so re-submitting after a
# preemption/timeout continues where it left off. Raise/lower live with:
#   scontrol update JobId=<id> ArrayTaskThrottle=<k>
#
# Config decisions: goal_jitter_pos=3.0 (wider revisit spread), goal_tries=100 (keep yield
# under the wider jitter), covis_hi left at default 1.0. N2=15 two-leg, N3=25 three-leg.
#
# Submit:
#   sbatch MemNavData/data_download/generate_mp3d_all.sbatch
# Override counts / output:
#   N2=15 N3=25 OUT_ROOT=/path sbatch MemNavData/data_download/generate_mp3d_all.sbatch

set -euo pipefail
mkdir -p /scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen

REPO=/scratch/lg154/Research/Nav/MemNavData
cd "${REPO}"

source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/habitat

MP3D_ROOT=/scratch/lg154/Research/datasets/mp3d/mp3d
OUT_ROOT=${OUT_ROOT:-/scratch/lg154/Research/datasets/mp3d_revisit_v0/vln_n1/traj_data}
N2=${N2:-15}
N3=${N3:-25}

# deterministic scene list (sorted); index by array task id
mapfile -t SCENES < <(ls -d ${MP3D_ROOT}/*/ | xargs -n1 basename | sort)
SCENE_ID=${SCENES[${SLURM_ARRAY_TASK_ID}]}
SCENE=${MP3D_ROOT}/${SCENE_ID}/${SCENE_ID}.glb
NAVMESH=${MP3D_ROOT}/${SCENE_ID}/${SCENE_ID}.navmesh

# headless EGL rendering knobs
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER}/mp3dgen_${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"; mkdir -p "${TMPDIR}"

[[ -f "${SCENE}" ]] || { echo "Missing scene: ${SCENE}" >&2; exit 1; }

echo "===== MP3D GEN [task ${SLURM_ARRAY_TASK_ID} / ${#SCENES[@]} scenes] ====="
echo "host:  $(hostname)"; nvidia-smi -L || true
echo "scene: ${SCENE_ID}"
echo "out:   ${OUT_ROOT}"
echo "n2=${N2} (2-leg) n3=${N3} (3-leg) seed=${SLURM_ARRAY_TASK_ID}"

exec python generate_twoleg.py \
    --scene "${SCENE}" \
    --navmesh "${NAVMESH}" \
    --out "${OUT_ROOT}" \
    --n2 "${N2}" \
    --n3 "${N3}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --goal_jitter_pos 3.0 \
    --goal_tries 100
