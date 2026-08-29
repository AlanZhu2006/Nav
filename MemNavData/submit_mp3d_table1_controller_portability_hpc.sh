#!/usr/bin/env bash
# Thin, explicit entry point for the verifier-gated MP3D Table-1 rollout.
set -euo pipefail

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
: "${CONSTRUCTION_RUN:?set the completed MP3D construction run root}"

export DATASET=MP3D
export CONSTRUCTION_RUN
exec bash "${ROOT}/MemNavData/submit_hm3d_table1_controller_portability_hpc.sh"
