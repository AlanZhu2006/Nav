#!/usr/bin/env bash
# T0 defaults to the frozen pLe scene.  Set SCENE_INDICES=0,...,19 and a fresh
# OUT_ROOT for the full consumed-pool T1 run.

set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
SCENE_INDICES=${SCENE_INDICES:-6}
OUT_ROOT=${OUT_ROOT:-${ROOT}/.diagnostics/revisit_front_support_t0_20260811}

exec env \
  ROOT="${ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
  PROTOCOL="${ROOT}/MemNavData/REVISIT_FRONT_SUPPORT_PROTOCOL_20260811.md" \
  FRONT_SUPPORT_SUMMARIZER="${ROOT}/MemNavData/summarize_revisit_front_support.py" \
  RUN_KNOWN_REVISIT_FRONT_SUPPORT=1 \
  SCENE_INDICES="${SCENE_INDICES}" \
  bash "${ROOT}/MemNavData/run_revisit_known_phase_ablation_local.sh"
