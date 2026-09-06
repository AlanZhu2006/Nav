#!/usr/bin/env bash
set -euo pipefail
source "${REPLAY_ROOT}/MemNavData/final14_spl_replay_env.sh"
export PYTHONPATH="${REPLAY_ROOT}:${FROZEN_BASE}:${FROZEN_BASE}/MemNavData:${DEPENDENCY_ROOT}:${LIGHTGLUE_REPO}"
(cd "${REPLAY_ROOT}" && sha256sum -c --quiet source_inputs.sha256)
[[ "$(sha256sum "${FROZEN_BASE}/source_inputs.sha256" | awk '{print $1}')" == "${BASE_RECEIPT_SHA}" ]]
(cd "${FROZEN_BASE}" && sha256sum -c --quiet source_inputs.sha256)
if [[ "${MODE}" == preflight ]]; then
  "${MEMNAV_PY}" -m unittest MemNavData.test_final14_spl_replay
  "${MEMNAV_PY}" -c 'import torch,cv2,kornia,kornia_rs,lightglue; print("model dependencies import correctly")'
  HAB_SITE=$("${HAB_PY}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  PYTHONPATH="${PYTHONPATH}:${HAB_SITE}/pip/_vendor" "${HAB_PY}" -c 'import habitat_sim, pandas, requests; print("simulator dependencies import correctly")'
  for spec in "${MEMNAV_CKPT}:9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7" \
              "${NAVDP_CKPT}:3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947" \
              "${LINGBOT_WEIGHTS}:832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409"; do
    IFS=: read -r path expected <<<"${spec}"
    [[ "$(sha256sum "${path}" | awk '{print $1}')" == "${expected}" ]]
  done
  "${HAB_PY}" "${FROZEN_BASE}/MemNavData/audit_final14_mono_factorial_inputs.py" \
    --manifest "${BENCH_ROOT}/manifest.json" --expected-manifest-sha256 "${EXPECTED_MANIFEST_SHA}"
else
  [[ "${MODE}" == summary ]]
  "${MEMNAV_PY}" "${REPLAY_ROOT}/MemNavData/final14_spl_replay.py" audit \
    --run-root "${RUN_ROOT}" --bench-root "${BENCH_ROOT}" --out "${RUN_ROOT}/summary_verified.json"
fi
