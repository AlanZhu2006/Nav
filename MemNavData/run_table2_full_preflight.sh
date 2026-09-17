#!/usr/bin/env bash
# Exact imports, payloads and CLI validation; no model inference on a login node.
set -euo pipefail
: "${REPAIRED_BUNDLE:?}" "${TABLE2_PLAN:?}" "${TABLE2_RUN:?}" "${EXPECTED_RUNTIME_SHA:?}"
[[ "$(id -un)" == yz11502 ]]
source "${REPAIRED_BUNDLE}/MemNavData/repaired_hm3d_hpc_env.sh"
TABLE2_PREFLIGHT_TMP=$(mktemp -d /tmp/table2_preflight_XXXXXX)
export TMPDIR=${TABLE2_PREFLIGHT_TMP} LIBFFI_TMPDIR=${TABLE2_PREFLIGHT_TMP}
[[ "$(sha256sum "${REPAIRED_SOURCE_RECEIPT}" | awk '{print $1}')" == "${EXPECTED_RUNTIME_SHA}" ]]
[[ ! -e "${TABLE2_RUN}/preflight" ]]
mkdir -p "${TABLE2_RUN}/preflight"
cd "${REPAIRED_BUNDLE}"
sha256sum -c --quiet SOURCE_BUNDLE.sha256
for kind in habitat memnav navdp; do
  if [[ "${kind}" == habitat ]]; then
    "${REPAIRED_HAB_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${TABLE2_RUN}/preflight/${kind}.log" 2>&1
  else
    env PYTHONPATH="${REPAIRED_BUNDLE}/NavDP/baselines/${kind}:${PYTHONPATH}" \
      "${REPAIRED_MEM_PY}" MemNavData/preflight_repaired_fullmono.py "${kind}" >"${TABLE2_RUN}/preflight/${kind}.log" 2>&1
  fi
done
[[ "$(sha256sum "${REPAIRED_MEM_CKPT}" | awk '{print $1}')" == 9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7 ]]
[[ "$(sha256sum "${REPAIRED_NAV_CKPT}" | awk '{print $1}')" == 3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947 ]]
[[ "$(sha256sum "${REPAIRED_LINGBOT_REPO}/weights/lingbot-map-long.pt" | awk '{print $1}')" == 832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409 ]]
"${REPAIRED_MEM_PY}" MemNavData/preflight_table2_mixed_hpc.py \
  --plan "${TABLE2_PLAN}" --out "${TABLE2_RUN}/preflight/table2"
"${REPAIRED_MEM_PY}" MemNavData/verify_table2_full_setup.py \
  --payload "${TABLE2_RUN}/payload" --out "${TABLE2_RUN}/preflight/published_query_verification.json"
