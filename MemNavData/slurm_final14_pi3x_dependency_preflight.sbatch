#!/usr/bin/env bash
# CPU-only fail-fast dependency gate for Final14 Pi3X evaluation.
#SBATCH --job-name=final14_p3x_pre
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --export=ALL

set -euo pipefail
umask 0022
: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA}"
: "${BASE_SOURCE_ROOT:?set base source root}"
: "${BASE_SOURCE_RECEIPT_SHA:?set base source receipt SHA}"
: "${DEPENDENCY_RECEIPT:?set dependency receipt}"
: "${EXPECTED_DEPENDENCY_RECEIPT_SHA:?set dependency receipt SHA}"
: "${RUN_ROOT:?set repair run root}"
: "${PI3X_SNAPSHOT:?set Pi3X snapshot}"
: "${EXPECTED_PI3X_MODEL_SHA:?set Pi3X model SHA}"
: "${PI3X_PROOF_MANIFEST:?set proof manifest}"
: "${EXPECTED_PI3X_PROOF_SHA:?set proof manifest SHA}"
: "${SOURCE_OVERLAY:?set immutable episode overlay}"
: "${EXPECTED_SOURCE_OVERLAY_BYTES:?set episode overlay size}"
: "${OVERLAY_EPISODE_ROOT:?set episode root inside overlay}"

PY=/scratch/lg154/conda-envs/memnav/bin/python
HAB_PY=/scratch/lg154/conda-envs/habitat/bin/python
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
VERIFY=${SOURCE_ROOT}/MemNavData/verify_portable_checksum_manifest.py
fail() { echo "ABORT: $*" >&2; exit 2; }

[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}" --quiet) || \
  fail "source bundle changed"
[[ "$(find "${SOURCE_ROOT}" -perm /222 | wc -l)" -eq 0 ]] || \
  fail "source bundle is writable"
[[ "$(sha256sum "${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256" | awk '{print $1}')" == \
   "${BASE_SOURCE_RECEIPT_SHA}" ]] || fail "base source receipt changed"
[[ "$(sha256sum "${DEPENDENCY_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_DEPENDENCY_RECEIPT_SHA}" ]] || fail "dependency receipt changed"
[[ "$(sha256sum "${PI3X_SNAPSHOT}/model.safetensors" | awk '{print $1}')" == \
   "${EXPECTED_PI3X_MODEL_SHA}" ]] || fail "Pi3X model changed"
[[ "$(sha256sum "${PI3X_PROOF_MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_PI3X_PROOF_SHA}" ]] || fail "proof manifest changed"
"${PY}" "${VERIFY}" "$(dirname "${PI3X_PROOF_MANIFEST}")/OUTPUTS.sha256" \
  --quiet || fail "nested proof receipt failed"

[[ -f "${RUN_ROOT}/benchmarks/SEALED" ]] || fail "benchmark not sealed"
[[ "$(sha256sum "${RUN_ROOT}/benchmarks/BENCHMARK_FILES.sha256" | awk '{print $1}')" == \
   "$(awk '{print $1}' "${RUN_ROOT}/benchmarks/BENCHMARK_FILES.sha256.sha256")" ]] || \
  fail "benchmark aggregate receipt changed"
(cd "${RUN_ROOT}/benchmarks" && sha256sum -c BENCHMARK_FILES.sha256 --quiet) || \
  fail "benchmark payload changed"
[[ "${EXPECTED_SOURCE_OVERLAY_BYTES}" =~ ^[1-9][0-9]*$ ]] || \
  fail "invalid overlay size contract"
[[ -r "${SOURCE_OVERLAY}" && -r "${BASE_SIF}" && -x "${HAB_PY}" ]] || \
  fail "episode-overlay runtime dependency missing"
[[ "$(stat -c %s "${SOURCE_OVERLAY}")" -eq \
   "${EXPECTED_SOURCE_OVERLAY_BYTES}" ]] || fail "episode overlay size changed"

# Exercise the exact container/overlay/Habitat path used by formal evaluation.
# Merely checking the squashfs file on the host would not catch a missing
# --overlay argument in the GPU job (the Attempt-5 failure mode).
singularity exec --overlay "${SOURCE_OVERLAY}:ro" \
  -B /scratch/lg154 -B /scratch/yz11502 "${BASE_SIF}" \
  "${HAB_PY}" - "${RUN_ROOT}" "${OVERLAY_EPISODE_ROOT}" <<'PY'
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

run_root = Path(sys.argv[1])
overlay_root = Path(sys.argv[2])
assert overlay_root.is_absolute() and overlay_root.is_dir()
episodes = set()
for protocol in ("natural_direction", "hard_support"):
    manifest = json.loads(
        (run_root / "benchmarks" / protocol / "manifest.json").read_text()
    )
    for row in manifest["episodes"]:
        receipt = json.loads(
            (Path(row["online_a_episode"]) / "receipt.json").read_text()
        )
        episode = Path(receipt["source_episode"])
        assert episode.is_absolute()
        assert episode == overlay_root or overlay_root in episode.parents
        parquet = episode / "data/chunk-000/episode_000000.parquet"
        assert parquet.is_file(), parquet
        assert pq.ParquetFile(parquet).metadata.num_rows > 0
        episodes.add(str(episode))
assert len(episodes) == 21, len(episodes)
print("final14_overlay_episode_preflight=passed episodes=21")
PY

export PYTHONPYCACHEPREFIX=${SLURM_TMPDIR:-/tmp}/final14_pi3x_preflight_${SLURM_JOB_ID}
export PYTHONPATH=${SOURCE_ROOT}:${SOURCE_ROOT}/MemNavData:${BASE_SOURCE_ROOT}/InternNav/src/diffusion-policy
"${PY}" -m py_compile \
  "${SOURCE_ROOT}/MemNavData/pi3x_online_relocalizer.py" \
  "${SOURCE_ROOT}/MemNavData/pi3x_spatial_proof_runtime.py" \
  "${SOURCE_ROOT}/NavDP/baselines/memnav/memnav_server.py"
"${PY}" - "${PI3X_PROOF_MANIFEST}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text())
assert manifest["authorization"]["consensus_numerator"] == 2
assert manifest["authorization"]["consensus_denominator"] == 4
for member in manifest["members"]:
    checkpoint = manifest_path.parent / member["checkpoint"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == member["checkpoint_sha256"]
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["member"] == member["member"]
    assert float(payload["threshold"]) == float(member["threshold"])
print("final14_pi3x_dependency_preflight=passed")
PY
