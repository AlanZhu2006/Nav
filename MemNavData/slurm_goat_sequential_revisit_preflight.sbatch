#!/usr/bin/env bash
# CPU-only dependency closure gate for the frozen GOAT sequential-Revisit run.
# No episode, simulator rollout, or method decision is executed here.
#SBATCH --job-name=goat_cec_dep
#SBATCH --output=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.out
#SBATCH --error=/scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_769_tandon_advanced

set -euo pipefail
umask 0022

: "${SOURCE_ROOT:?set immutable source root}"
: "${SOURCE_RECEIPT:?set source receipt}"
: "${EXPECTED_SOURCE_RECEIPT_SHA:?set source receipt SHA-256}"
: "${MANIFEST:?set frozen smoke manifest}"
: "${EXPECTED_MANIFEST_SHA:?set smoke manifest SHA-256}"
: "${RUN_ROOT:?set immutable smoke result root}"

readonly DATA_ROOT="${GOAT_HM3D_ROOT:-/scratch/yz11502/Research/datasets/goat_bench_20260814}"
readonly ENV_PREFIX="${GOAT_ENV_PREFIX:-/scratch/yz11502/conda_envs/goat-bench-habitat023-20260814}"
readonly GOAT_CODE="${GOAT_CODE_ROOT:-${DATA_ROOT}/code/goat-bench}"
readonly BASE_SIF="${GOAT_BASE_SIF:-/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif}"
readonly OPENAI_CLIP_ROOT="${SOURCE_ROOT}/third_party/openai_clip"
readonly GOAT_RUNTIME_DEPS="${SOURCE_ROOT}/third_party/goat_runtime"
readonly OPENAI_CLIP_RN50="${DATA_ROOT}/data/goat-assets/checkpoints/openai_clip/RN50.pt"
readonly OPENAI_CLIP_DEFAULT_CACHE="/home/yz11502/.cache/clip/RN50.pt"
readonly EXPECTED_OPENAI_CLIP_COMMIT="d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
readonly EXPECTED_OPENAI_CLIP_RN50_SHA="afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
readonly EXPECTED_FTFY_VERSION="6.1.1"
readonly EXPECTED_REGEX_DISTRIBUTION_VERSION="2024.4.16"
readonly EXPECTED_REGEX_MODULE_VERSION="2.5.141"
readonly EXPECTED_WCWIDTH_VERSION="0.2.14"

fail() { echo "ABORT: $*" >&2; exit 2; }
for command_name in singularity sha256sum readlink; do
  command -v "${command_name}" >/dev/null || fail "${command_name} unavailable"
done
[[ "$(sha256sum "${SOURCE_RECEIPT}" | awk '{print $1}')" == \
   "${EXPECTED_SOURCE_RECEIPT_SHA}" ]] || fail "source receipt changed"
(cd "${SOURCE_ROOT}" && sha256sum -c "${SOURCE_RECEIPT}") >/dev/null || \
  fail "source bundle changed"
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == \
   "${EXPECTED_MANIFEST_SHA}" ]] || fail "smoke manifest changed"
[[ -x "${ENV_PREFIX}/bin/python" ]] || fail "GOAT Python is missing"
[[ -f "${BASE_SIF}" ]] || fail "base Singularity image is missing"
[[ -L "${OPENAI_CLIP_DEFAULT_CACHE}" ]] || fail "CLIP cache link is missing"
[[ "$(readlink -f "${OPENAI_CLIP_DEFAULT_CACHE}")" == \
   "${OPENAI_CLIP_RN50}" ]] || fail "CLIP cache target changed"
[[ "$(sha256sum "${OPENAI_CLIP_RN50}" | awk '{print $1}')" == \
   "${EXPECTED_OPENAI_CLIP_RN50_SHA}" ]] || fail "CLIP RN50 changed"

mkdir -p "${RUN_ROOT}/preflight" "${RUN_ROOT}/logs"
readonly RECORD="${RUN_ROOT}/preflight/dependency_preflight.json"
readonly TEMP_RECORD="${RECORD}.tmp.${SLURM_JOB_ID}"
[[ ! -e "${RECORD}" && ! -e "${TEMP_RECORD}" ]] || \
  fail "preflight output already exists"

cd "${GOAT_CODE}"
singularity exec -B /scratch/lg154 -B /scratch/yz11502 \
  "${BASE_SIF}" env -u PYTORCH_CUDA_ALLOC_CONF \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="${SOURCE_ROOT}:${OPENAI_CLIP_ROOT}:${GOAT_RUNTIME_DEPS}" \
  "${ENV_PREFIX}/bin/python" -u - \
    "${TEMP_RECORD}" "${GOAT_CODE}" "${DATA_ROOT}" "${MANIFEST}" \
    "${OPENAI_CLIP_ROOT}" "${GOAT_RUNTIME_DEPS}" "${OPENAI_CLIP_RN50}" \
    "${EXPECTED_OPENAI_CLIP_COMMIT}" "${EXPECTED_OPENAI_CLIP_RN50_SHA}" \
    "${EXPECTED_FTFY_VERSION}" "${EXPECTED_REGEX_DISTRIBUTION_VERSION}" \
    "${EXPECTED_REGEX_MODULE_VERSION}" "${EXPECTED_WCWIDTH_VERSION}" <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import sys
import time
import importlib_metadata

(out_raw, goat_raw, data_raw, manifest_raw, clip_root_raw, deps_root_raw, rn50_raw,
 clip_commit, rn50_sha, ftfy_version, regex_distribution_version,
 regex_module_version, wcwidth_version) = sys.argv[1:]
out = pathlib.Path(out_raw)
goat_root = pathlib.Path(goat_raw).resolve()
data_root = pathlib.Path(data_raw).resolve()
manifest = json.loads(pathlib.Path(manifest_raw).read_text())
clip_root = pathlib.Path(clip_root_raw).resolve()
deps_root = pathlib.Path(deps_root_raw).resolve()
rn50 = pathlib.Path(rn50_raw).resolve()

if sys.version_info[:2] != (3, 7):
    raise SystemExit("GOAT preflight must use Python 3.7")
if platform.machine() != "x86_64":
    raise SystemExit("unexpected preflight architecture")

import ftfy
import regex
import wcwidth

def under(path, root):
    resolved = pathlib.Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise RuntimeError("dependency escaped pinned root: {}".format(resolved))
    return str(resolved)

module_files = {
    "ftfy": under(ftfy.__file__, deps_root),
    "regex": under(regex.__file__, deps_root),
    "wcwidth": under(wcwidth.__file__, deps_root),
}
if ftfy.__version__ != ftfy_version:
    raise RuntimeError("ftfy version changed")
if importlib_metadata.version("ftfy") != ftfy_version:
    raise RuntimeError("ftfy distribution version changed")
if importlib_metadata.version("regex") != regex_distribution_version:
    raise RuntimeError("regex distribution version changed")
if regex.__version__ != regex_module_version:
    raise RuntimeError("regex module version changed")
if importlib_metadata.version("wcwidth") != wcwidth_version:
    raise RuntimeError("wcwidth distribution version changed")
if wcwidth.__version__ != wcwidth_version:
    raise RuntimeError("wcwidth version changed")
if ftfy.fix_text("FranÃ§ais") != "Français":
    raise RuntimeError("ftfy functional smoke failed")
if regex.fullmatch(r"\p{L}+", "Revisit") is None:
    raise RuntimeError("regex functional smoke failed")
if wcwidth.wcswidth("Revisit") != 7:
    raise RuntimeError("wcwidth functional smoke failed")

import clip
import torch
clip_file = under(clip.__file__, clip_root)
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is not None:
    raise RuntimeError("GOAT allocator option was not removed")
tokens = clip.tokenize(["frozen sequential revisit"])
if tuple(tokens.shape) != (1, 77):
    raise RuntimeError("CLIP tokenizer shape changed")

started = time.monotonic()
model, preprocess = clip.load("RN50", device="cpu", jit=False)
load_seconds = time.monotonic() - started
parameter_count = sum(parameter.numel() for parameter in model.parameters())
if parameter_count <= 0:
    raise RuntimeError("CLIP RN50 loaded without parameters")
if not getattr(preprocess, "transforms", None):
    raise RuntimeError("CLIP preprocessing pipeline is absent")

from MemNavData.goat_sequential_revisit_pilot import (
    _build_task_config,
    _register_policy_modules,
)
scene_id = str(manifest["episodes"][0]["scene_id"])
_build_task_config(goat_root, data_root, [scene_id], 0)
_register_policy_modules(goat_root)
import goat_bench.models.clip_policy as goat_clip_policy
policy_file = under(goat_clip_policy.__file__, goat_root)

digest = hashlib.sha256()
with rn50.open("rb") as handle:
    for block in iter(lambda: handle.read(8 << 20), b""):
        digest.update(block)
if digest.hexdigest() != rn50_sha:
    raise RuntimeError("CLIP RN50 hash changed inside Singularity")

payload = {
    "schema_version": "goat_sequential_revisit_dependency_preflight_v1_20260815",
    "complete": True,
    "episodes_executed": 0,
    "method_or_threshold_selection_allowed": False,
    "python_version": list(sys.version_info[:3]),
    "torch_version": torch.__version__,
    "pytorch_cuda_alloc_conf": None,
    "platform_machine": platform.machine(),
    "openai_clip_file": clip_file,
    "openai_clip_upstream_commit": clip_commit,
    "openai_clip_rn50_sha256": rn50_sha,
    "openai_clip_rn50_parameter_count": parameter_count,
    "openai_clip_rn50_cpu_load_seconds": load_seconds,
    "ftfy_version": ftfy_version,
    "regex_distribution_version": regex_distribution_version,
    "regex_module_version": regex_module_version,
    "wcwidth_version": wcwidth_version,
    "module_files": module_files,
    "goat_clip_policy_file": policy_file,
}
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

mv "${TEMP_RECORD}" "${RECORD}"
sha256sum "${RECORD}" > "${RECORD}.sha256"
echo "GOAT_SEQUENTIAL_REVISIT_DEPENDENCY_PREFLIGHT_OK=${RECORD}"
