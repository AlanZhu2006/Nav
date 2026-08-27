#!/usr/bin/env bash
# Freeze, upload, and submit consumed-scene smoke -> 34-scene formal -> audits.

set -euo pipefail
umask 0022

LOCAL_ROOT=${LOCAL_ROOT:-$(git rev-parse --show-toplevel)}
REMOTE_HOST=${REMOTE_HOST:-alantorch}
EXPECTED_REMOTE_USER=${EXPECTED_REMOTE_USER:-yz11502}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-}
REMOTE_BUNDLE_BASE=${REMOTE_BUNDLE_BASE:-/scratch/yz11502/Research/source_bundles}
REMOTE_RESULT_BASE=${REMOTE_RESULT_BASE:-/scratch/yz11502/Research/Nav-axis-uturn-results/goat_sequential_revisit_20260815}
BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT:-/scratch/yz11502/Research/Nav-axis-uturn-source-bundles/certified_relocalization_closed_loop_d3bd281fc374cc80}
EXPECTED_BASE_SOURCE_RECEIPT_SHA=74001a9e0150c38c599a206fa0f4dd5e1279b9bed5d167119f4d14cb77995e98
FORMAL_MANIFEST_SHA=aaedc6fb0c6d3787b5c8c61eed2c2d943320f595f9b1783f881febc544121397
SMOKE_MANIFEST_SHA=7e23655af2578c39c8435981584dbe65b2de8eb2025478f3cfd50921d07628ab
MEMNAV_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt
NAVDP_CKPT=/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt
LINGBOT_WEIGHTS=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt
GOAT_POLICY_CKPT=/scratch/yz11502/Research/datasets/goat_bench_20260814/data/goat-assets/checkpoints/sense_act_nn_monolithic/ckpt_best.pth
OPENAI_CLIP_RN50=/scratch/yz11502/Research/datasets/goat_bench_20260814/data/goat-assets/checkpoints/openai_clip/RN50.pt
OPENAI_CLIP_DEFAULT_CACHE=/home/yz11502/.cache/clip/RN50.pt
EXPECTED_MEMNAV_CKPT_SHA=9b7a5811ff0aea212503f58b45258ba4f66b06420f87c350946aead39db6fdb7
EXPECTED_NAVDP_CKPT_SHA=3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947
EXPECTED_LINGBOT_WEIGHTS_SHA=832bc82cbae0bc9bbe946ef5ee1f7226abd8c0e183ccf8beddbb3d133576f409
EXPECTED_GOAT_POLICY_SHA=55e89c3d083198d4add4e9e70164b54ff892900963a2925471362e2d4761b3eb
EXPECTED_OPENAI_CLIP_RN50_SHA=afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762
LOCAL_TEST_PY=${LOCAL_TEST_PY:-/home/asus/miniconda3/envs/memnav/bin/python}
LOCAL_GOAT_PY=${LOCAL_GOAT_PY:-${LOCAL_ROOT}/.diagnostics/envs/goat-bench-policy-local-20260815/bin/python}
LOCAL_OPENAI_CLIP_ROOT=${LOCAL_OPENAI_CLIP_ROOT:-${LOCAL_ROOT}/.diagnostics/dependencies/openai-clip}
EXPECTED_OPENAI_CLIP_COMMIT=d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
LOCAL_GOAT_SITE_PACKAGES=${LOCAL_GOAT_SITE_PACKAGES:-${LOCAL_ROOT}/.diagnostics/envs/goat-bench-policy-local-20260815/lib/python3.7/site-packages}
EXPECTED_FTFY_VERSION=6.1.1
EXPECTED_REGEX_DISTRIBUTION_VERSION=2024.4.16
EXPECTED_REGEX_MODULE_VERSION=2.5.141
EXPECTED_WCWIDTH_VERSION=0.2.14
REMOTE_PY=${REMOTE_PY:-/scratch/lg154/conda-envs/memnav/bin/python}
FORMAL_CONCURRENCY=${FORMAL_CONCURRENCY:-4}
DRY_RUN=${DRY_RUN:-0}

fail() { echo "ABORT: $*" >&2; exit 2; }
[[ "${DRY_RUN}" =~ ^[01]$ ]] || fail "DRY_RUN must be 0 or 1"
[[ "${FORMAL_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || \
  fail "FORMAL_CONCURRENCY must be positive"

# Reuse only a live master authenticated as yz11502. A socket belonging to a
# different shared account is never used for uploads or Slurm mutations.
if [[ -z "${SSH_CONTROL_PATH}" ]]; then
  for candidate in /home/asus/.ssh/cm-*; do
    [[ -S "${candidate}" ]] || continue
    identity=$(timeout 8s ssh -o BatchMode=yes -o ConnectTimeout=3 -S "${candidate}" \
      -o ControlMaster=no "${REMOTE_HOST}" id -un 2>/dev/null || true)
    if [[ "${identity}" == "${EXPECTED_REMOTE_USER}" ]]; then
      SSH_CONTROL_PATH=${candidate}
      break
    fi
  done
fi
[[ -n "${SSH_CONTROL_PATH}" && -S "${SSH_CONTROL_PATH}" ]] || \
  fail "no live shared SSH master authenticated as ${EXPECTED_REMOTE_USER}"
SSH_ARGS=(-o BatchMode=yes -S "${SSH_CONTROL_PATH}" -o ControlMaster=no)
RSYNC_RSH="ssh -o BatchMode=yes -S ${SSH_CONTROL_PATH} -o ControlMaster=no"
remote() { ssh "${SSH_ARGS[@]}" "${REMOTE_HOST}" "$@"; }
[[ "$(remote 'id -un')" == "${EXPECTED_REMOTE_USER}" ]] || \
  fail "remote identity mismatch"

files=(
  MemNavData/build_goat_sequential_revisit_manifest.py
  MemNavData/goat_sequential_revisit_formal_manifest_20260815.json
  MemNavData/goat_sequential_revisit_smoke_manifest_20260815.json
  MemNavData/goat_sequential_revisit_pilot.py
  MemNavData/goat_contract_smoke.py
  MemNavData/goat_navdp_discrete_adapter.py
  MemNavData/goat_navdp_runtime_pilot.py
  MemNavData/goat_certified_arrival_contract.py
  MemNavData/goat_terminal_alignment.py
  MemNavData/revisit_bearing_adapter.py
  MemNavData/xnavdp_revisit_contract.py
  MemNavData/certified_relocalization_runtime.py
  MemNavData/lingbot_colored_registration.py
  MemNavData/lingbot_pnp_localization.py
  MemNavData/summarize_goat_sequential_revisit.py
  MemNavData/verify_goat_sequential_revisit.py
  MemNavData/test_build_goat_sequential_revisit_manifest.py
  MemNavData/test_goat_sequential_revisit_pilot.py
  MemNavData/test_goat_terminal_alignment.py
  MemNavData/test_summarize_goat_sequential_revisit.py
  MemNavData/test_policy_agent_graph.py
  MemNavData/slurm_goat_sequential_revisit_eval.sbatch
  MemNavData/slurm_goat_sequential_revisit_preflight.sbatch
  MemNavData/slurm_summarize_goat_sequential_revisit.sbatch
  MemNavData/slurm_verify_goat_sequential_revisit.sbatch
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  NavDP/baselines/memnav/router_candidates.py
)
for relative in "${files[@]}"; do
  path=${LOCAL_ROOT}/${relative}
  [[ -f "${path}" && ! -L "${path}" ]] || fail "missing input ${path}"
done
formal_manifest=${LOCAL_ROOT}/MemNavData/goat_sequential_revisit_formal_manifest_20260815.json
smoke_manifest=${LOCAL_ROOT}/MemNavData/goat_sequential_revisit_smoke_manifest_20260815.json
[[ "$(sha256sum "${formal_manifest}" | awk '{print $1}')" == \
   "${FORMAL_MANIFEST_SHA}" ]] || fail "formal manifest changed"
[[ "$(sha256sum "${smoke_manifest}" | awk '{print $1}')" == \
   "${SMOKE_MANIFEST_SHA}" ]] || fail "smoke manifest changed"
[[ -d "${LOCAL_OPENAI_CLIP_ROOT}/.git" ]] || \
  fail "local OpenAI CLIP checkout is missing"
[[ "$(git -C "${LOCAL_OPENAI_CLIP_ROOT}" rev-parse HEAD)" == \
   "${EXPECTED_OPENAI_CLIP_COMMIT}" ]] || fail "local OpenAI CLIP commit changed"
[[ -z "$(git -C "${LOCAL_OPENAI_CLIP_ROOT}" status --short)" ]] || \
  fail "local OpenAI CLIP checkout is dirty"
for relative in \
  clip/__init__.py \
  clip/clip.py \
  clip/model.py \
  clip/simple_tokenizer.py \
  clip/bpe_simple_vocab_16e6.txt.gz; do
  [[ -f "${LOCAL_OPENAI_CLIP_ROOT}/${relative}" ]] || \
    fail "local OpenAI CLIP file is missing: ${relative}"
done
for package in ftfy regex wcwidth; do
  [[ -f "${LOCAL_GOAT_SITE_PACKAGES}/${package}/__init__.py" ]] || \
    fail "local GOAT runtime package is missing: ${package}"
done
for dist_info in ftfy-6.1.1.dist-info regex-2024.4.16.dist-info \
  wcwidth-0.2.14.dist-info; do
  [[ -f "${LOCAL_GOAT_SITE_PACKAGES}/${dist_info}/METADATA" ]] || \
    fail "local GOAT runtime metadata is missing: ${dist_info}"
done

export PYTHONPATH="${LOCAL_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${LOCAL_TEST_PY}" -m pytest -q \
  MemNavData/test_build_goat_sequential_revisit_manifest.py \
  MemNavData/test_goat_sequential_revisit_pilot.py \
  MemNavData/test_goat_terminal_alignment.py \
  MemNavData/test_summarize_goat_sequential_revisit.py \
  MemNavData/test_policy_agent_graph.py
"${LOCAL_GOAT_PY}" -m py_compile \
  "${LOCAL_ROOT}/MemNavData/build_goat_sequential_revisit_manifest.py" \
  "${LOCAL_ROOT}/MemNavData/goat_sequential_revisit_pilot.py" \
  "${LOCAL_ROOT}/MemNavData/goat_certified_arrival_contract.py" \
  "${LOCAL_ROOT}/MemNavData/goat_terminal_alignment.py" \
  "${LOCAL_ROOT}/MemNavData/lingbot_colored_registration.py" \
  "${LOCAL_ROOT}/MemNavData/lingbot_pnp_localization.py" \
  "${LOCAL_ROOT}/MemNavData/summarize_goat_sequential_revisit.py" \
  "${LOCAL_ROOT}/MemNavData/verify_goat_sequential_revisit.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${LOCAL_ROOT}/NavDP/baselines/memnav/memnav_server.py"
PYTHONPATH="${LOCAL_OPENAI_CLIP_ROOT}:${LOCAL_GOAT_SITE_PACKAGES}" \
  "${LOCAL_GOAT_PY}" - "${LOCAL_OPENAI_CLIP_ROOT}" \
  "${LOCAL_GOAT_SITE_PACKAGES}" "${EXPECTED_OPENAI_CLIP_COMMIT}" \
  "${EXPECTED_FTFY_VERSION}" "${EXPECTED_REGEX_DISTRIBUTION_VERSION}" \
  "${EXPECTED_REGEX_MODULE_VERSION}" "${EXPECTED_WCWIDTH_VERSION}" <<'PY'
import clip
import ftfy
import json
import pathlib
import regex
import sys
import wcwidth
import importlib_metadata
clip_root = pathlib.Path(sys.argv[1]).resolve()
deps_root = pathlib.Path(sys.argv[2]).resolve()
for module, root in ((clip, clip_root), (ftfy, deps_root),
                     (regex, deps_root), (wcwidth, deps_root)):
    path = pathlib.Path(module.__file__).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise AssertionError("dependency import escaped the pinned checkout")
assert ftfy.__version__ == sys.argv[4]
assert importlib_metadata.version("ftfy") == sys.argv[4]
assert importlib_metadata.version("regex") == sys.argv[5]
assert regex.__version__ == sys.argv[6]
assert importlib_metadata.version("wcwidth") == sys.argv[7]
assert wcwidth.__version__ == sys.argv[7]
tokens = clip.tokenize(["revisit dependency smoke"])
assert tuple(tokens.shape) == (1, 77)
print(json.dumps({"clip_file": clip.__file__, "upstream_commit": sys.argv[3]}))
PY
for script in \
  MemNavData/slurm_goat_sequential_revisit_eval.sbatch \
  MemNavData/slurm_goat_sequential_revisit_preflight.sbatch \
  MemNavData/slurm_summarize_goat_sequential_revisit.sbatch \
  MemNavData/slurm_verify_goat_sequential_revisit.sbatch; do
  bash -n "${LOCAL_ROOT}/${script}"
done

remote "test \"\$(sha256sum '${BASE_SOURCE_ROOT}/SOURCE_BUNDLE.sha256' | awk '{print \$1}')\" = '${EXPECTED_BASE_SOURCE_RECEIPT_SHA}' && cd '${BASE_SOURCE_ROOT}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
remote "test -x /scratch/yz11502/conda_envs/goat-bench-habitat023-20260814/bin/python && test -f /scratch/yz11502/Research/datasets/goat_bench_20260814/data/scene_datasets/hm3d/val/SEALED"
remote "test -L '${OPENAI_CLIP_DEFAULT_CACHE}' && test \"\$(readlink -f '${OPENAI_CLIP_DEFAULT_CACHE}')\" = '${OPENAI_CLIP_RN50}'"

dependency_lines=$(remote "for p in '${MEMNAV_CKPT}' '${NAVDP_CKPT}' '${LINGBOT_WEIGHTS}' '${GOAT_POLICY_CKPT}' '${OPENAI_CLIP_RN50}'; do stat -c '%s' \"\$p\"; sha256sum \"\$p\" | awk '{print \$1}'; done")
mapfile -t dependency_values <<<"${dependency_lines}"
[[ "${#dependency_values[@]}" -eq 10 ]] || fail "dependency audit malformed"
[[ "${dependency_values[1]}" == "${EXPECTED_MEMNAV_CKPT_SHA}" ]] || \
  fail "remote MemNav checkpoint changed"
[[ "${dependency_values[3]}" == "${EXPECTED_NAVDP_CKPT_SHA}" ]] || \
  fail "remote NavDP checkpoint changed"
[[ "${dependency_values[5]}" == "${EXPECTED_LINGBOT_WEIGHTS_SHA}" ]] || \
  fail "remote LingBot weights changed"
[[ "${dependency_values[7]}" == "${EXPECTED_GOAT_POLICY_SHA}" ]] || \
  fail "remote GOAT policy changed"
[[ "${dependency_values[9]}" == "${EXPECTED_OPENAI_CLIP_RN50_SHA}" ]] || \
  fail "remote OpenAI CLIP RN50 changed"

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "DRY_RUN_REMOTE_USER=${EXPECTED_REMOTE_USER}"
  echo "DRY_RUN_SSH_CONTROL_PATH=${SSH_CONTROL_PATH}"
  echo "DRY_RUN_FORMAL_MANIFEST_SHA=${FORMAL_MANIFEST_SHA}"
  echo "DRY_RUN_SMOKE_MANIFEST_SHA=${SMOKE_MANIFEST_SHA}"
  echo "DRY_RUN_OPENAI_CLIP_RN50_SHA=${EXPECTED_OPENAI_CLIP_RN50_SHA}"
  exit 0
fi

stage=$(mktemp -d /tmp/goat_sequential_revisit.XXXXXX)
remote_partial=${REMOTE_BUNDLE_BASE}/goat_sequential_revisit.partial.$$
cleanup() { find "${stage}" -depth -delete 2>/dev/null || true; }
trap cleanup EXIT
for relative in "${files[@]}"; do
  mkdir -p "${stage}/$(dirname "${relative}")"
  cp --preserve=mode,timestamps "${LOCAL_ROOT}/${relative}" "${stage}/${relative}"
done
mkdir -p "${stage}/third_party/openai_clip/clip"
for relative in \
  clip/__init__.py \
  clip/clip.py \
  clip/model.py \
  clip/simple_tokenizer.py \
  clip/bpe_simple_vocab_16e6.txt.gz; do
  cp --preserve=mode,timestamps "${LOCAL_OPENAI_CLIP_ROOT}/${relative}" \
    "${stage}/third_party/openai_clip/${relative}"
done
"${LOCAL_TEST_PY}" - "${stage}/third_party/openai_clip/upstream.json" \
  "${EXPECTED_OPENAI_CLIP_COMMIT}" <<'PY'
import json
import sys
open(sys.argv[1], "x").write(json.dumps({
    "package": "openai/CLIP",
    "upstream_commit": sys.argv[2],
}, indent=2, sort_keys=True) + "\n")
PY
mkdir -p "${stage}/third_party/goat_runtime"
for package in ftfy regex wcwidth; do
  rsync -a --exclude='__pycache__/' --exclude='*.py[co]' \
    "${LOCAL_GOAT_SITE_PACKAGES}/${package}/" \
    "${stage}/third_party/goat_runtime/${package}/"
done
for dist_info in ftfy-6.1.1.dist-info regex-2024.4.16.dist-info \
  wcwidth-0.2.14.dist-info; do
  rsync -a --exclude='RECORD' \
    "${LOCAL_GOAT_SITE_PACKAGES}/${dist_info}/" \
    "${stage}/third_party/goat_runtime/${dist_info}/"
done
"${LOCAL_TEST_PY}" - \
  "${stage}/third_party/goat_runtime/runtime_dependencies.json" \
  "${EXPECTED_FTFY_VERSION}" "${EXPECTED_REGEX_DISTRIBUTION_VERSION}" \
  "${EXPECTED_REGEX_MODULE_VERSION}" "${EXPECTED_WCWIDTH_VERSION}" <<'PY'
import json
import sys
open(sys.argv[1], "x").write(json.dumps({
    "packages": {
        "ftfy": {"distribution_version": sys.argv[2],
                 "module_version": sys.argv[2]},
        "regex": {"distribution_version": sys.argv[3],
                  "module_version": sys.argv[4]},
        "wcwidth": {"distribution_version": sys.argv[5],
                    "module_version": sys.argv[5]},
    },
    "python_abi": ".cpython-37m-x86_64-linux-gnu.so",
}, indent=2, sort_keys=True) + "\n")
PY
"${LOCAL_TEST_PY}" - "${stage}/dependency_receipt.json" \
  "${MEMNAV_CKPT}" "${dependency_values[0]}" "${EXPECTED_MEMNAV_CKPT_SHA}" \
  "${NAVDP_CKPT}" "${dependency_values[2]}" "${EXPECTED_NAVDP_CKPT_SHA}" \
  "${LINGBOT_WEIGHTS}" "${dependency_values[4]}" "${EXPECTED_LINGBOT_WEIGHTS_SHA}" \
  "${GOAT_POLICY_CKPT}" "${dependency_values[6]}" "${EXPECTED_GOAT_POLICY_SHA}" \
  "${OPENAI_CLIP_RN50}" "${dependency_values[8]}" \
  "${EXPECTED_OPENAI_CLIP_RN50_SHA}" <<'PY'
import json
import sys
names = ("gatecurr600", "navdp_checkpoint", "lingbot_map_long", "goat_policy",
         "openai_clip_rn50")
raw = sys.argv[2:]
dependencies = {}
for index, name in enumerate(names):
    path, size, digest = raw[index * 3:(index + 1) * 3]
    dependencies[name] = {"path": path, "bytes": int(size), "sha256": digest}
open(sys.argv[1], "x").write(json.dumps({
    "schema_version": "goat_sequential_revisit_dependencies_v1_20260815",
    "dependencies": dependencies,
}, indent=2, sort_keys=True) + "\n")
PY
"${LOCAL_TEST_PY}" - "${stage}/source_bundle_manifest.json" \
  "${FORMAL_MANIFEST_SHA}" "${SMOKE_MANIFEST_SHA}" \
  "${EXPECTED_BASE_SOURCE_RECEIPT_SHA}" \
  "${EXPECTED_OPENAI_CLIP_COMMIT}" \
  "${EXPECTED_OPENAI_CLIP_RN50_SHA}" \
  "${EXPECTED_FTFY_VERSION}" "${EXPECTED_REGEX_DISTRIBUTION_VERSION}" \
  "${EXPECTED_REGEX_MODULE_VERSION}" "${EXPECTED_WCWIDTH_VERSION}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import sys
out = pathlib.Path(sys.argv[1])
root = out.parent
payload = {
    "schema_version": "goat_sequential_revisit_bundle_v1_20260815",
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "frozen targeted external sequential-Revisit evaluation",
    "is_full_goat_benchmark_score": False,
    "method_or_threshold_selection_allowed": False,
    "formal_manifest_sha256": sys.argv[2],
    "smoke_manifest_sha256": sys.argv[3],
    "base_source_receipt_sha256": sys.argv[4],
    "openai_clip_upstream_commit": sys.argv[5],
    "openai_clip_rn50_sha256": sys.argv[6],
    "ftfy_version": sys.argv[7],
    "regex_distribution_version": sys.argv[8],
    "regex_module_version": sys.argv[9],
    "wcwidth_version": sys.argv[10],
    "files": {},
}
for path in sorted(root.rglob("*")):
    if path.is_file() and path != out:
        payload["files"][path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()).hexdigest()
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

remote "test ! -e '${remote_partial}' && mkdir -p '${remote_partial}'"
rsync -e "${RSYNC_RSH}" -a --chmod=Du=rwx,Dgo=rx,Fu=rw,Fgo=r \
  "${stage}/" "${REMOTE_HOST}:${remote_partial}/"
remote "cd '${remote_partial}' && find . -type f ! -name SOURCE_BUNDLE.sha256 -print0 | sort -z | xargs -0 sha256sum > SOURCE_BUNDLE.sha256 && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"
source_receipt_sha=$(remote "sha256sum '${remote_partial}/SOURCE_BUNDLE.sha256' | awk '{print \$1}'")
[[ "${source_receipt_sha}" =~ ^[0-9a-f]{64}$ ]] || fail "bad source receipt SHA"
remote_bundle=${REMOTE_BUNDLE_BASE}/goat_sequential_revisit_${source_receipt_sha:0:16}
remote "test ! -e '${remote_bundle}' && chmod -R a-w '${remote_partial}' && mv '${remote_partial}' '${remote_bundle}' && cd '${remote_bundle}' && sha256sum -c SOURCE_BUNDLE.sha256 >/dev/null"

tag=$(date -u +%Y%m%dT%H%M%SZ)
smoke_root=${REMOTE_RESULT_BASE}/smoke_${tag}
formal_root=${REMOTE_RESULT_BASE}/formal_${tag}
eval_launcher=${remote_bundle}/MemNavData/slurm_goat_sequential_revisit_eval.sbatch
preflight_launcher=${remote_bundle}/MemNavData/slurm_goat_sequential_revisit_preflight.sbatch
summary_launcher=${remote_bundle}/MemNavData/slurm_summarize_goat_sequential_revisit.sbatch
verify_launcher=${remote_bundle}/MemNavData/slurm_verify_goat_sequential_revisit.sbatch
common="SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${remote_bundle}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${source_receipt_sha},BASE_SOURCE_ROOT=${BASE_SOURCE_ROOT},EXPECTED_BASE_SOURCE_RECEIPT_SHA=${EXPECTED_BASE_SOURCE_RECEIPT_SHA}"
smoke_exports="ALL,${common},MANIFEST=${remote_bundle}/MemNavData/goat_sequential_revisit_smoke_manifest_20260815.json,EXPECTED_MANIFEST_SHA=${SMOKE_MANIFEST_SHA},RUN_ROOT=${smoke_root},RUN_MODE=smoke"
preflight_exports="ALL,${common},MANIFEST=${remote_bundle}/MemNavData/goat_sequential_revisit_smoke_manifest_20260815.json,EXPECTED_MANIFEST_SHA=${SMOKE_MANIFEST_SHA},RUN_ROOT=${smoke_root}"
formal_exports="ALL,${common},MANIFEST=${remote_bundle}/MemNavData/goat_sequential_revisit_formal_manifest_20260815.json,EXPECTED_MANIFEST_SHA=${FORMAL_MANIFEST_SHA},RUN_ROOT=${formal_root},RUN_MODE=formal"
post_exports="ALL,SOURCE_ROOT=${remote_bundle},SOURCE_RECEIPT=${remote_bundle}/SOURCE_BUNDLE.sha256,EXPECTED_SOURCE_RECEIPT_SHA=${source_receipt_sha},MANIFEST=${remote_bundle}/MemNavData/goat_sequential_revisit_formal_manifest_20260815.json,EXPECTED_MANIFEST_SHA=${FORMAL_MANIFEST_SHA},RUN_ROOT=${formal_root}"

remote "sbatch --test-only --export='${preflight_exports}' '${preflight_launcher}' >/dev/null"
remote "sbatch --test-only --array=1-1 --export='${smoke_exports}' '${eval_launcher}' >/dev/null"
remote "sbatch --test-only --array=0-33%${FORMAL_CONCURRENCY} --export='${formal_exports}' '${eval_launcher}' >/dev/null"
remote "test ! -e '${smoke_root}' && test ! -e '${formal_root}' && mkdir -p '${smoke_root}' '${formal_root}'"
preflight_raw=$(remote "sbatch --begin=now+3minutes --parsable --export='${preflight_exports}' '${preflight_launcher}'")
preflight_job=${preflight_raw%%;*}
[[ "${preflight_job}" =~ ^[0-9]+$ ]] || fail "bad preflight job id"
smoke_raw=$(remote "sbatch --array=1-1 --parsable --dependency=afterok:${preflight_job} --export='${smoke_exports}' '${eval_launcher}'")
smoke_job=${smoke_raw%%;*}
[[ "${smoke_job}" =~ ^[0-9]+$ ]] || fail "bad smoke job id"
formal_raw=$(remote "sbatch --array=0-33%${FORMAL_CONCURRENCY} --parsable --dependency=afterok:${smoke_job} --export='${formal_exports}' '${eval_launcher}'")
formal_job=${formal_raw%%;*}
[[ "${formal_job}" =~ ^[0-9]+$ ]] || fail "bad formal job id"
summary_raw=$(remote "sbatch --parsable --dependency=afterok:${formal_job} --export='${post_exports}' '${summary_launcher}'")
summary_job=${summary_raw%%;*}
[[ "${summary_job}" =~ ^[0-9]+$ ]] || fail "bad summary job id"
verify_raw=$(remote "sbatch --parsable --dependency=afterok:${summary_job} --export='${post_exports}' '${verify_launcher}'")
verify_job=${verify_raw%%;*}
[[ "${verify_job}" =~ ^[0-9]+$ ]] || fail "bad verifier job id"

remote "'${REMOTE_PY}' - '${smoke_root}/submission.json' '${formal_root}/submission.json' '${preflight_job}' '${smoke_job}' '${formal_job}' '${summary_job}' '${verify_job}' '${remote_bundle}' '${source_receipt_sha}' '${SMOKE_MANIFEST_SHA}' '${FORMAL_MANIFEST_SHA}' '${EXPECTED_OPENAI_CLIP_COMMIT}' '${EXPECTED_OPENAI_CLIP_RN50_SHA}' '${EXPECTED_FTFY_VERSION}' '${EXPECTED_REGEX_DISTRIBUTION_VERSION}' '${EXPECTED_REGEX_MODULE_VERSION}' '${EXPECTED_WCWIDTH_VERSION}'" <<'PY'
import json
import sys
import time
smoke_path, formal_path = sys.argv[1:3]
preflight_job, smoke_job, formal_job, summary_job, verify_job = map(
    int, sys.argv[3:8])
(bundle, receipt, smoke_manifest, formal_manifest, clip_commit, clip_rn50_sha,
 ftfy_version, regex_distribution_version, regex_module_version,
 wcwidth_version) = sys.argv[8:]
common = {
    "schema_version": "goat_sequential_revisit_submission_v1_20260815",
    "source_bundle": bundle,
    "source_receipt_sha256": receipt,
    "openai_clip_upstream_commit": clip_commit,
    "openai_clip_rn50_sha256": clip_rn50_sha,
    "ftfy_version": ftfy_version,
    "regex_distribution_version": regex_distribution_version,
    "regex_module_version": regex_module_version,
    "wcwidth_version": wcwidth_version,
    "is_full_goat_benchmark_score": False,
    "method_or_threshold_selection_allowed": False,
    "submission_unix_time": time.time(),
}
smoke = dict(common, mode="engineering_smoke", job_id=smoke_job,
             preflight_dependency_job_id=preflight_job,
             manifest_sha256=smoke_manifest)
formal = dict(common, mode="formal_targeted_external_evaluation",
              preflight_dependency_job_id=preflight_job,
              eval_job_id=formal_job, smoke_dependency_job_id=smoke_job,
              summary_job_id=summary_job, verifier_job_id=verify_job,
              manifest_sha256=formal_manifest, selected_scenes=34)
open(smoke_path, "x").write(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
open(formal_path, "x").write(json.dumps(formal, indent=2, sort_keys=True) + "\n")
PY
remote "chmod a-w '${smoke_root}/submission.json' '${formal_root}/submission.json'"

echo "GOAT_SEQ_PREFLIGHT_JOB=${preflight_job}"
echo "GOAT_SEQ_SMOKE_JOB=${smoke_job}"
echo "GOAT_SEQ_FORMAL_JOB=${formal_job}"
echo "GOAT_SEQ_SUMMARY_JOB=${summary_job}"
echo "GOAT_SEQ_VERIFY_JOB=${verify_job}"
echo "GOAT_SEQ_BUNDLE=${remote_bundle}"
echo "GOAT_SEQ_SMOKE_ROOT=${smoke_root}"
echo "GOAT_SEQ_FORMAL_ROOT=${formal_root}"
