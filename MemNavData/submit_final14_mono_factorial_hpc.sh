#!/usr/bin/env bash
# Freeze, upload, smoke-test, and submit the consumed Final14 mono factorial.
set -euo pipefail
umask 0022

ROOT=${ROOT:-/home/asus/Research/Nav-graph-blind}
SSH_ALIAS=${SSH_ALIAS:-alantorch}
REMOTE_BUNDLES=/scratch/yz11502/Research/Nav-axis-uturn-source-bundles
REMOTE_RESULTS=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_mono_factorial_20260819
BENCH_ROOT=/scratch/yz11502/Research/Nav-axis-uturn-results/final14_cec_learned_20260817/final14_learned_20260817T115533Z_attempt7_handoff/benchmarks/natural_direction
MANIFEST_SHA=7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a
SOURCE_OVERLAY=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
EXPECTED_SOURCE_OVERLAY_BYTES=128854888448
LOCAL_PY=/home/asus/miniconda3/envs/memnav/bin/python
cd "${ROOT}"

"${LOCAL_PY}" -m py_compile \
  MemNavData/final14_mono_factorial.py \
  MemNavData/run_final14_mono_factorial_episode.py \
  MemNavData/summarize_final14_mono_factorial.py \
  MemNavData/independent_verify_final14_mono_factorial.py \
  MemNavData/audit_final14_mono_factorial_inputs.py \
  MemNavData/eval_shared_online_role_pairs.py \
  NavDP/baselines/memnav/memnav_server.py \
  NavDP/baselines/navdp/navdp_server.py
PYTHONPATH="${ROOT}:${ROOT}/MemNavData" "${LOCAL_PY}" -m unittest \
  MemNavData.test_final14_mono_factorial \
  MemNavData.test_mdtec_raw_depth_gate_d \
  MemNavData.test_monocular_depth_runtime \
  MemNavData.test_shared_online_role_pair_contract \
  MemNavData.test_navdp_memory_replay \
  MemNavData.test_policy_agent_graph \
  MemNavData.test_deterministic_eval_protocol
bash -n \
  MemNavData/run_final14_mono_factorial_history.sh \
  MemNavData/slurm_final14_mono_factorial.sbatch \
  MemNavData/slurm_final14_mono_factorial_summary.sbatch
"${LOCAL_PY}" -m json.tool \
  MemNavData/final14_mono_factorial_protocol_20260819.json >/dev/null

ssh "${SSH_ALIAS}" "test -r '${BENCH_ROOT}/manifest.json' && test \"\$(sha256sum '${BENCH_ROOT}/manifest.json' | cut -d' ' -f1)\" = '${MANIFEST_SHA}' && test -r '${SOURCE_OVERLAY}' && test \"\$(stat -c %s '${SOURCE_OVERLAY}')\" -eq '${EXPECTED_SOURCE_OVERLAY_BYTES}' && singularity exec --overlay '${SOURCE_OVERLAY}:ro' /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif test -r /mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg/8WUmhLawc2A/episode_0000/data/chunk-000/episode_000000.parquet"

staging=$(mktemp -d)
cleanup() { rm -rf -- "${staging}"; }
trap cleanup EXIT
files=(
  MemNavData/FINAL14_MONO_FACTORIAL_PROTOCOL_20260819.md
  MemNavData/final14_mono_factorial_protocol_20260819.json
  MemNavData/final14_mono_factorial.py
  MemNavData/run_final14_mono_factorial_episode.py
  MemNavData/summarize_final14_mono_factorial.py
  MemNavData/independent_verify_final14_mono_factorial.py
  MemNavData/audit_final14_mono_factorial_inputs.py
  MemNavData/run_final14_mono_factorial_history.sh
  MemNavData/slurm_final14_mono_factorial.sbatch
  MemNavData/slurm_final14_mono_factorial_summary.sbatch
  MemNavData/submit_final14_mono_factorial_hpc.sh
  MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT1_OVERLAY_INCIDENT_20260819.json
  MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT2_PREFLIGHT_INCIDENT_20260819.json
  MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT3_PY39_INCIDENT_20260819.json
  MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT4_PYCACHE_INCIDENT_20260819.json
  MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT5_BUNDLE_IDENTITY_INCIDENT_20260819.json
)
while IFS= read -r path; do files+=("${path}"); done < <(
  find MemNavData -maxdepth 1 -type f -name '*.py' -print | sort)
while IFS= read -r path; do files+=("${path}"); done < <(
  find NavDP/baselines/memnav NavDP/baselines/navdp -maxdepth 1 -type f \
    -name '*.py' -print | sort)
while IFS= read -r path; do files+=("${path}"); done < <(
  find NavDP/baselines/navdp/depth_anything/depth_anything_v2 -type f \
    ! -path '*/__pycache__/*' ! -name '*.pyc' -print | sort)
while IFS= read -r path; do files+=("${path}"); done < <(
  find InternNav/internnav InternNav/scripts/train/configs \
       InternNav/src/diffusion-policy -type f \
    ! -path '*/__pycache__/*' ! -name '*.pyc' -print | sort)
printf '%s\n' "${files[@]}" | sort -u >"${staging}/file_list.txt"
while IFS= read -r path; do
  [[ -r "${path}" ]] || { echo "missing bundle input ${path}" >&2; exit 1; }
  mkdir -p "${staging}/root/$(dirname "${path}")"
  cp -p -- "${path}" "${staging}/root/${path}"
done <"${staging}/file_list.txt"

(
  cd "${staging}/root"
  PYTHONPATH="${staging}/root:${staging}/root/MemNavData" "${LOCAL_PY}" \
    -m unittest \
      MemNavData.test_final14_mono_factorial \
      MemNavData.test_mdtec_raw_depth_gate_d \
      MemNavData.test_monocular_depth_runtime \
      MemNavData.test_shared_online_role_pair_contract
  PYTHONPATH="${staging}/root" "${LOCAL_PY}" -m py_compile \
    MemNavData/run_final14_mono_factorial_episode.py \
    MemNavData/summarize_final14_mono_factorial.py \
    MemNavData/independent_verify_final14_mono_factorial.py \
    MemNavData/eval_shared_online_role_pairs.py \
    NavDP/baselines/memnav/memnav_server.py \
    NavDP/baselines/navdp/navdp_server.py
)

LIGHTGLUE_SOURCE=${LIGHTGLUE_SOURCE:-${ROOT}/.diagnostics/dependencies/LightGlue}
DEPENDENCY_SOURCE=${DEPENDENCY_SOURCE:-${ROOT}/.diagnostics/dependencies/python}
TORCH_CHECKPOINT_SOURCE=${TORCH_CHECKPOINT_SOURCE:-/home/asus/.cache/torch/hub/checkpoints}
mkdir -p "${staging}/root/third_party/LightGlue" \
  "${staging}/root/third_party/python" \
  "${staging}/root/torch_home/hub/checkpoints"
cp -a "${LIGHTGLUE_SOURCE}/lightglue" "${staging}/root/third_party/LightGlue/"
cp --preserve=mode,timestamps "${LIGHTGLUE_SOURCE}/LICENSE" \
  "${staging}/root/third_party/LightGlue/LICENSE"
for dependency in kornia kornia-0.8.1.dist-info \
                  kornia_rs kornia_rs-0.1.9.dist-info; do
  cp -a "${DEPENDENCY_SOURCE}/${dependency}" "${staging}/root/third_party/python/"
done
cp --preserve=mode,timestamps \
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_v1.pth" \
  "${TORCH_CHECKPOINT_SOURCE}/superpoint_lightglue_v0-1_arxiv.pth" \
  "${staging}/root/torch_home/hub/checkpoints/"
find "${staging}/root" -type d -name __pycache__ -prune -exec rm -rf -- {} +
(
  cd "${staging}/root"
  find . -type f -print0 | sort -z | xargs -0 sha256sum \
    >"${staging}/source_inputs.sha256"
)
mv "${staging}/source_inputs.sha256" "${staging}/root/source_inputs.sha256"
receipt_sha=$(sha256sum "${staging}/root/source_inputs.sha256" | awk '{print $1}')
bundle_tag=final14_mono_factorial_${receipt_sha:0:16}
remote_root=${REMOTE_BUNDLES}/${bundle_tag}
run_tag=formal_$(date -u +%Y%m%dT%H%M%SZ)_${receipt_sha:0:8}
run_root=${REMOTE_RESULTS}/${run_tag}
smoke_root=${REMOTE_RESULTS}/${run_tag}_smoke

ssh "${SSH_ALIAS}" "test ! -e '${remote_root}' && mkdir -p '${remote_root}' '${REMOTE_RESULTS}' /scratch/yz11502/Research/Nav-axis-uturn-results/slurm_logs"
rsync -a "${staging}/root/" "${SSH_ALIAS}:${remote_root}/"
ssh "${SSH_ALIAS}" "cd '${remote_root}' && sha256sum -c --quiet source_inputs.sha256 && test \"\$(sha256sum source_inputs.sha256 | cut -d' ' -f1)\" = '${receipt_sha}' && chmod -R a-w '${remote_root}'"
ssh "${SSH_ALIAS}" "singularity exec --overlay '${SOURCE_OVERLAY}:ro' -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /scratch/lg154/conda-envs/memnav/bin/python '${remote_root}/MemNavData/audit_final14_mono_factorial_inputs.py' --manifest '${BENCH_ROOT}/manifest.json' --expected-manifest-sha256 '${MANIFEST_SHA}'"
ssh "${SSH_ALIAS}" "singularity exec --overlay '${SOURCE_OVERLAY}:ro' -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONPYCACHEPREFIX='/tmp/f14mono_${receipt_sha}' /scratch/lg154/conda-envs/habitat/bin/python -m py_compile '${remote_root}/MemNavData/run_final14_mono_factorial_episode.py' '${remote_root}/MemNavData/independent_verify_final14_mono_factorial.py'"
ssh "${SSH_ALIAS}" "singularity exec --overlay '${SOURCE_OVERLAY}:ro' -B /scratch/lg154 -B /scratch/yz11502 /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif env PYTHONPYCACHEPREFIX='/tmp/f14mono_${receipt_sha}' PYTHONPATH='${remote_root}:${remote_root}/MemNavData' /scratch/lg154/conda-envs/habitat/bin/python -m unittest MemNavData.test_final14_mono_factorial"

common_export="SOURCE_ROOT='${remote_root}',BENCH_ROOT='${BENCH_ROOT}',SOURCE_RECEIPT='${remote_root}/source_inputs.sha256',EXPECTED_SOURCE_RECEIPT_SHA='${receipt_sha}',SOURCE_OVERLAY='${SOURCE_OVERLAY}',EXPECTED_SOURCE_OVERLAY_BYTES='${EXPECTED_SOURCE_OVERLAY_BYTES}'"
smoke_job=$(ssh "${SSH_ALIAS}" "sbatch --parsable --array=0 --export=ALL,${common_export},RUN_ROOT='${smoke_root}',SMOKE=1,MAX_STEPS=80 '${remote_root}/MemNavData/slurm_final14_mono_factorial.sbatch'")
array_job=$(ssh "${SSH_ALIAS}" "sbatch --parsable --dependency=afterok:${smoke_job} --kill-on-invalid-dep=yes --array=0-20%10 --export=ALL,${common_export},RUN_ROOT='${run_root}',SMOKE=0,MAX_STEPS=600 '${remote_root}/MemNavData/slurm_final14_mono_factorial.sbatch'")
summary_job=$(ssh "${SSH_ALIAS}" "sbatch --parsable --dependency=afterok:${array_job} --kill-on-invalid-dep=yes --export=ALL,${common_export},RUN_ROOT='${run_root}' '${remote_root}/MemNavData/slurm_final14_mono_factorial_summary.sbatch'")
receipt=MemNavData/FINAL14_MONO_FACTORIAL_ATTEMPT6_SUBMISSION_RECEIPT_20260819.json
"${LOCAL_PY}" - "${receipt}" "${remote_root}" "${run_root}" "${smoke_root}" \
  "${receipt_sha}" "${smoke_job}" "${array_job}" "${summary_job}" <<'PY'
import json,sys
path,source,run,smoke,sha,smoke_job,array_job,summary_job=sys.argv[1:]
payload={
  "schema_version":"final14_mono_factorial_submission_receipt_v1_20260819",
  "source_root":source,"source_receipt_sha256":sha,
  "run_root":run,"smoke_root":smoke,
  "smoke_job":smoke_job,"formal_array_job":array_job,
  "summary_job":summary_job,
  "benchmark_manifest_sha256":"7468703a9efbb10e801ffdd226911f696a30fa9432ef9ab486d3134f6e40fe6a",
  "formal_history_count":21,
  "scope":"consumed_final14_query_controller_depth_attribution"
}
open(path,"x").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
printf 'SOURCE_ROOT=%s\nRUN_ROOT=%s\nSOURCE_RECEIPT_SHA=%s\nSMOKE_JOB=%s\nARRAY_JOB=%s\nSUMMARY_JOB=%s\n' \
  "${remote_root}" "${run_root}" "${receipt_sha}" \
  "${smoke_job}" "${array_job}" "${summary_job}"
