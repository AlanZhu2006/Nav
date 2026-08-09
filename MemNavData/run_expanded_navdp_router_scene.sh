#!/usr/bin/env bash
# Run paired arms for one scene from the frozen expanded benchmark. The
# optional P0 arm changes only top-8 candidate order with Phase-B features;
# native NavDP and both geometry controls retain their existing defaults.

set -euo pipefail
umask 0022
export GIT_OPTIONAL_LOCKS=0

ROOT=${ROOT:?set ROOT}
RUN_ROOT=${RUN_ROOT:?set RUN_ROOT}
MANIFEST=${MANIFEST:?set MANIFEST}
EXPECTED_MANIFEST_SHA=${EXPECTED_MANIFEST_SHA:?set EXPECTED_MANIFEST_SHA}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set EXPECTED_COMMIT}
SCENE_INDEX=${SCENE_INDEX:?set SCENE_INDEX}
HAB_PY=${HAB_PY:?set HAB_PY}
MEMNAV_PY=${MEMNAV_PY:?set MEMNAV_PY}
MAX_STEPS=${MAX_STEPS:-500}
EPISODE_LIMIT=${EPISODE_LIMIT:-0}
LEG1_MODE=${LEG1_MODE:-policy}
STOP_AFTER_LEG1=${STOP_AFTER_LEG1:-0}
WRITE_LEG1_TRACE=${WRITE_LEG1_TRACE:-0}
DETERMINISTIC_PLAN_SEEDS=${DETERMINISTIC_PLAN_SEEDS:-0}
NAVDP_GOAL_SWITCH_RESET=${NAVDP_GOAL_SWITCH_RESET:-carry}
TRAJECTORY_SELECTOR=${TRAJECTORY_SELECTOR:-server}
TRAJECTORY_SELECTOR_SCOPE=${TRAJECTORY_SELECTOR_SCOPE:-all}
ORACLE_SELECTOR_HORIZON=${ORACLE_SELECTOR_HORIZON:-0}
ORACLE_CANDIDATE_SEED_COUNT=${ORACLE_CANDIDATE_SEED_COUNT:-1}
ORACLE_GLOBAL_SUBGOAL_M=${ORACLE_GLOBAL_SUBGOAL_M:-0.0}
SHARED_LEG1_ROOT=${SHARED_LEG1_ROOT:-}
RUN_NAVDP_NATIVE=${RUN_NAVDP_NATIVE:-1}
RUN_GEOMETRY_TOP1=${RUN_GEOMETRY_TOP1:-1}
RUN_GEOMETRY_ROUTER=${RUN_GEOMETRY_ROUTER:-1}
RUN_LEARNED_RANK_GEOMETRY=${RUN_LEARNED_RANK_GEOMETRY:-0}
P0_SHARED_PREFIX=${P0_SHARED_PREFIX:-0}
XNAVDP_REVISIT_GATE=${XNAVDP_REVISIT_GATE:-0}
XNAVDP_PY=${XNAVDP_PY:-${MEMNAV_PY}}
XNAVDP_OFFICIAL_ROOT=${XNAVDP_OFFICIAL_ROOT:-${ROOT}/.diagnostics/xnavdp_official_878740a2011856d0/NavDP}
XNAVDP_CKPT=${XNAVDP_CKPT:-${ROOT}/.diagnostics/xnavdp_official_878740a2011856d0/x-navdp_posttrain.ckpt}
EXPECTED_XNAVDP_COMMIT=${EXPECTED_XNAVDP_COMMIT:-878740a2011856d0e3782dd6ccd880fd2eccd70f}
EXPECTED_XNAVDP_CKPT_SHA=${EXPECTED_XNAVDP_CKPT_SHA:-267089a81bbbe7a913debda6603f3f1b66a79520370ce953b2d888d793b89f24}
RETRIEVAL_CANDIDATE_MIN_GAP=${RETRIEVAL_CANDIDATE_MIN_GAP:-16}
GRAPH_SUBGOAL_SPACING_M=${GRAPH_SUBGOAL_SPACING_M:-0.0}
GRAPH_SUBGOAL_ARRIVAL_M=${GRAPH_SUBGOAL_ARRIVAL_M:-0.60}
TERMINAL_UTURN=${TERMINAL_UTURN:-off}
TERMINAL_VISUAL_REFINE=${TERMINAL_VISUAL_REFINE:-off}
ARRIVAL_SHADOW=${ARRIVAL_SHADOW:-off}
UNIT_TEST_MODULE=${UNIT_TEST_MODULE:-MemNavData.test_expanded_navdp_router_eval}
EXPECTED_HAB_REQUESTS_VERSION=${EXPECTED_HAB_REQUESTS_VERSION:-2.32.4}
EXPECTED_HAB_REQUESTS_INIT_BYTES=5057
EXPECTED_HAB_REQUESTS_INIT_SHA=1e507f1f386bcc6b5f0ff69a614c14875cd65cb67be7f6022f28adef9774573f
EXPECTED_HAB_REQUESTS_VERSION_BYTES=435
EXPECTED_HAB_REQUESTS_VERSION_SHA=${EXPECTED_HAB_REQUESTS_VERSION_SHA:-143abaf3563712f063743a7952aa65319dbcb934d894cfc989bd2c015f8da577}

# The frozen Habitat Python has requests only in pip's pure-Python vendor
# directory.  Scope that path to Habitat subprocesses so it cannot pollute the
# Python 3.10 policy environment.
HAB_SITE_PACKAGES=$("${HAB_PY}" -c \
  'import sysconfig; print(sysconfig.get_paths()["purelib"])')
HAB_REQUESTS_VENDOR=${HAB_REQUESTS_VENDOR:-${HAB_SITE_PACKAGES}/pip/_vendor}
HAB_PYTHONPATH=${HAB_REQUESTS_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}
hab_python() {
  env PYTHONPATH="${HAB_PYTHONPATH}" "${HAB_PY}" "$@"
}

EVALUATOR=${EVALUATOR:-${ROOT}/MemNavData/eval_2leg_habitat.py}
VALIDATOR=${VALIDATOR:-${ROOT}/MemNavData/validate_expanded_navdp_router_eval.py}
MEMNAV_SERVER=${ROOT}/NavDP/baselines/memnav/memnav_server.py
NAVDP_SERVER=${ROOT}/NavDP/baselines/navdp/navdp_server.py
XNAVDP_SERVER=${ROOT}/MemNavData/xnavdp_revisit_server.py
INTERNNAV_ROOT=${ROOT}/InternNav
LONGCLIP_ENTRY=${INTERNNAV_ROOT}/internnav/model/basemodel/LongCLIP/model/longclip.py
LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
EXPECTED_LINGBOT_COMMIT=${EXPECTED_LINGBOT_COMMIT:-7ff6f3ed0913d4d326f8f13bbb429c4ffc0195c2}
MEMNAV_CKPT=${MEMNAV_CKPT:-/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/gatecurr600.memnav.ckpt}
NAVDP_CKPT=${NAVDP_CKPT:-/scratch/yz11502/Research/Nav-axis-uturn/.diagnostics/unseen_scene_eval_20260803/checkpoints/navdp_checkpoint.ckpt}
PHASE_B_CKPT=${PHASE_B_CKPT:-}
EXPECTED_PHASE_B_CKPT_SHA=${EXPECTED_PHASE_B_CKPT_SHA:-}
ASSET_ROOT_OVERRIDE=${ASSET_ROOT_OVERRIDE:-}
EPISODE_ROOT_OVERRIDE=${EPISODE_ROOT_OVERRIDE:-}
RUN_CONDITIONAL_ORACLES=${RUN_CONDITIONAL_ORACLES:-0}
RUN_CONDITIONAL_ORACLE_ANCHOR=${RUN_CONDITIONAL_ORACLE_ANCHOR:-${RUN_CONDITIONAL_ORACLES}}
RUN_CONDITIONAL_ORACLE_POINT=${RUN_CONDITIONAL_ORACLE_POINT:-${RUN_CONDITIONAL_ORACLES}}

TASK_FILES=(
  MemNavData/eval_2leg_habitat.py
  MemNavData/validate_expanded_navdp_router_eval.py
  MemNavData/summarize_expanded_navdp_router_eval.py
  MemNavData/summarize_phase_b_p0.py
  MemNavData/test_expanded_navdp_router_eval.py
  MemNavData/test_summarize_phase_b_p0.py
  MemNavData/test_router_candidates.py
  MemNavData/test_reverse_memory_graph.py
  MemNavData/test_policy_agent_graph.py
  MemNavData/phase_b_feature_schema.py
  MemNavData/phase_b_model.py
  MemNavData/phase_b_runtime.py
  MemNavData/test_phase_b_runtime.py
  MemNavData/test_phase_b_policy_cache.py
  MemNavData/terminal_uturn.py
  MemNavData/visual_yaw_refinement.py
  MemNavData/summarize_terminal_uturn.py
  MemNavData/test_terminal_uturn.py
  MemNavData/arrival_shadow.py
  MemNavData/test_arrival_shadow.py
  MemNavData/summarize_arrival_shadow.py
  MemNavData/test_summarize_arrival_shadow.py
  MemNavData/deterministic_eval_protocol.py
  MemNavData/navdp_goal_switch.py
  MemNavData/test_navdp_goal_switch.py
  MemNavData/global_subgoal_protocol.py
  MemNavData/test_global_subgoal_protocol.py
  MemNavData/observed_frontier.py
  MemNavData/test_observed_frontier.py
  MemNavData/test_deterministic_eval_protocol.py
  MemNavData/test_navdp_memory_replay.py
  MemNavData/xnavdp_revisit_contract.py
  MemNavData/xnavdp_revisit_server.py
  MemNavData/test_xnavdp_revisit_contract.py
  MemNavData/test_xnavdp_revisit_server.py
  MemNavData/summarize_xnavdp_revisit_gate.py
  MemNavData/test_summarize_xnavdp_revisit_gate.py
  MemNavData/XNAVDP_REVISIT_CONTROLLER_PROTOCOL_20260809.md
  MemNavData/slurm_xnavdp_revisit_gate.sbatch
  MemNavData/slurm_xnavdp_revisit_summary.sbatch
  MemNavData/submit_xnavdp_revisit_gate_hpc.sh
  MemNavData/conditional_c_protocol.py
  MemNavData/test_conditional_c_protocol.py
  MemNavData/summarize_conditional_c_eval.py
  MemNavData/test_summarize_conditional_c_eval.py
  MemNavData/run_expanded_navdp_router_scene.sh
  MemNavData/submit_phase_b_p0_hpc.sh
  MemNavData/slurm_expanded_navdp_router_eval.sbatch
  MemNavData/expanded_navdp_router_eval_20260805.json
  NavDP/baselines/memnav/memnav_server.py
  NavDP/baselines/memnav/policy_agent.py
  NavDP/baselines/memnav/pose_alignment.py
  NavDP/baselines/memnav/router_candidates.py
  NavDP/baselines/memnav/reverse_memory_graph.py
  InternNav/internnav/model/basemodel/memnav/memnav_policy.py
  NavDP/baselines/navdp/navdp_server.py
  NavDP/baselines/navdp/deterministic_seed.py
  NavDP/baselines/navdp/policy_agent.py
  NavDP/baselines/navdp/policy_network.py
)
[[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: MAX_STEPS must be a positive integer" >&2
  exit 1
}
[[ "${EPISODE_LIMIT}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: EPISODE_LIMIT must be a non-negative integer" >&2
  exit 1
}
[[ "${UNIT_TEST_MODULE}" =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]] || {
  echo "ABORT: invalid UNIT_TEST_MODULE=${UNIT_TEST_MODULE}" >&2
  exit 1
}
for flag in RUN_CONDITIONAL_ORACLES RUN_CONDITIONAL_ORACLE_ANCHOR \
            RUN_CONDITIONAL_ORACLE_POINT; do
  [[ "${!flag}" =~ ^[01]$ ]] || {
    echo "ABORT: ${flag} must be 0 or 1" >&2; exit 1; }
done
for flag in RUN_NAVDP_NATIVE RUN_GEOMETRY_TOP1 RUN_GEOMETRY_ROUTER \
            RUN_LEARNED_RANK_GEOMETRY P0_SHARED_PREFIX \
            XNAVDP_REVISIT_GATE; do
  [[ "${!flag}" =~ ^[01]$ ]] || {
    echo "ABORT: ${flag} must be 0 or 1" >&2; exit 1; }
done
if [[ "${P0_SHARED_PREFIX}" -eq 1 && "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  echo "ABORT: P0_SHARED_PREFIX and XNAVDP_REVISIT_GATE are exclusive" >&2
  exit 1
fi
if [[ "${P0_SHARED_PREFIX}" -eq 1 ]]; then
  [[ "${RUN_NAVDP_NATIVE}" -eq 1 \
      && "${RUN_GEOMETRY_TOP1}" -eq 0 \
      && "${RUN_GEOMETRY_ROUTER}" -eq 1 \
      && "${RUN_LEARNED_RANK_GEOMETRY}" -eq 1 ]] || {
    echo "ABORT: P0_SHARED_PREFIX requires exactly native/geometry/learned arms" >&2
    exit 1
  }
  [[ "${LEG1_MODE}" == policy && "${STOP_AFTER_LEG1}" -eq 0 \
      && "${WRITE_LEG1_TRACE}" -eq 0 && -z "${SHARED_LEG1_ROOT}" ]] || {
    echo "ABORT: P0_SHARED_PREFIX owns the Goal-A trace protocol" >&2
    exit 1
  }
  [[ "${DETERMINISTIC_PLAN_SEEDS}" -eq 1 \
      && "${NAVDP_GOAL_SWITCH_RESET}" == carry \
      && "${TRAJECTORY_SELECTOR}" == server \
      && "${TRAJECTORY_SELECTOR_SCOPE}" == all ]] || {
    echo "ABORT: P0_SHARED_PREFIX requires the deterministic server policy" >&2
    exit 1
  }
  [[ "${RUN_CONDITIONAL_ORACLE_ANCHOR}" -eq 0 \
      && "${RUN_CONDITIONAL_ORACLE_POINT}" -eq 0 ]] || {
    echo "ABORT: P0_SHARED_PREFIX cannot mix conditional oracle arms" >&2
    exit 1
  }
fi
if [[ "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  [[ "${RUN_NAVDP_NATIVE}" -eq 1 \
      && "${RUN_GEOMETRY_TOP1}" -eq 0 \
      && "${RUN_GEOMETRY_ROUTER}" -eq 1 \
      && "${RUN_LEARNED_RANK_GEOMETRY}" -eq 0 ]] || {
    echo "ABORT: XNAVDP_REVISIT_GATE requires native + mixed geometry only" >&2
    exit 1
  }
  [[ "${LEG1_MODE}" == policy && "${STOP_AFTER_LEG1}" -eq 0 \
      && "${WRITE_LEG1_TRACE}" -eq 0 && -z "${SHARED_LEG1_ROOT}" ]] || {
    echo "ABORT: XNAVDP_REVISIT_GATE owns the shared Goal-A trace" >&2
    exit 1
  }
  [[ "${DETERMINISTIC_PLAN_SEEDS}" -eq 1 \
      && "${NAVDP_GOAL_SWITCH_RESET}" == carry \
      && "${TRAJECTORY_SELECTOR}" == server \
      && "${TRAJECTORY_SELECTOR_SCOPE}" == all \
      && "${ORACLE_CANDIDATE_SEED_COUNT}" -eq 1 ]] || {
    echo "ABORT: XNAVDP_REVISIT_GATE requires deterministic carried server control" >&2
    exit 1
  }
  [[ "${RUN_CONDITIONAL_ORACLE_ANCHOR}" -eq 0 \
      && "${RUN_CONDITIONAL_ORACLE_POINT}" -eq 0 ]] || {
    echo "ABORT: XNAVDP_REVISIT_GATE cannot mix conditional oracles" >&2
    exit 1
  }
fi
for flag in STOP_AFTER_LEG1 WRITE_LEG1_TRACE DETERMINISTIC_PLAN_SEEDS; do
  [[ "${!flag}" =~ ^[01]$ ]] || {
    echo "ABORT: ${flag} must be 0 or 1" >&2; exit 1; }
done
[[ "${NAVDP_GOAL_SWITCH_RESET}" =~ ^(carry|before_b|every_goal)$ ]] || {
  echo "ABORT: invalid NAVDP_GOAL_SWITCH_RESET=${NAVDP_GOAL_SWITCH_RESET}" >&2
  exit 1
}
[[ "${TRAJECTORY_SELECTOR}" =~ ^(server|oracle_geodesic)$ ]] || {
  echo "ABORT: invalid TRAJECTORY_SELECTOR=${TRAJECTORY_SELECTOR}" >&2
  exit 1
}
[[ "${TRAJECTORY_SELECTOR_SCOPE}" =~ ^(all|leg_a|leg_b|leg_c)$ ]] || {
  echo "ABORT: invalid TRAJECTORY_SELECTOR_SCOPE=${TRAJECTORY_SELECTOR_SCOPE}" >&2
  exit 1
}
[[ "${ORACLE_SELECTOR_HORIZON}" =~ ^[0-9]+$ ]] || {
  echo "ABORT: ORACLE_SELECTOR_HORIZON must be non-negative" >&2
  exit 1
}
[[ "${ORACLE_CANDIDATE_SEED_COUNT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: ORACLE_CANDIDATE_SEED_COUNT must be positive" >&2
  exit 1
}
[[ "${ORACLE_CANDIDATE_SEED_COUNT}" -le 100 ]] || {
  echo "ABORT: ORACLE_CANDIDATE_SEED_COUNT must be at most 100" >&2
  exit 1
}
if [[ "${TRAJECTORY_SELECTOR}" == server \
      && "${TRAJECTORY_SELECTOR_SCOPE}" != all ]]; then
  echo "ABORT: a scoped selector requires TRAJECTORY_SELECTOR=oracle_geodesic" >&2
  exit 1
fi
if [[ "${ORACLE_CANDIDATE_SEED_COUNT}" -gt 1 ]]; then
  [[ "${TRAJECTORY_SELECTOR}" == oracle_geodesic ]] || {
    echo "ABORT: multi-seed candidates require oracle_geodesic" >&2; exit 1; }
  [[ "${DETERMINISTIC_PLAN_SEEDS}" -eq 1 ]] || {
    echo "ABORT: multi-seed candidates require deterministic seeds" >&2; exit 1; }
  [[ "${RUN_GEOMETRY_TOP1}" -eq 0 && "${RUN_GEOMETRY_ROUTER}" -eq 0 \
      && "${RUN_LEARNED_RANK_GEOMETRY}" -eq 0 ]] || {
    echo "ABORT: multi-seed diagnostic currently supports native NavDP only" >&2
    exit 1
  }
fi
if [[ "${TRAJECTORY_SELECTOR}" == server \
      && "${ORACLE_SELECTOR_HORIZON}" -ne 0 ]]; then
  echo "ABORT: ORACLE_SELECTOR_HORIZON requires oracle_geodesic" >&2
  exit 1
fi
if [[ "${NAVDP_GOAL_SWITCH_RESET}" != carry \
      && "${DETERMINISTIC_PLAN_SEEDS}" -ne 1 ]]; then
  echo "ABORT: goal-switch reset ablation requires deterministic plan seeds" >&2
  exit 1
fi
[[ "${LEG1_MODE}" =~ ^(policy|replay|shared_trace)$ ]] || {
  echo "ABORT: invalid LEG1_MODE=${LEG1_MODE}" >&2; exit 1; }
[[ "${TERMINAL_UTURN}" =~ ^(off|oracle|lingbot_yaw|lingbot_local|lingbot)$ ]] || {
  echo "ABORT: invalid TERMINAL_UTURN=${TERMINAL_UTURN}" >&2; exit 1; }
[[ "${TERMINAL_VISUAL_REFINE}" =~ ^(off|verify|refine)$ ]] || {
  echo "ABORT: invalid TERMINAL_VISUAL_REFINE=${TERMINAL_VISUAL_REFINE}" >&2
  exit 1
}
[[ "${ARRIVAL_SHADOW}" =~ ^(off|diagnostic)$ ]] || {
  echo "ABORT: invalid ARRIVAL_SHADOW=${ARRIVAL_SHADOW}" >&2; exit 1; }
if [[ "${TERMINAL_UTURN}" == off && "${TERMINAL_VISUAL_REFINE}" != off ]]; then
  echo "ABORT: terminal visual refinement requires a terminal U-turn" >&2
  exit 1
fi
if [[ "${LEG1_MODE}" == shared_trace ]]; then
  [[ -n "${SHARED_LEG1_ROOT}" ]] || {
    echo "ABORT: shared_trace requires SHARED_LEG1_ROOT" >&2; exit 1; }
else
  [[ -z "${SHARED_LEG1_ROOT}" ]] || {
    echo "ABORT: SHARED_LEG1_ROOT requires shared_trace" >&2; exit 1; }
fi
if [[ "${STOP_AFTER_LEG1}" -eq 1 || "${WRITE_LEG1_TRACE}" -eq 1 ]]; then
  [[ "${LEG1_MODE}" == policy ]] || {
    echo "ABORT: trace writing/Goal-A-only requires policy LEG1_MODE" >&2
    exit 1
  }
fi
(( RUN_NAVDP_NATIVE + RUN_GEOMETRY_TOP1 + RUN_GEOMETRY_ROUTER \
   + RUN_LEARNED_RANK_GEOMETRY > 0 )) || {
  echo "ABORT: at least one evaluation arm must be enabled" >&2; exit 1; }
[[ "${RETRIEVAL_CANDIDATE_MIN_GAP}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ABORT: RETRIEVAL_CANDIDATE_MIN_GAP must be positive" >&2; exit 1; }
if [[ "${RUN_LEARNED_RANK_GEOMETRY}" -eq 1 ]]; then
  test -r "${PHASE_B_CKPT}" || {
    echo "ABORT: learned P0 arm requires readable PHASE_B_CKPT" >&2
    exit 1
  }
  [[ "${EXPECTED_PHASE_B_CKPT_SHA}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ABORT: learned P0 arm requires EXPECTED_PHASE_B_CKPT_SHA" >&2
    exit 1
  }
  [[ "$(sha256sum "${PHASE_B_CKPT}" | awk '{print $1}')" == \
      "${EXPECTED_PHASE_B_CKPT_SHA}" ]] || {
    echo "ABORT: Phase-B checkpoint SHA256 mismatch" >&2
    exit 1
  }
  [[ "${RETRIEVAL_CANDIDATE_MIN_GAP}" -eq 16 ]] || {
    echo "ABORT: learned P0 arm requires candidate gap 16" >&2
    exit 1
  }
fi
"${HAB_PY}" - "${GRAPH_SUBGOAL_SPACING_M}" \
  "${GRAPH_SUBGOAL_ARRIVAL_M}" "${ORACLE_GLOBAL_SUBGOAL_M}" <<'PY'
import math, sys
spacing, arrival, oracle_global = map(float, sys.argv[1:])
if not math.isfinite(spacing) or spacing < 0:
    raise SystemExit("GRAPH_SUBGOAL_SPACING_M must be finite and non-negative")
if not math.isfinite(arrival) or arrival <= 0:
    raise SystemExit("GRAPH_SUBGOAL_ARRIVAL_M must be finite and positive")
if not math.isfinite(oracle_global) or oracle_global < 0:
    raise SystemExit("ORACLE_GLOBAL_SUBGOAL_M must be finite and non-negative")
PY
if hab_python - "${ORACLE_GLOBAL_SUBGOAL_M}" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)
PY
then
  [[ "$(basename "${EVALUATOR}")" == "eval_3leg_habitat.py" ]] || {
    echo "ABORT: oracle global subgoals require eval_3leg_habitat.py" >&2
    exit 1
  }
  [[ "${RUN_NAVDP_NATIVE}" -eq 1 && "${RUN_GEOMETRY_TOP1}" -eq 0 \
      && "${RUN_GEOMETRY_ROUTER}" -eq 0 \
      && "${RUN_LEARNED_RANK_GEOMETRY}" -eq 0 ]] || {
    echo "ABORT: oracle global subgoals require the native-only arm" >&2
    exit 1
  }
  [[ "${DETERMINISTIC_PLAN_SEEDS}" -eq 1 ]] || {
    echo "ABORT: oracle global subgoals require deterministic seeds" >&2
    exit 1
  }
  [[ "${TRAJECTORY_SELECTOR}" == server \
      && "${TRAJECTORY_SELECTOR_SCOPE}" == all ]] || {
    echo "ABORT: oracle global subgoals cannot mix trajectory selectors" >&2
    exit 1
  }
  [[ "${NAVDP_GOAL_SWITCH_RESET}" == carry ]] || {
    echo "ABORT: oracle global subgoals require carried short memory" >&2
    exit 1
  }
fi
if (( RUN_CONDITIONAL_ORACLE_ANCHOR + RUN_CONDITIONAL_ORACLE_POINT > 0 )); then
  [[ "$(basename "${EVALUATOR}")" == "eval_conditional_c_habitat.py" ]] || {
    echo "ABORT: conditional oracle arms require eval_conditional_c_habitat.py" >&2
    exit 1
  }
fi
UNIT_TEST_PATH=${ROOT}/${UNIT_TEST_MODULE//./\/}.py
for dynamic_path in "${EVALUATOR}" "${VALIDATOR}" "${MANIFEST}" \
                    "${UNIT_TEST_PATH}"; do
  dynamic_relative=$(realpath --relative-to="${ROOT}" "${dynamic_path}")
  [[ "${dynamic_relative}" != ../* && "${dynamic_relative}" != ".." ]] || {
    echo "ABORT: benchmark source must live inside ${ROOT}: ${dynamic_path}" >&2
    exit 1
  }
  TASK_FILES+=("${dynamic_relative}")
done

actual_commit=$(git -C "${ROOT}" rev-parse HEAD)
[[ "${actual_commit}" == "${EXPECTED_COMMIT}" ]] || {
  echo "ABORT: code commit ${actual_commit} != ${EXPECTED_COMMIT}" >&2
  exit 1
}
git -C "${ROOT}" ls-files --error-unmatch -- "${TASK_FILES[@]}" \
  >/dev/null || {
    echo "ABORT: a benchmark task input is not tracked by the commit" >&2
    exit 1
  }
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "ABORT: benchmark worktree is not completely clean" >&2
  git -C "${ROOT}" status --short >&2
  exit 1
}
git -C "${ROOT}" diff --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: benchmark task files differ from the checked-out commit" >&2
  exit 1
}
git -C "${ROOT}" diff --cached --quiet -- "${TASK_FILES[@]}" || {
  echo "ABORT: staged benchmark task files differ from the checked-out commit" >&2
  exit 1
}

for required in "${HAB_PY}" "${MEMNAV_PY}" "${EVALUATOR}" "${VALIDATOR}" \
                "${MEMNAV_SERVER}" "${NAVDP_SERVER}" "${LINGBOT_WEIGHTS}" \
                "${MEMNAV_CKPT}" "${NAVDP_CKPT}" "${MANIFEST}" \
                "${UNIT_TEST_PATH}" "${LONGCLIP_ENTRY}" \
                "${HAB_REQUESTS_VENDOR}/requests/__init__.py" \
                "${HAB_REQUESTS_VENDOR}/requests/__version__.py"; do
  test -r "${required}" || { echo "ABORT: missing dependency ${required}" >&2; exit 1; }
done
if [[ "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  for required in "${XNAVDP_PY}" "${XNAVDP_SERVER}" \
                  "${XNAVDP_OFFICIAL_ROOT}" "${XNAVDP_CKPT}"; do
    test -r "${required}" || {
      echo "ABORT: missing X-NavDP dependency ${required}" >&2; exit 1; }
  done
  actual_xnavdp_commit=$(git -c safe.directory="${XNAVDP_OFFICIAL_ROOT}" \
    -C "${XNAVDP_OFFICIAL_ROOT}" rev-parse HEAD)
  [[ "${actual_xnavdp_commit}" == "${EXPECTED_XNAVDP_COMMIT}" ]] || {
    echo "ABORT: X-NavDP commit ${actual_xnavdp_commit} != ${EXPECTED_XNAVDP_COMMIT}" >&2
    exit 1
  }
  [[ -z "$(git -c safe.directory="${XNAVDP_OFFICIAL_ROOT}" \
      -C "${XNAVDP_OFFICIAL_ROOT}" status --porcelain --untracked-files=no)" ]] || {
    echo "ABORT: official X-NavDP checkout has tracked changes" >&2
    exit 1
  }
  [[ "$(sha256sum "${XNAVDP_CKPT}" | awk '{print $1}')" == \
      "${EXPECTED_XNAVDP_CKPT_SHA}" ]] || {
    echo "ABORT: X-NavDP checkpoint SHA256 mismatch" >&2
    exit 1
  }
fi
actual_lingbot_commit=$(git -c safe.directory="${LINGBOT_REPO}" \
  -C "${LINGBOT_REPO}" rev-parse HEAD)
[[ "${actual_lingbot_commit}" == "${EXPECTED_LINGBOT_COMMIT}" ]] || {
  echo "ABORT: LingBot commit ${actual_lingbot_commit} != ${EXPECTED_LINGBOT_COMMIT}" >&2
  exit 1
}
git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" \
  diff --quiet || {
    echo "ABORT: LingBot tracked files differ from its pinned commit" >&2
    exit 1
  }
git -c safe.directory="${LINGBOT_REPO}" -C "${LINGBOT_REPO}" \
  diff --cached --quiet || {
    echo "ABORT: LingBot has staged changes" >&2
    exit 1
  }
REQUESTS_INIT=${HAB_REQUESTS_VENDOR}/requests/__init__.py
REQUESTS_VERSION=${HAB_REQUESTS_VENDOR}/requests/__version__.py
[[ "$(stat -c '%s' "${REQUESTS_INIT}")" == "${EXPECTED_HAB_REQUESTS_INIT_BYTES}" ]] || {
  echo "ABORT: vendored requests size mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${REQUESTS_INIT}" | awk '{print $1}')" == \
    "${EXPECTED_HAB_REQUESTS_INIT_SHA}" ]] || {
  echo "ABORT: vendored requests SHA256 mismatch" >&2
  exit 1
}
[[ "$(stat -c '%s' "${REQUESTS_VERSION}")" == \
    "${EXPECTED_HAB_REQUESTS_VERSION_BYTES}" ]] || {
  echo "ABORT: vendored requests version-file size mismatch" >&2
  exit 1
}
[[ "$(sha256sum "${REQUESTS_VERSION}" | awk '{print $1}')" == \
    "${EXPECTED_HAB_REQUESTS_VERSION_SHA}" ]] || {
  echo "ABORT: vendored requests version-file SHA256 mismatch" >&2
  exit 1
}

mkdir -p "${RUN_ROOT}/preflight" "${RUN_ROOT}/scenes"
VALIDATOR_ARGS=(
  --manifest "${MANIFEST}"
  --expected-manifest-sha "${EXPECTED_MANIFEST_SHA}"
  --scene-index "${SCENE_INDEX}"
  --gatecurr-checkpoint "${MEMNAV_CKPT}"
  --navdp-checkpoint "${NAVDP_CKPT}"
  --lingbot-weights "${LINGBOT_WEIGHTS}"
)
[[ -z "${ASSET_ROOT_OVERRIDE}" ]] || \
  VALIDATOR_ARGS+=(--asset-root "${ASSET_ROOT_OVERRIDE}")
[[ -z "${EPISODE_ROOT_OVERRIDE}" ]] || \
  VALIDATOR_ARGS+=(--episode-root "${EPISODE_ROOT_OVERRIDE}")
hab_python "${VALIDATOR}" "${VALIDATOR_ARGS[@]}" \
  > "${RUN_ROOT}/preflight/scene_$(printf '%02d' "${SCENE_INDEX}").json"

scene=$(hab_python - "${MANIFEST}" "${SCENE_INDEX}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["selection"]["selected_scenes"][int(sys.argv[2])])
PY
)
mapfile -t EPISODE_IDS < <(hab_python - "${MANIFEST}" "${scene}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
print(*(row["episode"] for row in manifest["episodes"][sys.argv[2]]), sep="\n")
PY
)
if (( EPISODE_LIMIT > 0 && EPISODE_LIMIT < ${#EPISODE_IDS[@]} )); then
  EPISODE_IDS=("${EPISODE_IDS[@]:0:EPISODE_LIMIT}")
fi
(( ${#EPISODE_IDS[@]} > 0 )) || {
  echo "ABORT: no episodes remain after EPISODE_LIMIT" >&2; exit 1; }
episode_csv=$(IFS=,; echo "${EPISODE_IDS[*]}")

SCENE_ROOT=${RUN_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}
if [[ -e "${SCENE_ROOT}" ]]; then
  echo "ABORT: scene output already exists: ${SCENE_ROOT}" >&2
  exit 1
fi
mkdir -p "${SCENE_ROOT}/logs" "${SCENE_ROOT}/buffer"
exec > >(tee "${SCENE_ROOT}/run.log") 2>&1

hab_python -m py_compile \
  "${EVALUATOR}" \
  "${VALIDATOR}" \
  "${ROOT}/MemNavData/terminal_uturn.py" \
  "${ROOT}/MemNavData/arrival_shadow.py" \
  "${ROOT}/MemNavData/global_subgoal_protocol.py" \
  "${ROOT}/MemNavData/observed_frontier.py" \
  "${ROOT}/MemNavData/xnavdp_revisit_contract.py" \
  "${ROOT}/MemNavData/summarize_xnavdp_revisit_gate.py" \
  "${ROOT}/MemNavData/summarize_arrival_shadow.py" \
  "${ROOT}/MemNavData/visual_yaw_refinement.py" \
  "${ROOT}/MemNavData/summarize_terminal_uturn.py"
"${MEMNAV_PY}" -m py_compile \
  "${ROOT}/MemNavData/phase_b_model.py" \
  "${ROOT}/MemNavData/phase_b_runtime.py" \
  "${ROOT}/MemNavData/deterministic_eval_protocol.py" \
  "${ROOT}/MemNavData/navdp_goal_switch.py" \
  "${MEMNAV_SERVER}" \
  "${ROOT}/NavDP/baselines/memnav/policy_agent.py" \
  "${ROOT}/NavDP/baselines/memnav/router_candidates.py" \
  "${ROOT}/NavDP/baselines/memnav/reverse_memory_graph.py" \
  "${NAVDP_SERVER}" \
  "${ROOT}/NavDP/baselines/navdp/deterministic_seed.py"
(
  cd "${ROOT}"
  hab_python -m unittest \
    "${UNIT_TEST_MODULE}" \
    MemNavData.test_summarize_phase_b_p0 \
    MemNavData.test_router_candidates \
    MemNavData.test_reverse_memory_graph \
    MemNavData.test_terminal_uturn \
    MemNavData.test_arrival_shadow \
    MemNavData.test_summarize_arrival_shadow \
    MemNavData.test_conditional_c_protocol \
    MemNavData.test_global_subgoal_protocol \
    MemNavData.test_observed_frontier \
    MemNavData.test_summarize_xnavdp_revisit_gate \
    MemNavData.test_navdp_goal_switch \
    MemNavData.test_summarize_conditional_c_eval -v
)
if [[ "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  "${XNAVDP_PY}" -m py_compile \
    "${ROOT}/MemNavData/xnavdp_revisit_contract.py" \
    "${XNAVDP_SERVER}"
  (
    cd "${ROOT}"
    "${XNAVDP_PY}" -m unittest \
      MemNavData.test_xnavdp_revisit_contract \
      MemNavData.test_xnavdp_revisit_server -v
  )
  "${XNAVDP_PY}" -c \
    'import torch,cv2,flask,PIL,scipy; assert torch.cuda.is_available(); print("X-NavDP dependencies OK", torch.__version__)'
fi
(
  cd "${ROOT}"
  "${MEMNAV_PY}" -m unittest \
    MemNavData.test_policy_agent_graph \
    MemNavData.test_phase_b_policy_cache \
    MemNavData.test_phase_b_runtime \
    MemNavData.test_deterministic_eval_protocol \
    MemNavData.test_navdp_memory_replay -v
)
hab_python -c \
  'import habitat_sim,numpy,pandas,pyarrow,PIL,requests,scipy,quaternion,sys; assert requests.__version__ == sys.argv[1]; print("Habitat dependencies OK", habitat_sim.__version__, "requests", requests.__version__)' \
  "${EXPECTED_HAB_REQUESTS_VERSION}"
"${MEMNAV_PY}" -c \
  'import torch,torchvision,transformers,diffusers,cv2,flask,imageio; assert torch.cuda.is_available(); print("Policy dependencies OK", torch.__version__)'

port_key=$(( (${SLURM_JOB_ID:-1000} + SCENE_INDEX * 37) % 15000 ))
MEMNAV_PORT=${MEMNAV_PORT:-$((20000 + port_key * 2))}
NAVDP_PORT=${NAVDP_PORT:-$((MEMNAV_PORT + 1))}
XNAVDP_PORT=${XNAVDP_PORT:-$((MEMNAV_PORT + 2))}
PORTS=("${MEMNAV_PORT}" "${NAVDP_PORT}")
if [[ "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  PORTS+=("${XNAVDP_PORT}")
fi
for port in "${PORTS[@]}"; do
  if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
    echo "ABORT: port ${port} is already in use" >&2
    exit 1
  fi
done

RUNTIME_ROOT=${SLURM_TMPDIR:-/tmp}/memnav_expanded_${SLURM_JOB_ID:-local}_${SCENE_INDEX}
mkdir -p "${RUNTIME_ROOT}/memnav" "${RUNTIME_ROOT}/navdp" \
  "${RUNTIME_ROOT}/xnavdp"
MEMNAV_PID=
NAVDP_PID=
XNAVDP_PID=
cleanup() {
  for pid in "${XNAVDP_PID}" "${NAVDP_PID}" "${MEMNAV_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

PHASE_B_SERVER_ARGS=()
if [[ "${RUN_LEARNED_RANK_GEOMETRY}" -eq 1 ]]; then
  PHASE_B_SERVER_ARGS=(
    --phase_b_checkpoint "${PHASE_B_CKPT}"
    --phase_b_allow_unapproved
  )
fi

(
  cd "${RUNTIME_ROOT}/memnav"
  exec env \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${INTERNNAV_ROOT}/src/diffusion-policy:${PYTHONPATH:-}" \
    LINGBOT_REPO="${LINGBOT_REPO}" \
    LINGBOT_WEIGHTS="${LINGBOT_WEIGHTS}" \
    MEMNAV_WINDOW=32 \
    MEMNAV_NUM_SCALE=8 \
    MEMNAV_MAX_FRAME_NUM=2048 \
    MEMNAV_GROUND_SCALE_MAX=6.0 \
    MEMNAV_GATE_FUSION=complementary \
    MEMNAV_AUX_POSE_CALIBRATION=empirical \
    MEMNAV_COLLISION_SELECT=1 \
    MEMNAV_REPORT_TO=none \
    "${MEMNAV_PY}" -u "${MEMNAV_SERVER}" \
      --port "${MEMNAV_PORT}" \
      --checkpoint "${MEMNAV_CKPT}" \
      --internnav_root "${INTERNNAV_ROOT}" \
      --num_samples 16 \
      --exclude_recent 32 \
      --retrieval raw \
      --retrieval_candidate_top_k 32 \
      --retrieval_candidate_min_gap "${RETRIEVAL_CANDIDATE_MIN_GAP}" \
      --graph_subgoal_spacing_m "${GRAPH_SUBGOAL_SPACING_M}" \
      --graph_subgoal_arrival_m "${GRAPH_SUBGOAL_ARRIVAL_M}" \
      --flow_gate auto \
      "${PHASE_B_SERVER_ARGS[@]}" \
      --buffer_root "${SCENE_ROOT}/buffer"
) > "${SCENE_ROOT}/logs/server_memnav.log" 2>&1 &
MEMNAV_PID=$!

(
  cd "${RUNTIME_ROOT}/navdp"
  exec env NAVDP_DISABLE_VIDEO=1 PYTHONUNBUFFERED=1 \
    "${MEMNAV_PY}" -u "${NAVDP_SERVER}" \
      --port "${NAVDP_PORT}" --checkpoint "${NAVDP_CKPT}"
) > "${SCENE_ROOT}/logs/server_navdp.log" 2>&1 &
NAVDP_PID=$!

if [[ "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  (
    cd "${RUNTIME_ROOT}/xnavdp"
    exec env PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${XNAVDP_PY}" -u "${XNAVDP_SERVER}" \
        --official-root "${XNAVDP_OFFICIAL_ROOT}" \
        --checkpoint "${XNAVDP_CKPT}" \
        --device cuda:0 \
        --embodiment wheeled \
        --actor-mode posttrain \
        --host 127.0.0.1 \
        --port "${XNAVDP_PORT}"
  ) > "${SCENE_ROOT}/logs/server_xnavdp.log" 2>&1 &
  XNAVDP_PID=$!
fi

SERVER_SPECS=(
  "memnav:${MEMNAV_PID}:${MEMNAV_PORT}:${SCENE_ROOT}/logs/server_memnav.log"
  "navdp:${NAVDP_PID}:${NAVDP_PORT}:${SCENE_ROOT}/logs/server_navdp.log"
)
if [[ "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  SERVER_SPECS+=(
    "xnavdp:${XNAVDP_PID}:${XNAVDP_PORT}:${SCENE_ROOT}/logs/server_xnavdp.log")
fi
for spec in "${SERVER_SPECS[@]}"; do
  IFS=: read -r label pid port log <<<"${spec}"
  ready=0
  for _ in $(seq 1 240); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "ABORT: ${label} server exited during startup" >&2
      tail -n 120 "${log}" >&2
      exit 1
    fi
    if ss -ltn | awk '{print $4}' | grep -Eq "(^|:)${port}$"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "${ready}" -eq 1 ]] || {
    echo "ABORT: ${label} server did not bind port ${port}" >&2
    tail -n 120 "${log}" >&2
    exit 1
  }
done

MANIFEST_ASSET_ROOT=$(hab_python - "${MANIFEST}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["paths"]["asset_root"])
PY
)
MANIFEST_EPISODE_ROOT=$(hab_python - "${MANIFEST}" "${scene}" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
if "episode_root" in manifest["paths"]:
    key = "episode_root"
else:
    key = ("legacy_anchor_episode_root"
           if sys.argv[2] in manifest["selection"]["anchor_scenes"]
           else "expanded_episode_root")
print(manifest["paths"][key])
PY
)
ASSET_ROOT=${ASSET_ROOT_OVERRIDE:-${MANIFEST_ASSET_ROOT}}
EPISODE_ROOT=${EPISODE_ROOT_OVERRIDE:-${MANIFEST_EPISODE_ROOT}}
SCENE_FILE=${ASSET_ROOT}/${scene}/${scene}.glb
EPISODE_SCENE_ROOT=${EPISODE_ROOT}/${scene}
test -r "${SCENE_FILE}" || {
  echo "ABORT: missing scene asset ${SCENE_FILE}" >&2; exit 1; }
test -d "${EPISODE_SCENE_ROOT}" || {
  echo "ABORT: missing episode scene root ${EPISODE_SCENE_ROOT}" >&2; exit 1; }
COMMON_ARGS=(
  --episode_root "${EPISODE_SCENE_ROOT}"
  --scene "${SCENE_FILE}"
  --host 127.0.0.1
  --success_dist 1.0
  --max_steps "${MAX_STEPS}"
  --exec_horizon 8
  --trajectory_selector "${TRAJECTORY_SELECTOR}"
  --trajectory_selector_scope "${TRAJECTORY_SELECTOR_SCOPE}"
  --oracle_selector_horizon "${ORACLE_SELECTOR_HORIZON}"
  --oracle_candidate_seed_count "${ORACLE_CANDIDATE_SEED_COUNT}"
  --oracle_global_subgoal_m "${ORACLE_GLOBAL_SUBGOAL_M}"
  --navdp_goal_switch_reset "${NAVDP_GOAL_SWITCH_RESET}"
  --leg1_goal_source own
  --seed 20260803
  --terminal_uturn "${TERMINAL_UTURN}"
  --terminal_visual_refine "${TERMINAL_VISUAL_REFINE}"
  --arrival_shadow "${ARRIVAL_SHADOW}"
  --episode_ids "${episode_csv}"
)
DEFAULT_LEG1_ARGS=(--leg1_mode "${LEG1_MODE}")
if [[ "${STOP_AFTER_LEG1}" -eq 1 ]]; then
  DEFAULT_LEG1_ARGS+=(--stop_after_leg1)
fi
if [[ "${WRITE_LEG1_TRACE}" -eq 1 ]]; then
  DEFAULT_LEG1_ARGS+=(--write_leg1_trace)
fi
if [[ "${DETERMINISTIC_PLAN_SEEDS}" -eq 1 ]]; then
  COMMON_ARGS+=(--deterministic_plan_seeds)
fi
if [[ "${LEG1_MODE}" == shared_trace ]]; then
  SHARED_SCENE_ROOT=${SHARED_LEG1_ROOT}/scenes/$(printf '%02d' "${SCENE_INDEX}")_${scene}/geometry_router
  test -d "${SHARED_SCENE_ROOT}" || {
    echo "ABORT: shared Goal-A trace root missing: ${SHARED_SCENE_ROOT}" >&2
    exit 1
  }
  DEFAULT_LEG1_ARGS+=(--shared_leg1_trace_root "${SHARED_SCENE_ROOT}")
fi

ARMS=()
run_native_arm() {
  local -a leg1_args=("$@")
  echo "[eval] scene=${scene} arm=navdp_native episodes=${episode_csv}"
  mkdir -p "${SCENE_ROOT}/navdp_native"
  (
    cd "${RUNTIME_ROOT}"
    hab_python -u "${EVALUATOR}" \
      "${COMMON_ARGS[@]}" \
      "${leg1_args[@]}" \
      --port "${NAVDP_PORT}" \
      --out "${SCENE_ROOT}/navdp_native" \
      --server_backend navdp
  ) > "${SCENE_ROOT}/logs/eval_navdp_native.log" 2>&1
  ARMS+=(navdp_native)
}

run_geometry_arm() {
  local arm=$1
  local verify_top_k=$2
  local route=${3:-memory_geometry}
  shift 3
  local -a leg1_args=("$@")
  echo "[eval] scene=${scene} arm=${arm} route=${route} verify_top_k=${verify_top_k} episodes=${episode_csv}"
  mkdir -p "${SCENE_ROOT}/${arm}"
  (
    cd "${RUNTIME_ROOT}"
    hab_python -u "${EVALUATOR}" \
      "${COMMON_ARGS[@]}" \
      "${leg1_args[@]}" \
      --port "${MEMNAV_PORT}" \
      --novel_port "${NAVDP_PORT}" \
      --out "${SCENE_ROOT}/${arm}" \
      --server_backend hybrid_pose \
      --hybrid_route "${route}" \
      --router_visual_floor 0.88 \
      --router_min_matches 20 \
      --router_min_inliers 12 \
      --router_min_inlier_ratio 0.50 \
      --router_confirm_plans 2 \
      --router_verify_top_k "${verify_top_k}"
  ) > "${SCENE_ROOT}/logs/eval_${arm}.log" 2>&1
}

run_revisit_controller_arm() {
  local arm=$1
  local controller=$2
  shift 2
  local -a leg1_args=("$@")
  local -a xnavdp_args=()
  if [[ "${controller}" == xnavdp_point ]]; then
    xnavdp_args=(--xnavdp_port "${XNAVDP_PORT}")
  fi
  echo "[eval] scene=${scene} arm=${arm} controller=${controller} episodes=${episode_csv}"
  mkdir -p "${SCENE_ROOT}/${arm}"
  (
    cd "${RUNTIME_ROOT}"
    hab_python -u "${EVALUATOR}" \
      "${COMMON_ARGS[@]}" \
      "${leg1_args[@]}" \
      --port "${MEMNAV_PORT}" \
      --novel_port "${NAVDP_PORT}" \
      "${xnavdp_args[@]}" \
      --out "${SCENE_ROOT}/${arm}" \
      --server_backend hybrid_pose \
      --revisit_controller "${controller}" \
      --hybrid_route memory_geometry \
      --router_visual_floor 0.88 \
      --router_min_matches 20 \
      --router_min_inliers 12 \
      --router_min_inlier_ratio 0.50 \
      --router_confirm_plans 2 \
      --router_verify_top_k 8
  ) > "${SCENE_ROOT}/logs/eval_${arm}.log" 2>&1
}

if [[ "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  # The mixed arm materializes Goal A exactly once.  Every other controller
  # replays its byte-checked trace, so controller attribution begins only at B.
  run_revisit_controller_arm memory_mixed navdp_mixed \
    --leg1_mode policy --write_leg1_trace
  ARMS+=(memory_mixed)
  XNAVDP_SHARED_ARGS=(
    --leg1_mode shared_trace
    --shared_leg1_trace_root "${SCENE_ROOT}/memory_mixed"
  )
  # Alternate the two PointGoal controllers across scenes to expose any
  # process-order artifact without changing the frozen mixed trace source.
  if (( SCENE_INDEX % 2 == 0 )); then
    run_revisit_controller_arm memory_base_point navdp_point \
      "${XNAVDP_SHARED_ARGS[@]}"
    ARMS+=(memory_base_point)
    run_revisit_controller_arm memory_xnavdp_point xnavdp_point \
      "${XNAVDP_SHARED_ARGS[@]}"
    ARMS+=(memory_xnavdp_point)
  else
    run_revisit_controller_arm memory_xnavdp_point xnavdp_point \
      "${XNAVDP_SHARED_ARGS[@]}"
    ARMS+=(memory_xnavdp_point)
    run_revisit_controller_arm memory_base_point navdp_point \
      "${XNAVDP_SHARED_ARGS[@]}"
    ARMS+=(memory_base_point)
  fi
  run_native_arm "${XNAVDP_SHARED_ARGS[@]}"
elif [[ "${P0_SHARED_PREFIX}" -eq 1 ]]; then
  # Generate Goal A exactly once on this node, then replay the trace through
  # both comparison arms while preserving each server's short-memory stream.
  # This keeps every paired closed-loop claim within one job and one pair of
  # long-lived policy processes.
  run_geometry_arm geometry_router 8 memory_geometry \
    --leg1_mode policy --write_leg1_trace
  ARMS+=(geometry_router)
  P0_SHARED_ARGS=(
    --leg1_mode shared_trace
    --shared_leg1_trace_root "${SCENE_ROOT}/geometry_router"
  )
  run_geometry_arm learned_rank_geometry 8 learned_rank_geometry \
    "${P0_SHARED_ARGS[@]}"
  ARMS+=(learned_rank_geometry)
  run_native_arm "${P0_SHARED_ARGS[@]}"
else
  if [[ "${RUN_NAVDP_NATIVE}" -eq 1 ]]; then
    run_native_arm "${DEFAULT_LEG1_ARGS[@]}"
  fi
  # Run top-1 first so its state cannot inherit the top-K accepted anchor.
  # Each evaluator resets policy RNG, stream memory, and the router latch.
  if [[ "${RUN_GEOMETRY_TOP1}" -eq 1 ]]; then
    run_geometry_arm geometry_top1 1 memory_geometry \
      "${DEFAULT_LEG1_ARGS[@]}"
    ARMS+=(geometry_top1)
  fi
  if [[ "${RUN_GEOMETRY_ROUTER}" -eq 1 ]]; then
    run_geometry_arm geometry_router 8 memory_geometry \
      "${DEFAULT_LEG1_ARGS[@]}"
    ARMS+=(geometry_router)
  fi
  if [[ "${RUN_LEARNED_RANK_GEOMETRY}" -eq 1 ]]; then
    run_geometry_arm learned_rank_geometry 8 learned_rank_geometry \
      "${DEFAULT_LEG1_ARGS[@]}"
    ARMS+=(learned_rank_geometry)
  fi
fi
run_conditional_oracle_arm() {
  local arm=$1
  local mode=$2
  echo "[eval] scene=${scene} arm=${arm} conditional_mode=${mode} episodes=${episode_csv}"
  mkdir -p "${SCENE_ROOT}/${arm}"
  (
    cd "${RUNTIME_ROOT}"
    hab_python -u "${EVALUATOR}" \
      --conditional_c_mode "${mode}" \
      "${COMMON_ARGS[@]}" \
      "${DEFAULT_LEG1_ARGS[@]}" \
      --port "${MEMNAV_PORT}" \
      --novel_port "${NAVDP_PORT}" \
      --out "${SCENE_ROOT}/${arm}" \
      --server_backend hybrid_pose \
      --hybrid_route memory_geometry \
      --router_visual_floor 0.88 \
      --router_min_matches 20 \
      --router_min_inliers 12 \
      --router_min_inlier_ratio 0.50 \
      --router_confirm_plans 2 \
      --router_verify_top_k 8
  ) > "${SCENE_ROOT}/logs/eval_${arm}.log" 2>&1
}

if [[ "${RUN_CONDITIONAL_ORACLE_ANCHOR}" -eq 1 ]]; then
  run_conditional_oracle_arm oracle_anchor oracle_anchor
  ARMS+=(oracle_anchor)
fi
if [[ "${RUN_CONDITIONAL_ORACLE_POINT}" -eq 1 ]]; then
  run_conditional_oracle_arm oracle_point oracle_point
  ARMS+=(oracle_point)
fi

for arm in "${ARMS[@]}"; do
  test -s "${SCENE_ROOT}/${arm}/metric.csv" || {
    echo "ABORT: ${arm} did not produce metric.csv" >&2; exit 1; }
  test -s "${SCENE_ROOT}/${arm}/summary.json" || {
    echo "ABORT: ${arm} did not produce summary.json" >&2; exit 1; }
  count=$(($(wc -l < "${SCENE_ROOT}/${arm}/metric.csv") - 1))
  [[ "${count}" -eq "${#EPISODE_IDS[@]}" ]] || {
    echo "ABORT: ${arm} produced ${count} rows, expected ${#EPISODE_IDS[@]}" >&2
    exit 1
  }
done

if [[ "${XNAVDP_REVISIT_GATE}" -eq 1 ]]; then
  hab_python - "${ROOT}" "${SCENE_ROOT}" "${scene}" "${episode_csv}" \
      > "${SCENE_ROOT}/xnavdp_scene_audit.json" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from MemNavData.summarize_xnavdp_revisit_gate import (
    ARMS, _validate_pair, load_gate_arm, require)

scene_root = Path(sys.argv[2])
scene = sys.argv[3]
episodes = [value for value in sys.argv[4].split(",") if value]
expected = {(scene, episode) for episode in episodes}
rows = {arm: load_gate_arm(scene_root, arm, scene) for arm in ARMS}
for arm in ARMS:
    require(set(rows[arm]) == expected, f"{arm} scene result keys differ")
for arm in ARMS:
    if arm == "memory_mixed":
        continue
    for key in sorted(expected):
        _validate_pair(
            "memory_mixed", arm,
            rows["memory_mixed"][key], rows[arm][key], key)
print(json.dumps({
    "status": "ok",
    "scene": scene,
    "episodes": episodes,
    "arms": list(ARMS),
    "shared_goal_a_trace_match": True,
    "xnavdp_history_contract_valid": True,
    "xnavdp_checkpoint_coverage_valid": True,
}, indent=2, sort_keys=True))
PY
fi

if [[ "${RUN_LEARNED_RANK_GEOMETRY}" -eq 1 ]]; then
  hab_python - "${SCENE_ROOT}/learned_rank_geometry/summary.json" \
      "${EXPECTED_PHASE_B_CKPT_SHA}" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1]))
ranker = summary.get("phase_b_ranker") or {}
if ranker.get("checkpoint_sha256") != sys.argv[2]:
    raise SystemExit("learned arm checkpoint identity mismatch")
if summary.get("phase_b_p0_transport_valid") is not True:
    raise SystemExit("learned arm had ranking fallback/activation violation")
PY
fi

echo "[complete] scene=${scene} output=${SCENE_ROOT}"
