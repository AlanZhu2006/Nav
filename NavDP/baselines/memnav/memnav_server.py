"""MemNav Flask server — the model half of the two-process closed-loop eval.

Runs in the `memnav` conda env. The Habitat client (habitat env) streams RGB
frames over HTTP; this server maintains the live LingBot memory and plans
trajectories toward a goal image on request.

Endpoints (NavDP wire-contract style):
  POST /navigator_reset      JSON {camera_height?, seed?, episode_len?}
                             -> {"algo": "memnav"}   (starts a fresh episode)
  POST /memory_step          files: image (jpg)      -> {"frame_idx": i}
                             stream a frame into memory WITHOUT planning (leg replay)
  POST /monocular_depth_query form: expected_image_sha256
                             -> current raw LingBot depth PNG + immutable
                                first-40 scale receipt; never appends a frame
  POST /imagegoal_step       files: image (jpg), goal (jpg)
                             -> {"trajectory": [24,3] metres (x fwd, y left, theta),
                                 "all_trajectory": [N,24,3], "all_values": [N],
                                 "gate": float, "match_idx": int, "frame_idx": int,
                                 "goal_rel_yaw": float|null,
                                 "current_goal_cos": float}
                             streams the frame, then plans toward the goal.
  POST /posegoal_step        files: image (jpg), goal (jpg)
                             -> retrieval/gate/metric-pose diagnostics only;
                             streams the frame but skips MemNav diffusion.
  POST /posegoal_query       files: goal (jpg)
                             -> metric pose for the already-streamed latest
                             frame; does not append another observation.
  POST /retrieval_probe_step files: image (jpg), goal (jpg)
                             -> cheap retrieval diagnostics; streams the frame
                             but skips goal-pose recovery and diffusion.
  POST /retrieval_verify     files: goal (jpg), form: anchor
                             -> CPU two-view geometric overlap diagnostics;
                             does not mutate streaming model state.
  POST /phase_b_rank         files: goal (jpg), form: candidates (JSON)
                             -> learned ordering of the frozen DINO shortlist;
                             no activation decision and no stream mutation.
  POST /certified_anchor_image files: goal (jpg), form: selected_anchor,
                                  expected_anchor_sha256
                             -> exact causal history JPEG already authorized by
                                the cached accepted CEC proof; no stream mutation.
  POST /learned_relocalize   files: goal (jpg), form: candidates (JSON)
                             -> frozen Pi3X spatial-proof decision/bearing;
                             no frame append and fail-closed on every error.
  POST /imagegoal_similarity files: image (jpg), goal (jpg)
                             -> {"current_goal_cos": float}
                             stateless visual check; does not mutate memory.

Usage:
  conda activate memnav
  python memnav_server.py --port 18888 \
    --checkpoint /home/asus/Research/Nav/InternNav/checkpoints/memnav_2leg_axisfix/checkpoint-1500/memnav.ckpt
"""

import argparse
import json
import os
import sys

# must precede any torch import (policy_agent): reduces fragmentation OOMs from
# the large KV-cache alloc/free cycle each plan() runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from flask import Flask, jsonify, request
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=18888)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--internnav_root", type=str,
                    default="/home/asus/Research/Nav/InternNav")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_samples", type=int, default=16)
parser.add_argument("--exclude_recent", type=int, default=83,
                    help="retrieval candidate gap (dataset default)")
parser.add_argument("--retrieval", choices=["head", "raw"], default="raw",
                    help="match selector: trained projection vs raw dino-cls cosine")
parser.add_argument("--gate_skip_below", type=float, default=0.0,
                    help=("skip the goal-insert tower when trained gate < this "
                          "(0 = never skip); this does not hard-clamp the soft "
                          "decoder gate"))
parser.add_argument("--anchor_switch_margin", type=float, default=0.01,
                    help="sticky-anchor ratchet: switch match only on a clear score win")
parser.add_argument("--retrieval_candidate_top_k", type=int, default=32,
                    help="number of temporally diverse raw-DINO candidates to expose")
parser.add_argument("--retrieval_candidate_min_gap", type=int, default=16,
                    help="minimum frame gap between exposed retrieval candidates")
parser.add_argument(
    "--graph_subgoal_spacing_m", type=float, default=0.0,
    help=("metre spacing for reverse-memory graph subgoals; zero preserves "
          "the direct image-conditioned point goal"),
)
parser.add_argument(
    "--graph_subgoal_arrival_m", type=float, default=0.60,
    help="advance to the next reverse-memory node inside this radius",
)
parser.add_argument("--flow_gate", type=str, default="auto",
                    help="auto = training length tier; off = dense; or fixed px threshold")
parser.add_argument("--buffer_root", type=str, default="/tmp/memnav_server_buffer")
parser.add_argument(
    "--phase_b_checkpoint",
    type=str,
    default="",
    help=("experimental Phase-B checkpoint used only to rank a fixed DINO "
          "shortlist; empty disables the learned ranker"),
)
parser.add_argument(
    "--phase_b_allow_unapproved",
    action="store_true",
    help=("explicitly permit a deployment_approved=false checkpoint for the "
          "audited P0 closed-loop experiment"),
)
parser.add_argument(
    "--certified_relocalization",
    action="store_true",
    help=("enable the frozen SuperPoint+LightGlue + LingBot-depth PnP v2 "
          "endpoint; disabled leaves all historical routes unchanged"),
)
parser.add_argument(
    "--certified_counterfactual_audit",
    action="store_true",
    help=("diagnostic only: run the unchanged PnP certificate in canonical "
          "DINO order, stopping at the first accepted hypothesis, in addition "
          "to the deployed geometry proposal; extra attempts never change "
          "the action"),
)
parser.add_argument(
    "--certified_eager_depth_cache",
    action="store_true",
    help=("maintain an exact dense LingBot depth cache during history writes "
          "to remove selected-anchor replay latency; default keeps lazy replay"),
)
parser.add_argument(
    "--lightglue_repo", type=str, default="",
    help="pinned official LightGlue checkout (required when enabled)",
)
parser.add_argument(
    "--lightglue_dependency_root", type=str, default="",
    help="optional directory containing pinned kornia dependencies",
)
parser.add_argument(
    "--lightglue_max_keypoints", type=int, default=2048,
    help="frozen SuperPoint keypoint budget for certified relocalization",
)
parser.add_argument(
    "--cdec_pairwise_artifact", type=str, default="",
    help=("optional factorized learned anchor proposal; it is consulted only "
          "after the geometry proposal fails the unchanged certificate"),
)
parser.add_argument(
    "--cdec_pairwise_allow_unapproved",
    action="store_true",
    help=("explicit research-only override for a deployment_approved=false "
          "CDEC artifact"),
)
parser.add_argument(
    "--pi3x_learned_relocalizer",
    action="store_true",
    help=("enable the frozen DINO-top8/Pi3X-b16 learned spatial-proof "
          "endpoint; independent of LightGlue/PnP certificate support"),
)
parser.add_argument(
    "--pi3x_root", type=str, default="",
    help="pinned official Pi3 source checkout (required when enabled)",
)
parser.add_argument(
    "--pi3x_snapshot", type=str, default="",
    help="local Pi3X model snapshot containing model.safetensors",
)
parser.add_argument(
    "--pi3x_model_sha256", type=str, default="",
    help="required SHA256 of the frozen Pi3X model.safetensors",
)
parser.add_argument(
    "--pi3x_spatial_proof_manifest", type=str, default="",
    help="frozen four-member learned spatial-proof deployment manifest",
)
parser.add_argument(
    "--pi3x_inference_dtype",
    choices=["auto", "bfloat16", "float16", "float32"],
    default="auto",
)
parser.add_argument(
    "--synchronize_cuda_http_handoff",
    action="store_true",
    help=("finish this server's queued CUDA work before returning an HTTP "
          "response to a client that may immediately use another CUDA/EGL "
          "context on the same GPU; scheduling only, never a model input"),
)
args = parser.parse_args()

# The server changes directory below because LingBot historically resolves a
# few script-local resources from this module. Resolve every CLI filesystem
# argument first: otherwise a perfectly valid relative checkpoint silently
# becomes missing after chdir and ``from_pretrained`` leaves all trainable
# MemNav heads randomly initialized.
args.checkpoint = os.path.abspath(args.checkpoint)
args.internnav_root = os.path.abspath(args.internnav_root)
args.buffer_root = os.path.abspath(args.buffer_root)
args.phase_b_checkpoint = (
    os.path.abspath(args.phase_b_checkpoint)
    if args.phase_b_checkpoint else ""
)
args.lightglue_repo = (
    os.path.abspath(args.lightglue_repo) if args.lightglue_repo else "")
args.lightglue_dependency_root = (
    os.path.abspath(args.lightglue_dependency_root)
    if args.lightglue_dependency_root else "")
args.cdec_pairwise_artifact = (
    os.path.abspath(args.cdec_pairwise_artifact)
    if args.cdec_pairwise_artifact else "")
args.pi3x_root = (
    os.path.abspath(args.pi3x_root) if args.pi3x_root else "")
args.pi3x_snapshot = (
    os.path.abspath(args.pi3x_snapshot) if args.pi3x_snapshot else "")
args.pi3x_spatial_proof_manifest = (
    os.path.abspath(args.pi3x_spatial_proof_manifest)
    if args.pi3x_spatial_proof_manifest else "")
if not os.path.isfile(args.checkpoint):
    raise FileNotFoundError(f"MemNav checkpoint not found: {args.checkpoint}")
if not os.path.isdir(args.internnav_root):
    raise FileNotFoundError(f"InternNav root not found: {args.internnav_root}")
if args.phase_b_allow_unapproved and not args.phase_b_checkpoint:
    parser.error("--phase_b_allow_unapproved requires --phase_b_checkpoint")
if args.phase_b_checkpoint and not os.path.isfile(args.phase_b_checkpoint):
    raise FileNotFoundError(
        f"Phase-B checkpoint not found: {args.phase_b_checkpoint}")
if args.certified_relocalization:
    if not args.lightglue_repo or not os.path.isdir(args.lightglue_repo):
        parser.error(
            "--certified_relocalization requires --lightglue_repo")
    if (args.lightglue_dependency_root
            and not os.path.isdir(args.lightglue_dependency_root)):
        parser.error("--lightglue_dependency_root is not a directory")
    if args.lightglue_max_keypoints != 2048:
        parser.error("the frozen certificate requires 2048 keypoints")
if args.certified_counterfactual_audit and not args.certified_relocalization:
    parser.error(
        "--certified_counterfactual_audit requires "
        "--certified_relocalization")
if args.certified_eager_depth_cache and not args.certified_relocalization:
    parser.error(
        "--certified_eager_depth_cache requires --certified_relocalization")
if args.cdec_pairwise_allow_unapproved and not args.cdec_pairwise_artifact:
    parser.error(
        "--cdec_pairwise_allow_unapproved requires --cdec_pairwise_artifact")
if args.cdec_pairwise_artifact:
    if not args.certified_relocalization:
        parser.error(
            "--cdec_pairwise_artifact requires --certified_relocalization")
    if not os.path.isfile(args.cdec_pairwise_artifact):
        raise FileNotFoundError(
            f"CDEC pairwise artifact not found: {args.cdec_pairwise_artifact}")
if args.pi3x_learned_relocalizer:
    if not args.pi3x_root or not os.path.isdir(args.pi3x_root):
        parser.error("--pi3x_learned_relocalizer requires --pi3x_root")
    if not args.pi3x_snapshot or not os.path.isdir(args.pi3x_snapshot):
        parser.error("--pi3x_learned_relocalizer requires --pi3x_snapshot")
    if not os.path.isfile(os.path.join(
            args.pi3x_snapshot, "model.safetensors")):
        parser.error("--pi3x_snapshot must contain model.safetensors")
    if (len(args.pi3x_model_sha256) != 64
            or any(character not in "0123456789abcdef"
                   for character in args.pi3x_model_sha256.lower())):
        parser.error("--pi3x_model_sha256 must be a 64-character SHA256")
    if (not args.pi3x_spatial_proof_manifest
            or not os.path.isfile(args.pi3x_spatial_proof_manifest)):
        parser.error(
            "--pi3x_learned_relocalizer requires "
            "--pi3x_spatial_proof_manifest")

repo_root = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
phase_b_ranker = None
if args.phase_b_checkpoint:
    from MemNavData.phase_b_runtime import PhaseBEnsembleRanker
    phase_b_ranker = PhaseBEnsembleRanker(
        args.phase_b_checkpoint,
        allow_unapproved=args.phase_b_allow_unapproved,
    )

certified_relocalization_matcher = None
if args.certified_relocalization:
    from pathlib import Path
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher

    certified_relocalization_matcher = LightGluePointMatcher(
        Path(args.lightglue_repo),
        dependency_root=(Path(args.lightglue_dependency_root)
                         if args.lightglue_dependency_root else None),
        device=args.device,
        max_keypoints=args.lightglue_max_keypoints,
    )

cdec_pairwise_ranker = None
if args.cdec_pairwise_artifact:
    from MemNavData.cdec_pairwise_runtime import CDECPairwiseRanker
    cdec_pairwise_ranker = CDECPairwiseRanker(
        args.cdec_pairwise_artifact,
        allow_unapproved=args.cdec_pairwise_allow_unapproved,
    )

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from policy_agent import MemNavAgent  # noqa: E402  (after chdir so lingbot paths resolve)

agent = MemNavAgent(
    checkpoint=args.checkpoint,
    internnav_root=args.internnav_root,
    device=args.device,
    exclude_recent=args.exclude_recent,
    num_samples=args.num_samples,
    buffer_root=args.buffer_root,
    gate_skip_below=args.gate_skip_below,
    retrieval_mode=args.retrieval,
    anchor_switch_margin=args.anchor_switch_margin,
    retrieval_candidate_top_k=args.retrieval_candidate_top_k,
    retrieval_candidate_min_gap=args.retrieval_candidate_min_gap,
    graph_subgoal_spacing_m=args.graph_subgoal_spacing_m,
    graph_subgoal_arrival_m=args.graph_subgoal_arrival_m,
    flow_gate=args.flow_gate,
    phase_b_ranker=phase_b_ranker,
    certified_relocalization_matcher=certified_relocalization_matcher,
    certified_counterfactual_audit=args.certified_counterfactual_audit,
    certified_eager_depth_cache=args.certified_eager_depth_cache,
    cdec_pairwise_ranker=cdec_pairwise_ranker,
)

if args.pi3x_learned_relocalizer:
    from pathlib import Path
    from MemNavData.pi3x_online_relocalizer import Pi3XOnlineRelocalizer

    # Load after MemNav/LingBot so co-residency failures are explicit at server
    # startup rather than appearing halfway through an episode.
    agent.pi3x_online_relocalizer = Pi3XOnlineRelocalizer(
        pi3_root=Path(args.pi3x_root),
        snapshot=Path(args.pi3x_snapshot),
        expected_model_sha256=args.pi3x_model_sha256.lower(),
        proof_manifest=Path(args.pi3x_spatial_proof_manifest),
        device=args.device,
        inference_dtype=args.pi3x_inference_dtype,
    )

app = Flask(__name__)


@app.after_request
def synchronize_cuda_http_handoff(response):
    """Make the cross-process GPU ownership boundary explicit when requested."""
    if args.synchronize_cuda_http_handoff and torch.cuda.is_available():
        torch.cuda.synchronize(agent.device)
    return response


@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    payload = request.get_json(silent=True) or {}
    cam_h = float(payload.get("camera_height", 0.5))
    seed = payload.get("seed")
    episode_len = payload.get("episode_len")
    agent.reset(
        camera_height=cam_h,
        seed=seed,
        episode_len=episode_len,
        camera_intrinsic=payload.get("camera_intrinsic"),
    )
    return jsonify({
        "algo": "memnav",
        "flow_threshold": agent.flow_threshold,
        "retrieval": agent.retrieval_mode,
        "checkpoint": args.checkpoint,
        "exclude_recent": agent.exclude_recent,
        "retrieval_candidate_min_gap": agent.retrieval_candidate_min_gap,
        "graph_subgoal_spacing_m": agent.graph_subgoal_spacing_m,
        "graph_subgoal_arrival_m": agent.graph_subgoal_arrival_m,
        "synchronize_cuda_http_handoff": bool(
            args.synchronize_cuda_http_handoff),
        "phase_b_ranker": agent.phase_b_status(),
        "certified_relocalization": agent.certified_relocalization_status(),
        "learned_pi3x_relocalization": (
            agent.learned_pi3x_relocalization_status()),
        "certified_arrival": agent.certified_arrival_status(),
        "monocular_depth": agent.monocular_depth_status(),
    })


@app.route("/navigator_reset_env", methods=["POST"])
def navigator_reset_env():
    # single-env server: same as a full reset (used by the cold/reset-memory arm)
    agent.reset(camera_height=agent.camera_height, seed=agent._last_seed,
                episode_len=agent._last_episode_len,
                camera_intrinsic=agent.camera_intrinsic)
    return jsonify({"algo": "memnav"})


@app.route("/memory_step", methods=["POST"])
def memory_step():
    idx = agent.add_frame(request.files["image"].read())
    return jsonify({
        "frame_idx": idx,
        "monocular_depth": agent.monocular_depth_status(),
    })


@app.route("/monocular_depth_query", methods=["POST"])
def monocular_depth_query():
    """Read the latest frozen-LingBot depth without advancing the stream."""

    expected = request.form.get("expected_image_sha256")
    if expected is None:
        return jsonify({
            "error": "expected_image_sha256 is required",
            "metric_depth_sensor_consumed": False,
        }), 400
    try:
        payload = agent.monocular_depth_observation()
    except Exception as error:
        return jsonify({
            "error": f"{type(error).__name__}: {error}",
            "metric_depth_sensor_consumed": False,
        }), 500
    if payload.get("image_sha256") != expected:
        return jsonify({
            "error": "latest sidecar frame does not match current NavDP JPEG",
            "expected_image_sha256": expected,
            "latest_image_sha256": payload.get("image_sha256"),
            "frame_index": payload.get("frame_index"),
            "metric_depth_sensor_consumed": False,
        }), 409
    return jsonify(payload)


@app.route("/arrival_query", methods=["POST"])
def arrival_query():
    """Return read-only LingBot/PnP evidence for the latest streamed frame.

    This endpoint cannot emit or authorize GOAT ``SUBTASK_STOP`` because it
    does not receive the native-zero trigger.  Authorization remains a pure
    client-side conjunction under the frozen contract.
    """

    if "goal" not in request.files:
        return jsonify({"error": "goal image is required"}), 400
    raw_goal_intrinsic = request.form.get("goal_camera_intrinsic")
    goal_camera_intrinsic = None
    if raw_goal_intrinsic not in (None, ""):
        try:
            goal_camera_intrinsic = json.loads(raw_goal_intrinsic)
        except json.JSONDecodeError as error:
            return jsonify({
                "status": "invalid_goal_camera_intrinsic_json",
                "error": str(error),
            }), 400
    result = agent.certify_current_image_goal_arrival(
        request.files["goal"].read(),
        goal_camera_intrinsic=goal_camera_intrinsic)
    return jsonify(result)


def candidate_ceiling_override():
    value = request.form.get("candidate_ceiling_override")
    return int(value) if value is not None else None


def with_goal_session_receipt(payload):
    payload = dict(payload)
    payload.update(agent.goal_session_status())
    return payload


@app.route("/imagegoal_step", methods=["POST"])
def imagegoal_step():
    agent.add_frame(request.files["image"].read())
    forced_anchor = request.form.get("forced_anchor")
    forced_gate = request.form.get("forced_gate")
    out = agent.plan(
        request.files["goal"].read(),
        forced_anchor=(int(forced_anchor) if forced_anchor is not None else None),
        forced_gate=(float(forced_gate) if forced_gate is not None else None),
        candidate_ceiling_override=candidate_ceiling_override(),
    )
    return jsonify(with_goal_session_receipt(out))


@app.route("/posegoal_step", methods=["POST"])
def posegoal_step():
    """Append a decision frame and recover the retrieved metric point-goal."""
    agent.add_frame(request.files["image"].read())
    forced_anchor = request.form.get("forced_anchor")
    forced_gate = request.form.get("forced_gate")
    out = agent.plan(
        request.files["goal"].read(),
        forced_anchor=(int(forced_anchor) if forced_anchor is not None else None),
        forced_gate=(float(forced_gate) if forced_gate is not None else None),
        pose_only=True,
        candidate_ceiling_override=candidate_ceiling_override(),
    )
    return jsonify(with_goal_session_receipt(out))


@app.route("/posegoal_query", methods=["POST"])
def posegoal_query():
    """Recover metric pose after retrieval_probe_step, without double append."""
    forced_anchor = request.form.get("forced_anchor")
    forced_gate = request.form.get("forced_gate")
    out = agent.plan(
        request.files["goal"].read(),
        forced_anchor=(int(forced_anchor) if forced_anchor is not None else None),
        forced_gate=(float(forced_gate) if forced_gate is not None else None),
        pose_only=True,
        candidate_ceiling_override=candidate_ceiling_override(),
    )
    return jsonify(with_goal_session_receipt(out))


@app.route("/retrieval_probe_step", methods=["POST"])
def retrieval_probe_step():
    """Append a frame and return scores without allocating a goal-pose cache."""
    agent.add_frame(request.files["image"].read())
    forced_anchor = request.form.get("forced_anchor")
    forced_gate = request.form.get("forced_gate")
    out = agent.plan(
        request.files["goal"].read(),
        forced_anchor=(int(forced_anchor) if forced_anchor is not None else None),
        forced_gate=(float(forced_gate) if forced_gate is not None else None),
        retrieval_only=True,
        candidate_ceiling_override=candidate_ceiling_override(),
    )
    return jsonify(with_goal_session_receipt(out))


@app.route("/retrieval_verify", methods=["POST"])
def retrieval_verify():
    """Verify one candidate without appending or mutating LingBot KV state."""
    anchor = request.form.get("anchor")
    if anchor is None:
        return jsonify({"error": "anchor is required", "matches": 0,
                        "inliers": 0, "inlier_ratio": 0.0}), 400
    out = agent.verify_retrieval_overlap(
        request.files["goal"].read(), int(anchor))
    return jsonify(out)


@app.route("/phase_b_rank", methods=["POST"])
def phase_b_rank():
    """Rank a causal DINO shortlist; never decide memory activation."""
    raw_candidates = request.form.get("candidates")
    if raw_candidates is None:
        return jsonify({"ok": False, "error": "candidates is required"}), 400
    try:
        candidates = json.loads(raw_candidates)
    except json.JSONDecodeError as error:
        return jsonify({
            "ok": False,
            "error": f"candidates is invalid JSON: {error}",
        }), 400
    goal = request.files.get("goal")
    if goal is None:
        return jsonify({"ok": False, "error": "goal is required"}), 400
    return jsonify(agent.rank_retrieval_candidates(goal.read(), candidates))


@app.route("/certified_relocalize", methods=["POST"])
def certified_relocalize():
    """Localize one already-probed goal without appending another frame."""
    raw_candidates = request.form.get("candidates")
    if raw_candidates is None:
        return jsonify({
            "ok": False, "accepted": False,
            "reason": "candidates_required",
        }), 400
    try:
        candidates = json.loads(raw_candidates)
    except json.JSONDecodeError as error:
        return jsonify({
            "ok": False, "accepted": False,
            "reason": "invalid_candidate_json", "error": str(error),
        }), 400
    goal = request.files.get("goal")
    if goal is None:
        return jsonify({
            "ok": False, "accepted": False, "reason": "goal_required",
        }), 400
    raw_route_start = request.form.get("route_start_anchor")
    try:
        route_start_anchor = (
            None if raw_route_start in (None, "") else int(raw_route_start))
    except (TypeError, ValueError, OverflowError):
        return jsonify({
            "ok": False, "accepted": False,
            "reason": "invalid_route_start_anchor",
        }), 400
    raw_graph_rescue = request.form.get("graph_rescue", "0")
    if raw_graph_rescue not in ("0", "1"):
        return jsonify({
            "ok": False, "accepted": False,
            "reason": "invalid_graph_rescue",
        }), 400
    raw_learned_rescue = request.form.get("learned_rescue", "0")
    if raw_learned_rescue not in ("0", "1"):
        return jsonify({
            "ok": False, "accepted": False,
            "reason": "invalid_learned_rescue",
        }), 400
    proposal_order = request.form.get("proposal_order", "geometry_first")
    if proposal_order not in ("geometry_first", "dino_first_certified"):
        return jsonify({
            "ok": False, "accepted": False,
            "reason": "invalid_proposal_order",
        }), 400
    raw_goal_intrinsic = request.form.get("goal_camera_intrinsic")
    goal_camera_intrinsic = None
    if raw_goal_intrinsic not in (None, ""):
        try:
            goal_camera_intrinsic = json.loads(raw_goal_intrinsic)
        except json.JSONDecodeError as error:
            return jsonify({
                "ok": False, "accepted": False,
                "reason": "invalid_goal_camera_intrinsic_json",
                "error": str(error),
            }), 400
    # Runtime/model failures are represented as fail-closed JSON decisions;
    # candidate-contract errors remain visible rather than becoming a 500 that
    # could accidentally skip the native fallback.
    return jsonify(agent.certified_relocalize(
        goal.read(), candidates,
        route_start_anchor=route_start_anchor,
        graph_rescue=(raw_graph_rescue == "1"),
        allow_learned_rescue=(raw_learned_rescue == "1"),
        proposal_order=proposal_order,
        goal_camera_intrinsic=goal_camera_intrinsic,
    ))


@app.route("/certified_anchor_image", methods=["POST"])
def certified_anchor_image():
    """Expose only the history image bound to an accepted cached CEC proof."""
    goal = request.files.get("goal")
    if goal is None:
        return jsonify({"ok": False, "reason": "goal_required"}), 400
    raw_anchor = request.form.get("selected_anchor")
    expected_sha256 = request.form.get("expected_anchor_sha256")
    try:
        if raw_anchor is None:
            raise ValueError("selected_anchor is required")
        record = agent.certified_anchor_image(
            goal.read(), int(raw_anchor), expected_sha256=expected_sha256)
    except (FileNotFoundError, TypeError, ValueError, OverflowError) as error:
        return jsonify({
            "ok": False,
            "reason": "certified_anchor_not_authorized",
            "error": f"{type(error).__name__}: {error}",
        }), 409
    response = app.response_class(record["image"], mimetype="image/jpeg")
    response.headers["X-CEC-Anchor-Index"] = str(record["anchor"])
    response.headers["X-CEC-Anchor-SHA256"] = record["sha256"]
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/learned_relocalize", methods=["POST"])
def learned_relocalize():
    """Run learned relocalization after an already-appended causal probe."""
    raw_candidates = request.form.get("candidates")
    if raw_candidates is None:
        return jsonify({
            "ok": False, "accepted": False,
            "reason": "candidates_required",
        }), 400
    try:
        candidates = json.loads(raw_candidates)
    except json.JSONDecodeError as error:
        return jsonify({
            "ok": False, "accepted": False,
            "reason": "invalid_candidate_json", "error": str(error),
        }), 400
    goal = request.files.get("goal")
    if goal is None:
        return jsonify({
            "ok": False, "accepted": False, "reason": "goal_required",
        }), 400
    result = agent.learned_pi3x_relocalize(goal.read(), candidates)
    # Evaluation telemetry only.  This is appended after the relocalization
    # decision and is never read by the controller.
    result["peak_gpu_memory_allocated_bytes"] = (
        int(torch.cuda.max_memory_allocated())
        if torch.cuda.is_available() else None
    )
    return jsonify(result)


@app.route("/imagegoal_similarity", methods=["POST"])
def imagegoal_similarity():
    score = agent.image_goal_similarity(
        request.files["image"].read(), request.files["goal"].read())
    return jsonify({"current_goal_cos": score})


if __name__ == "__main__":
    print(f"[memnav_server] ready on :{args.port} "
          f"(W={agent.W}, S={agent.S}, amargin={agent.amargin}, "
          f"exclude_recent={agent.exclude_recent}, samples={agent.num_samples}, "
          f"retrieval={agent.retrieval_mode}, flow_gate={agent.flow_gate}, "
          f"candidate_top_k={agent.retrieval_candidate_top_k}, "
          f"candidate_gap={agent.retrieval_candidate_min_gap}, "
          f"graph_spacing_m={agent.graph_subgoal_spacing_m}, "
          f"graph_arrival_m={agent.graph_subgoal_arrival_m}, "
          f"phase_b_ranker={agent.phase_b_status().get('enabled')}, "
          f"certified_relocalization="
          f"{agent.certified_relocalization_status().get('enabled')}, "
          f"learned_pi3x_relocalization="
          f"{agent.learned_pi3x_relocalization_status().get('enabled')}, "
          f"cuda_http_handoff_sync="
          f"{args.synchronize_cuda_http_handoff}, "
          f"checkpoint={os.path.basename(args.checkpoint)})")
    app.run(host=args.host, port=args.port, threaded=False)
