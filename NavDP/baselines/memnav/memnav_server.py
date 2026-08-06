"""MemNav Flask server — the model half of the two-process closed-loop eval.

Runs in the `memnav` conda env. The Habitat client (habitat env) streams RGB
frames over HTTP; this server maintains the live LingBot memory and plans
trajectories toward a goal image on request.

Endpoints (NavDP wire-contract style):
  POST /navigator_reset      JSON {camera_height?, seed?, episode_len?}
                             -> {"algo": "memnav"}   (starts a fresh episode)
  POST /memory_step          files: image (jpg)      -> {"frame_idx": i}
                             stream a frame into memory WITHOUT planning (leg replay)
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
  POST /imagegoal_similarity files: image (jpg), goal (jpg)
                             -> {"current_goal_cos": float}
                             stateless visual check; does not mutate memory.

Usage:
  conda activate memnav
  python memnav_server.py --port 18888 \
    --checkpoint /home/asus/Research/Nav/InternNav/checkpoints/memnav_2leg_axisfix/checkpoint-1500/memnav.ckpt
"""

import argparse
import os

# must precede any torch import (policy_agent): reduces fragmentation OOMs from
# the large KV-cache alloc/free cycle each plan() runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from flask import Flask, jsonify, request

parser = argparse.ArgumentParser()
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
args = parser.parse_args()

# The server changes directory below because LingBot historically resolves a
# few script-local resources from this module. Resolve every CLI filesystem
# argument first: otherwise a perfectly valid relative checkpoint silently
# becomes missing after chdir and ``from_pretrained`` leaves all trainable
# MemNav heads randomly initialized.
args.checkpoint = os.path.abspath(args.checkpoint)
args.internnav_root = os.path.abspath(args.internnav_root)
args.buffer_root = os.path.abspath(args.buffer_root)
if not os.path.isfile(args.checkpoint):
    raise FileNotFoundError(f"MemNav checkpoint not found: {args.checkpoint}")
if not os.path.isdir(args.internnav_root):
    raise FileNotFoundError(f"InternNav root not found: {args.internnav_root}")

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
)

app = Flask(__name__)


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
    return jsonify({"frame_idx": idx})


@app.route("/imagegoal_step", methods=["POST"])
def imagegoal_step():
    agent.add_frame(request.files["image"].read())
    forced_anchor = request.form.get("forced_anchor")
    forced_gate = request.form.get("forced_gate")
    out = agent.plan(
        request.files["goal"].read(),
        forced_anchor=(int(forced_anchor) if forced_anchor is not None else None),
        forced_gate=(float(forced_gate) if forced_gate is not None else None),
    )
    return jsonify(out)


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
    )
    return jsonify(out)


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
    )
    return jsonify(out)


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
    )
    return jsonify(out)


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
          f"checkpoint={os.path.basename(args.checkpoint)})")
    app.run(host="0.0.0.0", port=args.port, threaded=False)
