"""Single-frame visualization of the full MemNav planning forward pass.

Picks one raw episode + one time frame, streams frames 0..k through the live
MemNavAgent (real checkpoint: LingBot memory -> retrieval -> DDPM -> geometric
collision selection), then renders:

  * current RGB, goal RGB, LingBot predicted depth (the selector's input)
  * BEV: robot, FoV wedge, LingBot obstacle points vs GT-depth obstacle points,
    all 16 candidate trajectories colored by collision score, the selected one,
    GT past/future path + start/goal positions (parquet world poses projected
    into the current camera planar frame).

Poses: parquet `action` is the per-frame cam-to-world 4x4 (local axes x right,
y up, z BACKWARD — OpenGL camera; world z up). Planar mapping of a world point
p in frame k: p_l = R_k^T (p_w - t_k); x_fwd = -p_l.z, y_left = -p_l.x.

Run on a GPU node in the memnav env (see viz_collision_select.sbatch):
  python scripts/diag_collision_geo/viz_collision_select.py \
      --traj .../mp3d_3leg/17DRP5sb8fy/episode_0002 --frame 160
"""
import argparse
import json
import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402

INTERNNAV_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMNAV_DIR = os.path.join(os.path.dirname(INTERNNAV_ROOT), "NavDP/baselines/memnav")


def load_poses(pq_path):
    import pandas as pd
    df = pd.read_parquet(pq_path)
    T = np.stack([np.stack([np.asarray(r) for r in a])
                  for a in df["action"]]).astype(np.float64)      # [n,4,4]
    K = np.stack([np.asarray(r) for r in
                  df["observation.camera_intrinsic"].iloc[0]]).astype(np.float64)
    return T, K


def world_to_planar(p_w, T_k):
    """World points [M,3] -> current camera planar frame (x fwd, y left) [M,2]."""
    R, t = T_k[:3, :3], T_k[:3, 3]
    p_l = (p_w - t) @ R                                           # R^T (p - t)
    return np.stack([-p_l[:, 2], -p_l[:, 0]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="/scratch/lg154/Research/datasets/mp3d_revisit_v0/"
                    "vln_n1/traj_data/mp3d_3leg/17DRP5sb8fy/episode_0002")
    ap.add_argument("--frame", type=int, default=160, help="current time frame k")
    ap.add_argument("--goal_frame", type=int, default=None,
                    help="goal image frame; default k - exclude_recent - 10 (a revisit)")
    ap.add_argument("--checkpoint", default=os.path.join(
        INTERNNAV_ROOT, "checkpoints/memnav_mp3d_gs6/ckpts/checkpoint-1620/memnav.ckpt"))
    ap.add_argument("--out_dir", default=os.path.join(INTERNNAV_ROOT, "logs/viz"))
    args = ap.parse_args()

    sys.path.insert(0, MEMNAV_DIR)
    os.chdir(MEMNAV_DIR)                      # lingbot relative paths resolve
    from policy_agent import MemNavAgent      # noqa: E402

    rgb_dir = os.path.join(args.traj, "videos/chunk-000/observation.images.rgb")
    dep_dir = os.path.join(args.traj, "videos/chunk-000/observation.images.depth")
    pq = os.path.join(args.traj, "data/chunk-000/episode_000000.parquet")
    T, K = load_poses(pq)
    n_total = len(T)
    k = args.frame
    assert k < n_total, f"frame {k} >= episode length {n_total}"

    agent = MemNavAgent(args.checkpoint, INTERNNAV_ROOT,
                        buffer_root=os.path.join(args.out_dir, "_agent_buffer"))
    g = args.goal_frame if args.goal_frame is not None else k - agent.exclude_recent - 10
    assert 0 <= g < n_total, f"goal frame {g} out of range"
    print(f"episode {args.traj}\n  frames 0..{k} streamed, goal = frame {g}, "
          f"{n_total} total")

    for i in range(k + 1):
        with open(os.path.join(rgb_dir, f"{i}.jpg"), "rb") as f:
            agent.add_frame(f.read())
        if i % 40 == 0:
            print(f"  streamed {i}/{k}")
    with open(os.path.join(rgb_dir, f"{g}.jpg"), "rb") as f:
        goal_bytes = f.read()
    res = agent.plan(goal_bytes)
    if "error" in res:
        raise SystemExit(res["error"])
    paths = np.asarray(res["all_trajectory"])                     # [N,24,3]
    values = np.asarray(res["all_values"])
    sel = np.asarray(res["trajectory"])
    pick = int(np.argmin([np.abs(paths[i] - sel).sum() for i in range(len(paths))]))
    print(f"gate={res['gate']:.3f} anchor={res['anchor']} raw_score={res['raw_score']}"
          f"\nscores: {np.round(values, 3).tolist()}\npicked candidate {pick}")

    # --- selector inputs, recomputed exactly as plan() did (pure heads) -------
    from internnav.model.basemodel.memnav.collision_check import (  # noqa: E402
        obstacle_points_from_depth)
    from internnav.model.basemodel.memnav.lingbot_stream import (   # noqa: E402
        GROUND_BIAS_CORRECTION)
    sys.path.insert(0, os.path.join(INTERNNAV_ROOT, "scripts/diag_collision_geo"))
    from diag_collision_geometry import gt_obstacle_points          # noqa: E402

    dev = agent.device
    cur_img = agent._window_imgs[-1]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        pred = agent.lb.model._predict_depth(
            agent._last_agg, cur_img[None][None].to(dev), agent._psi)
    d_cur = pred["depth"][0, -1, ..., 0].float()
    c_cur = pred["depth_conf"][0, -1].float()
    cur_pose = agent.cam_pose[k]
    fov_v, fov_h = float(cur_pose[7]), float(cur_pose[8])
    ms = float(agent._get_metric_scale())
    h_est = GROUND_BIAS_CORRECTION * agent.camera_height / ms
    obs_lb = obstacle_points_from_depth(
        d_cur, c_cur, fov_v, fov_h, h_est, ms).cpu().numpy()
    obs_gt = gt_obstacle_points(os.path.join(dep_dir, f"{k}.png"), K)
    print(f"scale={ms:.2f}  fov_h={np.degrees(fov_h):.1f}deg  "
          f"obstacle pts: lingbot={len(obs_lb)} gt={len(obs_gt)}")

    # --- GT geometry in the current planar frame ------------------------------
    centers = T[:, :3, 3]
    past = world_to_planar(centers[: k + 1], T[k])
    fut = world_to_planar(centers[k: min(k + 150, n_total)], T[k])
    p_start = world_to_planar(centers[0:1], T[k])[0]
    p_goal = world_to_planar(centers[g:g + 1], T[k])[0]

    # --- figure ---------------------------------------------------------------
    scene, ep = args.traj.rstrip("/").split(os.sep)[-2:]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.6])
    for r, (fr, title) in enumerate([(k, f"current frame {k}"),
                                     (g, f"goal frame {g}")]):
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(cv2.imread(os.path.join(rgb_dir, f"{fr}.jpg"))[:, :, ::-1])
        ax.set_title(title, fontsize=10); ax.axis("off")
    ax = fig.add_subplot(gs[2, 0])
    ax.imshow(d_cur.cpu().numpy() * ms, cmap="turbo", vmin=0, vmax=6)
    ax.set_title("LingBot depth (m) — selector input", fontsize=10); ax.axis("off")
    ax = fig.add_subplot(gs[2, 1])
    dgt = cv2.imread(os.path.join(dep_dir, f"{k}.png"), cv2.IMREAD_UNCHANGED)
    ax.imshow(dgt.astype(np.float32) / 10000.0, cmap="turbo", vmin=0, vmax=6)
    ax.set_title("GT sensor depth (m) — reference only", fontsize=10); ax.axis("off")
    ax = fig.add_subplot(gs[0:2, 1])
    ax.imshow(c_cur.cpu().numpy(), cmap="viridis")
    ax.set_title("depth confidence (keep top 75%)", fontsize=10); ax.axis("off")

    ax = fig.add_subplot(gs[:, 2])
    # FoV wedge + range
    for s in (-1, 1):
        a = s * (fov_h / 2 - 0.05)
        ax.plot([0, 5 * np.cos(a)], [0, 5 * np.sin(a)], ":", c="gray", lw=1)
    if len(obs_gt):
        ax.scatter(obs_gt[:, 0], obs_gt[:, 1], s=2, c="lightgray",
                   label=f"GT-depth obstacles ({len(obs_gt)})")
    if len(obs_lb):
        ax.scatter(obs_lb[:, 0], obs_lb[:, 1], s=3, c="crimson", alpha=0.5,
                   label=f"LingBot obstacles ({len(obs_lb)})")
    vmin, vmax = values.min(), values.max()
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-6))
    for i, p in enumerate(paths):
        if i == pick:
            continue
        ax.plot(p[:, 0], p[:, 1], "-", c=cm.viridis(norm(values[i])), lw=1.2, alpha=0.9)
    ax.plot(paths[pick, :, 0], paths[pick, :, 1], "-", c="k", lw=3,
            label=f"selected #{pick} (score {values[pick]:.2f})")
    ax.plot(past[:, 0], past[:, 1], "-", c="tab:blue", lw=1, alpha=0.6, label="GT past path")
    ax.plot(fut[:, 0], fut[:, 1], "--", c="tab:green", lw=1.5, label="GT future path")
    ax.plot(*p_start, "s", c="tab:green", ms=9, label=f"start (frame 0)")
    ax.plot(*p_goal, "*", c="gold", mec="k", ms=18, label=f"goal (frame {g})")
    ax.add_patch(plt.Circle((0, 0), 0.30, color="tab:blue", alpha=0.35))
    ax.plot(0, 0, "^", c="tab:blue", ms=10, label="robot (r=0.30 m)")
    sm = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, label="collision score")
    ax.set_xlim(-2.5, 6); ax.set_ylim(-4.5, 4.5); ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x forward (m)"); ax.set_ylabel("y left (m)")
    ax.set_title(f"{scene}/{ep} @ frame {k} — BEV, current camera frame\n"
                 f"gate={res['gate']:.2f} anchor={res['anchor']} scale={ms:.2f}",
                 fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"collision_select_{scene}_{ep}_f{k}.png")
    fig.tight_layout(); fig.savefig(out, dpi=140)
    with open(out.replace(".png", ".json"), "w") as f:
        json.dump(dict(res, obstacle_counts=dict(lingbot=len(obs_lb), gt=int(len(obs_gt))),
                       scale=ms, pick=pick, goal_frame=g), f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
