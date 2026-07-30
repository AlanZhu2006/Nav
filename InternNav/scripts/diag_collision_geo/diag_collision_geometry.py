"""Phase-0 validation for geometric trajectory selection (collision_check.py):
is LingBot predicted depth x ground_scale accurate enough to collision-check
candidate waypoints, judged against the GT sensor depth?

Per sampled episode (GPU; streams frames exactly like compute_metric_scale):
  * LingBot side: per-frame depth+conf from the frozen depth head, unprojected to
    BEV obstacle points in the CURRENT camera planar frame via
    collision_check.obstacle_points_from_depth (h_est + ground scale from the cam
    cache — the same inputs the eval agent has).
  * GT side: videos/chunk-000/observation.images.depth/<i>.png (uint16/10000 -> m,
    valid [0.1, 5.0]) unprojected with the parquet camera_intrinsic, same height
    band above the floor (camera 0.5 m, level).
  * Both clouds are sampled on a query grid of would-be waypoint positions
    (0.4..3.6 m ahead, |y| <= 1.6, inside both FoV wedges). Metrics:
      dist_err   |d_lb(q) - d_gt(q)| of nearest-obstacle distance at each q
      collision  precision/recall of (d_lb < r) predicting (d_gt < r), r = 0.30 m
        -- the exact decision the selector makes; recall misses = walls the filter
        would not see, precision misses = phantom obstacles vetoing good paths.

Run on a GPU node (frames + model). Example:
  python scripts/diag_collision_geometry.py --episodes 16 --max_frames 200
"""
import argparse
import glob
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from internnav.model.basemodel.memnav.collision_check import (  # noqa: E402
    OBSTACLE_H_BAND_M,
    ROBOT_RADIUS_M,
    obstacle_points_from_depth,
)
from internnav.model.basemodel.memnav.lingbot_stream import (  # noqa: E402
    LingBotStream,
    ground_scale_from_h_est,
)


def gt_obstacle_points(depth_png, K, cam_h=0.5, h_band=OBSTACLE_H_BAND_M,
                       max_range_m=5.0, stride=2):
    """GT sensor depth -> BEV obstacle points, planar frame (x fwd, y left), meters."""
    d = cv2.imread(depth_png, cv2.IMREAD_UNCHANGED)
    if d is None:
        return None
    d = d.astype(np.float32) / 10000.0
    d = d[::stride, ::stride]
    H, W = d.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = (np.arange(0, W) * stride)[None, :].repeat(H, 0).astype(np.float32)
    v = (np.arange(0, H) * stride)[:, None].repeat(W, 1).astype(np.float32)
    x_c = (u - cx) * d / fx                    # right
    y_c = (v - cy) * d / fy                    # down
    h_above = cam_h - y_c                      # camera level, mounted at cam_h
    keep = (d >= 0.1) & (d <= max_range_m) \
        & (h_above >= h_band[0]) & (h_above <= h_band[1])
    return np.stack([d[keep], -x_c[keep]], -1)  # x_p = z_cam, y_p = -x_cam


def query_grid(fov_h_min, x_rng=(0.4, 3.6), y_rng=(-1.6, 1.6), step=0.2,
               fov_margin=0.05):
    xs = np.arange(x_rng[0], x_rng[1] + 1e-6, step, dtype=np.float32)
    ys = np.arange(y_rng[0], y_rng[1] + 1e-6, step, dtype=np.float32)
    q = np.stack(np.meshgrid(xs, ys, indexing="ij"), -1).reshape(-1, 2)
    ang = np.abs(np.arctan2(q[:, 1], q[:, 0]))
    return q[ang <= fov_h_min / 2.0 - fov_margin]


def nearest_dist(q, pts, cap=5.0):
    if pts is None or len(pts) == 0:
        return np.full(len(q), cap, dtype=np.float32)
    d = np.linalg.norm(q[:, None, :] - pts[None, :, :], axis=-1).min(1)
    return np.minimum(d, cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_root", default="/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data")
    ap.add_argument("--raw_root", default="/scratch/lg154/Research/datasets/mp3d_revisit_v0/vln_n1/traj_data")
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=8)
    ap.add_argument("--radius", type=float, default=ROBOT_RADIUS_M)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    # enumerate from the RAW side: most raw frame dirs were pruned after squashfs
    # packing — only episodes that still have rgb+depth on host are usable here.
    raw_eps = sorted(glob.glob(os.path.join(
        args.raw_root, "*/*/*/videos/chunk-000/observation.images.rgb")))
    caches = []
    for rd in raw_eps:
        rel = os.path.relpath(rd, args.raw_root).split(os.sep)
        c = os.path.join(args.feat_root, *rel[:3],
                         "videos/chunk-000/lingbot_cam_cache.npz")
        if os.path.exists(c):
            caches.append(c)
    idx = np.linspace(0, len(caches) - 1, min(args.episodes, len(caches))).astype(int)
    picks = [caches[i] for i in sorted(set(idx))]
    print(f"{len(raw_eps)} raw episodes on host, {len(caches)} with cam caches; "
          f"sampling {len(picks)}")

    lb = LingBotStream(device="cuda")
    S = lb.num_scale

    # post-hoc operating-point sweep: conf-quantile x LingBot-side veto radius,
    # GT truth fixed at args.radius. Reuses the (expensive) predicted depths.
    sweep_cqs = [0.0, 0.25, 0.5]
    sweep_rs = [0.30, 0.40, 0.50]
    sweep = {(cq, r): [0, 0, 0, 0] for cq in sweep_cqs for r in sweep_rs}

    rows, all_err, n_tp, n_fp, n_fn, n_tn = [], [], 0, 0, 0, 0
    for cpath in picks:
        rel = os.path.relpath(cpath, args.feat_root)
        group, scene, ep = rel.split(os.sep)[:3]
        traj = os.path.join(args.raw_root, group, scene, ep)
        rgb_dir = os.path.join(traj, "videos/chunk-000/observation.images.rgb")
        dep_dir = os.path.join(traj, "videos/chunk-000/observation.images.depth")
        pq = os.path.join(traj, "data/chunk-000/episode_000000.parquet")
        if not (os.path.isdir(rgb_dir) and os.path.isdir(dep_dir) and os.path.exists(pq)):
            print(f"skip {rel}: raw episode incomplete"); continue
        cc = np.load(cpath)
        if "ground_h_est" not in cc.files or not np.isfinite(float(cc["ground_h_est"])):
            print(f"skip {rel}: no ground_h_est"); continue
        h_est = float(cc["ground_h_est"])
        pose = torch.as_tensor(cc["cam_pose_enc"], dtype=torch.float32)
        scale = ground_scale_from_h_est(h_est)
        if scale is None:
            print(f"skip {rel}: invalid h_est"); continue

        import pandas as pd
        K = np.stack(pd.read_parquet(pq)["observation.camera_intrinsic"][0])
        gt_fov_h = 2.0 * math.atan(K[0, 2] / K[0, 0])

        n_rgb = len([p for p in os.listdir(rgb_dir) if p.endswith(".jpg")])
        n = min(args.max_frames, n_rgb, len(pose))
        rgb_paths = [os.path.join(rgb_dir, f"{i}.jpg") for i in range(n)]
        eval_f = [f for f in range(S, n) if (f - S) % args.eval_every == 0]

        # stream exactly like compute_metric_scale: scale block, then per-frame
        imgs = lb._preprocess(rgb_paths, mode="pad",
                              image_size=lb.img_size, patch_size=lb.patch_size)
        lb.model.clean_kv_cache()
        want = set(eval_f)
        depths = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            blk = imgs[:S][None].cuda()
            agg, psi = lb.model._aggregate_features(
                blk, num_frame_for_scale=S, num_frame_per_block=S)
            for j in range(S, n):
                fj = imgs[j:j + 1][None].cuda()
                a, _ = lb.model._aggregate_features(
                    fj, num_frame_for_scale=S, num_frame_per_block=1)
                if j in want:
                    pj = lb.model._predict_depth(a, fj, psi)
                    depths[j] = (pj["depth"][0, -1, ..., 0].float(),
                                 pj["depth_conf"][0, -1].float())
        lb.model.clean_kv_cache()

        ep_err, ep_cnt = [], [0, 0, 0, 0]  # tp fp fn tn
        for f in eval_f:
            d_lb, c_lb = depths[f]
            fov_v, fov_h = float(pose[f, 7]), float(pose[f, 8])
            obs_gt = gt_obstacle_points(os.path.join(dep_dir, f"{f}.png"), K)
            if obs_gt is None:
                continue
            q = query_grid(min(fov_h, gt_fov_h))
            dq_gt = nearest_dist(q, obs_gt)
            hit_gt = dq_gt < args.radius
            dq_by_cq = {}
            for cq in sweep_cqs:
                pts = obstacle_points_from_depth(
                    d_lb, c_lb, fov_v, fov_h, h_est, scale,
                    conf_quantile=cq).cpu().numpy()
                dq_by_cq[cq] = nearest_dist(q, pts)
                for r_lb in sweep_rs:
                    hl = dq_by_cq[cq] < r_lb
                    c = sweep[(cq, r_lb)]
                    c[0] += int((hl & hit_gt).sum())
                    c[1] += int((hl & ~hit_gt).sum())
                    c[2] += int((~hl & hit_gt).sum())
                    c[3] += int((~hl & ~hit_gt).sum())
            dq_lb = dq_by_cq[0.5]                  # default config for headline stats
            ep_err.append(np.abs(dq_lb - dq_gt))
            hit_lb = dq_lb < args.radius
            ep_cnt[0] += int((hit_lb & hit_gt).sum())
            ep_cnt[1] += int((hit_lb & ~hit_gt).sum())
            ep_cnt[2] += int((~hit_lb & hit_gt).sum())
            ep_cnt[3] += int((~hit_lb & ~hit_gt).sum())
        if not ep_err:
            continue
        err = np.concatenate(ep_err)
        all_err.append(err)
        n_tp += ep_cnt[0]; n_fp += ep_cnt[1]; n_fn += ep_cnt[2]; n_tn += ep_cnt[3]
        prec = ep_cnt[0] / max(1, ep_cnt[0] + ep_cnt[1])
        rec = ep_cnt[0] / max(1, ep_cnt[0] + ep_cnt[2])
        rows.append(dict(group=group, scene=scene, ep=ep, n_eval=len(ep_err),
                         scale=scale, err_med=float(np.median(err)),
                         err_p90=float(np.percentile(err, 90)),
                         precision=prec, recall=rec, counts=ep_cnt))
        print(f"{scene}/{ep}: scale={scale:.2f} err med={np.median(err):.3f} "
              f"p90={np.percentile(err, 90):.3f}  P={prec:.3f} R={rec:.3f}")

    if not all_err:
        print("no episodes evaluated"); return
    err = np.concatenate(all_err)
    prec = n_tp / max(1, n_tp + n_fp)
    rec = n_tp / max(1, n_tp + n_fn)
    print("\n=== OVERALL (query-grid nearest-obstacle distance, meters) ===")
    print(f"  episodes={len(rows)}  queries={len(err)}")
    print(f"  |d_lb - d_gt|  median={np.median(err):.3f}  p90={np.percentile(err, 90):.3f}"
          f"  mean={err.mean():.3f}")
    print(f"  collision @ r={args.radius:.2f}: precision={prec:.3f} recall={rec:.3f}"
          f"  (tp={n_tp} fp={n_fp} fn={n_fn} tn={n_tn})")
    print("  interpretation: recall = walls the filter would actually see;")
    print("  precision = 1 - phantom-obstacle rate vetoing good candidates.")
    print(f"\n=== SWEEP (conf_quantile x lb veto radius; GT truth @ r={args.radius:.2f}) ===")
    sweep_out = []
    for (cq, r_lb), (tp, fp, fn, tn) in sorted(sweep.items()):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        sweep_out.append(dict(conf_quantile=cq, r_lb=r_lb, precision=p, recall=r,
                              tp=tp, fp=fp, fn=fn, tn=tn))
        print(f"  cq={cq:.2f} r_lb={r_lb:.2f}: P={p:.3f} R={r:.3f} "
              f"(tp={tp} fp={fp} fn={fn} tn={tn})")
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(rows=rows, overall=dict(
                err_med=float(np.median(err)), err_p90=float(np.percentile(err, 90)),
                precision=prec, recall=rec,
                tp=n_tp, fp=n_fp, fn=n_fn, tn=n_tn), sweep=sweep_out), f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
