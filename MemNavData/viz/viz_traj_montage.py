"""Montage of several generated episodes (2-leg or 3-leg) drawn over the scene BEV
navmesh occupancy. CPU-only: loads the precomputed .navmesh directly with
habitat_sim.PathFinder (no Simulator / no GL context), so it runs on a plain CPU node
while the GPUs stay busy generating.

Drawing logic mirrors viz_multileg.py: each leg gets its own colour, start/A are marked,
each goal is a star with an orientation arrow (black) and its covisibility-matched history
frame (diamond). Handles 2- and 3-leg via gen_meta.json["switches"].

Run (habitat env, CPU node):
  python viz_traj_montage.py \
    --scene_dir /.../mp3d_revisit_v0/vln_n1/traj_data/mp3d_2leg/b8cTxDM8gDG \
    --navmesh   /.../mp3d/mp3d/b8cTxDM8gDG/b8cTxDM8gDG.navmesh \
    --n 4 --out /.../viz_out/b8cTxDM8gDG_2leg.png
"""
import argparse, os, json, glob, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_episode(ep_dir):
    import pandas as pd
    m = json.load(open(os.path.join(ep_dir, "meta/gen_meta.json")))
    pq = os.path.join(ep_dir, "data/chunk-000/episode_000000.parquet")
    A = np.stack([np.array(a.tolist(), float).reshape(4, 4) for a in pd.read_parquet(pq)["action"]])
    return m, A


def occupancy(pf, xy, heights, res, margin):
    """Multi-slice BEV occupancy. A long revisit episode can span a floor-level change
    (e.g. episode_0008 jumps 0.6 m between legs), so a single-height slice paints the
    other level as 'obstacle' and the (fully-navigable) path appears to cut through walls.
    We snap at every candidate height in `heights` and UNION them: a cell is free if the
    navmesh is horizontally-exact there at ANY level. (No vertical band test needed — the
    horizontal-exact match already localises the cell; snapping downward from the camera
    height to the floor is expected.)"""
    x0, x1 = xy[:, 0].min() - margin, xy[:, 0].max() + margin
    y0, y1 = xy[:, 1].min() - margin, xy[:, 1].max() + margin
    xs = np.arange(x0, x1, res); ys = np.arange(y0, y1, res)
    occ = np.zeros((len(ys), len(xs)), np.uint8)
    for iy, gy in enumerate(ys):
        for ix, gx in enumerate(xs):
            for hy in heights:
                q = pf.snap_point([gx, hy, -gy])        # data (x,y) -> habitat (x, y=hy, z=-y)
                if pf.is_navigable(q) and abs(q[0] - gx) < res and abs(q[2] + gy) < res:
                    occ[iy, ix] = 1
                    break
    return occ, (x0, x1, y0, y1)


def draw_episode(ax, pf, ep_dir, res, margin, arrow_every):
    m, A = load_episode(ep_dir)
    xy = A[:, :2, 3]
    yaw = np.unwrap(np.arctan2(A[:, 1, 0], A[:, 0, 0]))
    z = A[:, 2, 3]                                       # data_z == habitat_y (camera height)
    n = m.get("n_frames", len(A)); sw = m.get("switches", [])
    # candidate height slices spanning the episode's z-range (handles floor-level changes)
    heights = sorted(set(np.round(np.linspace(z.min(), z.max(), 4), 2)))
    occ, ext = occupancy(pf, xy, heights, res, margin)
    ax.imshow(occ, origin="lower", extent=ext, cmap="gray", vmin=-0.5, vmax=1.0,
              interpolation="nearest", zorder=0)

    bounds = [0] + list(sw) + [n]
    cols = ["royalblue", "red", "darkorange"]
    labs = ["leg1 start->A", "leg2 ->B", "leg3 ->C"]
    for k in range(len(bounds) - 1):
        seg = xy[bounds[k]:bounds[k + 1]]
        ax.plot(seg[:, 0], seg[:, 1], "-", color=cols[k % 3], lw=1.8,
                label=labs[k] if k < 3 else None, zorder=3)
    for i in range(0, n, arrow_every):
        ax.arrow(xy[i, 0], xy[i, 1], .1 * np.cos(yaw[i]), .1 * np.sin(yaw[i]),
                 head_width=.04, color="dimgray", alpha=.5, zorder=4)
    ax.scatter(*xy[0], c="lime", s=90, ec="k", zorder=6, label="start")
    if sw:
        ax.scatter(xy[sw[0] - 1, 0], xy[sw[0] - 1, 1], c="cyan", s=80, ec="k", zorder=6, label="A")
    for g, c in zip(m.get("goals", []), ["magenta", "gold"]):
        p = g["pos"]
        ax.scatter(p[0], p[1], s=170, marker="*", ec="k", c=c, zorder=7,
                   label=f"{g['name']} [{g['kind']}] covis={g.get('covis', float('nan')):.2f}")
        ax.arrow(p[0], p[1], .35 * np.cos(g["yaw_habitat"]), .35 * np.sin(g["yaw_habitat"]),
                 head_width=.11, color="k", lw=2, zorder=8)
        ai = g.get("covis_argmax", -1)
        if ai is not None and ai >= 0:
            ax.scatter(xy[ai, 0], xy[ai, 1], s=70, marker="D", ec="k", c=c, zorder=7)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{os.path.basename(ep_dir)}  n_legs={m.get('n_legs')} n={n} sw={sw}", fontsize=9)


def pick_eps(scene_dir, n):
    eps = sorted(glob.glob(os.path.join(scene_dir, "episode_*")))
    if not eps:
        raise SystemExit(f"no episodes under {scene_dir}")
    if len(eps) <= n:
        return eps
    idx = np.linspace(0, len(eps) - 1, n).round().astype(int)   # spread across the set
    return [eps[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_dir", required=True, help="mp3d_{2,3}leg/<scene_id> directory")
    ap.add_argument("--navmesh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4, help="how many episodes to draw")
    ap.add_argument("--eps", default="", help="comma list of episode dir names (overrides --n auto-pick)")
    ap.add_argument("--res", type=float, default=0.06)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--arrow_every", type=int, default=16)
    args = ap.parse_args()

    import habitat_sim
    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(args.navmesh)
    if not pf.is_loaded:
        raise SystemExit(f"navmesh failed to load: {args.navmesh}")

    if args.eps:
        eps = [os.path.join(args.scene_dir, e.strip()) for e in args.eps.split(",")]
    else:
        eps = pick_eps(args.scene_dir, args.n)

    ncol = min(len(eps), 2); nrow = math.ceil(len(eps) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(9 * ncol, 7 * nrow), squeeze=False)
    for k, ep in enumerate(eps):
        ax = axes[k // ncol][k % ncol]
        draw_episode(ax, pf, ep, args.res, args.margin, args.arrow_every)
    for k in range(len(eps), nrow * ncol):          # blank any unused panels
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle(f"{os.path.basename(args.scene_dir)}  ({len(eps)} episodes)  "
                 f"BEV: free=light / walls=dark", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=120)
    print("saved", args.out)


if __name__ == "__main__":
    main()
