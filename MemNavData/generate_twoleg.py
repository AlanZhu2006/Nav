"""Generate causal two- or three-leg ImageGoal episodes in Habitat.

Two-leg data keeps the historical ``initial_imagegoal -> revisit`` contract.
Three-leg data supports two explicit contracts.  The default role-paired
contract is:

``initial_imagegoal A -> novel B -> long-term revisit C``.

The opt-in double-Revisit diagnostic is:

``initial_imagegoal A -> revisit B -> distinct long-term revisit C``.

For the A/B role comparison, both goals use the same geodesic band, their
per-episode distances are matched within a fixed tolerance, the first yaw is
not path-prealigned, and both goal images are exact expert-arrival frames.
Goal C is anchored on leg A and must remain below the negative co-visibility
threshold throughout leg B, so the recent leg-B window cannot satisfy the
nominal long-term revisit by itself.

All rendering is performed in Habitat's Y-up frame.  Camera poses are written
in the InternData-N1-compatible Z-up convention defined by ``M_W`` below.
"""
import argparse, os, json
import numpy as np
import quaternion
from PIL import Image

W, H = 480, 270
FX, FY, CX, CY = 355.81464, 351.687, 240.0, 135.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], float)
HFOV_DEG = float(np.degrees(2 * np.arctan(CX / FX)))  # ~68.0

# Habitat(Y-up) -> stored data(Z-up) rotation:  (x,y,z)_hab -> (x,-z,y)_data  (det=+1)
M_W = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)

MULTILEG_ROLE_SYMMETRIC_PROTOCOL = "multileg_v4_role_paired_20260812"
MULTILEG_DOUBLE_REVISIT_PROTOCOL = "multileg_v5_double_revisit_20260812"
GOAL_A_SOURCE_PROTOCOL = "goal_a_source_carrier_v1_20260814"
MAX_ROLE_DISTANCE_MATCH_TOLERANCE_M = 0.50
INITIAL_YAW_MODES = ("auto", "path_aligned", "uniform")
THREE_LEG_ROLE_MODES = ("novel_revisit", "double_revisit")


def multileg_protocol(n_legs, three_leg_roles):
    if int(n_legs) < 3:
        return "multileg_v2_symmetric_20260807"
    if three_leg_roles == "novel_revisit":
        return MULTILEG_ROLE_SYMMETRIC_PROTOCOL
    if three_leg_roles == "double_revisit":
        return MULTILEG_DOUBLE_REVISIT_PROTOCOL
    raise ValueError(f"unknown three-leg role mode: {three_leg_roles!r}")


def _bump(counters, key):
    if counters is not None:
        counters[key] = int(counters.get(key, 0)) + 1


def resolve_initial_yaw_mode(mode, n_legs):
    """Resolve the first-leg yaw contract without perturbing 2-leg Revisit.

    The historical generator pre-aligned the first camera with its shortest
    path, while every later goal inherited an arbitrary arrival heading.  That
    makes first-vs-later ImageGoal success incomparable.  Three-leg data now
    defaults to a uniform initial yaw; two-leg Revisit data keeps its historical
    path-aligned start unless explicitly overridden.
    """
    if mode not in INITIAL_YAW_MODES:
        raise ValueError(f"unknown initial yaw mode: {mode!r}")
    if mode == "auto":
        return "uniform" if int(n_legs) >= 3 else "path_aligned"
    return mode


def wrap_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def first_path_yaw(points, origin, min_baseline=0.30):
    """Habitat yaw of the first non-trivial shortest-path segment."""
    origin_xz = np.asarray(origin, dtype=float)[[0, 2]]
    for point in points[1:]:
        delta = np.asarray(point, dtype=float)[[0, 2]] - origin_xz
        if float(np.linalg.norm(delta)) >= float(min_baseline):
            return float(yaw_facing(delta))
    delta = np.asarray(points[-1], dtype=float)[[0, 2]] - origin_xz
    return float(yaw_facing(delta))


def make_sim(
    glb, navmesh, agent_radius=0.30, agent_height=1.5,
    *, recompute_navmesh=True,
):
    import habitat_sim, magnum as mn
    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = glb; bk.enable_physics = False
    def cam(uuid, typ):
        s = habitat_sim.CameraSensorSpec(); s.uuid = uuid; s.sensor_type = typ
        s.resolution = [H, W]; s.hfov = HFOV_DEG; s.position = mn.Vector3(0, 0, 0); return s
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = [cam("color", habitat_sim.SensorType.COLOR),
                               cam("depth", habitat_sim.SensorType.DEPTH)]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ac]))
    loaded_navmesh = False
    if navmesh and os.path.isfile(navmesh):
        loaded_navmesh = bool(sim.pathfinder.load_nav_mesh(navmesh))
        if not loaded_navmesh:
            sim.close()
            raise RuntimeError(f"failed to load pinned navmesh: {navmesh}")
    # Re-bake the navmesh at the robot radius so the free space (and every geodesic) keeps
    # real clearance from walls and centres through doorways (PythonRobotics/iPlanner: inflate
    # by robot radius before planning). agent_radius ~0.3 matches iPlanner's robot_size.
    if recompute_navmesh:
        ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
        ns.agent_radius = agent_radius; ns.agent_height = agent_height
        ok = sim.recompute_navmesh(sim.pathfinder, ns)
        if not ok:
            sim.close()
            raise RuntimeError("Habitat navmesh recomputation failed")
        print(f"[make_sim] recompute_navmesh(agent_radius={agent_radius}) ok={ok} "
              f"navigable_area={sim.pathfinder.navigable_area:.1f} m^2")
    else:
        if not loaded_navmesh:
            sim.close()
            raise RuntimeError(
                "recompute_navmesh=False requires a readable pinned navmesh")
        print(f"[make_sim] pinned_navmesh={navmesh} "
              f"navigable_area={sim.pathfinder.navigable_area:.1f} m^2")
    return sim


def detect_floors(pf, n=5000, gap=1.0):
    """MP3D scenes are multi-floor; the navmesh spans all floors. Cluster navigable-point
    heights into floors (split where the height gap > `gap` m). Returns [(floor_y, count)]
    sorted by count desc (count ~ floor area, used to weight floor choice)."""
    ys = np.sort(np.array([pf.get_random_navigable_point()[1] for _ in range(n)]))
    floors, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] > gap:
            floors.append(cur); cur = []
        cur.append(y)
    floors.append(cur)
    out = [(float(np.median(f)), len(f)) for f in floors]
    return sorted(out, key=lambda t: -t[1])


def build_esdf(pf, floor_y, res=0.05, pad=0.5, floor_tol=0.8):
    """2D ESDF (distance-to-navmesh-boundary, metres) over habitat x-z FOR ONE FLOOR: a cell is
    free only if snapping (x, floor_y, z) lands on this floor (|q.y - floor_y| < floor_tol), so
    other floors don't leak into the 2D map. = iPlanner cost map, per floor."""
    from scipy import ndimage
    lo, hi = pf.get_bounds()
    x0, z0 = float(lo[0]) - pad, float(lo[2]) - pad
    x1, z1 = float(hi[0]) + pad, float(hi[2]) + pad
    nx = int((x1 - x0) / res) + 1; nz = int((z1 - z0) / res) + 1
    free = np.zeros((nz, nx), bool)
    for iz in range(nz):
        for ix in range(nx):
            gx, gz = x0 + ix * res, z0 + iz * res
            q = pf.snap_point([gx, floor_y, gz]) # nearest navmesh point to the query
            free[iz, ix] = (pf.is_navigable(q) and abs(q[0] - gx) < res and abs(q[2] - gz) < res
                            and abs(q[1] - floor_y) < floor_tol)
    dist = ndimage.distance_transform_edt(free) * res # distance to the nearest non-free cell (the nearest boundary)
    gzd, gxd = np.gradient(dist, res)
    return dict(dist=dist, gx=gxd, gz=gzd, x0=x0, z0=z0, res=res, nx=nx, nz=nz, floor_y=float(floor_y))


def sample_esdf(E, x, z):
    """Bilinear (clearance, grad_xz) at habitat (x,z)."""
    fx = (x - E["x0"]) / E["res"]; fz = (z - E["z0"]) / E["res"]
    ix = int(np.clip(fx, 0, E["nx"] - 1)); iz = int(np.clip(fz, 0, E["nz"] - 1))
    return float(E["dist"][iz, ix]), np.array([float(E["gx"][iz, ix]), float(E["gz"][iz, ix])])


def geodesic(pf, a, b):
    import habitat_sim
    p = habitat_sim.ShortestPath(); p.requested_start = a; p.requested_end = b
    ok = pf.find_path(p)
    return (ok, float(p.geodesic_distance), [np.array(x, float) for x in p.points])


def densify(points, step_m=0.20):
    """polyline -> list of positions spaced ~step_m along it."""
    out = []
    for i in range(len(points) - 1):
        p, q = points[i], points[i + 1]
        seg = q - p; L = np.linalg.norm(seg)
        n = max(1, int(np.ceil(L / step_m)))
        for k in range(n):
            out.append(p + seg * (k / n))
    out.append(points[-1])
    return out


def yaw_facing(delta_xz):
    """heading (Habitat, rot about +Y) so camera -Z faces horizontal direction delta."""
    dx, dz = delta_xz
    return np.arctan2(-dx, -dz)  # camera forward = -Z


def elastic_smooth(pts, pf, E, iters=60, kc=0.5, kr=0.8, rho0=0.6, step=0.04, res=0.05):
    """ElasticBands smoothing of a geodesic (PythonRobotics ElasticBands / iPlanner ESDF cost):
    each interior point feels a contraction force (smoothness, toward neighbour midpoint) and
    a repulsion force up the clearance gradient when clearance rho < rho0. Endpoints fixed;
    every update snapped back onto the navmesh so it stays feasible. Works in habitat x-z;
    returns smoothed 3D points (x, floor_y, z)."""
    P = np.array(densify(pts, res))          # dense, habitat (x,y,z)
    g = P[:, [0, 2]].astype(float)           # ground plane (x,z)
    y = float(P[0, 1])
    for _ in range(iters):
        ng = g.copy()
        for i in range(1, len(g) - 1):
            dp = g[i - 1] - g[i]; dn = g[i + 1] - g[i]
            contraction = kc * (dp / (np.linalg.norm(dp) + 1e-6) + dn / (np.linalg.norm(dn) + 1e-6))
            rho, grad = sample_esdf(E, g[i, 0], g[i, 1])
            repulsion = kr * (rho0 - rho) * grad if rho < rho0 else np.zeros(2)
            cand = g[i] + step * (contraction + repulsion)
            q = pf.snap_point([cand[0], y, cand[1]])              # keep feasible
            ng[i] = [q[0], q[2]]
        g = ng
    out = np.stack([g[:, 0], np.full(len(g), y), g[:, 1]], axis=1)
    return [out[i] for i in range(len(out))]


def pursuit_track(ref_pts, pf, init_pos=None, init_theta=None,
                  v_max=0.0376, L=0.7, r_min=0.40, v_min_frac=0.48,
                  max_turn_deg=4.5, cam_h=0.5, stop_before=0.0, floor_y=0.0,
                  turn_flip_deg=110.0):
    """Pure-pursuit unicycle tracking of a geodesic polyline, matching InternData-N1's
    controller (v≈0.0376 m/frame, lookahead 0.7 m, min radius 0.4 m). Produces smooth,
    COUPLED motion (no in-place spin): cruises with pursuit curvature (radius >= r_min);
    for a target behind (|alpha|>90°, e.g. the A->B U-turn) it rotates at the max angular
    rate while creeping (a tight arc, not a frozen pivot). Snaps to navmesh for collisions."""
    ref = np.array(densify(ref_pts, 0.05))  # output waypoints from elastic_smooth
    refg = ref[:, [0, 2]]                   # ground plane (x,z)
    kappa_max = 1.0 / r_min                 # curvature: standard geometric measure of how sharply a path bends. defined as the reciprocal of the turning radius: 1 / R
    mturn = np.deg2rad(max_turn_deg)
    pos = (np.asarray(init_pos)[[0, 2]] if init_pos is not None else refg[0]).astype(float) # robot's position
    # psi is the HABITAT camera yaw everywhere (camera forward = travel dir = (-sin psi, -cos psi)).
    # ONE convention: the stored frame yaw is psi as-is (no conversion), and it winds with the right
    # sign, so any downstream heading/omega derived from the pose is consistent by construction.
    # psi: counterclockwise rotation about the +Y (up) axis 
    if init_theta is not None:
        psi = float(init_theta)                              # already a habitat yaw (prev leg's last frame)
    else:
        psi = yaw_facing(refg[min(5, len(refg) - 1)] - pos)  # face the initial path direction
    ci, frames, goal = 0, [], refg[-1]  # initialization. ci: path index. frames: output/the trajectory recorded so far. goal: the (x, z) endpoint of the reference path
    arclen = float(np.sum(np.linalg.norm(np.diff(refg, axis=0), axis=1))) #  total length of the reference path
    max_steps = int(arclen / v_max * 4) + 300     # generous vs ideal frame count
    stall = 0

    for _ in range(max_steps):
        prev = pos.copy()
        ci += int(np.argmin(np.linalg.norm(refg[ci:ci + 40] - pos, axis=1))) #  index of the point on reference path the robot is currently closest to
        li, acc = ci, 0.0  # li: look ahead index
        while li + 1 < len(refg) and acc < L:
            acc += np.linalg.norm(refg[li + 1] - refg[li])
            li += 1
        to = refg[li] - pos  # look ahead point
        alpha = (yaw_facing(to) - psi + np.pi) % (2 * np.pi) - np.pi   # heading error (habitat yaw)
        # clearance-aware turn direction: for a near-reversal, the shorter-angle turn (sign of alpha) may
        # sweep the r_min arc INTO a wall while the OTHER side is open. Probe both arc centres; if the
        # short side is tighter, turn the "long way" (flip alpha by 2*pi) through the clear side.
        if abs(alpha) > np.deg2rad(turn_flip_deg):
            tl = np.array([-np.cos(psi), np.sin(psi)])   # side the arc bulges toward when psi INCREASES
            cL = pf.distance_to_closest_obstacle([pos[0] + r_min * tl[0], floor_y, pos[1] + r_min * tl[1]])
            cR = pf.distance_to_closest_obstacle([pos[0] - r_min * tl[0], floor_y, pos[1] - r_min * tl[1]])
            if alpha > 0 and cL + 0.1 < cR:
                alpha -= 2 * np.pi                        # short turn bulges into the wall -> go the long way
            elif alpha < 0 and cR + 0.1 < cL:
                alpha += 2 * np.pi
        # Smooth, single-branch control law (no bang-bang) so v/omega/radius vary
        # continuously through a turn -> no delta-spikes, matches N1's motion:
        #  * speed eases off in turns (N1: corr(turn,speed)=-0.49, floor ~0.48*cruise)
        #  * curvature is proportional to heading error, capped at kappa_max=1/r_min,
        #    so the tightest turn (incl. the U-turn) holds radius r_min, not a spin.
        v = v_max * (v_min_frac + (1 - v_min_frac) * (1 + np.cos(alpha)) / 2) # Speed: slow down in turns
        kappa = np.clip(2.0 * alpha / L, -kappa_max, kappa_max)               # Curvature: steer proportionally, capped at the tightest arc
        dpsi = np.clip(kappa * v, -mturn, mturn)                              # Yaw increment: turning is proportional to distance travelled
        psi += dpsi
        fwd = np.array([-np.sin(psi), -np.cos(psi)])  # habitat camera forward = travel direction
        ng = pos + v * fwd                            # next ground position
        snap = pf.snap_point([ng[0], floor_y, ng[1]]) # project onto walkable space
        if (not pf.is_navigable(snap)) or np.hypot(snap[0] - ng[0], snap[2] - ng[1]) > 0.25:
            ng = pos + 0.3 * v * fwd                         # creep if blocked
            snap = pf.snap_point([ng[0], floor_y, ng[1]])    # Retry the step at 30% length (~1.2 cm instead of ~4 cm)
        pos = np.array([snap[0], snap[2]])                   # robot's 2D pos is updated from the snapped point
        frames.append((np.array([snap[0], snap[1] + cam_h, snap[2]]), psi))   # psi = commanded (fallback)
        # forward-difference re-derive: now this step is known, set the PREVIOUS frame's yaw to face its
        # REALIZED displacement (where the robot actually went), not the commanded psi — the navmesh snap
        # can slide the robot off psi by up to ~80deg on wall-grazing frames, and the policy/PSR read the
        # realized motion. Gate sub-1cm steps (creep/stall) -> keep commanded psi (dodge atan2 noise).
        d = pos - prev                                         # the step just realized
        if len(frames) >= 2 and np.linalg.norm(d) > 0.01:
            frames[-2] = (frames[-2][0], float(yaw_facing(d))) # patch the PREVIOUS frame
        if stop_before > 0 and np.linalg.norm(pos - goal) < stop_before:
            return frames, True           # early stop (unused: goal is a recognition target, not an arrival pose)
        if ci >= len(refg) - 2 and np.linalg.norm(pos - goal) < 0.15:  # Real success
            return frames, True
        stall = stall + 1 if np.linalg.norm(pos - prev) < 0.004 else 0
        if stall > 40:                       # wedged against geometry -> give up
            return frames, False
    return frames, np.linalg.norm(pos - goal) < 0.3


def agent_state(pos, yaw):
    import habitat_sim
    st = habitat_sim.agent.AgentState()
    st.position = pos
    st.rotation = quaternion.from_rotation_vector([0, yaw, 0])
    return st


def cam_to_world_hab(pos, yaw):
    """4x4 camera-to-world in Habitat frame (cam optical = OpenGL: -Z fwd, +Y up)."""
    R = quaternion.as_rotation_matrix(quaternion.from_rotation_vector([0, yaw, 0]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    return T


def project_point(T_wc_hab, p_world_hab, depth_img):
    """Project world point into camera; return (u,v,z_fwd, visible, unoccluded)."""
    Rwc = T_wc_hab[:3, :3]; t = T_wc_hab[:3, 3]
    p_cam = Rwc.T @ (p_world_hab - t)            # habitat optical (-Z fwd, +Y up)
    p_cv = np.array([p_cam[0], -p_cam[1], -p_cam[2]])  # -> OpenCV (+Z fwd, +Y down)
    z = p_cv[2]
    if z <= 1e-3:
        return None
    u = FX * p_cv[0] / z + CX; v = FY * p_cv[1] / z + CY
    inb = (0 <= u < W) and (0 <= v < H)
    if not inb:
        return (u, v, z, False, False)
    d_ren = float(depth_img[int(v), int(u)])
    unocc = (d_ren <= 0) or (z <= d_ren + 0.25)   # B not clearly behind a surface
    return (u, v, z, True, unocc)


def render(sim, pos, yaw):
    sim.get_agent(0).set_state(agent_state(pos, yaw))
    o = sim.get_sensor_observations()
    return o["color"][..., :3].copy(), o["depth"].copy()


# --------------------------------------------------------------------------- #
# Ground-truth co-visibility (occlusion-checked reprojection). This is the
# operational "revisit" measure: a goal view is a revisit of a history frame if
# they co-observe enough of the same 3D surface. (Shared with revisit_sweep_gen.py.)
# --------------------------------------------------------------------------- #
def backproject(depth, stride=6, d_min=0.15, d_max=10.0):
    """depth [H,W] -> surface points in the CAMERA frame (habitat optical: -Z fwd, +Y up),
    subsampled on a `stride` pixel grid. [Np, 3]."""
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    d = depth[vs, us].astype(float)
    m = (d > d_min) & (d < d_max)
    u, v, d = us[m].astype(float), vs[m].astype(float), d[m]
    x = (u - CX) / FX * d
    y = (v - CY) / FY * d
    return np.stack([x, -y, -d], axis=1)


def to_world(p_cam, T_wc):
    return p_cam @ T_wc[:3, :3].T + T_wc[:3, 3]


def covis_frac(p_world, T_wc, depth, tol=0.3):
    """Fraction of world surface points `p_world` co-observed by camera (T_wc, depth):
    inside the frustum AND rendered depth agrees with reprojected range (not occluded)."""
    if len(p_world) == 0:
        return 0.0
    pc = (p_world - T_wc[:3, 3]) @ T_wc[:3, :3]
    x, y, z = pc[:, 0], -pc[:, 1], -pc[:, 2]
    m = z > 0.05
    zs = np.maximum(z, 1e-6)
    u = FX * x / zs + CX; v = FY * y / zs + CY
    m &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.clip(u.astype(int), 0, W - 1); vi = np.clip(v.astype(int), 0, H - 1)
    good = m & (np.abs(z - depth[vi, ui]) <= tol)
    return float(good.sum()) / len(p_world)


def max_covis(goal_pts_world, poses, depths, stride=4, tol=0.3):
    """Max co-visibility of a goal view (its world surface points) over history frames,
    and the argmax frame index. poses/depths are the rendered leg frames (subsampled)."""
    best, bi = 0.0, -1
    for i in range(0, len(poses), stride):
        c = covis_frac(goal_pts_world, poses[i], depths[i], tol)
        if c > best:
            best, bi = c, i
    return best, bi


def covis_curve(goal_pts_world, poses, depths, tol=0.3):
    """Occlusion-aware co-visibility of a goal view vs EVERY history frame (stride 1). Index i
    aligns with global frame i (history = legs concatenated in order). This is the multi-positive
    retrieval label: the loader thresholds it into positives (>=pos_hi) / negatives (<=pos_lo) /
    ignore-band, and its argmax is the relocalization anchor."""
    return np.array([covis_frac(goal_pts_world, poses[i], depths[i], tol) for i in range(len(poses))], float)


def save_traj(out_dir, rgbs, depths, poses_hab, meta, goal_rgbs):
    import pandas as pd, shutil
    # Wipe any prior (re)generation of this episode first: writing in place with
    # os.makedirs(exist_ok=True) would leave orphan tail frames when the new
    # trajectory is shorter than a previous one (stale N..M jpgs/pngs never overwritten).
    shutil.rmtree(out_dir, ignore_errors=True)
    rgb_d = os.path.join(out_dir, "videos/chunk-000/observation.images.rgb")
    dep_d = os.path.join(out_dir, "videos/chunk-000/observation.images.depth")
    dat_d = os.path.join(out_dir, "data/chunk-000"); met_d = os.path.join(out_dir, "meta")
    for d in (rgb_d, dep_d, dat_d, met_d):
        os.makedirs(d, exist_ok=True)
    for i, (c, dep) in enumerate(zip(rgbs, depths)):
        Image.fromarray(c).save(os.path.join(rgb_d, f"{i}.jpg"), quality=95)
        du16 = np.clip(dep * 10000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(du16).save(os.path.join(dep_d, f"{i}.png"))
    # one goal image per stop (goal_1.jpg = B, goal_2.jpg = C, ...); goal_image.jpg = first (B)
    for k, g in enumerate(goal_rgbs, start=1):
        Image.fromarray(g).save(os.path.join(out_dir, f"goal_{k}.jpg"), quality=95)
    if goal_rgbs:
        Image.fromarray(goal_rgbs[0]).save(os.path.join(out_dir, "goal_image.jpg"), quality=95)
    # Poses are stored as Z-up camera-to-world. NavDP expects action_R =
    # base_R @ camera_mount_R and removes the mount before making planar labels.
    # At zero Habitat yaw, action_R is M_W, so the corresponding mount is M_W,
    # not identity. Its translation is the camera height in the Z-up data frame.
    ext = np.eye(4)
    ext[:3, :3] = M_W
    ext[:3, 3] = M_W @ np.array([0.0, float(meta.get("camera_height_m", 0.5)), 0.0])
    rows = []
    for i, Tw in enumerate(poses_hab):
        Td = np.eye(4); Td[:3, :3] = M_W @ Tw[:3, :3]; Td[:3, 3] = M_W @ Tw[:3, 3]
        rows.append({"index": i,
                     "observation.camera_intrinsic": K.astype(np.float32).tolist(),
                     "observation.camera_extrinsic": ext.astype(np.float32).tolist(),
                     "action": Td.astype(np.float32).tolist()})
    pd.DataFrame(rows).to_parquet(os.path.join(dat_d, "episode_000000.parquet"))
    json.dump(meta, open(os.path.join(met_d, "gen_meta.json"), "w"), indent=2)


def align_turn(pos, yaw0, yaw1, max_turn_deg):
    """Bounded-rate in-place rotation from yaw0 to the goal-image orientation yaw1 (image-goal:
    after reaching the goal POSITION, turn so the final view matches the goal image)."""
    dy = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
    step = np.deg2rad(max_turn_deg)
    n = int(abs(dy) // step)
    fr = [(pos.copy(), yaw0 + np.sign(dy) * step * k) for k in range(1, n + 1)]
    fr.append((pos.copy(), float(yaw1)))
    return fr


def heading_at_closest(frames, G):
    """(travel heading, index) of the frame closest to point G — the view the robot had when it
    passed G. Used as G's goal-image orientation (well-defined, unlike direction-to-G)."""
    pts = np.array([f[0] for f in frames])
    i = int(np.argmin(np.linalg.norm(pts - np.asarray(G, float), axis=1)))
    return float(frames[i][1]), i


def heading_at_closest_multi(legs, G):
    best = None
    for lg in legs:
        y, i = heading_at_closest(lg, G)
        d = float(np.linalg.norm(np.array(lg[i][0]) - np.asarray(G, float)))
        if best is None or d < best[0]:
            best = (d, y)
    return best[1]


def roll_leg(geo_pts, pf, E, eb, cp, init_pos, init_theta, goal=None, goal_yaw=None, arrive=False):
    """One leg: ElasticBands-smooth the geodesic -> pursuit-track (N1 dynamics) to the goal
    POSITION. NO terminal orientation alignment: the goal image is only a recognition /
    relocalization target (retrieval finds the best-match history frame X, inserts the goal to
    read its map pose), so the robot need only reach the goal position — its arrival heading is
    the natural approach heading. Keeps the whole trajectory smooth (no pivot / loop).
    (goal/goal_yaw/arrive kept for signature compatibility; unused for the trajectory.)"""
    s = elastic_smooth(geo_pts, pf, E, **eb)
    return pursuit_track(s, pf, init_pos=init_pos, init_theta=init_theta, **cp)


def _clear(pf, p, rmin):
    return pf.is_navigable(p) and pf.distance_to_closest_obstacle(p) >= rmin


def _goal_world_pts(sim, gpos_floor, gyaw, ch):
    """Render a candidate goal view; return its world surface points (for co-visibility)."""
    _, dep = render(sim, np.asarray(gpos_floor, float) + ch, gyaw)
    T = cam_to_world_hab(np.asarray(gpos_floor, float) + ch, gyaw)
    return to_world(backproject(dep, stride=6), T)


def _render_leg(sim, frames):
    poses, depths, rgbs = [], [], []
    for pos, yaw in frames:
        c, d = render(sim, pos, yaw); rgbs.append(c); depths.append(d)
        poses.append(cam_to_world_hab(pos, yaw))
    return poses, depths, rgbs


def sample_revisit(sim, pf, hist_frames, hist_poses, hist_depths, n_anchor, rng, args, ch,
                   floor_y, source=None, min_geo=0.0, diagnostics=None,
                   diagnostic_prefix="revisit"):
    """A perturbed REVISIT goal, ANCHOR-CENTRIC (same parameterisation as revisit_sweep_gen).
    hist_frames/poses/depths cover the FULL history the retrieval head sees at this goal's step
    (leg1 for B; leg1+leg2 for C); n_anchor = #leading frames that ARE the revisit target (leg1).
    When a non-anchor tail exists, every tail frame must stay below ``covis_pos_lo``;
    this makes leg2 a verified hard-negative segment rather than merely calling it one.
      1. pick a random anchor frame X from the target leg ([anchor_margin, n_anchor));
      2. sample B position in a UNIFORM DISK of radius goal_jitter_pos around X, snapped to navmesh;
      3. sample B heading in a +/- head_max_deg CONE around X's heading;
      4. cheap stride gate on covis in [covis_lo, covis_hi]; on pass, compute the stride-1 covisibility
         curve over EVERY history frame -> argmax = GT relocalization anchor, curve = multi-positive label.
    Returns (pos, goal_yaw, covis, matched_frame, head_off_deg, covis_curve) or None."""
    if int(n_anchor) <= int(args.anchor_margin):
        _bump(diagnostics, f"{diagnostic_prefix}.anchor_window_too_short")
        _bump(diagnostics, f"{diagnostic_prefix}.sampling_exhausted")
        return None
    lo = int(args.anchor_margin)
    # long-term (implicit-memory) vs recent (in-view): for a fraction of revisits, force the matched
    # frame OUTSIDE the current window. current frame = last history frame (len(hist)-1); gap = current-X.
    hi = n_anchor
    if rng.uniform() < args.long_term_frac:
        u2 = len(hist_frames) - args.min_recall_gap          # X <= current - min_recall_gap
        if u2 > lo:
            hi = min(n_anchor, u2)
    if hi <= lo:
        hi = n_anchor                                         # long-term range empty (short leg) -> free
    R = args.goal_jitter_pos
    for _ in range(args.goal_tries):
        xi = int(rng.integers(lo, hi)) if hi > lo else int(rng.integers(0, n_anchor))
        Xp = hist_frames[xi][0]; Xyaw = float(hist_frames[xi][1])
        r = R * np.sqrt(rng.uniform()); th = rng.uniform(0, 2 * np.pi)      # uniform in the disk
        p = np.array(pf.snap_point([float(Xp[0] + r * np.cos(th)), floor_y, float(Xp[2] + r * np.sin(th))]), float)
        if not _clear(pf, p, args.r_min) or not _on_floor(p, floor_y, args.floor_tol):
            _bump(diagnostics, f"{diagnostic_prefix}.clearance_or_floor")
            continue
        if source is not None:
            ok, gd, _ = geodesic(pf, source, p)
            if not ok or gd < min_geo:
                _bump(diagnostics, f"{diagnostic_prefix}.source_geodesic")
                continue
        gyaw = Xyaw + np.deg2rad(rng.uniform(-args.head_max_deg, args.head_max_deg))   # cone around X
        gpts = _goal_world_pts(sim, p, gyaw, ch)
        cov, _ = max_covis(gpts, hist_poses, hist_depths, stride=args.covis_stride, tol=args.covis_tol)
        if not (args.covis_lo <= cov <= args.covis_hi):          # cheap subsampled GATE (reject fast)
            _bump(diagnostics, f"{diagnostic_prefix}.stride_covis")
            continue
        # On accept, the true label is the FULL stride-1 covisibility curve over every history frame:
        # position jitter means the sampling anchor X is not necessarily the best match, so the GT
        # matched frame is the curve argmax (relocalization anchor) and the curve is the retrieval label.
        curve = covis_curve(gpts, hist_poses, hist_depths, tol=args.covis_tol)
        # The relocalization anchor must be goal_append-able and must belong to
        # the designated anchor leg.  The old code searched ``curve[vlo:]`` and
        # could silently select a recent leg-2 frame for nominal A-revisits.
        valid_hi = min(int(n_anchor), len(curve))
        vlo = int(args.anchor_margin)
        if valid_hi <= vlo:
            _bump(diagnostics, f"{diagnostic_prefix}.anchor_window_too_short")
            continue
        ai = vlo + int(curve[vlo:valid_hi].argmax())
        cov = float(curve[ai])
        non_anchor_max = (
            float(curve[valid_hi:].max()) if valid_hi < len(curve) else 0.0)
        if non_anchor_max > float(args.covis_pos_lo):
            _bump(diagnostics, f"{diagnostic_prefix}.non_anchor_not_negative")
            continue
        head_off = abs((gyaw - hist_frames[ai][1] + np.pi) % (2 * np.pi) - np.pi)       # vs GT matched frame
        if (args.covis_lo <= cov <= args.covis_hi) and head_off <= np.deg2rad(args.head_max_deg):
            _bump(diagnostics, f"{diagnostic_prefix}.accepted")
            return p, float(gyaw), float(cov), int(ai), float(np.degrees(head_off)), curve
        _bump(diagnostics, f"{diagnostic_prefix}.anchor_covis_or_heading")
    _bump(diagnostics, f"{diagnostic_prefix}.sampling_exhausted")
    return None


def sample_novel(sim, pf, A, hist_poses, hist_depths, rng, args, ch, floor_y,
                 desired_geo_m=None, distance_match_tolerance_m=None,
                 diagnostics=None, diagnostic_prefix="novel"):
    """A NOVEL goal: on THIS floor, navigable, clearance>=r_min, geodesic(A,.)>min_dist_AB, and
    max co-visibility with history < novel_covis (retrieval target = null -> all frames negative).
    Returns (pos_floor, goal_yaw, covis, covis_curve) or None."""
    for _ in range(args.goal_tries):
        p = np.array(pf.get_random_navigable_point(), float)
        if not _clear(pf, p, args.r_min) or not _on_floor(p, floor_y, args.floor_tol):
            _bump(diagnostics, f"{diagnostic_prefix}.clearance_or_floor")
            continue
        ok, gd, _ = geodesic(pf, A, p)
        if not ok or gd < args.min_dist_AB or gd > args.max_dist_AB:
            _bump(diagnostics, f"{diagnostic_prefix}.distance_band")
            continue
        if (desired_geo_m is not None
                and distance_match_tolerance_m is not None
                and abs(float(gd) - float(desired_geo_m))
                > float(distance_match_tolerance_m)):
            _bump(diagnostics, f"{diagnostic_prefix}.role_distance_match")
            continue
        gyaw = yaw_facing((p - np.asarray(A, float))[[0, 2]])   # view along the approach
        gpts = _goal_world_pts(sim, p, gyaw, ch)
        cov, _ai = max_covis(gpts, hist_poses, hist_depths, stride=args.covis_stride, tol=args.covis_tol)
        if cov >= args.novel_covis:                              # cheap subsampled reject
            _bump(diagnostics, f"{diagnostic_prefix}.stride_covis")
            continue
        # confirm genuinely unseen over EVERY frame: the stride gate can SKIP the one frame that
        # observed p and wrongly call it novel. The full curve doubles as the (all-negative) label.
        curve = covis_curve(gpts, hist_poses, hist_depths, tol=args.covis_tol)
        if float(curve.max()) < args.novel_covis:
            _bump(diagnostics, f"{diagnostic_prefix}.accepted")
            return p, float(gyaw), float(curve.max()), curve
        _bump(diagnostics, f"{diagnostic_prefix}.full_covis")
    _bump(diagnostics, f"{diagnostic_prefix}.sampling_exhausted")
    return None


def _on_floor(p, floor_y, tol):
    return abs(float(p[1]) - floor_y) < tol


def _geo_on_floor(pts, floor_y, tol):
    """True if every geodesic waypoint stays on the floor (no stairs to another level)."""
    return all(abs(float(w[1]) - floor_y) < tol for w in pts)


def _rand_on_floor(pf, floor_y, tol, tries=40):
    for _ in range(tries):
        p = pf.get_random_navigable_point()
        if _on_floor(p, floor_y, tol):
            return p
    return None


def _get_esdf(cache, pf, floor_y, args):
    """Per-floor ESDF, cached by height bucket (stairs are navigable so we can't pre-split
    floors by height gaps; instead each episode's floor is defined by its A, and we build/reuse
    one ESDF per floor bucket)."""
    key = round(floor_y / 0.5)
    if key not in cache:
        cache[key] = build_esdf(pf, floor_y, res=args.esdf_res, floor_tol=args.floor_tol)
    return cache[key]


def make_episode(sim, rng, args, ep_idx, esdf_cache, diagnostics=None):
    pf = sim.pathfinder
    ftol = args.floor_tol
    eb = dict(iters=args.eb_iters, kc=args.eb_kc, kr=args.eb_kr, rho0=args.eb_rho0,
              step=args.eb_step, res=args.esdf_res)
    ch = np.array([0, args.cam_h, 0])
    for _attempt in range(args.max_attempts):
        _bump(diagnostics, "episode_candidate_attempt")
        # --- A defines the episode's FLOOR (any navigable point); everything else stays on it ---
        A = pf.get_random_navigable_point()
        if not _clear(pf, A, args.r_min):
            _bump(diagnostics, "episode.a_clearance")
            continue
        floor_y = float(A[1])
        cp = dict(v_max=args.v, L=args.lookahead, r_min=args.r_min, v_min_frac=args.v_min_frac,
                  max_turn_deg=args.max_turn_deg, cam_h=args.cam_h, floor_y=floor_y)  # pure-pursuit controller parameters,
        # --- start on A's floor, geodesic start->A stays on floor ---
        start = None
        for _ in range(30):
            s = _rand_on_floor(pf, floor_y, ftol)
            if s is None:
                continue
            ok, gd, pts = geodesic(pf, s, A) # from start to A. gd:  the geodesic distance, pts: 3D waypoints of the path
            if ok and args.dA_min <= gd <= args.dA_max and _geo_on_floor(pts, floor_y, ftol):
                start, gdA, g1 = s, gd, pts
                break
        if start is None:
            _bump(diagnostics, "episode.a_start_sampling")
            continue
        # ESDF construction is the expensive, memory-heavy per-floor step.
        # Defer it until the floor has a valid 3--9 m start/goal pair; the old
        # order cached large grids even for floors rejected immediately after.
        E = _get_esdf(esdf_cache, pf, floor_y, args)
        initial_yaw_mode = resolve_initial_yaw_mode(
            args.initial_yaw_mode, args.n_legs)
        requested_initial_yaw = (
            float(rng.uniform(-np.pi, np.pi))
            if initial_yaw_mode == "uniform" else None)
        leg1, ok = roll_leg(
            g1, pf, E, eb, cp,
            start if requested_initial_yaw is not None else None,
            requested_initial_yaw,
        )
        if not ok:
            _bump(diagnostics, "episode.a_rollout")
            continue
        p1, d1, r1 = _render_leg(sim, leg1)
        sampled_A = np.asarray(A, dtype=float).copy()
        dataset_start = np.asarray(leg1[0][0], dtype=float) - ch
        start_for_metadata = np.asarray(start, dtype=float)
        start_path_points = g1
        if args.n_legs >= 3:
            # The evaluator uses the final expert frame as Goal A.  Bind the
            # target position to that exact frame and the episode start to the
            # first *stored* expert frame.  The historical metadata measured
            # from the unrecorded pre-step pose, creating a ~4 cm hidden length
            # offset and occasionally admitting an evaluated A below dA_min.
            A = np.asarray(leg1[-1][0], dtype=float) - ch
            start_for_metadata = dataset_start
            okA, gdA, start_path_points = geodesic(
                pf, start_for_metadata, A)
            if (not okA or not args.dA_min <= gdA <= args.dA_max
                    or not _on_floor(A, floor_y, ftol)):
                _bump(diagnostics, "episode.a_terminal_reanchor")
                continue
        start_path_yaw = first_path_yaw(
            start_path_points, start_for_metadata)
        actual_start_yaw = float(leg1[0][1])
        start_heading_offset_deg = wrap_degrees(np.degrees(
            actual_start_yaw - start_path_yaw))

        # Paper role-pair construction needs only a frozen start pose and a
        # native Goal-A image.  Requiring a subsequently generated Revisit-B
        # here is unrelated to that source contract and causes severe,
        # outcome-blind attrition in compact datasets such as Replica.  This
        # opt-in carrier keeps every Goal-A sampling, rollout, rendering, and
        # safety condition above unchanged.  The placeholder second goal only
        # satisfies the historical two-leg file schema; callers must use this
        # mode exclusively with eval_2leg_habitat --stop_after_leg1.
        if getattr(args, "goal_a_source_only", False):
            allframes = list(leg1)
            fy = float(allframes[0][0][1] - args.cam_h)
            xyt = np.array([[frame[0][0], frame[0][2]] for frame in allframes])
            seg = np.diff(xyt, axis=0)
            tang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
            if (len(tang) > 1
                    and float(np.abs(np.degrees(np.diff(tang))).max())
                    > args.max_frame_turn):
                _bump(diagnostics, "episode.trajectory_turn_spike")
                continue

            def _goal_a_nav_ok(x, z):
                q = pf.snap_point([x, fy, z])
                return (pf.is_navigable(q)
                        and abs(q[0] - x) < 0.06
                        and abs(q[2] - z) < 0.06)

            if any(
                    not _goal_a_nav_ok(
                        xyt[i - 1, 0] * (1 - t) + xyt[i, 0] * t,
                        xyt[i - 1, 1] * (1 - t) + xyt[i, 1] * t,
                    )
                    for i in range(1, len(xyt))
                    for t in (0.25, 0.5, 0.75)):
                _bump(diagnostics, "episode.trajectory_collision")
                continue

            def _d(p):
                return (M_W @ np.asarray(p, float)).tolist()

            switch = int(len(leg1))
            placeholder_yaw = float(leg1[-1][1])
            meta = dict(
                scene=os.path.basename(args.scene),
                ep_idx=ep_idx,
                generation_seed=int(args.seed),
                n_frames=switch,
                n_legs=2,
                switch_idx=switch,
                switches=[switch],
                start=_d(start_for_metadata),
                A=_d(A),
                goals=[dict(
                    name="B",
                    kind="source_only_placeholder",
                    pos=_d(A),
                    yaw_habitat=placeholder_yaw,
                    covis=1.0,
                    covis_argmax=switch - 1,
                    head_off_deg=0.0,
                    anchor_frame_limit=switch,
                    non_anchor_max_covis=0.0,
                    recall_gap=0,
                    covis_curve=[0.0] * max(0, switch - 1) + [1.0],
                )],
                geo_startA=float(gdA),
                geo_AB=0.0,
                geo_BC=None,
                gen_protocol=GOAL_A_SOURCE_PROTOCOL,
                role_sequence=[
                    "initial_imagegoal", "source_only_placeholder"],
                source_only_goal_a=True,
                source_only_usage="eval_2leg_habitat --stop_after_leg1",
                query_goal_present=False,
                initial_yaw_mode=initial_yaw_mode,
                start_yaw_habitat=actual_start_yaw,
                start_path_yaw_habitat=float(start_path_yaw),
                start_heading_offset_deg=float(start_heading_offset_deg),
                initial_distance_band_m=[
                    float(args.dA_min), float(args.dA_max)],
                initial_goal_pose_source="legacy_sampled_target",
                initial_start_pose_source="legacy_unrecorded_pre_step",
                covis_band=[args.covis_lo, args.covis_hi],
                novel_covis=args.novel_covis,
                covis_pos_hi=args.covis_pos_hi,
                covis_pos_lo=args.covis_pos_lo,
                window=args.window,
                num_scale=args.num_scale,
                anchor_margin=args.anchor_margin,
                camera_height_m=float(args.cam_h),
                frame_convention=(
                    "positions+parquet in data(Zup,M_W); "
                    "yaw_habitat in render frame"),
            )
            _bump(diagnostics, "episode.accepted")
            return r1, d1, p1, meta, [r1[-1]]

        # --- B: 2-leg -> REVISIT on leg A.  Three-leg defaults to NOVEL,
        #     while the explicit double-Revisit diagnostic makes B a Revisit. ---
        # history at B's step = leg1 (n_anchor = all of leg1: the revisit target).
        double_revisit = (
            args.n_legs >= 3
            and args.three_leg_roles == "double_revisit"
        )
        if args.n_legs == 2 or double_revisit:
            rv = sample_revisit(
                sim, pf, leg1, p1, d1, len(leg1), rng, args, ch,
                floor_y, diagnostics=diagnostics,
                diagnostic_prefix="revisit_b")
            if rv is None:
                _bump(diagnostics, "episode.revisit_b_sampling")
                continue
            B, yaw_B, covB, aiB, hoB, curveB = rv; kindB = "revisit"; arriveB = True
        else:
            nv = sample_novel(
                sim, pf, A, p1, d1, rng, args, ch, floor_y,
                desired_geo_m=float(gdA),
                distance_match_tolerance_m=float(
                    args.role_distance_match_tol_m),
                diagnostics=diagnostics,
                diagnostic_prefix="novel_b",
            )
            if nv is None:
                _bump(diagnostics, "episode.novel_b_sampling")
                continue
            # NOTE: roll_leg's goal_yaw/arrive are signature-compat no-ops; the goal-image
            # symmetry fix happens AFTER the leg is rolled (see the arrival-yaw re-anchor below).
            B, yaw_B, covB, curveB = nv; aiB = -1; hoB = None; kindB = "novel"; arriveB = False
            sampled_B = np.asarray(B, dtype=float).copy()
        okB, gdB, g2 = geodesic(pf, A, B)
        if not okB or gdB < args.b_min or not _geo_on_floor(g2, floor_y, ftol):
            _bump(diagnostics, "episode.b_geodesic")
            continue
        leg2, ok = roll_leg(g2, pf, E, eb, cp, leg1[-1][0], leg1[-1][1],
                            goal=B, goal_yaw=yaw_B, arrive=arriveB)
        if not ok:
            _bump(diagnostics, "episode.b_rollout")
            continue
        p2, d2, r2 = _render_leg(sim, leg2)
        if args.n_legs >= 3 and kindB == "novel":
            # v4 exact symmetry: Goal B is the expert's actual terminal pose
            # and exact terminal RGB, just as Goal A is its expert terminal
            # frame.  Re-check distance and novelty after re-anchoring.
            B = np.asarray(leg2[-1][0], dtype=float) - ch
            yaw_B = float(leg2[-1][1])
            okB, gdB, _ = geodesic(pf, A, B)
            if (not okB or not args.min_dist_AB <= gdB <= args.max_dist_AB
                    or abs(float(gdB) - float(gdA))
                    > float(args.role_distance_match_tol_m)
                    or not _on_floor(B, floor_y, ftol)):
                _bump(diagnostics, "episode.b_terminal_reanchor")
                continue
            curveB = covis_curve(_goal_world_pts(sim, B, yaw_B, ch), p1, d1,
                                 tol=args.covis_tol)
            covB = float(curveB.max())
            if covB >= args.novel_covis:
                _bump(diagnostics, "episode.b_terminal_covis")
                continue
        legs = [leg1, leg2]; R = [(p1, d1, r1), (p2, d2, r2)]
        goals = [dict(
            name="B", pos=B, yaw=yaw_B, kind=kindB, covis=covB,
            covis_argmax=aiB, head_off_deg=hoB, covis_curve=curveB,
            anchor_frame_limit=(len(leg1) if kindB == "revisit" else None),
            non_anchor_max_covis=0.0,
        )]

        # --- 3-leg: C REVISITS leg A (leg1), reached from B. History at C's step = leg1+leg2, so leg2
        #     frames are hard NEGATIVES; anchor is still sampled from leg1 (n_anchor = len(leg1)). ---
        if args.n_legs >= 3:
            hist_fr = leg1 + leg2; hist_p = p1 + p2; hist_d = d1 + d2
            rv = sample_revisit(sim, pf, hist_fr, hist_p, hist_d, len(leg1), rng, args, ch, floor_y,
                                source=B, min_geo=args.c_min,
                                diagnostics=diagnostics,
                                diagnostic_prefix="revisit_c")
            if rv is None:
                _bump(diagnostics, "episode.revisit_c_sampling")
                continue
            C, yaw_C, covC, aiC, hoC, curveC = rv
            if (double_revisit
                    and abs(int(aiC) - int(aiB))
                    < int(args.double_revisit_min_anchor_gap)):
                _bump(diagnostics, "revisit_c.anchor_too_close_to_b")
                continue
            okC, gdC, g3 = geodesic(pf, B, C)
            if not okC or gdC < args.c_min or not _geo_on_floor(g3, floor_y, ftol):
                _bump(diagnostics, "episode.c_geodesic")
                continue
            leg3, ok = roll_leg(g3, pf, E, eb, cp, leg2[-1][0], leg2[-1][1],
                                goal=C, goal_yaw=yaw_C, arrive=True)
            if not ok:
                _bump(diagnostics, "episode.c_rollout")
                continue
            p3, d3, r3 = _render_leg(sim, leg3)
            legs.append(leg3); R.append((p3, d3, r3))
            goals.append(dict(
                name="C", pos=C, yaw=yaw_C, kind="revisit",
                covis=covC, covis_argmax=aiC, head_off_deg=hoC,
                covis_curve=curveC, anchor_frame_limit=len(leg1),
                non_anchor_max_covis=(
                    float(curveC[len(leg1):].max())
                    if len(curveC) > len(leg1) else 0.0),
            ))

        # --- assemble (already rendered per leg) ---
        rgbs = [x for (_p, _d, rr) in R for x in rr]
        depths = [x for (_p, dd, _r) in R for x in dd]
        poses = [x for (pp, _d, _r) in R for x in pp]
        allframes = [f for lg in legs for f in lg]
        switches = [int(s) for s in np.cumsum([len(lg) for lg in legs])[:-1]]

        # --- trajectory safety gate: reject (retry) the WHOLE episode if any sharp turn-rate spike or
        #     any segment clips geometry. Cheap net guaranteeing the written trajectory is smooth +
        #     collision-free even where the controller's per-frame caps didn't (e.g. snap-slide jumps). ---
        fy = float(allframes[0][0][1] - args.cam_h)
        xyt = np.array([[f[0][0], f[0][2]] for f in allframes])
        # (1) turn rate: no frame-to-frame heading change sharper than max_frame_turn (radius >= r_min ->
        #     ~v/turn; a spike means the realized motion kinked, e.g. a navmesh snap-slide).
        seg = np.diff(xyt, axis=0)
        tang = np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))
        if len(tang) > 1 and float(np.abs(np.degrees(np.diff(tang))).max()) > args.max_frame_turn:
            _bump(diagnostics, "episode.trajectory_turn_spike")
            continue
        # (2) collision: sample densely along every inter-frame segment; all must stay on the navmesh
        #     (a straight hop between two navigable frames can still clip an obstacle corner).
        def _gnav(x, z):
            q = pf.snap_point([x, fy, z])
            return pf.is_navigable(q) and abs(q[0] - x) < 0.06 and abs(q[2] - z) < 0.06
        if any(not _gnav(xyt[i - 1, 0] * (1 - t) + xyt[i, 0] * t, xyt[i - 1, 1] * (1 - t) + xyt[i, 1] * t)
               for i in range(1, len(xyt)) for t in (0.25, 0.5, 0.75)):
            _bump(diagnostics, "episode.trajectory_collision")
            continue

        goal_rgbs = []
        for goal in goals:
            if (args.n_legs >= 3 and goal["name"] == "B"
                    and goal["kind"] == "novel"):
                # Saving the same RGB array with the same JPEG settings makes
                # goal_1.jpg byte-identical to expert frame switch_b-1.
                goal_rgbs.append(r2[-1])
            else:
                goal_rgbs.append(render(
                    sim, np.asarray(goal["pos"], float) + ch,
                    goal["yaw"])[0])

        def _d(p):
            return (M_W @ np.asarray(p, float)).tolist()
        gen_protocol = multileg_protocol(
            args.n_legs, args.three_leg_roles)
        meta = dict(scene=os.path.basename(args.scene), ep_idx=ep_idx,
                    generation_seed=int(args.seed), n_frames=len(rgbs),
                    n_legs=len(legs), switch_idx=int(switches[0]), switches=switches,
                    start=_d(start_for_metadata), A=_d(A),
                    goals=[dict(name=g["name"], kind=g["kind"], pos=_d(g["pos"]),
                                yaw_habitat=g["yaw"], covis=round(float(g["covis"]), 4),
                                covis_argmax=int(g["covis_argmax"]),
                                head_off_deg=(round(g["head_off_deg"], 1) if g.get("head_off_deg") is not None else None),
                                anchor_frame_limit=g.get("anchor_frame_limit"),
                                non_anchor_max_covis=round(
                                    float(g.get("non_anchor_max_covis", 0.0)), 4),
                                # recall gap = current(=len history-1) - matched frame; large => long-term memory.
                                recall_gap=(len(g["covis_curve"]) - 1 - int(g["covis_argmax"])
                                            if int(g["covis_argmax"]) >= 0 else None),
                                # multi-positive retrieval label: covis vs every history frame [0..step-1];
                                # loader thresholds into positive/negative/ignore (pos_hi/pos_lo below).
                                covis_curve=[round(float(c), 4) for c in g["covis_curve"]])
                           for g in goals],
                    geo_startA=float(gdA),
                    geo_AB=(float(gdB) if args.n_legs >= 3 else None),
                    geo_BC=(float(gdC) if args.n_legs >= 3 else None),
                    gen_protocol=gen_protocol,
                    role_sequence=(
                        (["initial_imagegoal", "revisit", "revisit"]
                         if double_revisit else
                         ["initial_imagegoal", "novel", "revisit"])
                        if args.n_legs >= 3
                        else ["initial_imagegoal", "revisit"]),
                    initial_yaw_mode=initial_yaw_mode,
                    start_yaw_habitat=actual_start_yaw,
                    start_path_yaw_habitat=float(start_path_yaw),
                    start_heading_offset_deg=float(start_heading_offset_deg),
                    initial_distance_band_m=[float(args.dA_min), float(args.dA_max)],
                    novel_distance_band_m=[float(args.min_dist_AB), float(args.max_dist_AB)],
                    initial_goal_pose_source=(
                        "expert_arrival_frame_exact"
                        if args.n_legs >= 3 else "legacy_sampled_target"),
                    initial_start_pose_source=(
                        "first_stored_expert_frame_exact"
                        if args.n_legs >= 3 else "legacy_unrecorded_pre_step"),
                    novel_b_max_dist=float(args.max_dist_AB),
                    novel_b_goal_yaw=(
                        "expert_arrival_heading" if kindB == "novel" else None),
                    novel_b_goal_image_source=(
                        "expert_arrival_frame_exact"
                        if args.n_legs >= 3 and kindB == "novel" else None),
                    role_pairing=(
                        "same_episode_geodesic"
                        if args.n_legs >= 3 and kindB == "novel" else None),
                    role_distance_match_tolerance_m=(
                        float(args.role_distance_match_tol_m)
                        if args.n_legs >= 3 and kindB == "novel" else None),
                    role_distance_error_m=(
                        abs(float(gdA) - float(gdB))
                        if args.n_legs >= 3 and kindB == "novel" else None),
                    sampled_target_error_m=(
                        {"A": float(np.linalg.norm(A - sampled_A)),
                         "B": float(np.linalg.norm(B - sampled_B))}
                        if args.n_legs >= 3 and kindB == "novel" else None),
                    double_revisit_min_anchor_gap=(
                        int(args.double_revisit_min_anchor_gap)
                        if double_revisit else None),
                    double_revisit_anchor_gap=(
                        abs(int(aiC) - int(aiB))
                        if double_revisit else None),
                    double_revisit_goal_image_source=(
                        "metadata_pose_render" if double_revisit else None),
                    double_revisit_distance_min_m=(
                        {"B": float(args.b_min), "C": float(args.c_min)}
                        if double_revisit else None),
                    covis_band=[args.covis_lo, args.covis_hi], novel_covis=args.novel_covis,
                    covis_pos_hi=args.covis_pos_hi, covis_pos_lo=args.covis_pos_lo,
                    # LingBot streaming: valid match range = [anchor_margin, step); loader masks [0,anchor_margin)
                    # from retrieval positives (goal_append can't reconstruct a match below num_scale+window-1).
                    window=args.window, num_scale=args.num_scale, anchor_margin=args.anchor_margin,
                    camera_height_m=float(args.cam_h),
                    frame_convention="positions+parquet in data(Zup,M_W); yaw_habitat in render frame")
        _bump(diagnostics, "episode.accepted")
        return rgbs, depths, poses, meta, goal_rgbs
    _bump(diagnostics, "make_episode.exhausted")
    return None


def _count_complete(out_dir, n):
    """# of consecutive episode_XXXX dirs already fully written (parquet + meta), capped at n."""
    c = 0
    while c < n:
        ep = os.path.join(out_dir, f"episode_{c:04d}")
        if (os.path.isfile(os.path.join(ep, "data/chunk-000/episode_000000.parquet"))
                and os.path.isfile(os.path.join(ep, "meta/gen_meta.json"))):
            c += 1
        else:
            break
    return c


def run_legs(sim, args, rng, esdf_cache, n_legs, n, out_dir):
    """Generate `n` episodes of `n_legs` legs into out_dir, reusing the loaded sim + per-floor ESDF
    cache. Resumable: if out_dir already holds n complete episodes, skip; otherwise (re)generate from
    scratch (deterministic given args.seed, so any already-written episodes are reproduced identically)."""
    if _count_complete(out_dir, n) >= n:
        print(f"[skip] {out_dir} already complete ({n} episodes)")
        return n
    args.n_legs = n_legs
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    diagnostics = {}
    # Episode validity is intentionally strict (same-floor geodesic, revisit
    # visibility, smooth tracking, and collision checks).  Small MP3D scenes can
    # therefore need many more rejected candidates than the historical fixed
    # six calls per requested episode.  Keep the acceptance distribution and RNG
    # stream unchanged, but make the label-blind sampling budget explicit so a
    # confirmation run can fail closed or allocate more attempts without relaxing
    # any episode criterion.
    attempt_budget = n * args.episode_attempt_multiplier
    attempts_used = 0
    for attempt_index in range(attempt_budget):
        if made >= n:
            break
        attempts_used = attempt_index + 1
        res = make_episode(
            sim, rng, args, made, esdf_cache, diagnostics=diagnostics)
        if res is None:
            continue
        rgbs, depths, poses, meta, goal_rgbs = res
        ep_dir = os.path.join(out_dir, f"episode_{made:04d}")
        save_traj(ep_dir, rgbs, depths, poses, meta, goal_rgbs)
        gsum = " ".join(f"{g['name']}[{g['kind']}]covis{g['covis']:.2f}"
                        + (f"/head{g['head_off_deg']:.0f}deg" if g.get('head_off_deg') is not None else "")
                        + (f"/gap{g['recall_gap']}" if g.get('recall_gap') is not None else "")
                        for g in meta["goals"])
        print(f"[{n_legs}leg ep {made}] sample_attempt={attempts_used}/{attempt_budget} "
              f"frames={meta['n_frames']} switches={meta['switches']} "
              f"geo start->A={meta['geo_startA']:.1f} goals: {gsum}")
        made += 1
    print(f"DONE: {made}/{n} attempts={attempts_used}/{attempt_budget} "
          f"n_legs={n_legs} -> {out_dir}")
    generation_summary = {
        "protocol": (
            GOAL_A_SOURCE_PROTOCOL
            if getattr(args, "goal_a_source_only", False) else
            multileg_protocol(n_legs, args.three_leg_roles)),
        "n_legs": int(n_legs),
        "requested_episodes": int(n),
        "generated_episodes": int(made),
        "complete": bool(made >= n),
        "outer_attempts_used": int(attempts_used),
        "outer_attempt_budget": int(attempt_budget),
        "acceptance_per_outer_attempt": (
            float(made / attempts_used) if attempts_used else None),
        "rejection_counts": dict(sorted(diagnostics.items())),
    }
    with open(os.path.join(out_dir, "generation_summary.json"), "w") as handle:
        json.dump(generation_summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    ranked_rejections = sorted(
        ((count, name) for name, count in diagnostics.items()
         if not name.endswith(".accepted")),
        reverse=True,
    )[:8]
    if ranked_rejections:
        print("[generation diagnostics] " + ", ".join(
            f"{name}={count}" for count, name in ranked_rejections))
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--navmesh", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=5)
    # dual-leg mode: generate BOTH leg types for one loaded scene, reusing the sim + ESDF cache.
    # When either is set, --out is a ROOT and output nests as <out>/mp3d_{2,3}leg/<scene_id>/episode_XXXX
    # (single-scene --n/--n_legs behavior is unchanged when both are None).
    ap.add_argument("--n2", type=int, default=None, help="dual-leg: #2-leg episodes for this scene")
    ap.add_argument("--n3", type=int, default=None, help="dual-leg: #3-leg episodes for this scene")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--initial_yaw_mode",
        choices=INITIAL_YAW_MODES,
        default="auto",
        help=("first-leg camera yaw: auto keeps historical path alignment for "
              "2-leg Revisit but uses a uniform yaw for 3-leg role-symmetric "
              "evaluation; path_aligned reproduces the legacy bias"),
    )
    ap.add_argument("--episode_attempt_multiplier", type=int, default=6,
                    help="maximum make_episode calls per requested episode; changes only the "
                         "sampling budget, never episode acceptance criteria")
    ap.add_argument(
        "--allow_incomplete",
        action="store_true",
        help=("diagnostic only: return success when the attempt budget produces "
              "fewer episodes than requested; formal generation fails closed"),
    )
    ap.add_argument("--dA_min", type=float, default=3.0); ap.add_argument("--dA_max", type=float, default=9.0)
    ap.add_argument("--b_min", type=float, default=2.0)
    ap.add_argument("--n_legs", type=int, default=2, choices=[2, 3],
                    help="2: start->A->B (B revisit on leg A) ; 3: ->C (B novel off leg A, C revisit on leg A)")
    ap.add_argument(
        "--goal_a_source_only",
        action="store_true",
        help=("construct only the unchanged first ImageGoal leg and emit a "
              "two-leg-schema carrier for --stop_after_leg1 source collection; "
              "no Revisit query is generated or valid in this mode"),
    )
    ap.add_argument(
        "--three_leg_roles",
        choices=THREE_LEG_ROLE_MODES,
        default="novel_revisit",
        help=("3-leg role sequence: novel_revisit keeps the strict v4 "
              "A->Novel-B->Revisit-C benchmark; double_revisit samples "
              "two distinct leg-A revisits and makes leg B a hard negative "
              "for C"),
    )
    ap.add_argument(
        "--double_revisit_min_anchor_gap",
        type=int,
        default=32,
        help=("double_revisit only: minimum frame separation between the "
              "B and C leg-A relocalization anchors"),
    )
    ap.add_argument("--c_min", type=float, default=2.0, help="min geodesic B->C for the 3rd leg")
    # revisit / novel definition (co-visibility) + goal perturbation
    ap.add_argument("--covis_lo", type=float, default=0.20, help="revisit: min max-covisibility with history")
    ap.add_argument("--covis_hi", type=float, default=1.00, help="revisit: max max-covisibility (avoid exact copy)")
    # multi-positive retrieval label thresholds — RECORDED into meta only; the loader applies them to
    # covis_curve (positive >= pos_hi, negative <= pos_lo, ignore-band between). Not used at gen time.
    ap.add_argument("--covis_pos_hi", type=float, default=0.50, help="retrieval positive threshold on covis_curve")
    ap.add_argument("--covis_pos_lo", type=float, default=0.10, help="retrieval negative threshold on covis_curve")
    ap.add_argument("--head_max_deg", type=float, default=45.0,
                    help="revisit: max |goal yaw - matched frame yaw| (relocalizability envelope)")
    ap.add_argument("--novel_covis", type=float, default=0.10, help="novel B: max covisibility with history must be <")
    ap.add_argument("--min_dist_AB", type=float, default=3.0, help="novel B: min geodesic A->B (3-leg)")
    ap.add_argument("--max_dist_AB", type=float, default=9.0,
                    help="novel B: max geodesic A->B; v4 requires it to equal --dA_max")
    ap.add_argument(
        "--role_distance_match_tol_m",
        type=float,
        default=MAX_ROLE_DISTANCE_MATCH_TOLERANCE_M,
        help=("3-leg only: maximum within-episode |geo(start,A)-geo(A,B)|; "
              "the v4 formal contract permits at most 0.50 m"),
    )
    ap.add_argument("--goal_jitter_pos", type=float, default=1.50,
                    help="revisit: uniform-disk RADIUS (m) around the anchor frame X; covis+heading gates cap realized offset")
    # LingBot goal_append recomputes a FIXED W-frame window [m-W+1..m] around the match and injects it
    # at RoPE pos total_frames=m-W+1. For that window to have W real frames AND be disjoint from the
    # scale block [0,num_scale) (else RoPE position collision), need m-W+1 >= num_scale => m >= num_scale+W-1.
    ap.add_argument("--window", type=int, default=32, help="LingBot local sliding window W (must match precompute)")
    ap.add_argument("--num_scale", type=int, default=8, help="LingBot scale frames (full dense, injected)")
    ap.add_argument("--anchor_margin", type=int, default=None,
                    help="revisit anchor X >= this; default num_scale+window-1 (goal_append window disjoint from scale)")
    # recent (in-view) vs long-term (implicit-memory) revisit balance. long_term forces the matched frame
    # to sit OUTSIDE the current window (recall gap >= min_recall_gap), the memory-testing case.
    ap.add_argument("--long_term_frac", type=float, default=0.7,
                    help="fraction of revisits forced long-term (X outside current window); rest free (natural in-view mix)")
    ap.add_argument("--min_recall_gap", type=int, default=None,
                    help="long-term revisit: min (current - matched) frame gap; default = window")
    ap.add_argument("--goal_tries", type=int, default=40, help="rejection-sampling tries per goal")
    ap.add_argument("--covis_stride", type=int, default=4, help="history frame stride for covisibility")
    ap.add_argument("--covis_tol", type=float, default=0.30, help="depth-consistency tol for covisibility (m)")
    # multi-floor handling (MP3D)
    ap.add_argument("--floor_tol", type=float, default=0.80, help="max |y-floor_y| to count as same floor (m)")
    ap.add_argument("--cam_h", type=float, default=0.5, help="camera height above floor navmesh (m)")
    ap.add_argument("--max_turn_deg", type=float, default=4.5, help="max heading change per frame")
    # pure-pursuit controller (measured from InternData-N1)
    ap.add_argument("--v", type=float, default=0.0376, help="speed m/frame")
    ap.add_argument("--lookahead", type=float, default=0.7)
    ap.add_argument("--r_min", type=float, default=0.40, help="min turning radius (m)")
    ap.add_argument("--v_min_frac", type=float, default=0.48,
                    help="speed floor as frac of cruise during sharp turns (N1-measured ~0.48)")
    ap.add_argument("--max_attempts", type=int, default=60); ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max_frame_turn", type=float, default=15.0,
                    help="safety gate: reject+retry the episode if any frame's heading changes more than this (deg)")
    # navmesh inflation + ElasticBands clearance smoothing (from PythonRobotics / iPlanner)
    ap.add_argument("--agent_radius", type=float, default=0.30, help="navmesh inflation = robot radius (m)")
    ap.add_argument("--esdf_res", type=float, default=0.05, help="scene ESDF grid resolution (m)")
    ap.add_argument("--eb_iters", type=int, default=60)
    ap.add_argument("--eb_kc", type=float, default=0.5, help="ElasticBands contraction (smoothness)")
    ap.add_argument("--eb_kr", type=float, default=0.8, help="ElasticBands repulsion (clearance)")
    ap.add_argument("--eb_rho0", type=float, default=0.6, help="clearance influence radius (m)")
    ap.add_argument("--eb_step", type=float, default=0.04)
    args = ap.parse_args()
    if args.goal_a_source_only and (
            args.n_legs != 2 or args.n2 is not None or args.n3 is not None):
        ap.error(
            "--goal_a_source_only requires single-output --n_legs 2 mode")
    if args.episode_attempt_multiplier < 1:
        ap.error("--episode_attempt_multiplier must be >= 1")
    generates_three_leg = (
        int(args.n_legs) == 3 or args.n3 is not None)
    if args.double_revisit_min_anchor_gap < 1:
        ap.error("--double_revisit_min_anchor_gap must be >= 1")
    if generates_three_leg and args.three_leg_roles == "novel_revisit":
        if (abs(float(args.dA_min) - float(args.min_dist_AB)) > 1e-9
                or abs(float(args.dA_max) - float(args.max_dist_AB)) > 1e-9):
            ap.error(
                "3-leg role-paired generation requires identical A/B "
                "distance bands")
        if not (0.0 < float(args.role_distance_match_tol_m)
                <= MAX_ROLE_DISTANCE_MATCH_TOLERANCE_M):
            ap.error(
                "--role_distance_match_tol_m must be in (0, 0.50] for "
                "the v4 formal contract")
    if args.anchor_margin is None:
        args.anchor_margin = args.num_scale + args.window - 1     # goal_append window disjoint from scale block
    if args.min_recall_gap is None:
        args.min_recall_gap = args.window                        # "outside the current window"
    print(f"[main] window={args.window} num_scale={args.num_scale} -> anchor_margin={args.anchor_margin}; "
          f"long_term_frac={args.long_term_frac} min_recall_gap={args.min_recall_gap}")

    sim = make_sim(args.scene, args.navmesh, agent_radius=args.agent_radius)
    assert sim.pathfinder.is_loaded, "navmesh not loaded"
    # Habitat's navigable-point sampler owns RNG state separately from NumPy.
    # Seed both sources so a confirmation manifest can be regenerated from
    # its recorded per-scene seed.
    sim.seed(args.seed)
    sim.pathfinder.seed(args.seed)
    lo, hi = sim.pathfinder.get_bounds()
    print(
        f"[main] scene navmesh height span={hi[1]-lo[1]:.1f} m; "
        "the scene may contain multiple floors, but every episode is "
        "strictly single-floor (all three geodesics are floor-gated; "
        "ESDF is cached per floor).")
    esdf_cache = {}                                          # floor bucket -> ESDF (built on demand)
    rng = np.random.default_rng(args.seed)
    scene_id = os.path.splitext(os.path.basename(args.scene))[0]

    completed = []
    if args.n2 is not None or args.n3 is not None:
        # dual-leg: one loaded scene -> both leg types, sharing sim + ESDF cache (built once per floor).
        # nests as <out>/mp3d_{2,3}leg/<scene_id>/episode_XXXX ; the SLURM array does the scene loop.
        plan = []
        if args.n2 is not None:
            plan.append((2, args.n2, os.path.join(args.out, "mp3d_2leg", scene_id)))
        if args.n3 is not None:
            plan.append((3, args.n3, os.path.join(args.out, "mp3d_3leg", scene_id)))
        for n_legs, n, out_dir in plan:
            completed.append((
                run_legs(sim, args, rng, esdf_cache, n_legs, n, out_dir),
                n, out_dir))
    else:
        os.makedirs(args.out, exist_ok=True)
        completed.append((
            run_legs(sim, args, rng, esdf_cache, args.n_legs, args.n, args.out),
            args.n, args.out))
    sim.close()
    incomplete = [
        (made, requested, out_dir)
        for made, requested, out_dir in completed if made < requested]
    if incomplete and not args.allow_incomplete:
        details = "; ".join(
            f"{made}/{requested} at {out_dir}"
            for made, requested, out_dir in incomplete)
        raise SystemExit(
            "generation incomplete after the configured attempt budget: "
            + details)


if __name__ == "__main__":
    main()
