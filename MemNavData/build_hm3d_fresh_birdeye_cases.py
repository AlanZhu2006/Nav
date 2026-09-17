#!/usr/bin/env python3
"""Bird's-eye HM3D case figure: native NavDP failure versus CEC on the fresh Full-Mono run.

Post-hoc visualization of already-sealed records from
``hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6``
(the paper's 17/56 -> 32/56 population). It creates no new evaluation row.

Inputs (all local mirrors of the sealed run):
  --pull        evaluation_natural_direction/<hist>/<arm>/*_plans.json + metric.csv, online_a/<hist>/online_a_trace.json
  --bench       benchmarks/natural_direction/<scene>/<episode>/{role_pairs.json,pair_00/<role>/goal.jpg}
  --topdown     topdown_maps.npz written by export_hm3d_fresh_topdown_maps.py (navmesh top-down view per scene/floor)

Stage 1 (habitat interpreter):
  /home/asus/miniconda3/envs/habitat/bin/python MemNavData/export_hm3d_fresh_topdown_maps.py --navmesh-root ... --out topdown_maps.npz
Stage 2 (memnav interpreter):
  /home/asus/miniconda3/envs/memnav/bin/python MemNavData/build_hm3d_fresh_birdeye_cases.py --out <dir>
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from PIL import Image

DIAG = '/data/diagnostics/nav-graph-blind/hm3d_fresh_fullmono_mixed_role_20260820'
PULL = f'{DIAG}/pulled_20260828'
BENCH = f'{DIAG}/birdeye_cases_20260910/benchmarks_natural_direction'
TOPDOWN = f'{DIAG}/birdeye_cases_20260910/topdown_maps.npz'
ARMS = ('mono_native', 'mono_raw_fixed', 'mono_cec')
ROLES = ('novel', 'revisit')
_maps = None
def maps():
    global _maps
    if _maps is None:
        z = np.load(TOPDOWN)
        _maps = {}
        for k in z.files:
            if k.endswith('__meta'):
                continue
            m = z[k + '__meta']
            _maps[k] = dict(topdown=z[k], lower=m[:3], upper=m[3:6], mpp=float(m[6]), height=float(m[7]))
    return _maps

def histories():
    return sorted(os.listdir(os.path.join(PULL, 'evaluation_natural_direction')))

def load_case(hist):
    scene = hist.split('_')[1]
    ep = '_'.join(hist.split('_')[2:])
    rp = json.load(open(f"{BENCH}/{scene}/{ep}/role_pairs.json"))
    a = json.load(open(f'{PULL}/online_a/{hist}/online_a_trace.json'))
    a_xy = np.asarray([[p['x'], p['z']] for p in a['poses']])
    y = rp['online_a_endpoint']['floor_position'][1]
    case = dict(hist=hist, scene=scene, ep=ep, a_xy=a_xy, a_end=np.asarray(rp['online_a_endpoint']['floor_position']),
                a_end_yaw=rp['online_a_endpoint']['yaw_rad'], map_key=f'{scene}_{y:.3f}', roles={})
    queries = {q['analysis_role']: q for q in rp['pairs'][0]['queries']}
    for role in ROLES:
        q = queries[role]
        r = dict(goal=np.asarray(q['floor_position']), goal_yaw=q['yaw_rad'], geodesic=q['geodesic_from_a_end_m'],
                 covis=q.get('max_online_a_covis'), src_frame=q.get('source_online_frame'), arms={})
        for arm in ARMS:
            pj = glob.glob(f'{PULL}/evaluation_natural_direction/{hist}/{arm}/{ep}_pair_00_{role}_plans.json')[0]
            p = json.load(open(pj))
            met = {m['analysis_role']: m for m in csv.DictReader(open(f'{PULL}/evaluation_natural_direction/{hist}/{arm}/metric.csv'))}[role]
            tr = np.asarray([[t['x'], t['z']] for t in p['rollout_traces']['query']])
            r['arms'][arm] = dict(xy=tr, reached=int(met['reached']), steps=int(met['steps']), final=float(met['final_goal_dist_m']),
                                  takeover=int(met['adapter_takeover_plans']), term=met['termination_reason'],
                                  path_len=float(met['path_len_m']), query_leg=p['query_leg'])
        case['roles'][role] = r
    return case
COL = {'mono_native': '#5F6670', 'mono_raw_fixed': '#D9822B', 'mono_cec': '#2F6DB3'}
NAV_FILL, WALL = '#FFFFFF', '#D6D8DC'
HIST = '#3F8F63'

CASES = [  # (history, role, panel title)
    ('009_FnSn2KSrALj_episode_0000', 'revisit', 'NavDP drifts away along a corridor'),
    ('014_QHhQZWdMpGJ_episode_0002', 'revisit', 'NavDP loops in place'),
    ('021_auFeVz9Go4m_episode_0000', 'revisit', 'NavDP explores the wrong rooms'),
    ('023_LEFTm3JecaC_episode_0001', 'revisit', 'NavDP gets stuck at clutter'),
]
NOVEL_CASE = ('019_66seV3BWPoX_episode_0002', 'novel', 'Unsupported Novel: raw memory interferes, GEM abstains')

def fwd(yaw):
    return np.array([-np.sin(yaw), -np.cos(yaw)])

def panel(ax, case, role, title, arms=('mono_native', 'mono_cec'), margin=0.7, inset=True, label=None):
    m = maps()[case['map_key']]; td = m['topdown']; lb = m['lower']
    r = case['roles'][role]
    goal = r['goal'][[0, 2]]
    pts = np.vstack([case['a_xy']] + [r['arms'][a]['xy'] for a in arms] + [goal[None]])
    x0, x1 = pts[:, 0].min() - margin, pts[:, 0].max() + margin
    z0, z1 = pts[:, 1].min() - margin, pts[:, 1].max() + margin
    # keep panels from becoming extreme slivers
    side = max(x1 - x0, z1 - z0)
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    x0, x1, z0, z1 = cx - side / 2, cx + side / 2, cz - side / 2, cz + side / 2
    rgb = np.empty((*td.shape, 3)); rgb[:] = matplotlib.colors.to_rgb(WALL)
    rgb[td.astype(bool)] = matplotlib.colors.to_rgb(NAV_FILL)
    ax.set_facecolor(WALL)
    ext = [lb[0], lb[0] + td.shape[1] * m['mpp'], lb[2] + td.shape[0] * m['mpp'], lb[2]]
    ax.imshow(rgb, origin='upper', extent=ext, interpolation='nearest', zorder=0)
    # history (Goal-A) path
    A = case['a_xy']
    ax.plot(A[:, 0], A[:, 1], color=HIST, lw=1.1, ls=(0, (3, 2)), alpha=0.85, zorder=2)
    ax.scatter(*A[0], marker='s', s=22, color=HIST, zorder=3)
    # query trajectories
    for a in arms:
        t = r['arms'][a]['xy']; ok = r['arms'][a]['reached']
        # native is drawn as the widest, lowest line so an exact CEC overlay (Novel fallback) stays visible as a grey halo
        lw = {'mono_native': 2.3, 'mono_raw_fixed': 1.6, 'mono_cec': 2.0}[a]
        z = {'mono_native': 3, 'mono_raw_fixed': 3.5, 'mono_cec': 4}[a]
        ax.plot(t[:, 0], t[:, 1], color=COL[a], lw=lw, alpha=0.95, zorder=z, solid_capstyle='round')
        ax.scatter(*t[-1], marker='o' if ok else 'x', s=26 if ok else 40, color=COL[a], zorder=6, linewidths=1.6)
    # certificate acceptances along the CEC path and the authenticated history frame
    if 'mono_cec' in arms:
        ql = r['arms']['mono_cec']['query_leg']; t = r['arms']['mono_cec']['xy']
        acc = [e for e in ql if e.get('certified_relocalization_accepted')]
        if acc:
            anchors = sorted({e.get('visual_anchor') for e in acc if e.get('visual_anchor') is not None})
            for k in anchors:
                if k < len(A):
                    ax.scatter(*A[k], marker='o', s=70, facecolor='none', edgecolor=HIST, linewidths=1.4, zorder=5)
    # start with heading, goal
    s = case['a_end'][[0, 2]]; d = fwd(case['a_end_yaw']) * 0.45
    ax.add_patch(FancyArrowPatch(s, s + d, arrowstyle='-|>', mutation_scale=9, color='#1F2328', lw=1.2, zorder=7))
    ax.scatter(*s, s=40, color='#1F2328', zorder=7)
    ax.scatter(*goal, marker='*', s=190, color='#E0B000', edgecolor='#1F2328', linewidth=0.6, zorder=8)
    ax.set_xlim(x0, x1); ax.set_ylim(z1, z0); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#C9CDD2'); sp.set_linewidth(0.7)
    corner = None
    # outcome text
    nm = {'mono_native': 'NavDP', 'mono_raw_fixed': 'Raw memory', 'mono_cec': 'GEM'}
    lines = []
    for a in arms:
        q = r['arms'][a]
        lines.append(f"{nm[a]}: {'reached' if q['reached'] else 'failed'}, {q['steps']} steps")
    if label:
        lines[0] = rf'$\mathbf{{({label})}}$ ' + lines[0]
        lines[1:] = ['       ' + l for l in lines[1:]]  # align the method lines under each other
    ax.set_xlabel('\n'.join(lines), fontsize=10, labelpad=5, ha='center', ma='left')
    if inset:
        im = Image.open(f"{BENCH}/{case['scene']}/{case['ep']}/pair_00/{role}/goal.jpg").convert('RGB')
        u = (pts[:, 0] - x0) / (x1 - x0); v = (z1 - pts[:, 1]) / (z1 - z0)  # axes fractions (v up)
        corners = {'tl': (0.015, 0.695), 'tr': (0.655, 0.695), 'bl': (0.015, 0.03), 'br': (0.655, 0.03)}
        def crowd(cx, cy):
            return np.sum((u > cx - 0.02) & (u < cx + 0.35) & (v > cy - 0.02) & (v < cy + 0.33))
        corner = min(corners, key=lambda k: crowd(*corners[k]) + (0.5 if k.startswith('b') else 0))  # prefer top corners on ties
        cx, cy = corners[corner]
        ia = ax.inset_axes([cx, cy, 0.33, 0.275]); ia.imshow(im); ia.set_xticks([]); ia.set_yticks([])
        for sp in ia.spines.values(): sp.set_color('#E0B000'); sp.set_linewidth(1.2)
        ia.set_title('goal image', fontsize=10, pad=2, color='#7A5F00')
    # scale bar: bottom-right unless the inset sits there
    bz = z1 - 0.06 * (z1 - z0)
    bx = x0 + 0.05 * (x1 - x0) if corner == 'br' else x1 - 0.05 * (x1 - x0) - 1.0
    ax.plot([bx, bx + 1], [bz, bz], color='#1F2328', lw=1.6, zorder=9)
    ax.text(bx + 0.5, bz - 0.03 * (z1 - z0), '1 m', ha='center', va='bottom', fontsize=9)

def build(out_base, cases, arms, legend_arms, ncol=None, figsize=None):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    n = len(cases); ncol = ncol or n
    # size the figure so that each square map fills its slot: slot width = 3.7 in, height chosen from the margins below
    top = 0.99
    bottom = (0.20 if n > 2 else 0.22) + 0.03 * (len(arms) - 2)
    wspace = 0.03
    panel_w = 3.7
    fig_w = panel_w * ncol / (0.995 - 0.005) * (1 + wspace * (ncol - 1) / ncol)
    fig, axes = plt.subplots(1, ncol, figsize=figsize or (fig_w, panel_w / (top - bottom)), facecolor='white')
    axes = np.atleast_1d(axes)
    meta = []
    for i, (ax, (h, role, title)) in enumerate(zip(axes, cases)):
        c = load_case(h)
        panel(ax, c, role, title, arms=arms, label='abcdef'[i])
        meta.append(dict(panel='abcdef'[i], history=h, scene=c['scene'], episode=c['ep'], role=role, title=title,
                         **{a: {k: v for k, v in c['roles'][role]['arms'][a].items() if k not in ('xy', 'query_leg')} for a in ARMS}))
    nm = {'mono_native': 'NavDP: goal-2 traj.', 'mono_raw_fixed': 'Raw memory: goal-2 traj.', 'mono_cec': 'GEM (ours): goal-2 traj.'}
    arm_handles = [Line2D([0], [0], color=COL[a], lw=2, label=nm[a]) for a in legend_arms]
    h_goal1_traj = Line2D([0], [0], color=HIST, lw=1.2, ls=(0, (3, 2)), label='goal-1 traj. (history)')
    h_ep_start = Line2D([0], [0], marker='s', color=HIST, lw=0, markersize=6, label='episode start')
    h_goal1 = Line2D([0], [0], marker='o', color='#1F2328', lw=0, markersize=5, label='goal 1 / goal-2 start')
    h_goal2 = Line2D([0], [0], marker='*', color='#E0B000', markeredgecolor='#1F2328', lw=0, markersize=11, label='goal 2')
    h_verified = Line2D([0], [0], marker='o', markerfacecolor='none', markeredgecolor=HIST, color='none', markersize=9, label='GEM-verified history frame')
    h_obstacle = Line2D([0], [0], marker='s', color=WALL, lw=0, markersize=9, label='obstacle')
    h_reached = Line2D([0], [0], marker='o', color=COL['mono_cec'], lw=0, markersize=5, label='end (reached)')
    h_failed = Line2D([0], [0], marker='x', color=COL['mono_native'], lw=0, markersize=6, markeredgewidth=1.6, label='end (failed)')
    # legend fills column-major: three-row layout (<=2 panels) reads goal-1 traj / NavDP / GEM, then the three positions,
    # then verified frame + obstacle, then the two end markers; the two-row layout pairs them the same way.
    if n <= 2:
        handles = [h_goal1_traj, *arm_handles, h_ep_start, h_goal1, h_goal2, h_verified, h_obstacle, h_reached, h_failed]
    else:
        handles = [*arm_handles, h_goal1_traj, h_ep_start, h_goal1, h_goal2, h_verified, h_obstacle, h_reached, h_failed]
    fig.legend(handles=handles, loc='lower center', ncol=(len(handles) + 1) // 2 if len(cases) > 2 else 4, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.01 - 0.02 * (len(arms) - 2)),
               handlelength=2.0, columnspacing=1.2, handletextpad=0.6)
    fig.subplots_adjust(left=0.005, right=0.995, top=top, bottom=bottom, wspace=wspace)
    fig.savefig(out_base + '.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_base + '.pdf', bbox_inches='tight', facecolor='white')
    plt.close(fig)
    json.dump(dict(source_run='hm3d_fresh_fullmono_mixed_role_20260820/formal_20260820T143609Z_e6dd44c6', local_pull=PULL,
                   map='habitat-sim PathFinder.get_topdown_view on the sealed HM3D navmesh at the Goal-A floor height, 2.5 cm/px',
                   panels=meta), open(out_base + '.json', 'w'), indent=2)
    print('wrote', out_base)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--pull', default=PULL); ap.add_argument('--bench', default=BENCH); ap.add_argument('--topdown', default=TOPDOWN)
    ap.add_argument('--cases', default='', help='comma-separated 0-based indices into CASES for a reduced figure (e.g. 1,2)')
    ap.add_argument('--name', default='hm3d_birdeye_cases', help='output basename for the reduced figure')
    a = ap.parse_args()
    PULL, BENCH, TOPDOWN = a.pull, a.bench, a.topdown
    os.makedirs(a.out, exist_ok=True)
    if a.cases:
        sel = [CASES[int(i)] for i in a.cases.split(',')]
        build(os.path.join(a.out, a.name), sel, ('mono_native', 'mono_cec'), ('mono_native', 'mono_cec'))
        raise SystemExit(0)
    build(os.path.join(a.out, 'hm3d_birdeye_cases'), CASES, ('mono_native', 'mono_cec'), ('mono_native', 'mono_cec'))
    build(os.path.join(a.out, 'hm3d_birdeye_cases_with_raw'), CASES, ARMS, ARMS)
    build(os.path.join(a.out, 'hm3d_birdeye_novel_case'), [NOVEL_CASE], ARMS, ARMS, figsize=(4.2, 4.2))
