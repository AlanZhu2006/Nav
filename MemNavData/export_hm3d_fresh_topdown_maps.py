"""Stage 1 (HAB_PY): export navmesh top-down occupancy for every fresh scene/height."""
import glob, json, os, sys
import numpy as np
import habitat_sim

S = None
NAV = os.environ.get('NAVMESH_ROOT', '/data/diagnostics/nav-graph-blind/hm3d_fresh_fullmono_mixed_role_20260820/navmesh_fresh_fullmono54')
PULL = '/data/diagnostics/nav-graph-blind/hm3d_fresh_fullmono_mixed_role_20260820/pulled_20260828'
BENCH = '/data/diagnostics/nav-graph-blind/hm3d_fresh_fullmono_mixed_role_20260820/birdeye_cases_20260910/benchmarks_natural_direction'
MPP = 0.025
out = {}
for hist in sorted(os.listdir(os.path.join(PULL, 'evaluation_natural_direction'))):
    scene = hist.split('_')[1]
    ep = '_'.join(hist.split('_')[2:])
    rp = json.load(open(glob.glob(f'{BENCH}/{scene}/{ep}/role_pairs.json')[0]))
    y = rp['online_a_endpoint']['floor_position'][1]
    key = f'{scene}_{y:.3f}'
    if key in out:
        continue
    nm = glob.glob(f'{NAV}/*-{scene}/{scene}.basis.navmesh')[0]
    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(nm)
    assert pf.is_loaded, nm
    lb, ub = pf.get_bounds()
    td = pf.get_topdown_view(MPP, y)  # rows: z, cols: x
    out[key] = dict(topdown=td.astype(np.uint8), lower=np.asarray(lb), upper=np.asarray(ub), mpp=MPP, height=y)
    print(hist, key, td.shape, td.mean().round(3), lb, ub)
np.savez_compressed(os.environ.get('TOPDOWN_OUT', '/data/diagnostics/nav-graph-blind/hm3d_fresh_fullmono_mixed_role_20260820/birdeye_cases_20260910/topdown_maps.npz'), **{k: v['topdown'] for k, v in out.items()},
                    **{k + '__meta': np.asarray([*v['lower'], *v['upper'], v['mpp'], v['height']]) for k, v in out.items()})
print('saved', len(out))
