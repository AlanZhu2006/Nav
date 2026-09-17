#!/usr/bin/env python3
"""Summarize the complete A/B audit; all contrasts are observational."""

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def mean(xs):
    xs = list(xs)
    return statistics.mean(xs) if xs else None


def sr(rows):
    return {'n': len(rows), 'successes': sum(r['success'] for r in rows),
            'sr': mean(r['success'] for r in rows)}


def direction(row):
    if row['group'] == 'table2/A':
        degrees = row['source_metadata']['start_heading_offset_deg']
    else:
        degrees = row['query']['initial_path_direction_relative_to_a_end_deg']
    return float(degrees)


def direction_band(row):
    angle = abs(direction(row))
    return 'front' if angle <= 60 else 'side' if angle <= 120 else 'rear'


def average_by(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row['success'])
    return {'groups': len(groups), 'macro_sr': mean(mean(x) for x in groups.values())}


def summarize(rows):
    failures = [r for r in rows if not r['success']]
    result = {**sr(rows), 'scenes': len({r['scene'] for r in rows}),
              'parent_histories': len({(r['scene'], r['parent_episode']) for r in rows}),
              'geodesic_mean_m': mean(r['geodesic_m'] for r in rows),
              'actual_path_mean_m': mean(r['path_m'] for r in rows),
              'corrected_spl': mean(r['spl'] for r in rows),
              'route_ratio_mean': mean(r['geodesic_to_euclidean_ratio'] for r in rows),
              'abs_initial_path_bearing_mean_deg': mean(abs(direction(r)) for r in rows),
              'first_low_critic_n': sum(r['first_plan']['navdp_critic_max'] < -0.5 for r in rows),
              'termination_reasons': dict(Counter(r['termination_reason'] for r in rows)),
              'failures_never_within_2m': sum(r['min_goal_distance_all_poses_m'] >= 2 for r in failures),
              'failures_within_1_5m': sum(r['min_goal_distance_all_poses_m'] < 1.5 for r in failures),
              'failures': len(failures),
              'depth_states': dict(sum((Counter(r['depth_states']) for r in rows), Counter())),
              'depth_shapes': dict(sum((Counter(r['depth_shapes']) for r in rows), Counter())),
              'scene_macro': average_by(rows, lambda r: r['scene']),
              'history_macro': average_by(rows, lambda r: (r['scene'], r['parent_episode'])),
              'takeover_count': sum(r['takeover_plans'] for r in rows)}
    result['directions'] = {k: sr([r for r in rows if direction_band(r) == k])
                            for k in ('front', 'side', 'rear')}
    result['distance_bins'] = {f'[{lo},{hi})': sr([r for r in rows if lo <= r['geodesic_m'] < hi])
                               for lo, hi in ((0, 3), (3, 5), (5, 7), (7, 10))}
    result['route_ratio_bins'] = {f'[{lo},{hi})': sr([r for r in rows if lo <= r['geodesic_to_euclidean_ratio'] < hi])
                                  for lo, hi in ((0, 1.05), (1.05, 1.25), (1.25, 1000))}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audit_root', type=Path)
    args = parser.parse_args()
    payload = json.loads((args.audit_root / 'raw_audit_v2.json').read_text())
    expert = json.loads((args.audit_root / 'expert_target_audit.json').read_text())
    rows = payload['records']
    a = [r for r in rows if r['group'] == 'table2/A']
    b = [r for r in rows if r['group'] == 'table2/B']
    assert len(a) == 196 and len(b) == 183
    assert sum(r['success'] for r in a) == 131 and sum(r['success'] for r in b) == 54
    assert all(r['native_route'] == 'native_sidecar' and r['takeover_plans'] == 0 for r in rows)
    assert len(expert) == len(a)
    for r in a:
        error = (r['start_pose']['yaw'] - r['source_metadata']['start_yaw_habitat'] + math.pi) % (2*math.pi) - math.pi
        assert abs(error) < 1e-6
        assert r['source_metadata']['n_legs'] == 2
        assert r['source_metadata']['initial_yaw_mode'] == 'path_aligned'
    scale_differences = [abs(r['scale_hats'][0] / r['parent_scale_hats'][0] - 1) for r in b]
    b_scenes = {r['scene'] for r in b}
    result = {'schema': 'table2_novel_gap_summary_v1_20260910',
              'scope': 'post-hoc full-population attribution, not a causal ablation or new SR',
              'A': summarize(a), 'B': summarize(b),
              'A_in_B_scenes': sr([r for r in a if r['scene'] in b_scenes]),
              'B_populations': {p: sr([r for r in b if r['population'] == p])
                               for p in sorted({r['population'] for r in b})},
              'B_replay_scale_relative_change': {'mean': mean(scale_differences),
                                                  'max': max(scale_differences)},
              'A_expert_goal_photo': {
                  'n': len(expert),
                  'mean_camera_to_scored_goal_offset_m': mean(r['goal_photo_position_offset_m'] for r in expert),
                  'max_camera_to_scored_goal_offset_m': max(r['goal_photo_position_offset_m'] for r in expert),
                  'mean_abs_yaw_vs_last10_expert_displacement_deg': mean(abs(r['expert_goal_heading_vs_last10_displacement_deg']) for r in expert),
                  'max_abs_yaw_vs_last10_expert_displacement_deg': max(abs(r['expert_goal_heading_vs_last10_displacement_deg']) for r in expert)}}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
