#!/usr/bin/env python3
"""Read-only export of the frozen Table-II A/B population for attribution.

No policy calls, simulator execution, threshold changes, or remote writes.
Run on the evidence host and redirect stdout into a new local JSON artifact.
"""

import collections
import csv
import getpass
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys


ROOT = Path('/scratch/yz11502/Research/Nav-axis-uturn-results')
PARENT = ROOT / ('hm3d_fresh_fullmono_mixed_role_20260820/'
                 'formal_20260820T143609Z_e6dd44c6')
CORRECTION = ROOT / 'paper_executed_spl_correction_20260906/correction.json'
EXPECTED_CORRECTION = '41bd2c68c498863d5cf92452b2a7564d517698c38d26918ce7820b39cd49a942'
EXPECTED_PARENT = 'a96a0b96fab7b7b47709b36cb8eeb9410b42b09f095f87ef01304a68de716dd5'


def read_bound(path, expected=None):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert sha == expected, (str(path), sha, expected)
    return json.loads(raw), sha


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


def xyz(p):
    return [p[k] for k in ('x', 'y', 'z')]


def distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def stat(values):
    finite = [float(x) for x in values
              if x is not None and math.isfinite(float(x))]
    if not finite:
        return {'n': 0}
    return {'n': len(finite), 'mean': statistics.mean(finite),
            'median': statistics.median(finite), 'min': min(finite),
            'max': max(finite)}


def main():
    assert getpass.getuser() == 'yz11502', 'Use the documented shared SSH account'
    correction, correction_sha = read_bound(CORRECTION, EXPECTED_CORRECTION)
    parent, parent_sha = read_bound(PARENT / 'sealed_inputs/parent_manifest.json', EXPECTED_PARENT)
    parent_items = {(s, x['episode']): x for s, v in parent['episodes'].items() for x in v}
    manifests = {}
    records = []
    for i, original in enumerate(x for x in correction['rows'] if x['group'] in ('table2/A', 'table2/B')):
        trace_path = Path(original['trace'])
        trace, _ = read_bound(trace_path, original['trace_sha256'])
        metric_raw = Path(original['metric']).read_bytes()
        assert hashlib.sha256(metric_raw).hexdigest() == original['metric_sha256']
        metrics = list(csv.DictReader(io.StringIO(metric_raw.decode())))
        assert len(metrics) == 1
        metric = metrics[0]
        assert int(bool(trace['reached'])) == original['success']
        scene, episode = trace['source_scene'], trace['episode']
        row = dict(original)
        row.update(scene=scene, metric_values=metric,
                   start_pose=trace['poses'][0], end_position=trace['end_position'],
                   end_yaw=trace['end_yaw'], steps=trace['steps'],
                   blocked_steps=trace['blocked_step_count'],
                   final_goal_dist_m=trace['final_goal_dist_m'],
                   termination_reason=trace['termination_reason'],
                   native_route=trace['source_hybrid_route'])
        assert row['native_route'] == 'native_sidecar'
        if original['group'].endswith('/A'):
            item = parent_items[(scene, episode)]
            metadata, sha = read_bound(item['files']['metadata']['path'], item['files']['metadata']['sha256'])
            row.update(source_metadata=metadata, source_metadata_sha256=sha,
                       source_item=item, parent_episode=episode)
            # In the sealed evaluator, Habitat = M_W.T @ data-Z-up.
            # The stored A metadata is the scored position, not the last
            # expert camera position; do not silently reanchor this target.
            data_goal = metadata['A']
            goal_position = [data_goal[0], data_goal[2], -data_goal[1]]
        else:
            source_root = trace_path.parents[2]
            if source_root not in manifests:
                obj, sha = read_bound(source_root / 'ab_population/role_pairs/manifest.json')
                manifests[source_root] = (obj, sha, {(x['scene'], x['episode']): x for x in obj['episodes']})
            obj, sha, lookup = manifests[source_root]
            item = lookup[(scene, episode)]
            query = next(q for pair in item['pairs'] for q in pair['queries'] if q['analysis_role'] == 'novel')
            assert query['goal_rgb_sha256'] == trace['goal_sha256']
            assert distance(item['online_a_endpoint']['floor_position'], xyz(trace['poses'][0])) < 1e-5
            assert abs(wrap(item['online_a_endpoint']['yaw_rad'] - trace['poses'][0]['yaw'])) < 1e-6
            assert abs(query['geodesic_from_a_end_m'] - original['geodesic_m']) < 0.05
            row.update(query={k: v for k, v in query.items() if k != 'covis_curve'},
                       construction=item.get('lifelong_construction'),
                       manifest_sha256=sha, population=str(source_root),
                       parent_episode=Path(item['online_a_episode']).name,
                       parent_trace_sha256=item['online_a_trace_sha256'])
            goal_position = query['floor_position']
        poses = trace['poses']
        positions = [xyz(p) for p in poses] + [trace['end_position']]
        lengths = [distance(a, b) for a, b in zip(positions, positions[1:])]
        yaws = [p['yaw'] for p in poses] + [trace['end_yaw']]
        turns = [abs(wrap(b-a)) for a, b in zip(yaws, yaws[1:])]
        row.update(total_turn_degrees=math.degrees(sum(turns)),
                   zero_translation_steps_1mm=sum(x < 0.001 for x in lengths),
                   near_end_stationary_steps_1mm=sum(x < 0.001 for x in lengths[-100:]),
                   cumulative_actual_path_m=sum(lengths),
                   start_end_displacement_m=distance(positions[0], positions[-1]))
        goal_distances = [distance(p, goal_position) for p in positions]
        assert abs(goal_distances[-1] - trace['final_goal_dist_m']) < 1e-5
        assert bool(min(goal_distances) < 1.0) == bool(trace['reached'])
        delta_x = goal_position[0] - positions[0][0]
        delta_z = goal_position[2] - positions[0][2]
        direct_yaw = math.atan2(-delta_x, -delta_z)
        row.update(goal_position=goal_position,
                   initial_euclidean_m=goal_distances[0],
                   min_goal_distance_all_poses_m=min(goal_distances),
                   min_goal_distance_step=goal_distances.index(min(goal_distances)),
                   direct_initial_relative_bearing_deg=math.degrees(wrap(direct_yaw-yaws[0])),
                   goal_distance_after_40_steps_m=goal_distances[min(40, len(goal_distances)-1)],
                   goal_distance_after_100_steps_m=goal_distances[min(100, len(goal_distances)-1)],
                   geodesic_to_euclidean_ratio=original['geodesic_m']/max(1e-8, goal_distances[0]))
        plans = trace['plans']
        row['plan_count'] = len(plans)
        row['plan_steps'] = [p['step'] for p in plans]
        row['goal_distance_at_plans'] = [p.get('evaluation_gt_goal_distance_m') for p in plans]
        row['goal_distance_plan_stats'] = stat(row['goal_distance_at_plans'])
        row['critic'] = stat([p.get('navdp_critic_max') for p in plans])
        row['critic_below_minus05'] = sum(float(p['navdp_critic_max']) < -0.5 for p in plans if p.get('navdp_critic_max') is not None)
        row['critic_stop_evidence_count'] = sum(bool(p.get('navdp_stop_evidence')) for p in plans)
        row['candidate_heading_resultant'] = stat([p.get('candidate_heading_resultant') for p in plans])
        row['depth_states'] = dict(collections.Counter((p.get('monocular_depth_receipt') or {}).get('scale_state') for p in plans))
        row['depth_shapes'] = dict(collections.Counter(str((p.get('monocular_depth_receipt') or {}).get('depth_shape')) for p in plans))
        row['depth_median'] = stat([(p.get('monocular_depth_receipt') or {}).get('depth_nonzero_median_m') for p in plans])
        row['scale_hats'] = sorted(set(float(s['scale_hat']) for p in plans if (s := (p.get('monocular_depth_receipt') or {}).get('scale_receipt')) and s.get('scale_hat') is not None))
        row['first_plan'] = {k: v for k, v in plans[0].items() if v is not None}
        row['last_plan'] = {k: v for k, v in plans[-1].items() if v is not None}
        row['takeover_plans'] = sum(bool(p.get('router_active') or p.get('cec_takeover') or p.get('certified_relocalization_accepted')) for p in plans)
        assert row['takeover_plans'] == 0
        records.append(row)
        if (i+1) % 50 == 0:
            print(f'AUDITED {i+1}/379', file=sys.stderr, flush=True)
    assert len(records) == 379
    assert sum(x['success'] for x in records if x['group'] == 'table2/A') == 131
    assert sum(x['success'] for x in records if x['group'] == 'table2/B') == 54
    a_lookup = {(x['scene'], x['episode']): x for x in records if x['group'] == 'table2/A'}
    for row in records:
        if row['group'] == 'table2/B':
            a = a_lookup[row['scene'], row['parent_episode']]
            assert row['parent_trace_sha256'] == a['trace_sha256']
            row['parent_steps'] = a['steps']
            row['parent_geodesic_m'] = a['geodesic_m']
            row['parent_path_m'] = a['path_m']
            row['parent_scale_hats'] = a['scale_hats']
    result = {'schema': 'table2_novel_gap_raw_audit_v1_20260910',
              'scope': 'read-only retrospective, no new navigation',
              'correction_sha256': correction_sha, 'parent_manifest_sha256': parent_sha,
              'all_379_trace_and_metric_hashes_verified': True,
              'all_183_b_starts_match_parent_a_endpoint': True,
              'records': records}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
