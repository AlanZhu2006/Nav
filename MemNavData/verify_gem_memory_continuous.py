"""Audit persistent A-to-B-to-A execution, including failed/unfinished chains.

Per-leg views under audit_views are derived copies with action/plan indices
translated to zero. The immutable original observations, geometry and raster
artifacts remain the authority. The established independent motion, depth and
heading auditor verifies each derived view; no navigation is reexecuted.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]


def load(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if path.exists():
        assert path.read_text() == encoded, f'Existing derived record differs: {path}'
    else:
        path.write_text(encoded)


def save_lines(path, rows):
    encoded = ''.join(json.dumps(r, allow_nan=False) + '\n' for r in rows)
    path = Path(path)
    if path.exists():
        assert path.read_text() == encoded
    else:
        path.write_text(encoded)


def audit(root):
    from MemNavData.table1_repaired_eval import history
    from MemNavData.verify_repaired_fullmono_local import verify_rollout

    root = Path(root).resolve()
    manifest, summary = load(root / 'manifest.json'), load(root / 'summary.json')
    assert summary['completed'] and not (root / 'failure.json').exists()
    assert manifest['cell'] == summary['cell'] and manifest['mode'] == summary['mode']
    plan = load(manifest['plan'])
    assert sha(manifest['plan']) == summary['plan_sha256'] == manifest['plan_sha256']
    entry = plan['cells'][manifest['cell']['index']]
    assert entry['cell'] == manifest['cell'] and manifest['mode'] in plan['modes']
    for path, digest in manifest['sources_sha256'].items():
        saved = root / 'source_snapshot' / Path(path).relative_to(ROOT)
        assert sha(saved) == digest
    _, frozen, _ = history(manifest['cell'])
    run = root / 'evaluation'
    chain = load(run / 'chain_summary.json')
    assert chain == summary['chain'] and chain['completed'] and chain['initial_episode_denominator'] == 1
    assert chain['history_frames'] == entry['history_frames'] == len(frozen['trace']['poses'])
    assert chain['replay'] == load(run / 'prefix_replay.json')
    assert chain['replay']['all_rgb_hashes_verified'] and chain['replay']['diffusion_samples_during_replay'] == 0
    poses = lines(root / 'lingbot_pose_readout.jsonl')
    assert [r['frame_idx'] for r in poses] == list(range(len(poses)))
    assert all(r['motion_receipt_recorded'] is None for r in poses)
    assert [r['image_sha256'] for r in poses[:entry['history_frames']]] == [r['jpg_sha256'] for r in frozen['trace']['poses']]
    actions, plans = lines(run / 'executor_actions.jsonl'), lines(run / 'full_plan_outputs.jsonl')
    http, boundary = lines(run / 'navdp_http_receipts.jsonl'), lines(run / 'memory_http_boundary.jsonl')
    front = load(run / 'front_goal_receipt.json')
    assert sum(r['path'] == '/navigator_reset' for r in http) == 1
    assert sum(r['path'] == '/navigator_reset' for r in boundary) == 1
    assert not any(r['path'] in ('/navigator_reset_env', '/goal_session_replay') for r in http + boundary)
    for r in boundary:
        assert not set(r['sent_fields']) & {'analysis_role', 'role', 'floor_position', 'gt_pose',
            'executed_translation_m', 'executed_yaw_rad', 'executed_forward_m',
            'executed_left_m', 'executor_local_se2_source'}
        if r['frame_idx'] is not None and r['image_sha256'] is not None:
            assert r['image_sha256'] == poses[r['frame_idx']]['image_sha256']
    # Replay is permitted once before the first query, never between goals.
    assert not any(not r['query_active'] and r['next_action_index'] > 0
                   and r['path'] == '/memory_replay_step' for r in http)
    scale_ids = {r['monocular_depth_receipt']['scale_receipt_sha256'] for r in http
                 if r.get('monocular_depth_receipt') and r['monocular_depth_receipt']['frame_index'] >= 40}
    assert len(scale_ids) == 1
    previous_position = frozen['trace']['end_position']
    previous_yaw = frozen['trace']['end_yaw']
    previous_memory = entry['history_frames'] - 1
    action_offset, plan_offset, checked, failed = 0, 0, [], False
    for record in chain['leg_records']:
        index = record['stage']
        assert index == len(checked) and not failed
        original = run / f'leg_{index}'
        query, measured = load(original / 'query.json'), load(original / 'measurement.json')
        continuity = load(original / 'continuity.json')
        evidence = load(original / 'rollout_evidence.json')
        trace = load(original / 'actual_trace.json')
        assert record['query'] == query and record['measurement'] == measured and record['continuity'] == continuity
        assert all(query[k] == v for k, v in entry['goals'][index].items())
        assert sha(query['goal_rgb']) == query['goal_rgb_sha256'] == trace['goal_sha256']
        assert trace['poses'] == evidence['rollout_trace']
        assert trace['end_position'] == measured['end_position'] == continuity['end_position']
        assert continuity == chain['legs'][index]
        np.testing.assert_allclose(query['start_position'], previous_position, rtol=0, atol=1e-8)
        assert abs(query['start_yaw'] - previous_yaw) < 1e-8
        count = measured['steps']
        own_actions = deepcopy(actions[action_offset:action_offset + count])
        own_plans = deepcopy([p for p in plans if action_offset <= p['next_action_index'] < action_offset + count])
        own_http = deepcopy([r for r in http if r['query_active']
            and action_offset <= r['next_action_index'] < action_offset + count])
        assert count > 0, 'A goal required no observation; report this scope explicitly before changing the auditor'
        assert continuity['first_memory_index'] == previous_memory + 1
        assert continuity['last_memory_index'] == previous_memory + count
        for a in own_actions:
            a['action_index'] -= action_offset
            a['plan_index'] -= plan_offset
        for p in own_plans:
            p['plan_index'] -= plan_offset
            p['next_action_index'] -= action_offset
        for r in own_http:
            r['next_action_index'] -= action_offset
        events = deepcopy([e for e in front['events'] if action_offset <= e['action_index'] < action_offset + count])
        for event in events:
            event['action_index'] -= action_offset
        view = root / 'audit_views' / f'leg_{index}'
        save(view / 'rollout_evidence.json', evidence)
        save(view / 'lingbot_frame_poses.json', poses[:continuity['last_memory_index'] + 1])
        save(view / 'front_goal_receipt.json', dict(front, events=events,
            actions=sum(a['action_kind'] == 'heading' for a in own_actions),
            active=front['active'] if index == len(chain['leg_records']) - 1 else False))
        save_lines(view / 'executor_actions.jsonl', own_actions)
        save_lines(view / 'full_plan_outputs.jsonl', own_plans)
        save_lines(view / 'navdp_http_receipts.jsonl', own_http)
        save_lines(view / 'memory_http_boundary.jsonl', [r for r in boundary
            if action_offset <= r['next_action_index'] < action_offset + count])
        mesh = view / 'execution.navmesh'
        if not mesh.exists():
            mesh.write_bytes((run / 'execution.navmesh').read_bytes())
        assert sha(mesh) == sha(run / 'execution.navmesh')
        result, _, verified_plans, _ = verify_rollout(dict(measured, directory=str(view),
            role='historical_goal_cycle', arm='cec', scene=manifest['cell']['scene']))
        resources = load(original / 'memory_resources.json')
        memory = resources['memory']
        assert memory['frames'] == continuity['last_memory_index'] + 1
        if manifest['mode'] != 'legacy':
            assert memory['failure'] is None and memory['module'] == manifest['mode']
            assert memory['frames'] == memory['pose_frames'] == memory['descriptor_frames'] == memory['online_depth_frames']
            assert [r['frame'] for r in resources['write_timings']] == list(range(memory['frames']))
            assert max(r['live_committed'] for r in resources['write_timings']) <= (
                16 if manifest['mode'] == 'connected_reciprocal' else 320)
        query_plans = trace['plans']
        assert len(query_plans) == len(verified_plans)
        if query_plans:
            first = query_plans[0]
            assert first['goal_start_frame'] == continuity['first_memory_index']
            assert first['certified_relocalization_cached'] is False
            assert all(p['goal_start_frame'] == continuity['first_memory_index'] for p in query_plans)
        checked.append(dict(result, stage=index, goal=query['name'],
            geodesic_m=query['geodesic_m'], first_frame=continuity['first_memory_index'],
            last_frame=continuity['last_memory_index'], first_localization_cached=first['certified_relocalization_cached'],
            first_anchor=first['anchor'], first_accepted=first['certified_relocalization_accepted'],
            takeover_plans=sum(p['receipt']['revisit_adapter_takeover'] is True for p in own_plans)))
        action_offset += count
        plan_offset += len(own_plans)
        previous_memory = continuity['last_memory_index']
        previous_position, previous_yaw = trace['end_position'], trace['end_yaw']
        failed = not measured['reached']
    assert action_offset == len(actions) and plan_offset == len(plans)
    assert len(poses) == entry['history_frames'] + len(actions)
    assert chain['goals_completed'] == sum(r['reached'] for r in checked)
    assert chain['joint_success'] == (len(checked) == 3 and not failed)
    assert chain['cumulative_success'] == [int(chain['goals_completed'] > i) for i in range(3)]
    assert all(r['attempted'] is False and r['reached'] is None for r in chain['legs'][len(checked):])
    files = [root / 'manifest.json', root / 'summary.json', root / 'lingbot_pose_readout.jsonl',
        *sorted(run.rglob('*.json')), *sorted(run.rglob('*.jsonl')), run / 'execution.navmesh']
    result = dict(verified=True, index=manifest['cell']['index'], mode=manifest['mode'],
        scene=manifest['cell']['scene'], scope=plan['scope'], per_leg=checked,
        joint_success=chain['joint_success'], goals_completed=chain['goals_completed'],
        one_initial_history_replay=True, no_goal_boundary_reset_or_replay=True,
        continuous_frames=len(poses), scale_receipt_sha256=next(iter(scale_ids)),
        plan_sha256=sha(manifest['plan']), artifact_sha256={str(p): sha(p) for p in files},
        verifier_sha256=sha(__file__))
    save(root / 'independent_verification.json', result)
    print(json.dumps(dict(index=result['index'], mode=result['mode'], verified=True,
        goals_completed=result['goals_completed'], continuous_frames=len(poses))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    audit(parser.parse_args().root)
