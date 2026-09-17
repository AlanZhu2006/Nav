"""Independent execution audit and paired, scene-clustered GEM reduction.

The task auditor reconstructs SR/SPL from actual positions and uses the
existing independent motion/depth auditor. The reducer never replaces a
failed or missing task by another history or treats partial data as final.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ.get('GEM_VERIFY_SOURCE_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
MODES = ('legacy', 'native_interval7', 'connected_reciprocal')
ROLES = ('novel', 'revisit')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def audit_task(folder):
    from MemNavData.table1_repaired_eval import history
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows

    folder = Path(folder).resolve()
    manifest, summary = load(folder / 'manifest.json'), load(folder / 'summary.json')
    assert not (folder / 'failure.json').exists(), 'Task has a recorded runtime failure'
    assert summary['completed'] and len(summary['records']) == 2
    assert manifest['cell'] == summary['cell'] and manifest['mode'] == summary['mode']
    cell, mode = summary['cell'], summary['mode']
    plan = load(manifest['plan'])
    assert sha(manifest['plan']) == manifest['plan_sha256']
    assert plan['cells'][cell['index']] == cell and tuple(plan['modes']) == MODES
    assert mode in MODES and cell['controller'] == 'navdp'
    assert all(sha(p) == h for p, h in manifest['sources_sha256'].items())
    queries, frozen, query_dir = history(cell)
    query_by_role = {q['analysis_role']: q for p in queries['pairs'] for q in p['queries']}
    assert set(query_by_role) == set(ROLES)
    assert {r['role'] for r in summary['records']} == set(ROLES)
    # The observer writes both reset-separated episodes to one append-only
    # log. Materialize its exact rows for the existing per-rollout auditor.
    observed, segments = read_rows(folder / 'lingbot_pose_readout.jsonl'), []
    for row in observed:
        if row['frame_idx'] == 0:
            segments.append([])
        assert segments and row['frame_idx'] == len(segments[-1])
        segments[-1].append(row)
    assert len(segments) == len(summary['records']) == 2
    rows, artifacts = [], {str(folder / 'lingbot_pose_readout.jsonl'): sha(folder / 'lingbot_pose_readout.jsonl')}
    for record, segment in zip(summary['records'], segments):
        role = record['role']
        target = folder / 'evaluation' / role
        assert Path(record['directory']).resolve() == target
        assert record['mode'] == mode and record['scene'] == cell['scene']
        terminal = load(target / 'terminal_measurements.json')
        assert len(terminal) == 1
        assert all(record[k] == v for k, v in terminal[0].items())
        assert len(segment) == len(frozen['trace']['poses']) + record['steps']
        derived = target / 'lingbot_frame_poses.json'
        if derived.exists():
            assert load(derived) == segment
        else:
            save(derived, segment)
        checked, evidence, plans, actions = verify_rollout(dict(record, arm='cec'))
        data = load(target / 'query_plans.json')
        assert data['controller'] == cell['controller']
        assert data['replay']['all_rgb_hashes_verified']
        assert data['replay']['diffusion_samples_during_replay'] == 0
        assert data['rollout_traces']['legA'] == frozen['trace']['poses']
        assert data['rollout_traces']['query'] == evidence['rollout_trace']
        np.testing.assert_array_equal(actions[0]['position_before'], frozen['trace']['end_position'])
        assert actions[0]['yaw_before'] == frozen['trace']['end_yaw']
        query = query_by_role[role]
        assert data['original_goal_sha256'] == query['goal_rgb_sha256'] == sha(query_dir / query['goal_rgb'])
        np.testing.assert_array_equal(record['goal_xz_evaluator_only'], np.asarray(query['floor_position'])[[0, 2]])
        assert abs(record['geodesic_m'] - query['geodesic_from_a_end_m']) <= .05
        for request in read_rows(target / 'memory_http_boundary.jsonl'):
            assert not set(request['sent_fields']) & {
                'role', 'analysis_role', 'floor_position', 'gt_pose', 'executed_translation_m',
                'executed_yaw_rad', 'executed_forward_m', 'executed_left_m', 'executor_local_se2_source'}
            if request['frame_idx'] is not None and request['image_sha256'] is not None:
                assert request['image_sha256'] == segment[request['frame_idx']]['image_sha256']
        resources = load(target / 'memory_resources.json')
        memory = resources['memory']
        assert memory['frames'] == len(frozen['trace']['poses']) + record['steps']
        if mode != 'legacy':
            assert memory['module'] == mode and memory['failure'] is None
            assert memory['frames'] == memory['pose_frames'] == memory['descriptor_frames'] == memory['online_depth_frames']
            writes = resources['write_timings']
            assert [r['frame'] for r in writes] == list(range(memory['frames']))
            assert max(r['live_committed'] for r in writes) <= (16 if mode == 'connected_reciprocal' else 320)
        row = dict(checked, index=cell['index'], dataset=cell['dataset'], mode=mode,
            episode=cell['episode'], candidate_design_scene=cell['candidate_design_scene'],
            first_query_rgb_sha256=record['first_query_rgb_sha256'],
            goal_sha256=data['original_goal_sha256'], goal_xz=record['goal_xz_evaluator_only'],
            geodesic_m=record['geodesic_m'], history_trace_sha256=queries['online_a_trace_sha256'],
            wall_seconds=record['wall_seconds'], memory_resources=resources,
            takeover_plans=sum(p['receipt']['revisit_adapter_takeover'] is True for p in plans))
        rows.append(row)
        for name in ('terminal_measurements.json', 'rollout_evidence.json', 'executor_actions.jsonl',
                'full_plan_outputs.jsonl', 'navdp_http_receipts.jsonl', 'memory_http_boundary.jsonl',
                'memory_resources.json', 'query_plans.json', 'execution.navmesh', 'lingbot_frame_poses.json'):
            artifacts[str(target / name)] = sha(target / name)
    result = dict(verified=True, cell=cell, mode=mode, records=rows,
        manifest_sha256=sha(folder / 'manifest.json'), summary_sha256=sha(folder / 'summary.json'),
        plan_sha256=manifest['plan_sha256'], artifact_sha256=artifacts,
        verifier_sha256=sha(__file__), verifier_dependencies={
            str(ROOT / 'MemNavData' / name): sha(ROOT / 'MemNavData' / name)
            for name in ('verify_repaired_fullmono_local.py', 'verify_habitat_minimal_repair.py')})
    save(folder / 'independent_verification.json', result)
    return dict(index=cell['index'], mode=mode, verified=True,
        outcomes=[(r['role'], r['reached'], r['steps']) for r in rows])


def clustered_difference(rows, candidate, control, field, draws=10000):
    """Episode-weighted difference, resampling whole scenes, fixed seed."""
    delta = np.array([r[candidate][field] - r[control][field] for r in rows], float)
    if not len(delta):
        return dict(n=0, scenes=0, mean=None, scene_bootstrap_ci95=None)
    scene_keys = [(r[candidate]['dataset'], r[candidate]['scene']) for r in rows]
    scenes = sorted(set(scene_keys))
    ci = None
    if len(scenes) > 1:
        sums = np.array([delta[[k == scene for k in scene_keys]].sum() for scene in scenes])
        counts = np.array([scene_keys.count(scene) for scene in scenes])
        indices = np.random.default_rng(20260913).integers(len(scenes), size=(draws, len(scenes)))
        estimates = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        ci = np.quantile(estimates, [.025, .975]).tolist()
    return dict(n=len(delta), scenes=len(scenes), mean=float(delta.mean()), scene_bootstrap_ci95=ci)


def reduce(plan_path, task_root, *, require_complete=False):
    plan = load(plan_path)
    assert tuple(plan['modes']) == MODES and plan['total_rollouts'] == len(plan['cells']) * 6
    expected = {(c['index'], m) for c in plan['cells'] for m in MODES}
    indexed, states = {}, {}
    for folder in sorted(Path(task_root).iterdir()):
        if not folder.is_dir() or not (folder / 'manifest.json').exists():
            continue
        manifest = load(folder / 'manifest.json')
        key = (manifest['cell']['index'], manifest['mode'])
        assert key in expected and key not in states, 'Unexpected or duplicated task'
        assert manifest['plan_sha256'] == sha(plan_path)
        assert manifest['cell'] == plan['cells'][key[0]]
        if (folder / 'failure.json').exists():
            states[key] = dict(status='runtime_failed', error=load(folder / 'failure.json'))
        elif not load(folder / 'summary.json')['completed']:
            states[key] = dict(status='running_or_incomplete')
        elif not (folder / 'independent_verification.json').exists():
            states[key] = dict(status='completed_unverified')
        else:
            verified = load(folder / 'independent_verification.json')
            assert verified['verified'] and verified['cell'] == manifest['cell'] and verified['mode'] == key[1]
            assert verified['manifest_sha256'] == sha(folder / 'manifest.json')
            assert verified['summary_sha256'] == sha(folder / 'summary.json')
            assert verified['plan_sha256'] == sha(plan_path)
            assert verified['verifier_sha256'] == sha(__file__)
            assert all(sha(p) == h for p, h in verified['artifact_sha256'].items())
            states[key] = dict(status='verified')
            indexed[key] = {r['role']: r for r in verified['records']}
    for key in expected - states.keys():
        states[key] = dict(status='not_started')
    complete = all(r['status'] == 'verified' for r in states.values())
    pairs = []
    for cell in plan['cells']:
        if not all((cell['index'], mode) in indexed for mode in MODES):
            continue
        for role in ROLES:
            arm = {mode: indexed[(cell['index'], mode)][role] for mode in MODES}
            for name in ('first_query_rgb_sha256', 'goal_sha256', 'goal_xz', 'geodesic_m', 'history_trace_sha256'):
                assert all(arm[m][name] == arm['legacy'][name] for m in MODES), 'Unpaired ' + name
            pairs.append(arm)
    groups = {}
    for dataset in ['all'] + sorted({c['dataset'] for c in plan['cells']}):
        for scope in ('all_scenes', 'not_used_for_tonights_design', 'design_scene_overlap'):
            for role in ROLES:
                chosen = [p for p in pairs if p['legacy']['role'] == role
                    and (dataset == 'all' or p['legacy']['dataset'] == dataset)
                    and (scope == 'all_scenes' or p['legacy']['candidate_design_scene'] == (scope == 'design_scene_overlap'))]
                if not chosen:
                    continue
                arm_stats = {}
                for mode in MODES:
                    rows = [p[mode] for p in chosen]
                    arm_stats[mode] = dict(n=len(rows), successes=sum(r['reached'] for r in rows),
                        sr=float(np.mean([r['reached'] for r in rows])), spl=float(np.mean([r['spl'] for r in rows])),
                        wall_seconds_mean=float(np.mean([r['wall_seconds'] for r in rows])),
                        service_peak_allocated_gib_mean=float(np.mean([r['memory_resources']['gpu_peak_allocated_bytes'] / 2**30 for r in rows])))
                differences = {}
                for control in ('legacy', 'native_interval7'):
                    differences[control] = {metric: clustered_difference(chosen, 'connected_reciprocal', control, metric)
                        for metric in ('reached', 'spl')}
                    differences[control]['success_discordance'] = dict(
                        candidate_only=sum(p['connected_reciprocal']['reached'] == 1 and p[control]['reached'] == 0 for p in chosen),
                        control_only=sum(p['connected_reciprocal']['reached'] == 0 and p[control]['reached'] == 1 for p in chosen))
                groups[f'{dataset}/{scope}/{role}'] = dict(arms=arm_stats, paired_differences=differences)
    result = dict(complete=complete, plan_sha256=sha(plan_path), expected_tasks=len(expected),
        expected_rollouts=plan['total_rollouts'], verified_tasks=len(indexed), paired_histories=len(pairs)//2,
        states=[dict(index=i, mode=m, **states[(i, m)]) for i, m in sorted(states)], groups=groups,
        interpretation='Final fixed-population comparison' if complete else 'INCOMPLETE: matched completed histories only; no population conclusion',
        resource_scope='Torch memory-service process peak (both role runs); excludes NavDP, Habitat, NVML and CPU RSS. Wall time includes history replay.',
        resampling='10000 fixed-seed bootstrap draws of whole dataset/scene clusters; episode-weighted mean difference',
        reducer_sha256=sha(__file__))
    save(Path(task_root).parent / 'independent_reduction.json', result)
    if require_complete and not complete:
        raise RuntimeError('Full planned population has not completed and passed independent verification')
    return {k: result[k] for k in ('complete', 'expected_tasks', 'expected_rollouts', 'verified_tasks', 'paired_histories')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('task', 'reduce'))
    parser.add_argument('path', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = audit_task(args.path) if args.action == 'task' else reduce(args.plan, args.path, require_complete=args.require_complete)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
