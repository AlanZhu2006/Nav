"""Four-arm consumed-history pilot using the existing repaired Habitat stack."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.habitat_executor_audit import dump, sha

HERE = Path(__file__).resolve()
ARMS = ('native', 'raw_fixed', 'cec', 'cec_no_coverage')
HISTORIES = (2, 12)
QUERIES = ('c10_30', 'natural_novel')
SCOPE = 'consumed targeted mechanism pilot; not independent confirmation'


def load(path):
    return json.loads(Path(path).read_text())


def task(inputs, index):
    if not 0 <= index < 4:
        raise ValueError('four fixed queries only')
    history, query_id = HISTORIES[index // 2], QUERIES[index % 2]
    manifest = load(inputs / 'manifest.json')
    item = next(h for h in manifest['histories'] if h['history'] == history)
    folder = inputs / item['prefix']
    construction = load(folder / 'construction.json')
    query = next(q for q in construction['queries'] if q['query_id'] == query_id)
    staged = next(q for q in item['queries'] if q['query_id'] == query_id)
    source = dict(construction['source'], asset=str(inputs / item['asset']))
    online = folder / 'online'
    frozen = dict(source=online, trace=load(online / 'online_a_trace.json'),
                  receipt=load(online / 'receipt.json'))
    return item, construction, query, staged, source, frozen


def command(inputs, index, arm, out, mem, nav):
    from MemNavData.run_repaired_fullmono_local import evaluator_command
    _, _, query, _, source, _ = task(inputs, index)
    cmd = evaluator_command(source, out, mem, nav,
        arm='cec' if arm == 'cec_no_coverage' else arm,
        role=query['analysis_role'], benchmark=inputs)
    cmd[2:4] = [str(HERE), 'eval']
    if arm == 'cec_no_coverage':
        cmd += ['--certified_authority_policy', 'certificate_without_coverage']
    return cmd


def evaluate_query():
    import numpy as np
    import pandas as pd
    import eval_shared_online_role_pairs as shared
    base, args = shared.base, shared.args
    _, backend = shared.validate_cli()
    if args.contract_dry_run:
        print('LOCAL COVERAGE PILOT CLI OK', args.certified_authority_policy)
        return
    inputs = Path(os.environ['LOW_COVIS_INPUTS'])
    index = int(os.environ['LOW_COVIS_QUERY_INDEX'])
    item, payload, query, staged, _, frozen = task(inputs, index)
    trace, receipt = frozen['trace'], frozen['receipt']
    assert args.scene_identity == item['scene'] and args.episode_ids == payload['episode']
    assert args.role_pair_query_role == query['analysis_role']
    folder = inputs / item['prefix']
    carrier = folder / 'camera_source.parquet'
    assert sha(carrier) == receipt['source_parquet_sha256']
    intrinsic = np.stack([np.asarray(r, float) for r in
                         pd.read_parquet(carrier).iloc[0]['observation.camera_intrinsic']])
    assert float(receipt['camera_height_m']) == base.CAM_H
    goal_path = inputs / staged['goal']
    assert sha(goal_path) == query['goal_rgb_sha256']
    goal = np.asarray(query['floor_position'], float)  # evaluator scoring only
    sim = base.make_sim(args.scene, '', agent_radius=args.agent_radius)
    try:
        base.srv_reset(camera_height=receipt['camera_height_m'], seed=int(trace['episode_seed']),
            episode_len=len(trace['poses']) + args.max_steps, camera_intrinsic=intrinsic,
            causal_history_sha256=payload['online_a_trace_sha256'])
        a, replay = shared.replay_prefix(frozen)
        ok, geo, _ = base.geodesic(sim.pathfinder, a['end_pos'], goal)
        assert ok and abs(geo - query['geodesic_from_a_end_m']) <= .05
        result = base.run_policy_leg(sim, sim.pathfinder, a['end_pos'], a['end_psi'],
            goal_path.read_bytes(), goal[[0, 2]], float(geo), None, terminal_mode='off',
            goal_yaw=query['yaw_rad'], camera_intrinsic=intrinsic, policy_backend=backend,
            episode_seed=int(trace['episode_seed']), leg_index=1)
        dump(Path(args.out) / 'query_plans.json', dict(
            scope=SCOPE, authority_policy=args.certified_authority_policy,
            query_leg=result['plans'], replay=replay,
            rollout_traces=dict(legA=trace['poses'], query=result['rollout_trace']),
            counts=shared.router_counts(result['plans']), depth=shared.depth_counts(result['plans']),
            runtime_inputs='goal RGB and causal observations; role/support/pose not forwarded'))
    finally:
        sim.close()


def run(args):
    from MemNavData.run_repaired_fullmono_local import (
        HAB_PY, execution_environment, private_servers, run_child)
    inputs, out = args.inputs.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'logs').mkdir()
    staged = load(inputs / 'manifest.json')
    for item in staged['files']:
        assert sha(inputs / item['path']) == item['sha256'], item['path']
    schedule = [dict(query_index=i, history=HISTORIES[i // 2], query=QUERIES[i % 2],
                     arms=list(ARMS[i:] + ARMS[:i])) for i in range(4)]
    source_paths = [HERE, ROOT / 'MemNavData/eval_2leg_habitat.py',
        ROOT / 'MemNavData/certified_relocalization_runtime.py',
        ROOT / 'MemNavData/certified_relocalization_contract.py',
        ROOT / 'MemNavData/run_habitat_minimal_repair_local.py',
        ROOT / 'MemNavData/run_repaired_fullmono_local.py',
        ROOT / 'MemNavData/bounded_pursuit.py',
        ROOT / 'MemNavData/navdp_front_goal_adapter.py',
        ROOT / 'NavDP/baselines/memnav/policy_agent.py',
        ROOT / 'NavDP/baselines/memnav/memnav_server.py']
    source_hashes = {str(p): sha(p) for p in source_paths}
    dump(out / 'manifest.json', dict(scope=SCOPE, inputs=str(inputs),
        input_manifest_sha256=sha(inputs / 'manifest.json'), source_hashes=source_hashes,
        schedule=schedule, new_a_rollouts=0, max_steps=600, exec_horizon=8,
        support_domain='CEC frame>=8 through A end; not annotation frame>=39',
        protocol_sha256=sha(ROOT / 'MemNavData/LOW_COVISIBILITY_NAVIGATION_PILOT_PROTOCOL_20260910.md')))
    # Validate all CLI variants before starting either model server.
    for arm in ARMS:
        cmd = command(inputs, 0, arm, out / 'dry', args.mem_port, args.nav_port)
        run_child(cmd + ['--contract_dry_run'], out / 'logs' / f'dry_{arm}.log',
                  environment=execution_environment())
    summary = dict(scope=SCOPE, completed=False, queries=[])
    dump(out / 'summary.json', summary)
    if args.wait_free_gib:
        # Wait for other workloads, without changing the experiment's history or model.
        deadline, stable = time.monotonic() + 6 * 3600, 0
        while stable < 3:
            free = float(subprocess.check_output([
                'nvidia-smi', '--id=0', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip()) / 1024
            stable = stable + 1 if free >= args.wait_free_gib else 0
            dump(out / 'gpu_wait.json', dict(waiting=stable < 3, free_gib=free,
                required_free_gib=args.wait_free_gib, consecutive_ready=stable,
                other_processes_untouched=True, checked_at=time.time()))
            print(f'GPU WAIT free={free:.2f} GiB required={args.wait_free_gib:.1f} stable={stable}/3', flush=True)
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU not available within six hours; no navigation started')
            if stable < 3:
                time.sleep(30)
        if any(sha(Path(p)) != digest for p, digest in source_hashes.items()):
            raise RuntimeError('source changed while queued; review before running')
    pose_log = out / 'lingbot_pose_readout.jsonl'
    os.environ.update(REPAIRED_BUFFER_ROOT=str(out / 'buffer'),
                      REPAIRED_RUNTIME_ROOT=str(out / 'runtime'))
    with private_servers(out, args.mem_port, args.nav_port):
        for scheduled in schedule:
            index = scheduled['query_index']
            item, _, query, _, _, _ = task(inputs, index)
            for arm in scheduled['arms']:
                if shutil.disk_usage(out).free < 1_000_000_000:
                    raise RuntimeError('less than 1 GB free; preserve outputs and stop before another rollout')
                folder = out / 'evaluation' / f'q{index:02d}' / arm
                cmd = command(inputs, index, arm, folder, args.mem_port, args.nav_port)
                dump(out / 'progress.json', dict(query_index=index, arm=arm,
                    completed_arms=len(summary['queries']), command=cmd))
                print(f'RUN q={index} history={item["history"]} goal={query["query_id"]} arm={arm}', flush=True)
                offset = pose_log.stat().st_size if pose_log.exists() else 0
                env = dict(execution_environment(), LOW_COVIS_INPUTS=str(inputs),
                           LOW_COVIS_QUERY_INDEX=str(index))
                seconds = run_child(cmd, out / 'logs' / f'q{index:02d}_{arm}.log', environment=env)
                with pose_log.open() as stream:
                    stream.seek(offset)
                    dump(folder / 'lingbot_frame_poses.json', [json.loads(l) for l in stream if l.strip()])
                terminal = load(folder / 'terminal_measurements.json')
                assert len(terminal) == 1
                row = dict(terminal[0], query_index=index, history=item['history'], scene=item['scene'],
                    role=query['analysis_role'], arm=arm, directory=str(folder), wall_seconds=seconds,
                    counts=load(folder / 'query_plans.json')['counts'])
                summary['queries'].append(row)
                dump(out / 'summary.json', summary)
                print(f'DONE q={index} arm={arm} SR={row["reached"]} steps={row["steps"]} SPL={row["spl"]:.6f}', flush=True)
    summary['completed'] = True
    dump(out / 'summary.json', summary)
    run_child([HAB_PY, str(HERE), 'verify', '--out', str(out)],
              out / 'logs/verification.log', environment=execution_environment())


def verify(out):
    import numpy as np
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows
    manifest, summary = load(out / 'manifest.json'), load(out / 'summary.json')
    assert summary['completed'] and len(summary['queries']) == 16
    assert all(sha(Path(p)) == d for p, d in manifest['source_hashes'].items())
    results, paired = [], {}
    for row in summary['queries']:
        checked, evidence, plans, actions = verify_rollout(row)
        data = load(Path(row['directory']) / 'query_plans.json')
        _, _, _, _, _, frozen = task(Path(manifest['inputs']), row['query_index'])
        np.testing.assert_array_equal(actions[0]['position_before'], frozen['trace']['end_position'])
        assert actions[0]['yaw_before'] == frozen['trace']['end_yaw']
        assert data['replay']['all_rgb_hashes_verified']
        assert data['replay']['diffusion_samples_during_replay'] == 0
        assert data['rollout_traces']['legA'] == frozen['trace']['poses']
        forbidden = {'analysis_role', 'q_eligible', 'covis_curve', 'floor_position',
            'executed_translation_m', 'executed_yaw_rad', 'executed_forward_m', 'executed_left_m'}
        for request in read_rows(Path(row['directory']) / 'memory_http_boundary.jsonl'):
            assert not forbidden.intersection(request['sent_fields'])
        if row['arm'] in ('cec', 'cec_no_coverage'):
            policy = 'strict_certificate' if row['arm'] == 'cec' else 'certificate_without_coverage'
            assert data['authority_policy'] == policy
            assert all(p.get('certified_relocalization_authority_policy') == policy
                       for p in data['query_leg'])
        results.append(dict(checked, query_index=row['query_index'], counts=data['counts']))
        paired.setdefault(row['query_index'], {})[row['arm']] = (row, data, plans, actions)
    for index, arms in paired.items():
        assert set(arms) == set(ARMS)
        native = arms['native']
        for arm, (row, data, plans, actions) in arms.items():
            assert row['first_query_rgb_sha256'] == native[0]['first_query_rgb_sha256']
            assert row['goal_xz_evaluator_only'] == native[0]['goal_xz_evaluator_only']
            assert data['replay'] == native[1]['replay']
            if arm in ('cec', 'cec_no_coverage') and not any(
                    p['receipt']['revisit_adapter_takeover'] is True for p in plans):
                assert len(plans) == len(native[2]) and len(actions) == len(native[3])
                for p, n in zip(plans, native[2]):
                    for key in ('selected_trajectory', 'all_trajectory', 'all_values', 'position', 'yaw'):
                        np.testing.assert_array_equal(p[key], n[key])
                for a, n in zip(actions, native[3]):
                    np.testing.assert_array_equal(a['actual_position'], n['actual_position'])
                    assert a['actual_yaw'] == n['actual_yaw']
    dump(out / 'independent_verification.json', dict(verified=True, scope=SCOPE, results=results))
    print('VERIFIED 16 local rollouts; targeted mechanism sample, not confirmation', flush=True)


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'eval':
        from MemNavData.run_habitat_minimal_repair_local import evaluate
        evaluate(query_main=evaluate_query)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--out', type=Path, required=True)
        parser.add_argument('--inputs', type=Path)
        parser.add_argument('--mem-port', type=int, default=21810)
        parser.add_argument('--nav-port', type=int, default=21811)
        parser.add_argument('--wait-free-gib', type=float, default=0.)
        args = parser.parse_args()
        if mode == 'run':
            run(args)
        elif mode == 'verify':
            verify(args.out)
        else:
            parser.error('mode must be run, eval or verify')
