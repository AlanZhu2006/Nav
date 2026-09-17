"""Fixed A-to-B-to-A navigation with persistent, own-state GEM memory.

All four existing local integration histories are retained. A is the existing
perturbed Revisit image and B is the first actually observed historical RGB.
This is a chronological integration/stress experiment, not a new Novel/Revisit
benchmark. The history is replayed once; later goals inherit each arm's actual
endpoint and memory. A navigation failure ends the chain without replacement.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
from MemNavData.run_gem_memory_navigation import MODES, servers, existing
from MemNavData.habitat_executor_audit import dump, sha
from MemNavData.run_repaired_fullmono_local import execution_environment, run_child

HERE = Path(__file__).resolve()
SCHEMA = 'gem_memory_continuous_aba_v1'


def load(path):
    return json.loads(Path(path).read_text())


def freeze(source_plan, destination):
    if destination.exists():
        raise FileExistsError(destination)
    original = load(source_plan)
    assert original['local_interface_population'] and len(original['cells']) == 4
    cells = []
    for cell in original['cells']:
        payload, frozen, folder = existing.history(cell)
        query = next(q for pair in payload['pairs'] for q in pair['queries']
                     if q['analysis_role'] == 'revisit')
        first = frozen['trace']['poses'][0]
        a = dict(name='A', goal_rgb=str(folder / query['goal_rgb']),
            goal_rgb_sha256=query['goal_rgb_sha256'], floor_position=query['floor_position'],
            yaw_rad=query['yaw_rad'], source='existing perturbed Revisit query')
        b = dict(name='B', goal_rgb=str(frozen['source'] / 'rgb/000000.jpg'),
            goal_rgb_sha256=first['jpg_sha256'], floor_position=[first[k] for k in 'xyz'],
            yaw_rad=first['yaw'], source='first actually observed RGB; no outcome-based selection')
        assert a['goal_rgb_sha256'] != b['goal_rgb_sha256']
        assert all(sha(g['goal_rgb']) == g['goal_rgb_sha256'] for g in (a, b))
        cells.append(dict(cell=cell, history_frames=len(frozen['trace']['poses']),
            history_sha256=payload['online_a_trace_sha256'], goals=[a, b, deepcopy(a)]))
    api = ROOT / '.diagnostics/gem_connected_memory_20260913/api_final_main_002/result.json'
    proof = load(api)
    assert proof['passed']
    assert all(sha(ROOT / p) == v['delivered_sha256'] for p, v in proof['core_files'].items())
    dump(destination, dict(schema=SCHEMA, source_plan=str(source_plan.resolve()),
        source_plan_sha256=sha(source_plan), cells=cells, modes=MODES, episodes=12,
        max_steps_per_goal=600, success_radius_m=1., execution_horizon=8,
        sequence='A -> historical first view B -> A', history_replays_per_episode=1,
        runtime_reset_at_goal_switch=False, failure_terminates_chain=True,
        scope='All four previously used local integration histories; not independent held-out performance evidence',
        selection='All four histories, fixed original Revisit A and first observed B; no result-dependent resampling',
        api_proof=str(api), api_proof_sha256=sha(api),
        core_sha256={p: v['delivered_sha256'] for p, v in proof['core_files'].items()},
        created_at=time.time()))
    print(json.dumps(dict(episodes=12, maximum_goal_legs=36, plan_sha256=sha(destination))), flush=True)


def evaluate():
    from MemNavData.run_habitat_minimal_repair_local import evaluate as wrapped_evaluate

    def query():
        import numpy as np
        import pandas as pd
        import requests
        import eval_shared_online_role_pairs as shared
        from MemNavData.table2_continuous_chain import run_chain
        from MemNavData.final14_spl_replay import measurement
        from MemNavData.deterministic_eval_protocol import validate_leg1_trace

        base, args = shared.base, shared.args
        _, backend = shared.validate_cli()
        if args.contract_dry_run:
            print('GEM CONTINUOUS CLI OK: one prefix, persistent own-state A -> B -> A', flush=True)
            return
        plan = load(os.environ['GEM_CONTINUOUS_PLAN'])
        entry = plan['cells'][int(os.environ['GEM_CONTINUOUS_INDEX'])]
        cell = entry['cell']
        payload, frozen, _ = existing.history(cell)
        assert payload['online_a_trace_sha256'] == entry['history_sha256']
        trace, receipt = frozen['trace'], frozen['receipt']
        carrier = Path(receipt['source_episode']) / 'data/chunk-000/episode_000000.parquet'
        assert sha(carrier) == receipt['source_parquet_sha256']
        assert sha(receipt['source_asset']) == receipt['source_asset_sha256']
        intrinsic = np.stack([np.asarray(r, float) for r in
            pd.read_parquet(carrier).iloc[0]['observation.camera_intrinsic']])
        assert float(receipt['camera_height_m']) == base.CAM_H
        out = Path(args.out)
        base.runtime_executor_motion_form = lambda *a, **k: {}
        original_post = base.requests.post

        def rgb_post(url, *a, **kwargs):
            assert not set(kwargs.get('data') or {}) & {
                'executed_translation_m', 'executed_yaw_rad', 'executed_forward_m',
                'executed_left_m', 'executor_local_se2_source'}
            response = original_post(url, *a, **kwargs)
            if url.endswith('/certified_relocalize'):
                response.raise_for_status()
                value = response.json()
                assert value.get('pnp', {}).get('status') != 'runtime_exception'
                assert not any(r.get('error') for r in value.get('ranked_candidates', []))
            return response

        base.requests.post = rgb_post
        sim = base.make_sim(args.scene, '', agent_radius=.30)
        selected, leg_records = [], []
        try:
            base.srv_reset(camera_height=receipt['camera_height_m'], seed=trace['episode_seed'],
                episode_len=len(trace['poses']) + 3 * plan['max_steps_per_goal'],
                camera_intrinsic=intrinsic, causal_history_sha256=entry['history_sha256'])
            initial, replay = shared.replay_prefix(frozen)
            dump(out / 'prefix_replay.json', replay)

            def execute(index, goal, position, yaw):
                adapter = getattr(base, '_front_goal_adapter', None)
                if index and adapter is not None and adapter.active:
                    raise RuntimeError('A successful goal left an unfinished heading action')
                assert sha(goal['goal_rgb']) == goal['goal_rgb_sha256']
                target = np.asarray(goal['floor_position'], float)
                ok, distance, _ = base.geodesic(sim.pathfinder, position, target)
                if not ok or not np.isfinite(distance):
                    raise RuntimeError('Fixed successor is not reachable from the actual endpoint')
                issued = dict(goal, start_position=position.tolist(), start_yaw=yaw,
                              geodesic_m=float(distance), stage=index)
                selected.append(issued)
                (out / f'leg_{index}').mkdir(parents=True, exist_ok=False)
                dump(out / f'leg_{index}/query.json', issued)
                print(f'START {index} {goal["name"]}: actual geodesic={distance:.3f}m', flush=True)
                return base.run_policy_leg(sim, sim.pathfinder, position, yaw,
                    Path(goal['goal_rgb']).read_bytes(), target[[0, 2]], float(distance), None,
                    terminal_mode='off', goal_yaw=goal['yaw_rad'], camera_intrinsic=intrinsic,
                    policy_backend=backend, success_dist=plan['success_radius_m'],
                    episode_seed=int(trace['episode_seed']), leg_index=index + 1)

            def observe(index, goal, leg, continuity):
                directory = out / f'leg_{index}'
                query = selected[index]
                actual = base.leg1_trace_payload(episode=cell['episode'],
                    episode_seed=trace['episode_seed'], goal_jpg=Path(goal['goal_rgb']).read_bytes(),
                    goal_source_episode=cell['episode'], source_scene=cell['scene'], leg=leg)
                validate_leg1_trace(actual)
                measured = measurement(leg, np.asarray(goal['floor_position'])[[0, 2]], query['geodesic_m'])
                dump(directory / 'actual_trace.json', actual)
                dump(directory / 'measurement.json', measured)
                dump(directory / 'continuity.json', continuity)
                # The existing wrapper overwrites this per-leg view. Preserve
                # each exact returned record before the next goal runs.
                dump(directory / 'rollout_evidence.json', load(out / 'rollout_evidence.json'))
                response = requests.get(base.BASE + '/memory_status', timeout=30)
                response.raise_for_status()
                dump(directory / 'memory_resources.json', response.json())
                leg_records.append(dict(stage=index, query=query, measurement=measured,
                    continuity=continuity, directory=str(directory)))
                dump(out / 'leg_records.json', leg_records)
                print(f'DONE {index} {goal["name"]}: SR={int(leg["reached"])} '
                      f'steps={leg["steps"]} memory={continuity["first_memory_index"]}..'
                      f'{continuity["last_memory_index"]}', flush=True)

            result = run_chain(initial['end_pos'], initial['end_psi'], entry['goals'], execute, observe)
            result.update(completed=True, history_frames=len(trace['poses']),
                replay=replay, leg_records=leg_records, scope=plan['scope'])
            dump(out / 'chain_summary.json', result)
        finally:
            sim.close()

    wrapped_evaluate(query_main=query)


def run(args):
    plan = load(args.plan)
    assert plan['schema'] == SCHEMA and tuple(plan['modes']) == MODES
    assert sha(plan['source_plan']) == plan['source_plan_sha256']
    assert sha(plan['api_proof']) == plan['api_proof_sha256']
    assert all(sha(ROOT / p) == h for p, h in plan['core_sha256'].items())
    entry = plan['cells'][args.index]
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'logs').mkdir()
    sources = [HERE, ROOT / 'MemNavData/run_gem_memory_navigation.py',
        ROOT / 'MemNavData/table2_continuous_chain.py', ROOT / 'MemNavData/run_habitat_minimal_repair_local.py',
        ROOT / 'MemNavData/eval_2leg_habitat.py', ROOT / 'MemNavData/navdp_front_goal_adapter.py',
        *[ROOT / p for p in plan['core_sha256']]]
    hashes = {str(p): sha(p) for p in sources}
    dump(args.out / 'manifest.json', dict(schema=SCHEMA, cell=entry['cell'], mode=args.mode,
        plan=str(args.plan.resolve()), plan_sha256=sha(args.plan), sources_sha256=hashes,
        core_api_proof_sha256=plan['api_proof_sha256'], scope=plan['scope']))
    for p in sources:
        destination = args.out / 'source_snapshot' / p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    command = existing.command(entry['cell'], 'revisit', 'cec', args.out / 'evaluation',
                               args.mem_port, args.nav_port)
    command[2] = str(HERE)
    env = dict(execution_environment(), GEM_CONTINUOUS_PLAN=str(args.plan.resolve()),
               GEM_CONTINUOUS_INDEX=str(args.index))
    try:
        run_child(command + ['--contract_dry_run'], args.out / 'logs/preflight.log', environment=env)
        if args.prepare_only:
            dump(args.out / 'prepared.json', dict(prepared=True, navigation_started=False))
            return
        with servers(args.out, args.mode, args.mem_port, args.nav_port):
            duration = run_child(command, args.out / 'logs/evaluation.log', environment=env)
        assert all(sha(p) == h for p, h in hashes.items())
        result = load(args.out / 'evaluation/chain_summary.json')
        assert result['completed']
        dump(args.out / 'summary.json', dict(completed=True, cell=entry['cell'], mode=args.mode,
            plan_sha256=sha(args.plan), chain=result, wall_seconds=duration))
    except BaseException as error:
        dump(args.out / 'failure.json', dict(type=type(error).__name__, error=str(error)))
        raise


def main():
    action = sys.argv.pop(1)
    if action == 'eval':
        evaluate()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--mode', choices=MODES, default='connected_reciprocal')
    parser.add_argument('--mem-port', type=int, default=21870)
    parser.add_argument('--nav-port', type=int, default=21871)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if action == 'freeze':
        freeze(args.plan, args.out)
    elif action == 'run':
        run(args)
    else:
        parser.error('Expected freeze, run or eval')


if __name__ == '__main__':
    main()
