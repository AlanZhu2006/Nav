"""Four paired policies on the existing, immutable 159-query population.

The original query callback, simulator repair and archive format are reused.
Only the explicit coverage-authority option differs between the two CEC arms.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import repaired_covisibility_eval as original

load, dump, sha, require = original.load, original.dump, original.sha, original.require
ARMS = ('native', 'raw_fixed', 'cec', 'cec_no_coverage')
BINS = ('c10_30', 'c30_50', 'c50_70', 'c70_90', 'c90_100')
SCHEMA = 'cec_coverage_only_four_arm_20260910_v1'
SCOPE = 'consumed full support-spectrum component ablation; existing actual-mono A, repaired query execution'


def bin_for(q):
    require(.1 <= q <= 1., 'support query outside the prespecified range')
    return BINS[next((i for i, bound in enumerate((.3, .5, .7, .9)) if q < bound), 4)]


def freeze_plan(population, output):
    require(not output.exists(), 'do not overwrite a frozen plan')
    require(sha(population) == original.POPULATION_SHA, 'original population changed')
    tasks = []
    for index in range(len(load(population)['tasks'])):
        task, payload, query, _ = original.load_task(population, index)
        curve, n = query['covis_curve'], payload['online_a_steps']
        require(len(curve) == n and abs(max(curve[39:])-query['q_eligible']) < 1e-10,
                'source history domain changed')
        q = max(curve[8:])
        tasks.append(dict(task_index=index, history_index=task['history_index'],
            scene=payload['scene'], query_id=task['query_id'], original_bin=query['bin'],
            analysis_role=query['analysis_role'], q_cec_frame8=q,
            q_annotation_frame39=query['q_eligible'],
            q_raw_first_decision=max(curve[39:n-32+1]),
            analysis_bin='natural_novel' if query['analysis_role']=='novel' else bin_for(q),
            local_navigation_pilot_history=task['history_index'] in (2, 12),
            arm_order=list(ARMS[index % 4:] + ARMS[:index % 4])))
    counts = {b: sum(t['analysis_bin']==b for t in tasks) for b in (*BINS, 'natural_novel')}
    require(counts == dict(zip((*BINS, 'natural_novel'), (14,26,23,32,36,28))), 'frozen counts changed')
    require(len(tasks)==159 and len({t['history_index'] for t in tasks})==28
            and len({t['scene'] for t in tasks})==21, 'population identity changed')
    dump(output, dict(schema=SCHEMA, scope=SCOPE, population=str(population),
        population_sha256=original.POPULATION_SHA, arms=list(ARMS), tasks=tasks,
        counts=counts, primary='cec_no_coverage minus cec SR on support q8 in [0.1,0.5)',
        smoke_indices=[10,71], remaining_indices=[i for i in range(159) if i not in (10,71)],
        query_count=159, rollout_count=636, no_new_navigation_outcomes_read=True))
    print(json.dumps(dict(plan=str(output), sha256=sha(output), counts=counts)))


def read_plan(path):
    require(sha(path) == os.environ['EXPECTED_PLAN_SHA'], 'frozen plan changed')
    plan = load(path)
    require(plan['schema']==SCHEMA and plan['arms']==list(ARMS), 'wrong experiment')
    require(plan['query_count']==len(plan['tasks'])==159 and plan['rollout_count']==636, 'incomplete plan')
    require(sha(plan['population'])==original.POPULATION_SHA, 'original population changed')
    return plan


def command(source, out, mem, nav, *, arm, role):
    require(arm in ARMS, 'unknown arm')
    cmd = original.evaluator_command(source, out, mem, nav,
        arm='cec' if arm=='cec_no_coverage' else arm, role=role)
    cmd[2] = str(HERE)
    if arm=='cec_no_coverage':
        cmd += ['--certified_authority_policy', 'certificate_without_coverage']
    return cmd


def run(args):
    from MemNavData.run_repaired_fullmono_local import (
        HAB_PY, execution_environment, private_servers, run_child)
    plan = read_plan(args.plan)
    task = plan['tasks'][args.index]
    require(task['task_index']==args.index, 'task order changed')
    _, payload, query, _ = original.load_task(Path(plan['population']), args.index)
    require(task['query_id']==query['query_id'] and task['scene']==payload['scene'], 'target changed')
    out, durable = args.out.resolve(), args.durable.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out/'logs').mkdir()
    os.environ.update(COVIS_POPULATION=plan['population'], COVIS_TASK_INDEX=str(args.index),
        REPAIRED_BUFFER_ROOT=str(out/'buffer'), REPAIRED_RUNTIME_ROOT=str(out/'runtime'))
    manifest = dict(schema=SCHEMA, scope=SCOPE, plan=str(args.plan), plan_sha256=sha(args.plan),
        task=task, scene=payload['scene'], episode=payload['episode'], original_work_root=str(out),
        runtime_source_receipt=dict(path=os.environ['REPAIRED_SOURCE_RECEIPT'],
                                    sha256=sha(os.environ['REPAIRED_SOURCE_RECEIPT'])))
    dump(out/'manifest.json', manifest); dump(durable/'manifest.json', manifest)
    summary = dict(schema=SCHEMA, completed=False, queries=[])
    dump(out/'summary.json', summary)
    pose_log = out/'lingbot_pose_readout.jsonl'
    with private_servers(out, args.memnav_port, args.navdp_port):
        for arm in task['arm_order']:
            folder = out/'evaluation'/arm
            cmd = command(payload['source'], folder, args.memnav_port, args.navdp_port,
                          arm=arm, role=query['analysis_role'])
            progress = dict(task_index=args.index, arm=arm, completed_arms=len(summary['queries']), command=cmd)
            dump(durable/'progress.json', progress); dump(out/'progress.json', progress)
            print(f'RUN task={args.index} arm={arm}', flush=True)
            offset = pose_log.stat().st_size if pose_log.exists() else 0
            elapsed = run_child(cmd, out/'logs'/f'{arm}.log', environment=execution_environment())
            with pose_log.open() as stream:
                stream.seek(offset)
                dump(folder/'lingbot_frame_poses.json', [json.loads(s) for s in stream if s.strip()])
            terminal = load(folder/'terminal_measurements.json')
            require(len(terminal)==1, 'one terminal required per query arm')
            row = dict(terminal[0], scene=payload['scene'], episode=payload['episode'],
                role=query['analysis_role'], arm=arm, directory=str(folder), wall_seconds=elapsed)
            summary['queries'].append(row)
            dump(out/'summary.json', summary); dump(durable/'partial_summary.json', summary)
            print(f'DONE task={args.index} arm={arm} SR={row["reached"]} steps={row["steps"]} SPL={row["spl"]:.6f}', flush=True)
    summary['completed']=True
    dump(out/'summary.json', summary)
    run_child([HAB_PY, str(HERE), 'verify', '--out', str(out)], out/'logs/verification.log',
              environment=execution_environment())


def exact_fallback(native, candidate):
    import numpy as np
    from MemNavData.verify_habitat_minimal_repair import read_rows
    require(len(native[2])==len(candidate[2]) and len(native[3])==len(candidate[3]), 'fallback length differs')
    for a,b in zip(native[2], candidate[2]):
        for key in ('selected_trajectory','all_trajectory','all_values','position','yaw'):
            np.testing.assert_array_equal(a[key], b[key])
    for a,b in zip(native[3], candidate[3]):
        for key in ('actual_position','actual_yaw','action_kind'):
            require(a[key]==b[key], 'fallback action differs')
    depths=[]
    for row,*_ in (native,candidate):
        http=read_rows(Path(row['directory'])/'navdp_http_receipts.jsonl')
        depths.append([r['monocular_depth_receipt'] for r in http
            if r['query_active'] and r['path']!='/memory_replay_step' and r['audit']['image_calls']])
    require(len(depths[0])==len(depths[1])==len(native[2]), 'fallback depth count differs')
    for a,b in zip(*depths):
        for key in ('depth_png_sha256','image_sha256','scale_receipt_sha256','frame_index'):
            require(a[key]==b[key], 'fallback depth receipt differs')
        for key in ('input_tensor_sha256','output_tensor_sha256'):
            require(a['navdp_depth_raster'][key]==b['navdp_depth_raster'][key], 'fallback depth tensor differs')


def verify(out):
    import numpy as np
    from MemNavData.verify_repaired_fullmono_local import verify_rollout
    from MemNavData.verify_habitat_minimal_repair import read_rows
    manifest, summary = load(out/'manifest.json'), load(out/'summary.json')
    receipt=manifest['runtime_source_receipt']
    require(sha(receipt['path'])==receipt['sha256'], 'runtime receipt changed')
    plan=read_plan(Path(manifest['plan']))
    task=plan['tasks'][manifest['task']['task_index']]
    require(task==manifest['task'], 'frozen task changed')
    _, payload, query, frozen=original.load_task(Path(plan['population']), task['task_index'])
    require(summary['completed'] and len(summary['queries'])==4, 'incomplete four-arm query')
    require([r['arm'] for r in summary['queries']]==task['arm_order'], 'arm order changed')
    records, indexed = [], {}
    for row in summary['queries']:
        checked, evidence, plans, actions=verify_rollout(row)
        folder=Path(row['directory']); data=load(folder/'query_plans.json'); trace=frozen['trace']
        require(row['scene']==payload['scene'] and row['role']==query['analysis_role'], 'wrong query')
        require(data['rollout_traces']['legA']==trace['poses'] and
                data['rollout_traces']['query']==evidence['rollout_trace'], 'trace mismatch')
        require(data['replay']['all_rgb_hashes_verified'] and
                data['replay']['diffusion_samples_during_replay']==0, 'history replay mismatch')
        np.testing.assert_array_equal(actions[0]['position_before'], trace['end_position'])
        require(actions[0]['yaw_before']==trace['end_yaw'], 'start yaw differs')
        require(data['analysis_role_not_forwarded'] and 'analysis_role' not in data['query_runtime_fields'], 'role leak')
        forbidden={'analysis_role','max_online_a_covis','q_eligible','q_cec_frame8','covis_curve','floor_position',
                   'executed_translation_m','executed_yaw_rad','executed_forward_m','executed_left_m'}
        for boundary in read_rows(folder/'memory_http_boundary.jsonl'):
            require(not forbidden.intersection(boundary['sent_fields']), 'privileged memory input')
        if row['arm'] in ('cec','cec_no_coverage'):
            authority='strict_certificate' if row['arm']=='cec' else 'certificate_without_coverage'
            require(all(p.get('certified_relocalization_authority_policy')==authority for p in data['query_leg']),
                    'wrong authority or failed certificate request')
        checked.update(wall_seconds=row['wall_seconds'], counts=data['counts'],
            total_turn_deg=sum(abs(math.degrees(a['actual_yaw']-a['yaw_before'])) for a in actions))
        records.append(checked); indexed[row['arm']]=(row,data,plans,actions)
    native=indexed['native']
    for row,data,*_ in indexed.values():
        require(row['first_query_rgb_sha256']==native[0]['first_query_rgb_sha256'] and
                row['goal_xz_evaluator_only']==native[0]['goal_xz_evaluator_only'] and
                data['replay']==native[1]['replay'], 'unpaired arms')
    takeover, fallback={},{}
    for arm in ('cec','cec_no_coverage'):
        count=sum(p['receipt']['revisit_adapter_takeover'] is True for p in indexed[arm][2])
        takeover[arm]=count; fallback[arm]=count==0
        if not count:
            exact_fallback(native,indexed[arm])
    dump(out/'independent_verification.json', dict(verified=True, schema=SCHEMA, scope=SCOPE,
        plan_sha256=sha(manifest['plan']), population_sha256=original.POPULATION_SHA, task=task,
        query_arm_count=4, new_a_rollouts=0, records=records, takeover_plans=takeover,
        no_takeover_exact_native=fallback, verifier_sha256=sha(HERE)))
    print(f'VERIFIED task={task["task_index"]}: four arms, terminal SPL, RGB-only repaired control', flush=True)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='eval':
        sys.argv.pop(1)
        from MemNavData.run_habitat_minimal_repair_local import evaluate
        evaluate(query_main=original.evaluate_query)
        return
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=('freeze-plan','preflight-inputs','dry-run','run','verify'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--population', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--durable', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--memnav-port', type=int, default=21750)
    parser.add_argument('--navdp-port', type=int, default=21751)
    args=parser.parse_args()
    if args.mode=='freeze-plan':
        freeze_plan(args.population.resolve(), args.out.resolve())
    elif args.mode=='preflight-inputs':
        plan=read_plan(args.plan)
        for task in plan['tasks']:
            original.load_task(Path(plan['population']),task['task_index'])
        print('VERIFIED all 159 original query inputs; no new outcomes read')
    elif args.mode=='run':
        run(args)
    elif args.mode=='verify':
        verify(args.out.resolve())
    else:
        from MemNavData.run_repaired_fullmono_local import execution_environment
        source=dict(scene='s',episode='episode_0000',asset='/unused.glb',source_episode='/unused/episode_0000',seed=1)
        for role in ('novel','revisit'):
            for arm in ARMS:
                cmd=command(source,args.out/role/arm,args.memnav_port,args.navdp_port,arm=arm,role=role)
                subprocess.run(cmd+['--contract_dry_run'],env=execution_environment(),check=True)
        require(not args.out.exists(), 'dry-run created output')


if __name__=='__main__':
    main()
