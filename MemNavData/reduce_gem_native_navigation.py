"""Verify paired GPU/configuration provenance and reduce the fixed population."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
from run_gem_native_navigation import ARMS, check_plan, load, sha
from verify_gem_memory_navigation import clustered_difference


def audit_pair(plan_path, folder):
    plan = check_plan(plan_path)
    manifest, completion = load(folder / 'pair_manifest.json'), load(folder / 'completion.json')
    index = manifest['index']
    assert plan['cells'][index] == manifest['cell']
    assert manifest['plan_sha256'] == completion['plan_sha256'] == sha(plan_path)
    assert completion['completed'] and completion['index'] == index
    assert not (folder / 'failure.json').exists()
    shift = index % 3
    order = list(ARMS[shift:] + ARMS[:shift])
    assert manifest['order'] == [a['name'] for a in order]
    assert [r['arm'] for r in completion['arms']] == order
    rows = []
    sources = {str(folder / name):sha(folder / name) for name in ('pair_manifest.json','completion.json')}
    for arm, receipt in zip(order, completion['arms']):
        target = folder / arm['name']
        identity = load(target / 'native_arm_receipt.json')
        window, binding = load(target / 'fixed_window_receipt.json'), load(target / 'paired_gpu_binding.json')
        assert identity['completed'] and identity['index'] == index and identity['arm'] == arm
        assert identity['plan_sha256'] == sha(plan_path)
        assert identity['fixed_window_sha256'] == sha(target / 'fixed_window_receipt.json')
        assert identity['gpu_binding_sha256'] == sha(target / 'paired_gpu_binding.json')
        assert identity['summary_sha256'] == sha(target / 'summary.json')
        assert receipt['receipt_sha256'] == sha(target / 'native_arm_receipt.json')
        assert window['mode'] == arm['mode'] and window['dense_window'] == arm['dense_window']
        assert window['environment']['MEMNAV_WINDOW'] == str(arm['dense_window'])
        assert all(binding[k] == completion['gpu'][k] for k in (
            'gpu_uuid','hostname','slurm_job_id','slurm_array_task_id'))
        assert binding['owned_processes_sha256'] == sha(target / 'owned_processes.json')
        assert all(v == [binding['gpu_uuid']] for v in binding['owned_pid_gpu'].values())
        assert load(folder / (arm['name'] + '_exit.json'))['exit_code'] == 0
        assert load(folder / (arm['name'] + '_audit_exit.json'))['exit_code'] == 0
        audit = load(target / 'independent_verification.json')
        assert receipt['independent_verification_sha256'] == sha(target / 'independent_verification.json')
        assert audit['verified'] and audit['cell'] == manifest['cell'] and audit['mode'] == arm['mode']
        assert audit['plan_sha256'] == plan['population_plan_sha256']
        assert audit['manifest_sha256'] == sha(target / 'manifest.json')
        assert audit['summary_sha256'] == sha(target / 'summary.json')
        assert audit['verifier_sha256'] == sha(ROOT / 'MemNavData/verify_gem_memory_navigation.py')
        assert all(sha(p) == h for p,h in audit['verifier_dependencies'].items())
        assert all(sha(p) == h for p,h in audit['artifact_sha256'].items())
        backend_manifest = load(target / 'manifest.json')
        assert all(sha(p) == h for p,h in backend_manifest['sources_sha256'].items())
        assert backend_manifest['mode'] == arm['mode']
        for record in audit['records']:
            assert record['index'] == index and record['mode'] == arm['mode']
            rows.append(dict(record, configuration=arm['name'], dense_window=arm['dense_window'],
                paired_gpu_uuid=binding['gpu_uuid'], paired_hostname=binding['hostname']))
        for name in ('native_arm_receipt.json','fixed_window_receipt.json',
                'paired_gpu_binding.json','independent_verification.json'):
            sources[str(target / name)] = sha(target / name)
    assert len(rows) == 6
    for role in ('novel','revisit'):
        same_role = [r for r in rows if r['role'] == role]
        assert len(same_role) == 3
        for key in ('goal_sha256','first_query_rgb_sha256','history_trace_sha256'):
            assert len({r[key] for r in same_role}) == 1, key
        assert all(r['goal_xz'] == same_role[0]['goal_xz'] for r in same_role)
    return rows, sources


def run(args):
    plan = check_plan(args.plan)
    rows, states, sources = [], [], {}
    assert all(not p.is_dir() or p.name.isdigit() for p in args.tasks.iterdir()), 'Unexpected task directory'
    for cell in plan['cells']:
        folder = args.tasks / str(cell['index'])
        if (folder / 'failure.json').exists():
            states.append(dict(index=cell['index'], status='failed', failure=load(folder / 'failure.json')))
        elif not (folder / 'completion.json').exists():
            states.append(dict(index=cell['index'], status='running' if folder.exists() else 'not_started'))
        else:
            verified, inputs = audit_pair(args.plan, folder)
            rows.extend(verified); sources.update(inputs)
            states.append(dict(index=cell['index'], status='verified'))
    complete = len(rows) == plan['total_rollouts']
    groups, comparisons = {}, {}
    for dataset in ('all','hm3d','mp3d'):
        for role in ('novel','revisit'):
            selected = [r for r in rows if r['role'] == role and (dataset == 'all' or r['dataset'] == dataset)]
            key = dataset + '/' + role
            groups[key] = {}
            for arm in ARMS:
                chosen = [r for r in selected if r['configuration'] == arm['name']]
                groups[key][arm['name']] = dict(n=len(chosen), successes=sum(r['reached'] for r in chosen),
                    sr=float(np.mean([r['reached'] for r in chosen])) if chosen else None,
                    spl=float(np.mean([r['spl'] for r in chosen])) if chosen else None,
                    any_takeover=sum(r['takeover_plans'] > 0 for r in chosen))
            paired = []
            for index in sorted({r['index'] for r in selected}):
                pair = {r['configuration']:r for r in selected if r['index'] == index}
                assert set(pair) == {a['name'] for a in ARMS}
                paired.append(pair)
            comparisons[key] = {}
            for control in ('legacy','native64'):
                comparisons[key]['native16_vs_' + control] = dict(
                    gained=sum(p['native16']['reached'] and not p[control]['reached'] for p in paired),
                    lost=sum(p[control]['reached'] and not p['native16']['reached'] for p in paired),
                    sr=clustered_difference(paired, 'native16', control, 'reached'),
                    spl=clustered_difference(paired, 'native16', control, 'spl'))
    report = dict(complete=complete, planned_pairs=70, verified_pairs=len(rows)//6,
        rollouts=len(rows), plan_sha256=sha(args.plan), states=states, groups=groups,
        comparisons=comparisons, rows=rows, source_sha256=sources,
        reducer_sha256=sha(__file__),
        scope='Same-GPU fixed-window comparison on the full previously used population; scene-cluster bootstrap is descriptive, not a prespecified noninferiority test')
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(args.out)
    print(json.dumps(dict(complete=complete, verified_pairs=len(rows)//6, rollouts=len(rows), groups=groups)), flush=True)
    if args.require_complete and not complete:
        raise RuntimeError('The frozen 70-pair experiment has not completed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    run(parser.parse_args())
