"""Audit monitored fixed-RGB resource runs without selecting favorable trials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def distribution(values):
    a = np.asarray(values, float)
    if not len(a):
        return dict(n=0, mean=None, median=None, p95=None, min=None, max=None)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
        p95=float(np.quantile(a, .95)), min=float(a.min()), max=float(a.max()))


def audit_task(plan_path, task):
    plan = load(plan_path)
    root = plan_path.parent/'tasks'/str(task['index'])
    result, process = load(root/'result.json'), load(root/'process.json')
    case = next(c for c in plan['cases'] if c['id'] == task['case'])
    assert result['completed'] and result['task'] == task == process['task']
    assert result['plan_sha256'] == process['plan_sha256'] == sha(plan_path)
    assert load(root/'process_exit.json') == dict(exit_code=0, task=task)
    assert not (root/'failure.json').exists()
    assert result['source_sha256'] == plan['source_sha256']
    assert [r['frame'] for r in result['writes']] == list(range(case['frame_count']))
    assert result['final_status']['frames'] == case['frame_count']
    assert [q['id'] for q in result['queries']] == [q['id'] for q in case['queries']]
    assert all(len(q['cached_ms']) == 20 for q in result['queries'])
    events = [json.loads(line) for line in (root/'events.jsonl').open()]
    samples = [json.loads(line) for line in (root/'resources.jsonl').open()]
    write_events = [r for r in events if r['phase'] == 'write']
    assert [r['frame'] for r in write_events] == list(range(case['frame_count']))
    assert events[-1]['phase'] == 'done'
    assert all(b['time'] >= a['time'] for a, b in zip(events, events[1:]))
    assert all(r['pid'] == process['pid'] for r in samples)
    assert samples[0]['time'] <= events[0]['time'] <= events[-1]['time'] <= samples[-1]['time'] + 2.
    assert all(b['time'] >= a['time'] for a, b in zip(samples, samples[1:]))
    nvml = [r for r in samples if 'sampled_gpu_bytes' in r]
    assert nvml and max(r['sampled_gpu_bytes'] for r in nvml) > 0
    start, end = write_events[0]['time'], next(r['time'] for r in events if r['phase'] == 'dense')
    during_write = [r for r in samples if start <= r['time'] < end]
    assert during_write
    costs = [r['ms'] for r in result['writes']]
    assert np.isfinite(costs).all() and min(costs) > 0
    # All-call statistics retain initial buffering, scale freeze and rebases.
    transitions = [r['ms'] for r in result['writes'] if task['mode'] == 'connected_reciprocal'
        and r['frame'] >= 64 and (r['frame']-64)%56 == 0]
    row = dict(task=task, frames=case['frame_count'], gpu=result['gpu'],
        write_seconds=result['write_seconds'], write_ms=distribution(costs),
        transition_write_ms=distribution(transitions),
        initialization_write_ms=distribution(costs[:8]), calibration_frame39_write_ms=costs[39],
        gpu_peak_allocated_gib=result['service_peak_allocated_bytes']/2**30,
        gpu_peak_reserved_gib=result['service_peak_reserved_bytes']/2**30,
        sampled_gpu_peak_gib=max(r['sampled_gpu_bytes'] for r in nvml)/2**30,
        cpu_rss_process_peak_gib=max(r['rss_bytes'] for r in samples)/2**30,
        cpu_rss_write_peak_gib=max(r['rss_bytes'] for r in during_write)/2**30,
        cpu_rss_write_final_gib=during_write[-1]['rss_bytes']/2**30,
        sample_gap_seconds_max=max(b['time']-a['time'] for a,b in zip(samples,samples[1:])),
        gpu_sample_gap_seconds_max=max(b['time']-a['time'] for a,b in zip(nvml,nvml[1:])),
        depth_archive_gib=result['final_status'].get('depth_archive_bytes',0)/2**30,
        dense_first_ms=result['dense_first_ms'], dense_cached_ms=distribution(result['dense_cached_ms']),
        queries=[dict(id=q['id'],accepted=q['first_result']['accepted'], first_ms=q['first_ms'],
            cached_ms=distribution(q['cached_ms'])) for q in result['queries']],
        checkpoints=result['checkpoints'], artifacts_sha256={str(root/name): sha(root/name)
            for name in ('result.json','process.json','process_exit.json','events.jsonl','resources.jsonl')})
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('plan', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    plan = load(args.plan)
    states, rows = [], []
    for task in plan['tasks']:
        root = args.plan.parent/'tasks'/str(task['index'])
        if (root/'failure.json').exists():
            states.append(dict(task=task, status='failed', error=load(root/'failure.json')))
        elif not (root/'process_exit.json').exists():
            states.append(dict(task=task, status='running' if root.exists() else 'not_started'))
        elif load(root/'process_exit.json')['exit_code'] != 0:
            states.append(dict(task=task, status='process_failed', exit=load(root/'process_exit.json')))
        else:
            rows.append(audit_task(args.plan, task))
            states.append(dict(task=task, status='verified'))
    groups = {}
    for case in plan['cases']:
        for mode in ('legacy','native_interval7','connected_reciprocal'):
            chosen = [r for r in rows if r['task']['case']==case['id'] and r['task']['mode']==mode]
            if not chosen:
                continue
            assert len({r['task']['repeat'] for r in chosen}) == len(chosen)
            groups[case['id']+'/'+mode] = dict(repeats=len(chosen),
                metrics={key:distribution([r[key] for r in chosen]) for key in (
                    'write_seconds','gpu_peak_allocated_gib','gpu_peak_reserved_gib',
                    'sampled_gpu_peak_gib','cpu_rss_process_peak_gib','cpu_rss_write_peak_gib',
                    'cpu_rss_write_final_gib','depth_archive_gib')},
                queries={q['id']:dict(first_ms=distribution([next(x for x in r['queries'] if x['id']==q['id'])['first_ms'] for r in chosen]),
                    cached_median_ms=distribution([next(x for x in r['queries'] if x['id']==q['id'])['cached_ms']['median'] for r in chosen]))
                    for q in case['queries']})
    complete = len(rows) == len(plan['tasks'])
    result = dict(complete=complete, plan_sha256=sha(args.plan), planned_tasks=len(plan['tasks']),
        verified_tasks=len(rows), states=states, rows=rows, groups=groups,
        interpretation='Complete preplanned 3-repeat cost experiment' if complete else 'INCOMPLETE cost experiment; trial counts shown for each group',
        timing='Write milliseconds start with JPEG bytes at the memory API and include decoding, neural inference, sync and archival. Whole-write seconds also include input-file reading and SHA checks.',
        allocation='Torch CUDA peaks reset after model initialization; sampled process GPU usage and CPU RSS also cover loading. NVML sampling can miss transient peaks.',
        query='Real complete-history first query then 20 cached queries; per-query acceptance and candidate geometry may differ across modes.',
        reducer_sha256=sha(__file__))
    (args.plan.parent/'independent_reduction.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','planned_tasks','verified_tasks')}),flush=True)
    if args.require_complete and not complete:
        raise RuntimeError('Preplanned resource experiment has not completed')


if __name__ == '__main__':
    main()
