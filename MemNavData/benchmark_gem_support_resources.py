"""Fixed W64, dense versus detector-support storage, complete paired costs."""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from MemNavData import benchmark_gem_native_budget_resources as benchmark
from MemNavData.reduce_gem_native_budget_resources import audit_task,distribution,sha,load
from MemNavData.verify_gem_support_online import semantic_result

STORAGE=('dense','detector_support')


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def freeze(args):
    parent=load(args.source)
    assert len(parent['cases'])==2 and len(parent['tasks'])==12
    assert {c['frame_count'] for c in parent['cases']}=={234,1948}
    assert sha(parent['source_input'])==parent['source_input_sha256']
    proof=load(args.proof)
    assert proof['passed'] and all(sha(ROOT/p)==h for p,h in proof['core_sha256'].items())
    args.out.mkdir(parents=True,exist_ok=False)
    tasks=[]
    for repeat in range(3):
        for case in parent['cases']:
            for storage in (STORAGE if repeat%2==0 else STORAGE[::-1]):
                tasks.append(dict(index=len(tasks),case=case['id'],repeat=repeat,
                    mode='native_interval7',dense_window=64,geometry_storage=storage))
    files=[ROOT/p for p in proof['core_sha256']]+[
        Path(__file__),ROOT/'MemNavData/benchmark_gem_native_budget_resources.py',
        ROOT/'MemNavData/reduce_gem_native_budget_resources.py',
        ROOT/'MemNavData/verify_gem_support_online.py']
    plan=dict(parent,cases=parent['cases'],tasks=tasks,
        parent_resource_plan=str(args.source.resolve()),parent_resource_plan_sha256=sha(args.source),
        api_proof=str(args.proof.resolve()),api_proof_sha256=sha(args.proof),
        source_sha256={str(p):sha(p) for p in files},
        protocol='Fixed native W64; two historical geometry representations; 2 preselected history lengths x 2 storage forms x 3 fresh processes; alternating storage order by repetition. Original real matcher, scale calibration and current dense readout.',
        primary_costs=['historical_depth_and_confidence_disk_bytes','all_history_write_seconds',
            'Torch_peak_GPU_bytes','sampled_process_GPU_bytes','CPU_RSS','first_query_ms','cached_query_ms'],
        expected_invariance='Matching/PnP/acceptance/bearing fields compared across every repeat and storage form; no old window results substituted',
        created_at=time.time())
    save(args.out/'plan.json',plan)
    print(json.dumps(dict(tasks=12,plan_sha256=sha(args.out/'plan.json'))),flush=True)


def reduce(args):
    plan=load(args.plan)
    assert len(plan['tasks'])==12
    assert all(sha(p)==h for p,h in plan['source_sha256'].items())
    execution=load(args.plan.parent/'execution_complete.json')
    assert execution['execution_complete'] and execution['tasks']==12
    assert execution['plan_sha256']==sha(args.plan)
    rows,results=[],{}
    for task in plan['tasks']:
        row=audit_task(args.plan,task)
        result=load(args.plan.parent/'tasks'/str(task['index'])/'result.json')
        assert task['dense_window']==64 and task['geometry_storage'] in STORAGE
        assert result['final_status'].get('geometry_storage','dense')==task['geometry_storage']
        if task['geometry_storage']=='detector_support':
            assert result['final_status']['support_validated_match_calls']>0
            assert result['final_status']['support_detector_milliseconds']>0
        rows.append(row)
        results[task['index']]=result
    metrics=('write_seconds','gpu_peak_allocated_gib','sampled_gpu_peak_gib',
        'cpu_rss_process_peak_gib','cpu_rss_write_final_gib','depth_archive_gib')
    groups,parity={},[]
    for case in plan['cases']:
        reference=None
        for storage in STORAGE:
            chosen=[r for r in rows if r['task']['case']==case['id'] and r['task']['geometry_storage']==storage]
            assert sorted(r['task']['repeat'] for r in chosen)==[0,1,2]
            groups[case['id']+'/'+storage]=dict(frames=case['frame_count'],repeats=3,geometry_storage=storage,
                metrics={k:distribution([r[k] for r in chosen]) for k in metrics},
                queries={q['id']:dict(first_ms=distribution([next(v for v in r['queries'] if v['id']==q['id'])['first_ms'] for r in chosen]),
                    cached_median_ms=distribution([next(v for v in r['queries'] if v['id']==q['id'])['cached_ms']['median'] for r in chosen]))
                    for q in case['queries']})
            for row in chosen:
                current=[dict(id=q['id'],result=semantic_result(q['first_result'])) for q in results[row['task']['index']]['queries']]
                if reference is None: reference=current
                parity.append(dict(task=row['task'],first_readout_exact=current==reference))
    result=dict(complete=True,planned_tasks=12,verified_tasks=12,rows=rows,groups=groups,
        readout_parity=parity,all_first_readouts_exact=all(r['first_readout_exact'] for r in parity),
        plan_sha256=sha(args.plan),execution_sha256=sha(args.plan.parent/'execution_complete.json'),
        scope=plan['protocol'],gpu_scope=plan['gpu_scope'],sampling=plan['sampling'])
    save(args.plan.parent/'independent_reduction.json',result)
    lines=['# 在线历史几何表示：完整资源对照','',plan['protocol'],'',
        '|历史帧数|存储|全部写入 s|Torch 峰值 GiB|进程 GPU 采样峰值 GiB|历史几何 GiB|',
        '|---:|---|---:|---:|---:|---:|']
    for group in groups.values():
        m=group['metrics']
        lines.append(f"|{group['frames']}|{group['geometry_storage']}|{m['write_seconds']['mean']:.3f}|{m['gpu_peak_allocated_gib']['mean']:.3f}|{m['sampled_gpu_peak_gib']['mean']:.3f}|{m['depth_archive_gib']['mean']:.4f}|")
    lines.extend(['',f"全部 12 次结果已核验；完整首次读出是否逐字段一致：{result['all_first_readouts_exact']}。",
        '', '成本范围为实际 MemNavAgent 和 SP/LightGlue、40 帧标定、历史写入、当前稠密读出及目标查询；不包含 NavDP、Habitat 和网络。进程采样可能漏过瞬时峰值。缩小历史几何磁盘字节不能解释为整机显存同比缩小。'])
    (args.plan.parent/'RESULT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(complete=True,verified_tasks=12,all_first_readouts_exact=result['all_first_readouts_exact'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run','reduce'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--proof',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--plan',type=Path)
    args=parser.parse_args()
    if args.action=='freeze': freeze(args)
    elif args.action=='reduce': reduce(args)
    else:
        benchmark.run(SimpleNamespace(plan=args.plan))
        reduce(args)
