"""Complete paired online-storage A/B/A navigation, using the original executor."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
STORAGE=('dense','detector_support')


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def check(plan):
    assert plan['storage_arms']==list(STORAGE) and plan['server_dense_window']==64
    assert plan['episodes']==8 and len(plan['cells'])==4
    assert sha(plan['api_proof'])==plan['api_proof_sha256']
    assert sha(plan['parent_continuous_plan'])==plan['parent_continuous_plan_sha256']
    assert all(sha(ROOT/p)==h for p,h in plan['core_sha256'].items())
    assert all(sha(ROOT/p)==h for p,h in plan['launcher_sha256'].items())


def freeze(args):
    from MemNavData.run_gem_memory_continuous import SCHEMA

    parent=load(args.source)
    proof=load(args.proof)
    assert parent['schema']==SCHEMA and len(parent['cells'])==4
    assert proof['passed'] and load(proof['gpu_result'])['passed']
    assert sha(proof['gpu_result'])==proof['gpu_result_sha256']
    plan=dict(parent,parent_continuous_plan=str(args.source.resolve()),
        parent_continuous_plan_sha256=sha(args.source),modes=['native_interval7'],episodes=8,
        storage_arms=list(STORAGE),server_dense_window=64,
        api_proof=str(args.proof.resolve()),api_proof_sha256=sha(args.proof),
        core_sha256=proof['core_sha256'],
        launcher_sha256={p:sha(ROOT/p) for p in (
            'MemNavData/run_gem_support_continuous.py','MemNavData/run_gem_memory_continuous.py',
            'MemNavData/run_gem_memory_navigation.py','MemNavData/run_gem_native_navigation.py',
            'MemNavData/verify_gem_memory_continuous.py')},
        scope='All four existing development histories, both storage representations, same W64 production writer and original SP/LG/PnP/controller. Fresh services per arm on the same actual GPU. No goal-dependent storage/window choice; not new held-out scenes.',
        ordering='Rotate dense/support order by history index modulo2; preserve failed chains and audit every completed run',
        created_at=time.time())
    check(plan)
    save(args.out,plan)
    print(json.dumps(dict(histories=4,episodes=8,maximum_legs=24,plan_sha256=sha(args.out))),flush=True)


def task(args):
    from MemNavData import run_gem_memory_continuous as base
    from MemNavData.run_gem_memory_navigation import servers
    from MemNavData.run_gem_native_navigation import gpu_binding

    plan=load(args.plan)
    check(plan)
    assert args.storage in STORAGE

    @contextmanager
    def configured(out,mode,mem_port,nav_port):
        with servers(out,mode,mem_port,nav_port,dense_window=64,geometry_storage=args.storage):
            yield
            save(out/'paired_gpu_binding.json',dict(gpu_binding(out),
                capture_phase='after_rollouts_before_service_shutdown'))

    base.MODES=('native_interval7',)
    base.servers=configured
    args.mode='native_interval7'
    args.prepare_only=False
    base.run(args)
    check(plan)
    for record in load(args.out/'summary.json')['chain']['leg_records']:
        status=load(args.out/'evaluation'/f"leg_{record['stage']}"/'memory_resources.json')['memory']
        assert status['module']=='native_interval7'
        assert status.get('geometry_storage','dense')==args.storage
        if args.storage=='detector_support':
            assert status['support_detector_milliseconds']>0
            assert status['support_validated_match_calls']>0
    save(args.out/'storage_arm_receipt.json',dict(completed=True,storage=args.storage,
        history_index=args.index,plan_sha256=sha(args.plan),
        actual_storage_sha256=sha(args.out/'archive_storage_receipt.json'),
        actual_gpu_sha256=sha(args.out/'paired_gpu_binding.json'),summary_sha256=sha(args.out/'summary.json')))


def driver(args):
    from MemNavData.run_repaired_fullmono_local import MEM_PY,HAB_PY

    plan=load(args.plan)
    check(plan)
    args.out.mkdir(parents=True,exist_ok=False)
    save(args.out/'driver.json',dict(pid=os.getpid(),plan=str(args.plan.resolve()),
        plan_sha256=sha(args.plan),started_at=time.time()))
    tasks=[]
    for index in range(4):
        order=STORAGE if index%2==0 else STORAGE[::-1]
        for storage in order:
            target=args.out/f'{index}_{storage}'
            command=[MEM_PY,'-u',str(Path(__file__).resolve()),'task','--plan',str(args.plan.resolve()),
                '--index',str(index),'--storage',storage,'--out',str(target.resolve()),
                '--mem-port',str(args.mem_port),'--nav-port',str(args.nav_port)]
            with (args.out/f'{index}_{storage}.log').open('x') as log:
                code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
            row=dict(index=index,storage=storage,directory=str(target.resolve()),exit_code=code,command=command)
            if code==0:
                audit=[HAB_PY,str(ROOT/'MemNavData/verify_gem_memory_continuous.py'),str(target.resolve())]
                with (args.out/f'{index}_{storage}_audit.log').open('x') as log:
                    row['audit_exit_code']=subprocess.call(audit,stdout=log,stderr=subprocess.STDOUT)
                row['audit_command']=audit
            save(args.out/f'{index}_{storage}_exit.json',row)
            tasks.append(row)
            print(json.dumps({k:v for k,v in row.items() if not k.endswith('command')}),flush=True)
    check(plan)
    save(args.out/'execution_complete.json',dict(execution_complete=True,tasks=tasks,
        plan_sha256=sha(args.plan),completed_at=time.time()))
    reduce(SimpleNamespace(plan=args.plan,out=args.out))


def reduce(args):
    plan=load(args.plan)
    check(plan)
    completion=load(args.out/'execution_complete.json')
    assert completion['plan_sha256']==sha(args.plan) and len(completion['tasks'])==8
    rows=[]
    for task in completion['tasks']:
        assert task['exit_code']==task['audit_exit_code']==0,task
        directory=Path(task['directory'])
        arm=load(directory/'storage_arm_receipt.json')
        assert arm['completed'] and arm['plan_sha256']==sha(args.plan)
        for field,filename in (('actual_storage_sha256','archive_storage_receipt.json'),
                ('actual_gpu_sha256','paired_gpu_binding.json'),('summary_sha256','summary.json')):
            assert arm[field]==sha(directory/filename)
        verification=load(directory/'independent_verification.json')
        assert verification['verified']
        assert verification['plan_sha256']==sha(args.plan)
        assert all(sha(p)==h for p,h in verification['artifact_sha256'].items())
        summary=load(directory/'summary.json')
        legs=summary['chain']['leg_records']
        rows.append(dict(index=task['index'],storage=task['storage'],directory=str(directory),
            verification_sha256=sha(directory/'independent_verification.json'),
            gpu=load(directory/'paired_gpu_binding.json')['gpu_uuid'],
            completed_legs=len(legs),successes=sum(r['measurement']['reached'] for r in legs),
            chain_success=len(legs)==3 and all(r['measurement']['reached'] for r in legs),
            steps=sum(r['measurement']['steps'] for r in legs)))
    paired=[]
    for index in range(4):
        a,b=[next(r for r in rows if r['index']==index and r['storage']==s) for s in STORAGE]
        assert a['gpu']==b['gpu']
        paths=[Path(r['directory']) for r in (a,b)]
        actions_equal=(paths[0]/'evaluation/executor_actions.jsonl').read_bytes()==(paths[1]/'evaluation/executor_actions.jsonl').read_bytes()
        poses=[[json.loads(line) for line in (p/'lingbot_pose_readout.jsonl').read_text().splitlines()] for p in paths]
        pose_fields=('frame_idx','image_sha256','camera_pose9')
        # Preserve complete source rows as evidence; equality is reported,
        # never required to hide a navigation divergence.
        common_pose_keys=set(poses[0][0]) & set(poses[1][0])
        assert set(pose_fields).issubset(common_pose_keys)
        compared=list(pose_fields)
        geometry_equal=[{k:r[k] for k in compared} for r in poses[0]]==[{k:r[k] for k in compared} for r in poses[1]]
        paired.append(dict(index=index,actions_byte_equal=actions_equal,
            observed_sequence_equal=geometry_equal,compared_pose_fields=compared,
            dense_chain_success=a['chain_success'],support_chain_success=b['chain_success']))
    save(args.out/'independent_reduction.json',dict(complete=True,episodes=8,rows=rows,pairs=paired,
        plan_sha256=sha(args.plan),execution_sha256=sha(args.out/'execution_complete.json'),
        scope=plan['scope'],totals={s:dict(chains=sum(r['chain_success'] for r in rows if r['storage']==s),
            legs=sum(r['completed_legs'] for r in rows if r['storage']==s),
            successful_legs=sum(r['successes'] for r in rows if r['storage']==s)) for s in STORAGE}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run','task','reduce'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--proof',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--index',type=int)
    parser.add_argument('--storage',choices=STORAGE)
    parser.add_argument('--mem-port',type=int,default=23240)
    parser.add_argument('--nav-port',type=int,default=23241)
    args=parser.parse_args()
    {'freeze':freeze,'run':driver,'task':task,'reduce':reduce}[args.action](args)
