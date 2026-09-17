"""Audit a completed paired gate and submit the remaining frozen histories.

Scheduler polling and bookkeeping only: no inference on the login node. The
gate's outcomes never determine whether the rest of the population is run.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def run(args):
    assert subprocess.check_output(['id','-un'],text=True).strip()=='yz11502'
    assert args.kind in ('native','support') and args.gate.isdigit()
    sys.path[:0]=[str(args.bundle),str(args.bundle/'MemNavData')]
    launch=importlib.import_module('run_gem_'+args.kind+'_navigation')
    reducer=importlib.import_module('reduce_gem_'+args.kind+'_navigation')
    plan_path=args.run/'plan.json'
    plan=launch.check_plan(plan_path)
    assert plan['gate_index']==14 and len(plan['cells'])==70
    submission=load(args.run/'gate_submission.json')
    assert submission['job_id']==args.gate and submission['plan_sha256']==sha(plan_path)
    assert submission['bundle']==str(args.bundle)
    save(args.run/'followup_process.json',dict(pid=os.getpid(),kind=args.kind,gate=args.gate,
        source=str(Path(__file__).resolve()),source_sha256=sha(__file__),started_at=time.time()))
    with (args.run/'followup_scheduler.jsonl').open('x',buffering=1) as events:
        while True:
            command=['sacct','-X','--noheader','--parsable2','-j',args.gate+'_14',
                '--format=JobID,State%40,ElapsedRaw']
            try:
                raw=subprocess.check_output(command,text=True,timeout=30)
                rows=[line.strip().split('|') for line in raw.splitlines() if line.strip()]
                row=next((r for r in rows if r[0]==args.gate+'_14'),None)
                state=row[1].strip().split()[0].rstrip('+') if row else 'UNOBSERVED'
                events.write(json.dumps(dict(time=time.time(),state=state,row=row))+'\n')
            except (subprocess.TimeoutExpired,subprocess.CalledProcessError) as error:
                events.write(json.dumps(dict(time=time.time(),observation_error=str(error)))+'\n')
                time.sleep(30)
                continue
            if state=='COMPLETED':
                elapsed=int(row[2]);break
            if state in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL',
                    'PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'):
                raise RuntimeError('The fixed gate terminated: '+state)
            time.sleep(30)
    folder=args.run/'tasks/14'
    rows,artifacts=reducer.audit_pair(plan_path,folder)
    assert len(rows)==6
    role_times=[]
    for arm in plan['arms']:
        summary=load(folder/arm['name']/'summary.json')
        assert summary['completed'] and len(summary['records'])==2
        for r in summary['records']:
            assert r['steps']>0 and r['wall_seconds']>0
            role_times.append(dict(arm=arm['name'],role=r['role'],steps=r['steps'],
                seconds=r['wall_seconds'],max_step_extrapolated_seconds=r['wall_seconds']*max(1.,600/r['steps'])))
    overhead=max(0.,elapsed-sum(r['seconds'] for r in role_times))
    bound=1.5*(overhead+sum(r['max_step_extrapolated_seconds'] for r in role_times))
    minutes=max(15,15*math.ceil(bound/900))
    walltime=f'{minutes//60:02d}:{minutes%60:02d}:00'
    budget=dict(gate_elapsed_seconds=elapsed,role_times=role_times,overhead_seconds=overhead,
        rule='Extrapolate every observed role to the frozen600-step cap, add measured startup/audit overhead, multiply by1.5, round up to15minutes',
        requested_walltime=walltime,margin=1.5,requested_seconds=minutes*60)
    upper=args.kind.upper()
    command=['sbatch','--parsable','--partition=a100_tandon','--account=torch_pr_769_tandon_advanced',
        '--qos=gpu48','--gres=gpu:1','--cpus-per-task=12','--mem=128G','--time='+walltime,
        '--array=0-13,15-69%8','--comment=gem_'+args.kind+'_'+sha(plan_path)[:12],
        '--export=ALL,REPAIRED_BUNDLE='+str(args.bundle)+',GEM_'+upper+'_PLAN='+str(plan_path)
        +',GEM_'+upper+'_RUN='+str(args.run)+',EXPECTED_'+upper+'_PLAN_SHA='+sha(plan_path),
        str(args.bundle/'MemNavData'/('slurm_gem_'+args.kind+'_navigation.sbatch'))]
    receipt=dict(command=command,gate_job=args.gate,gate_independently_verified=True,
        gate_artifact_sha256=artifacts,plan_sha256=sha(plan_path),remaining_histories=69,
        remaining_rollouts=414,selection='Every remaining frozen index regardless of gate SR',
        time_budget=budget,coordinator_sha256=sha(__file__),time=time.time())
    save(args.run/'full_submission_intent.json',receipt)
    # A submission intent without a receipt is deliberately not auto-retried;
    # inspect Slurm before any retry to avoid duplicate paid jobs.
    job=subprocess.check_output(command,text=True).strip()
    receipt['job_id']=job
    save(args.run/'full_submission.json',receipt)
    print(json.dumps(dict(submitted=job,remaining_histories=69,walltime=walltime,
        kind=args.kind,gate_verified=True)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--kind',choices=['native','support'],required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--gate',required=True)
    args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        save(args.run/'followup_failure.json',dict(type=type(error).__name__,error=str(error),time=time.time()))
        raise
