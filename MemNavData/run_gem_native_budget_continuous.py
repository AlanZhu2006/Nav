"""Same four continuous goal chains with one fixed native W16 control.

Only the experiment launcher's dense-window argument differs from native64.
The production memory, calibration, matching, controller and executor code
are reused unchanged. No outcome-dependent window choice is permitted.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from MemNavData import run_gem_memory_continuous as continuous
from MemNavData.run_gem_memory_navigation import servers
from MemNavData.habitat_executor_audit import sha,dump


def freeze(source,out):
    plan=continuous.load(source)
    assert plan['schema']==continuous.SCHEMA and len(plan['cells'])==4
    assert tuple(plan['modes'])==('legacy','native_interval7','connected_reciprocal')
    assert all(sha(ROOT/p)==h for p,h in plan['core_sha256'].items())
    plan.update(parent_continuous_plan=str(source.resolve()),parent_continuous_plan_sha256=sha(source),
        modes=['native_interval7'],episodes=4,server_dense_window=16,
        control='One fixed compact native configuration; upstream camera and old special states retained',
        launcher=str(Path(__file__).resolve()),launcher_sha256=sha(__file__),
        scope='Fixed compact-native development control on all four previously used continuous integration histories; same A/B/A goals; not held-out evaluation',
        created_at=time.time())
    dump(out,plan)


def run(args):
    plan=continuous.load(args.plan)
    assert plan['server_dense_window']==16 and plan['episodes']==4
    assert sha(plan['parent_continuous_plan'])==plan['parent_continuous_plan_sha256']
    assert sha(__file__)==plan['launcher_sha256']
    continuous.MODES=('native_interval7',)
    continuous.servers=partial(servers,dense_window=16)
    args.mode='native_interval7'
    continuous.run(args)
    assert sha(__file__)==plan['launcher_sha256']
    if args.prepare_only:
        return
    dump(args.out/'budget_control_receipt.json',dict(completed=True,
        dense_window=16,production_core_unchanged=True,
        plan_sha256=sha(args.plan),launcher_sha256=sha(__file__),
        actual_launch_sha256=sha(args.out/'fixed_window_receipt.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','run'])
    parser.add_argument('--source',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--mem-port',type=int,default=19217)
    parser.add_argument('--nav-port',type=int,default=19218)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.action=='freeze': freeze(args.source,args.out)
    else: run(args)
