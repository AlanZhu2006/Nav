"""Frozen paired navigation with three memory implementations and one controller.

The existing Habitat executor, NavDP and SP/LightGlue/PnP are reused. Each task
owns one history/memory configuration and runs both preconstructed query roles.
Only RGB crosses the memory HTTP boundary; evaluator odometry is omitted.
"""
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from MemNavData import table1_repaired_eval as existing
from MemNavData.habitat_executor_audit import dump,sha
from MemNavData.run_repaired_fullmono_local import (
    MEM_PY,HAB_PY,MEM_CKPT,NAV_CKPT,LINGBOT,execution_environment,open_port,run_child)

HERE=Path(__file__).resolve()
MODES=('legacy','native_interval7','connected_reciprocal')
KV_STORAGES=('reader_precision','int8_storage','lossless_bf16')
DEVELOPMENT_SCENES=('LT9Jq6dN3Ea','Qpor2mEya8F','8WUmhLawc2A','PuKPg4mmafe',
    'V2XKFyX4ASd','vBMLrTe4uLA','SiKqEZx7Ejt','5jp3fCRSRjc')


def freeze(path, local=False):
    if path.exists():
        raise FileExistsError(path)
    populations=existing.POPULATIONS
    if local:
        from MemNavData.run_cec_stream_depth_closed_loop import BENCH
        populations={'local_interface':(str(BENCH),sha(BENCH/'manifest.json'),4,4)}
    cells=[]
    for dataset,(directory,digest,count,scenes) in populations.items():
        directory=Path(directory)
        assert sha(directory/'manifest.json')==digest
        rows=json.loads((directory/'manifest.json').read_text())['episodes']
        assert len(rows)==count and len({h['scene'] for h in rows})==scenes
        for index,h in enumerate(rows):
            cell=dict(index=len(cells),dataset=dataset,controller='navdp',history_index=index,
                benchmark=str(directory),benchmark_sha256=digest,scene=h['scene'],episode=h['episode'],
                role_pairs_sha256=sha(directory/h['scene']/h['episode']/'role_pairs.json'),
                candidate_design_scene=h['scene'] in DEVELOPMENT_SCENES)
            existing.history(cell)
            cells.append(cell)
    dump(path,dict(schema='gem_memory_navigation_v1',cells=cells,modes=MODES,
        max_steps=600,success_radius_m=1.,exec_horizon=8,total_rollouts=len(cells)*len(MODES)*2,
        dataset_status='Existing frozen paper populations; unseen by tonight\'s candidate selection except marked scene overlap',
        primary_comparison='Published legacy canonical-depth GEM vs connected reciprocal memory',
        control='Native FP32-camera interval7 and first-write depth isolates bounded memory from frontend/storage changes',
        history_and_queries_shared=True,simulator_inputs_to_memory=False,
        local_interface_population=local,created_at=time.time()))
    print(json.dumps(dict(histories=len(cells),tasks=len(cells)*len(MODES),rollouts=len(cells)*len(MODES)*2)),flush=True)


@contextmanager
def servers(out, mode, mem_port, nav_port, *, dense_window=None, geometry_storage=None,
            kv_storage=None, memory_server_entrypoint=None):
    if mem_port==nav_port or any(open_port(p) for p in (mem_port,nav_port)):
        raise RuntimeError('Evaluation ports are already occupied')
    dependency=Path(os.environ.get('REPAIRED_DEPENDENCIES',ROOT/'.diagnostics/dependencies/python'))
    glue=Path(os.environ.get('REPAIRED_LIGHTGLUE',ROOT/'.diagnostics/dependencies/LightGlue'))
    intern=Path(os.environ.get('REPAIRED_INTERNNAV',ROOT/'InternNav'))
    shared=f'{ROOT}:{ROOT}/MemNavData:{dependency}:{glue}:{intern}/src/diffusion-policy'
    window = (32 if mode == 'legacy' else 64) if dense_window is None else int(dense_window)
    if window < 1:
        raise ValueError('Dense retention window must be positive')
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        LINGBOT_REPO=str(LINGBOT),LINGBOT_WEIGHTS=str(LINGBOT/'weights/lingbot-map-long.pt'),
        MEMNAV_WINDOW=str(window),MEMNAV_NUM_SCALE='8',MEMNAV_MAX_FRAME_NUM='4096',
        MEMNAV_GROUND_SCALE_MAX='6.0',MEMNAV_GATE_FUSION='complementary',
        MEMNAV_AUX_POSE_CALIBRATION='empirical',MEMNAV_COLLISION_SELECT='1',MEMNAV_REPORT_TO='none',
        NAVDP_DISABLE_VIDEO='1',LINGBOT_POSE_LOG=str(out/'lingbot_pose_readout.jsonl'))
    if dense_window is not None:
        dump(out/'fixed_window_receipt.json',dict(mode=mode,dense_window=window,
            scale_frames=8,max_frame_num=4096,environment={k:env[k] for k in
                ('MEMNAV_WINDOW','MEMNAV_NUM_SCALE','MEMNAV_MAX_FRAME_NUM')}))
    settings=[('memnav',mem_port,
        [MEM_PY,'-u',str(ROOT/'MemNavData/lingbot_pose_diagnostic_server.py'),
        '--host','127.0.0.1','--port',str(mem_port),'--checkpoint',str(MEM_CKPT),
        '--internnav_root',str(intern),'--num_samples','16','--exclude_recent','32','--retrieval','raw',
        '--retrieval_candidate_top_k','32','--retrieval_candidate_min_gap','16',
        '--graph_subgoal_spacing_m','0.0','--graph_subgoal_arrival_m','0.60',
        '--flow_gate','auto' if mode=='legacy' else 'off','--buffer_root',str(out/'buffer'),
        '--certified_relocalization','--certified_reference_depth_source',
        'canonical' if mode=='legacy' else 'online_history','--memory_mechanism',mode,
        '--lightglue_repo',str(glue),'--lightglue_dependency_root',str(dependency),'--lightglue_max_keypoints','2048'],
        dict(env,PYTHONPATH=f'{ROOT}/NavDP/baselines/memnav:{shared}')),
        ('navdp',nav_port,
        [MEM_PY,'-u',str(ROOT/'MemNavData/navdp_depth_raster_audit_server.py'),
        '--port',str(nav_port),'--checkpoint',str(NAV_CKPT),'--depth_source','monocular_sidecar',
        '--require_monocular_depth_transaction','--monocular_depth_url',f'http://127.0.0.1:{mem_port}/monocular_depth_query'],
        dict(env,PYTHONPATH=f'{ROOT}/NavDP/baselines/navdp:{shared}',
            NAVDP_EXECUTION_INPUT_LOG=str(out/'navdp_input_audit.jsonl'),
            NAVDP_DEPTH_RASTER_LOG_ROOT=str(out/'depth_raster_artifacts')))]
    if memory_server_entrypoint is not None:
        entrypoint = Path(memory_server_entrypoint).resolve()
        if not entrypoint.is_file() or entrypoint.parent != ROOT / 'MemNavData':
            raise ValueError('The diagnostic memory entrypoint must belong to this source bundle')
        settings[0][2][2] = str(entrypoint)
    if geometry_storage is not None:
        if mode == 'legacy' or geometry_storage not in ('dense', 'detector_support'):
            raise ValueError('Historical storage comparison requires an episodic writer')
        settings[0][2].extend(['--memory_geometry_storage', geometry_storage])
        dump(out/'archive_storage_receipt.json',dict(mode=mode,
            geometry_storage=geometry_storage,dense_window=window,
            match_algorithm='Original SuperPoint/LightGlue/PnP/certificate'))
    if kv_storage is not None:
        if kv_storage not in KV_STORAGES or mode != 'native_interval7' or window != 64:
            raise ValueError('Explicit SDPA storage requires the native interval7 W64 writer')
        settings[0][2].extend(['--memory_kv_storage', kv_storage])
        dump(out/'kv_storage_receipt.json',dict(mode=mode,kv_storage=kv_storage,
            dense_window=window,reader='Original LingBot SDPA at BF16 read precision'))
    children,handles=[],[]
    try:
        for name,port,command,environment in settings:
            cwd=out/'runtime'/name;cwd.mkdir(parents=True)
            handle=(out/'logs'/f'{name}.log').open('x');handles.append(handle)
            process=subprocess.Popen(command,cwd=cwd,env=environment,stdout=handle,stderr=subprocess.STDOUT)
            children.append(process)
            dump(out/'owned_processes.json',[dict(pid=p.pid,command=p.args) for p in children])
            deadline=time.monotonic()+600
            while not open_port(port):
                if process.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError(name+' failed to start')
                time.sleep(1)
            print('READY',name,process.pid,flush=True)
        yield
    finally:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        for handle in handles:handle.close()


def evaluate():
    from MemNavData.run_habitat_minimal_repair_local import evaluate as run_evaluator

    def query():
        import eval_shared_online_role_pairs as shared
        base=shared.base
        # Endpoint-bearing navigation does not consume executor odometry.
        # Omit it from HTTP for every comparison, including the old memory.
        base.runtime_executor_motion_form=lambda *a,**k:{}
        original_post=base.requests.post
        def observed_post(url,*a,**kwargs):
            forbidden=('executed_translation_m','executed_yaw_rad','executed_forward_m',
                'executed_left_m','executor_local_se2_source')
            data=kwargs.get('data',{})
            assert not any(k in data for k in forbidden),'Evaluator odometry crossed the memory boundary'
            result=original_post(url,*a,**kwargs)
            if url.endswith('/certified_relocalize'):
                result.raise_for_status()
                value=result.json()
                if value.get('pnp',{}).get('status')=='runtime_exception':
                    raise RuntimeError(value['pnp'])
                if any(r.get('error') for r in value.get('ranked_candidates',[])):
                    raise RuntimeError('Correspondence inference failed')
            return result
        base.requests.post=observed_post
        existing.evaluate_query()
    run_evaluator(query_main=query)


def run(args):
    import requests
    plan=json.loads(args.plan.read_text())
    cell=plan['cells'][args.index]
    assert cell['index']==args.index and args.mode in MODES
    existing.history(cell)
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'logs').mkdir()
    sources=[HERE,ROOT/'NavDP/baselines/memnav/policy_agent.py',ROOT/'NavDP/baselines/memnav/memnav_server.py',
        ROOT/'MemNavData/lingbot_pnp_localization.py',ROOT/'MemNavData/certified_relocalization_runtime.py',
        *sorted((ROOT/'NavDP/baselines/memnav/gem').glob('*.py'))]
    hashes={str(p):sha(p) for p in sources}
    memory_options={name:getattr(args,name,None) for name in
        ('dense_window','geometry_storage','kv_storage')}
    dump(args.out/'manifest.json',dict(cell=cell,mode=args.mode,plan=str(args.plan.resolve()),
        plan_sha256=sha(args.plan),sources_sha256=hashes,memory_http_inputs='RGB only',
        memory_options=memory_options,
        unchanged='NavDP, executor, DINO shortlist, SP/LightGlue/PnP/certificate parameters',
        reference_depth_source='canonical' if args.mode=='legacy' else 'online_history'))
    env=dict(execution_environment(),TABLE1_PLAN=str(args.plan.resolve()),TABLE1_INDEX=str(args.index))
    summary=dict(completed=False,cell=cell,mode=args.mode,records=[])
    dump(args.out/'summary.json',summary)
    try:
        with servers(args.out,args.mode,args.mem_port,args.nav_port,**memory_options):
            for role in ('novel','revisit'):
                target=args.out/'evaluation'/role
                command=existing.command(cell,role,'cec',target,args.mem_port,args.nav_port)
                command[2]=str(HERE)
                print('START',args.mode,cell['scene'],role,flush=True)
                seconds=run_child(command,args.out/'logs'/f'{role}.log',environment=env)
                resource=requests.get(f'http://127.0.0.1:{args.mem_port}/memory_status',timeout=30)
                resource.raise_for_status();dump(target/'memory_resources.json',resource.json())
                measurements=json.loads((target/'terminal_measurements.json').read_text())
                assert len(measurements)==1
                row=dict(measurements[0],role=role,mode=args.mode,scene=cell['scene'],
                    wall_seconds=seconds,directory=str(target))
                summary['records'].append(row)
                dump(args.out/'summary.json',summary)
                print('DONE',args.mode,role,'SR',row['reached'],'steps',row['steps'],flush=True)
        assert all(sha(p)==h for p,h in hashes.items())
        summary['completed']=True
        dump(args.out/'summary.json',summary)
    except BaseException as error:
        dump(args.out/'failure.json',dict(type=type(error).__name__,error=str(error)))
        raise


def main():
    action=sys.argv.pop(1)
    if action=='eval':
        evaluate();return
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--local',action='store_true')
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--mode',choices=MODES,default='connected_reciprocal')
    parser.add_argument('--mem-port',type=int,default=21730)
    parser.add_argument('--nav-port',type=int,default=21731)
    parser.add_argument('--dense-window',type=int,
        help='Explicit neural context window; optimized KV storage requires 64')
    parser.add_argument('--geometry-storage',choices=('dense','detector_support'),
        help='Historical geometry representation for an episodic writer')
    parser.add_argument('--kv-storage',choices=KV_STORAGES,
        help='Explicit KV storage for native_interval7/W64; omitted keeps native storage')
    args=parser.parse_args()
    if action=='freeze':freeze(args.out,args.local)
    elif action=='run':run(args)
    else:parser.error('Expected freeze, run, or eval')


if __name__=='__main__':main()
