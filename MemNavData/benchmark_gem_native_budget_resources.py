"""Paired W16/W64 native-memory costs in fresh monitored processes.

Two preselected history lengths, two fixed native windows, three
repetitions. Per-frame writes include JPEG loading, geometry and archiving.
Sparse queries use the original real matcher; repeated reads retain caches.
NVML (via nvidia-smi) is sampled, not an exact instantaneous peak.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
MODES = ('legacy', 'native_interval7', 'connected_reciprocal')
CASES = ('8WUmhLawc2A_episode_0000', 'LT9Jq6dN3Ea_episode_table3_survey_435')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze(args):
    source = json.loads(args.inputs.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    cases = [next(c for c in source['cases'] if c['id'] == identity) for identity in CASES]
    # No positions/labels are given to the model worker.
    cases = [dict(id=c['id'], seed=c['seed'], frame_count=c['frame_count'],
        rgb_paths=c['rgb_paths'], rgb_sha256=c['rgb_sha256'],
        queries=source['queries'][c['id']]) for c in cases]
    sources = [Path(__file__), ROOT / 'NavDP/baselines/memnav/policy_agent.py',
        ROOT / 'MemNavData/lingbot_pnp_localization.py', ROOT / 'MemNavData/certified_relocalization_runtime.py',
        *sorted((ROOT / 'NavDP/baselines/memnav/gem').glob('*.py'))]
    tasks = []
    # Rotate order by repetition; every task starts a fresh CUDA process.
    for repeat in range(3):
        for case in cases:
            for j in range(2):
                tasks.append(dict(index=len(tasks), case=case['id'], repeat=repeat,
                    mode='native_interval7', dense_window=(16,64)[(j+repeat)%2]))
    save(args.out / 'plan.json', dict(cases=cases, tasks=tasks, seed=20260913,
        source_input=str(args.inputs.resolve()), source_input_sha256=sha(args.inputs),
        source_sha256={str(p): sha(p) for p in sources},
        protocol='Paired native W16/W64, real MemNavAgent, same RGB and queries; 3 fresh-process repetitions, window order alternates by repeat',
        sparse_queries='Existing goal images at complete history, one first read and 20 cached reads each',
        sampling='CPU RSS every 100 ms; per-process nvidia-smi memory every 1 s',
        gpu_scope='All resident MemNavAgent and SP/LightGlue models, temporary calibration/replay, CUDA context',
        created_at=time.time()))
    print(json.dumps(dict(tasks=len(tasks), frames=sum(next(c['frame_count'] for c in cases if c['id']==t['case']) for t in tasks))), flush=True)


def worker(args):
    import cv2
    import numpy as np
    import torch
    from MemNavData.run_repaired_fullmono_local import MEM_CKPT
    from NavDP.baselines.memnav.policy_agent import MemNavAgent
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher

    plan = json.loads(args.plan.read_text())
    task = plan['tasks'][args.index]
    case = next(c for c in plan['cases'] if c['id'] == task['case'])
    assert all(sha(p) == h for p, h in plan['source_sha256'].items())
    mode = task['mode']
    dependency = ROOT / '.diagnostics/dependencies/python'
    glue = ROOT / '.diagnostics/dependencies/LightGlue'
    torch.set_num_threads(4)
    torch.manual_seed(case['seed'])
    cv2.setRNGSeed(0)
    matcher = LightGluePointMatcher(glue, dependency_root=dependency, device='cuda:0',
        max_keypoints=2048, reference_cache_size=8)
    agent = MemNavAgent(str(MEM_CKPT), str(ROOT/'InternNav'), buffer_root=str(args.out/'buffer'),
        exclude_recent=32, num_samples=16, retrieval_mode='raw', retrieval_candidate_top_k=32,
        retrieval_candidate_min_gap=16, graph_subgoal_spacing_m=0., graph_subgoal_arrival_m=.60,
        certified_relocalization_matcher=matcher,
        certified_reference_depth_source='canonical' if mode=='legacy' else 'online_history',
        memory_mechanism=mode, memory_geometry_storage=task.get('geometry_storage','dense'),
        flow_gate='auto' if mode=='legacy' else 'off')
    assert mode=='native_interval7' and task['dense_window'] in (16,64)
    assert int(agent.W)==int(agent.lb.model.kv_cache_sliding_window)==task['dense_window']
    agent.reset(seed=case['seed'], episode_len=case['frame_count'])
    assert next(agent.lb.model.camera_head.parameters()).dtype==torch.float32
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    writes, checkpoints, queries = [], [], []
    selected = {64, 128, 256, 512, 1024, case['frame_count']}
    events = (args.out/'events.jsonl').open('x', buffering=1)

    def event(phase, **kwargs):
        events.write(json.dumps(dict(time=time.time(), phase=phase, **kwargs))+'\n')

    try:
        for frame, (path, digest) in enumerate(zip(case['rgb_paths'], case['rgb_sha256'])):
            jpeg = Path(path).read_bytes()
            assert hashlib.sha256(jpeg).hexdigest() == digest
            event('write', frame=frame)
            tick = time.perf_counter()
            assert agent.add_frame(jpeg) == frame
            torch.cuda.synchronize()
            milliseconds = 1000*(time.perf_counter()-tick)
            writes.append(dict(frame=frame, ms=milliseconds))
            if frame+1 in selected:
                row = dict(frames=frame+1, allocated_bytes=torch.cuda.memory_allocated(),
                    reserved_bytes=torch.cuda.memory_reserved(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(), status=agent.memory.status())
                checkpoints.append(row)
                print(json.dumps(dict(mode=mode, repeat=task['repeat'], **row)), flush=True)
        write_seconds = time.perf_counter()-start
        event('dense')
        dense_times = []
        for _ in range(21):
            tick = time.perf_counter()
            dense = agent.monocular_depth_observation()
            torch.cuda.synchronize()
            dense_times.append(1000*(time.perf_counter()-tick))
        for query in case['queries']:
            goal = Path(query['goal_path']).read_bytes()
            assert hashlib.sha256(goal).hexdigest() == query['goal_sha256']
            elapsed, first = [], None
            for repeat in range(21):
                event('query_first' if repeat == 0 else 'query_cached', query=query['id'], repeat=repeat)
                tick = time.perf_counter()
                proposal = agent.plan(goal, retrieval_only=True)
                result = agent.certified_relocalize(goal, proposal['certified_visual_candidates'])
                torch.cuda.synchronize()
                elapsed.append(1000*(time.perf_counter()-tick))
                if result.get('pnp', {}).get('status') == 'runtime_exception' or any(r.get('error') for r in result.get('ranked_candidates', [])):
                    raise RuntimeError('Correspondence/PnP execution failed')
                if first is None:
                    first = result
                else:
                    assert result['cached'] is True and result['accepted'] == first['accepted']
                    if first['accepted']:
                        np.testing.assert_array_equal(result['aux_pose'], first['aux_pose'])
            queries.append(dict(id=query['id'], first_ms=elapsed[0], cached_ms=elapsed[1:], first_result=first))
        event('done')
        assert all(sha(p) == h for p, h in plan['source_sha256'].items())
        save(args.out/'result.json', dict(completed=True, task=task, plan_sha256=sha(args.plan),
            gpu=torch.cuda.get_device_name(), torch_version=torch.__version__, writes=writes,
            checkpoints=checkpoints, write_seconds=write_seconds, dense_first_ms=dense_times[0],
            dense_cached_ms=dense_times[1:], dense_status=dense, queries=queries,
            final_status=agent.memory.status(), service_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            service_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            duration_seconds=time.perf_counter()-start,
            source_sha256=plan['source_sha256']))
    except BaseException as error:
        save(args.out/'failure.json', dict(type=type(error).__name__, error=str(error), frames=len(writes)))
        raise
    finally:
        events.close()


def run(args):
    import psutil
    from MemNavData.run_repaired_fullmono_local import MEM_PY, LINGBOT

    plan = json.loads(args.plan.read_text())
    assert all(sha(p) == h for p, h in plan['source_sha256'].items())
    for task in plan['tasks']:
        out = args.plan.parent / 'tasks' / str(task['index'])
        out.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', LINGBOT_REPO=str(LINGBOT),
            LINGBOT_WEIGHTS=str(LINGBOT/'weights/lingbot-map-long.pt'),
            MEMNAV_WINDOW=str(task['dense_window']), MEMNAV_NUM_SCALE='8',
            MEMNAV_MAX_FRAME_NUM='4096', MEMNAV_GROUND_SCALE_MAX='6.0',
            MEMNAV_GATE_FUSION='complementary', MEMNAV_AUX_POSE_CALIBRATION='empirical',
            MEMNAV_COLLISION_SELECT='1', MEMNAV_REPORT_TO='none')
        command = [MEM_PY, '-u', str(Path(__file__).resolve()), 'worker', '--plan', str(args.plan.resolve()),
            '--index', str(task['index']), '--out', str(out.resolve())]
        with (out/'worker.log').open('x') as log, (out/'resources.jsonl').open('x', buffering=1) as samples:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            save(out/'process.json', dict(pid=process.pid, command=command, task=task, plan_sha256=sha(args.plan)))
            inspected = psutil.Process(process.pid)
            next_gpu = 0.
            while process.poll() is None:
                tick = time.time()
                try:
                    row = dict(time=tick, rss_bytes=inspected.memory_info().rss, pid=process.pid)
                except psutil.NoSuchProcess:
                    break
                if tick >= next_gpu:
                    report = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
                    values = [line.split(',') for line in report.stdout.splitlines() if line.strip()]
                    row['sampled_gpu_bytes'] = sum(int(float(v[1]))*2**20 for v in values if int(v[0])==process.pid)
                    row['all_compute_process_memory_mib'] = {v[0].strip():int(float(v[1])) for v in values}
                    next_gpu = tick+1.
                samples.write(json.dumps(row)+'\n')
                time.sleep(.1)
            code = process.wait()
        save(out/'process_exit.json', dict(exit_code=code, task=task))
        print(json.dumps(dict(task=task, exit_code=code)), flush=True)
        # A failed arm is recorded, and other preplanned arms still run.
    save(args.plan.parent/'execution_complete.json', dict(execution_complete=True, tasks=len(plan['tasks']), plan_sha256=sha(args.plan)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run', 'worker'))
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    {'freeze': freeze, 'run': run, 'worker': worker}[args.action](args)


if __name__ == '__main__':
    main()
