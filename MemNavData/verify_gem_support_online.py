"""Real online detector-support writing, matching and goal-lifecycle parity.

One production model runs the original dense archive and then the support
archive after an ordinary episode reset. Both perform real RGB writes and
real SuperPoint/LightGlue queries. Saved match outputs are never supplied as
inputs. This verifies component equivalence, not closed-loop navigation cost.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'MemNavData')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def array_sha(value):
    import numpy as np
    array = np.ascontiguousarray(value)
    return hashlib.sha256(str((array.dtype.str,array.shape)).encode()+array.tobytes()).hexdigest()


def semantic_result(result):
    # Storage counters and wall-clock time differ by design. All ranking,
    # acceptance, PnP and goal-direction fields remain compared recursively.
    excluded = {'relocalization_ms','uncached_relocalization_ms','reference_depth_cache'}
    if isinstance(result, dict):
        return {k:semantic_result(v) for k,v in result.items() if k not in excluded}
    if isinstance(result, list):
        return [semantic_result(v) for v in result]
    return result


def run(args):
    import cv2
    import numpy as np
    import torch
    from MemNavData.run_repaired_fullmono_local import MEM_CKPT
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher
    from NavDP.baselines.memnav.policy_agent import MemNavAgent
    from NavDP.baselines.memnav.gem.episodic import EpisodicGEM

    args.out.mkdir(parents=True, exist_ok=False)
    parent = json.loads(args.inputs.read_text())
    case = next(c for c in parent['cases'] if c['id']=='8WUmhLawc2A_episode_0000')
    assert case['frame_count']==234
    files = [Path(__file__), ROOT/'NavDP/baselines/memnav/policy_agent.py',
        ROOT/'MemNavData/lingbot_pnp_localization.py',
        ROOT/'MemNavData/certified_relocalization_runtime.py',
        ROOT/'MemNavData/certified_relocalization_contract.py',
        *sorted((ROOT/'NavDP/baselines/memnav/gem').glob('*.py'))]
    sources = {str(p):sha(p) for p in files}
    schedule = {127:'A',143:'A',160:'B',192:'A',233:'A'}
    save(args.out/'manifest.json',dict(inputs=str(args.inputs.resolve()),
        input_sha256=sha(args.inputs),case=case,source_sha256=sources,
        schedule=schedule,window=64,arms=['dense','detector_support'],
        scope='Actual production writes and fresh matching, same model and GPU, full 234-frame history; component test, not navigation or unbiased cost benchmark',
        created_at=time.time()))
    torch.set_num_threads(4)
    torch.manual_seed(case['seed'])
    cv2.setRNGSeed(0)
    matcher = LightGluePointMatcher(ROOT/'.diagnostics/dependencies/LightGlue',
        dependency_root=ROOT/'.diagnostics/dependencies/python',device='cuda:0',
        max_keypoints=2048,reference_cache_size=8)
    agent = MemNavAgent(str(MEM_CKPT),str(ROOT/'InternNav'),buffer_root=str(args.out/'buffer'),
        exclude_recent=32,num_samples=16,retrieval_mode='raw',retrieval_candidate_top_k=32,
        retrieval_candidate_min_gap=16,graph_subgoal_spacing_m=0.,graph_subgoal_arrival_m=.60,
        certified_relocalization_matcher=matcher,certified_reference_depth_source='online_history',
        memory_mechanism='native_interval7',memory_geometry_storage='dense',flow_gate='off')
    assert int(agent.W)==int(agent.lb.model.kv_cache_sliding_window)==64
    goal_info = parent['queries'][case['id']][0]
    goal = Path(goal_info['goal_path']).read_bytes()
    assert hashlib.sha256(goal).hexdigest()==goal_info['goal_sha256']
    alternate = Path(case['rgb_paths'][96]).read_bytes()
    goals = dict(A=goal,B=alternate)
    assert goal != alternate
    arms, baseline_memory = {}, None
    started = time.monotonic()
    try:
        for storage in ('dense','detector_support'):
            agent.memory_geometry_storage = storage
            agent.memory = EpisodicGEM(agent,bounded=False,geometry_storage=storage)
            agent.reset(seed=case['seed'],episode_len=case['frame_count'])
            cv2.setRNGSeed(0)
            writes,queries,matched = [],[],[]
            actual_match = matcher.match_paths

            def observed_match(reference,query,**kwargs):
                value = actual_match(reference,query,**kwargs)
                matched.append(dict(frame=agent.n-1,reference=int(Path(reference).stem),
                    reference_rgb_sha256=sha(reference),query_rgb_sha256=sha(query),
                    value={k:array_sha(v) if isinstance(v,np.ndarray) else v for k,v in value.items()}))
                return value

            matcher.match_paths = observed_match
            try:
                for frame,(path,digest) in enumerate(zip(case['rgb_paths'],case['rgb_sha256'])):
                    jpeg = Path(path).read_bytes()
                    assert hashlib.sha256(jpeg).hexdigest()==digest
                    before_features = list(matcher._reference_feature_cache)
                    assert agent.add_frame(jpeg)==frame
                    assert list(matcher._reference_feature_cache)==before_features
                    if frame>=7:
                        row = dict(frame=frame,pose=array_sha(agent.cam_pose[frame].numpy()),
                            descriptor=array_sha(agent.memory.descriptors[frame].numpy()),
                            dense=array_sha(agent.memory.current_relative_depth.numpy()),
                            metric_scale=agent.memory.metric_scale_receipt)
                        writes.append(row)
                        if storage=='detector_support':
                            assert row==arms['dense']['writes'][frame-7],frame
                            for identity in (range(8) if frame==7 else [frame]):
                                archive=agent.memory.online_depths
                                record=archive._record(identity)
                                indices=record['indices'].astype(np.int64)
                                dense,confidence=baseline_memory.online_depths[identity]
                                restored,restored_confidence=archive[identity]
                                np.testing.assert_array_equal(restored.ravel()[indices],dense.ravel()[indices])
                                np.testing.assert_array_equal(restored_confidence.ravel()[indices],confidence.ravel()[indices])
                    if frame in schedule:
                        label=schedule[frame]
                        query_goal=goals[label]
                        before=(agent.lb.agg.kv_cache,agent.lb.model.camera_head.kv_cache,
                            agent.lb.agg.total_frames_processed,agent.lb.model.camera_head.frame_idx)
                        proposal=agent.plan(query_goal,retrieval_only=True)
                        candidates=proposal['certified_visual_candidates']
                        first=agent.certified_relocalize(query_goal,candidates)
                        cached=agent.certified_relocalize(query_goal,candidates)
                        assert cached['cached'] and first['cached'] is (frame in (143,233))
                        after=(agent.lb.agg.kv_cache,agent.lb.model.camera_head.kv_cache,
                            agent.lb.agg.total_frames_processed,agent.lb.model.camera_head.frame_idx)
                        assert before[0] is after[0] and before[1] is after[1] and before[2:]==after[2:]
                        queries.append(dict(frame=frame,goal=label,session=agent.memory.goal_session_index,
                            candidates=candidates,result=semantic_result(first),cached=semantic_result(cached)))
                        if storage=='detector_support':
                            assert queries[-1]==arms['dense']['queries'][len(queries)-1],frame
                    if (frame+1)%64==0 or frame==233:
                        print(json.dumps(dict(storage=storage,frames=frame+1,queries=len(queries))),flush=True)
            finally:
                matcher.match_paths=actual_match
            assert [q['session'] for q in queries]==[1,1,2,3,3]
            arms[storage]=dict(writes=writes,queries=queries,matches=matched,
                status=agent.memory.status(),write_timings=agent.memory.write_timings)
            save(args.out/(storage+'.json'),arms[storage])
            if storage=='dense':
                baseline_memory=agent.memory
            else:
                assert matched==arms['dense']['matches']
                assert agent.memory.online_depths.validated_match_calls==len(matched)>0
        old_archive=agent.memory.online_depths
        archive_digest=sha(old_archive.directory/'000000.npz')
        agent.reset(seed=case['seed'],episode_len=234)
        assert not agent.memory.certificates and agent.memory.goal_session_index==0
        for frame in range(8):
            agent.add_frame(Path(case['rgb_paths'][frame]).read_bytes())
        assert sha(old_archive.directory/'000000.npz')==archive_digest
        assert array_sha(agent.cam_pose[7].numpy())==arms['dense']['writes'][0]['pose']
        assert all(sha(p)==h for p,h in sources.items())
        result=dict(passed=True,frames=234,arms=2,actual_match_pairs=len(arms['dense']['matches']),
            matches_exact=True,all_retained_geometry_exact=True,poses_descriptors_current_depth_exact=True,
            query_results_exact=True,goal_cycle=['A','A','B','A','A'],sessions=[1,1,2,3,3],
            next_writes_after_queries_exact=True,query_does_not_mutate_lingbot_state=True,
            write_does_not_mutate_query_feature_cache=True,reset_exact=True,
            statuses={k:v['status'] for k,v in arms.items()},manifest_sha256=sha(args.out/'manifest.json'),
            result_sha256={k:sha(args.out/(k+'.json')) for k in arms},
            seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name())
        save(args.out/'result.json',result)
        print(json.dumps(result),flush=True)
    except BaseException as error:
        save(args.out/'failure.json',dict(type=type(error).__name__,error=str(error),time=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args())
