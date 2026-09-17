"""Real production-model parity, readout and lifecycle test for EpisodicGEM."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from NavDP.baselines.memnav.policy_agent import MemNavAgent
from MemNavData.lingbot_pnp_localization import LightGluePointMatcher
from NavDP.baselines.memnav.gem.connected import ConnectedEpisodeMemory


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--lightglue',type=Path,required=True)
    parser.add_argument('--dependencies',type=Path,required=True)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    inputs=json.loads((args.inputs/'manifest.json').read_text())
    for file,digest in inputs['files_sha256'].items():
        assert hashlib.sha256((args.inputs/file).read_bytes()).hexdigest()==digest,file
    with np.load(args.inputs/'connected.npz') as a:
        keys,sampled,uv=a['keys'],a['sample_depth'],a['uv'].astype(int)
    with np.load(args.inputs/'reciprocal.npz') as a:
        expected_pose,expected_scale=a['pose9'],a['scale']
    matcher=LightGluePointMatcher(args.lightglue,dependency_root=args.dependencies,
        device='cuda:0',max_keypoints=2048,reference_cache_size=8)
    agent=MemNavAgent(str(args.checkpoint),str(ROOT/'InternNav'),buffer_root=str(args.out/'buffer'),
        certified_relocalization_matcher=matcher,certified_reference_depth_source='online_history',
        memory_mechanism='connected_reciprocal',flow_gate='off')
    # Compare runtime integration to a direct writer on the SAME GPU. BF16
    # attention on different GPU architectures need not be bit-identical.
    # Keep the saved development outputs for reporting device differences.
    direct=ConnectedEpisodeMemory(agent.lb.model,uv,
        np.load(args.inputs/'connected.npz')['valid_pixels'],transport='reciprocal')
    direct_pose, direct_keys, direct_depth, direct_scale = [], [], [], []
    for frame in range(len(keys)):
        image=agent.lb.load_images([str(args.inputs/'rgb'/f'{frame}.jpg')])[0]
        value=direct.write(frame,image)
        if value is None:
            continue
        direct_pose.extend(value.world_pose9)
        direct_keys.extend(torch.stack(agent._pop_cls(len(value.frames))).numpy())
        direct_depth.extend(value.predictions['depth'][0,:,uv[:,1],uv[:,0],0].float().cpu().numpy())
        direct_scale.extend([value.world_scale]*len(value.frames))
        del value
    direct_pose, direct_keys, direct_depth, direct_scale = map(np.asarray,
        (direct_pose,direct_keys,direct_depth,direct_scale))
    device_difference=dict(pose_max_abs=float(np.max(np.abs(direct_pose-expected_pose))),
        descriptor_max_abs=float(np.max(np.abs(direct_keys-keys))),
        sampled_depth_max_abs=float(np.max(np.abs(direct_depth-sampled))))
    expected_pose, keys, sampled, expected_scale=direct_pose,direct_keys,direct_depth,direct_scale
    del direct
    agent.reset(seed=inputs['seed'],episode_len=len(keys))
    errors=[]
    queries=[]
    goal=(args.inputs/'goal.jpg').read_bytes()
    goal_key=hashlib.md5(goal).hexdigest()
    alternate_goal=(args.inputs/'rgb'/'96.jpg').read_bytes()
    alternate_key=hashlib.md5(alternate_goal).hexdigest()
    assert alternate_key!=goal_key
    query_schedule={127:('A',goal),143:('A',goal),160:('B',alternate_goal),
        192:('A',goal),len(keys)-1:('A',goal)}
    assert len(keys)>193 and len(query_schedule)==5
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    try:
        for frame in range(len(keys)):
            jpeg=(args.inputs/'rgb'/f'{frame}.jpg').read_bytes()
            assert agent.add_frame(jpeg)==frame
            if frame<7:
                continue
            lo=0 if frame==7 else frame
            actual_pose=torch.stack(agent.memory.poses[lo:frame+1]).numpy()
            maximum=float(np.max(np.abs(actual_pose-expected_pose[lo:frame+1])))
            errors.append(maximum)
            np.testing.assert_allclose(actual_pose,expected_pose[lo:frame+1],atol=2e-7,rtol=2e-7)
            np.testing.assert_array_equal(torch.stack(agent.memory.descriptors[lo:frame+1]).numpy(),keys[lo:frame+1])
            np.testing.assert_array_equal(agent.memory.current_relative_depth.numpy()[uv[:,1],uv[:,0]],
                sampled[frame]*np.float32(expected_scale[frame]))
            assert agent.lb.agg.total_frames_processed<=16
            if frame==40:
                depth=agent.monocular_depth_observation()
                assert depth['frame_index']==frame and depth['depth_prediction_runtime_ms']==0
                assert depth['stream_observation_count']==41
            if frame in query_schedule:
                label,query_goal=query_schedule[frame]
                before=(agent.lb.agg.kv_cache,agent.lb.model.camera_head.kv_cache,
                    agent.lb.agg.total_frames_processed,agent.lb.model.camera_head.frame_idx)
                probe=agent.plan(query_goal,retrieval_only=True)
                candidates=probe['certified_visual_candidates']
                result=agent.certified_relocalize(query_goal,candidates)
                if result.get('pnp',{}).get('status')=='runtime_exception':
                    raise RuntimeError(result['pnp'])
                assert result['cached'] is (frame in (143,len(keys)-1))
                repeated=agent.certified_relocalize(query_goal,candidates)
                assert repeated.get('cached') is True
                assert repeated['accepted']==result['accepted']
                if result['accepted']:
                    np.testing.assert_allclose(result['aux_pose'],repeated['aux_pose'],atol=1e-12)
                after=(agent.lb.agg.kv_cache,agent.lb.model.camera_head.kv_cache,
                    agent.lb.agg.total_frames_processed,agent.lb.model.camera_head.frame_idx)
                assert before[0] is after[0] and before[1] is after[1] and before[2:]==after[2:]
                queries.append(dict(frame=frame,goal=label,goal_session=agent.memory.goal_session_index,
                    probe=probe,result=result,repeated=repeated))
            if (frame+1)%64==0:
                print(json.dumps(dict(frames=frame+1,pose_max_abs=max(errors),status=agent.memory.status())),flush=True)
        assert [q['goal'] for q in queries]==['A','A','B','A','A']
        assert [q['goal_session'] for q in queries]==[1,1,2,3,3]
        first_read=queries[0]['result']
        if first_read['accepted']:
            assert first_read['pnp']==queries[1]['result']['pnp']
        returned_read=queries[3]['result']
        final_read=queries[-1]['result']
        assert final_read['cached'] is True
        assert agent.memory.goal_start_frames[goal_key]==192
        assert alternate_key not in agent.memory.goal_start_frames
        if returned_read['accepted']:
            assert returned_read['pnp']==final_read['pnp']
        status=agent.memory.status()
        timings=agent.memory.write_timings
        peak=torch.cuda.max_memory_allocated()
        archive_dir=Path(agent.rgb_dir)/'geometry'
        previous_digest=hashlib.sha256((archive_dir/'000000.npz').read_bytes()).hexdigest()
        agent.reset(seed=inputs['seed'])
        for frame in range(8):
            agent.add_frame((args.inputs/'rgb'/f'{frame}.jpg').read_bytes())
        np.testing.assert_allclose(torch.stack(agent.memory.poses).numpy(),expected_pose[:8],atol=2e-7,rtol=2e-7)
        assert not agent.memory.certificates and agent.memory.goal_session_index==0
        assert hashlib.sha256((archive_dir/'000000.npz').read_bytes()).hexdigest()==previous_digest
        result=dict(passed=True,frames=len(keys),pose_max_abs=max(errors),
            gpu=torch.cuda.get_device_name(),same_gpu_direct_reference=True,
            differences_from_saved_development_gpu=device_difference,
            descriptor_exact=True,depth_sample_exact=True,queries=queries,
            actual_goal_cycle='A->B->A',goal_sessions=[1,2,3],
            next_writes_match_after_goal_switches=True,
            next_write_matches_after_scale_and_goal_reads=True,reset_parity=True,
            sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),
                ROOT/'NavDP/baselines/memnav/policy_agent.py',*sorted((ROOT/'NavDP/baselines/memnav/gem').glob('*.py'))]},
            status=status,write_timings=timings,gpu_peak_allocated_bytes=peak,wall_seconds=time.perf_counter()-start)
        (args.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('queries','sources','write_timings')}),flush=True)
    except BaseException as error:
        (args.out/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,error=str(error)),indent=2)+'\n')
        raise


if __name__=='__main__':main()
