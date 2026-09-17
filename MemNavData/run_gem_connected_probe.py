"""Bounded LingBot episode writing with identical-camera connections.

The native baseline, frames, interval, model weights and preprocessing are
bound to a completed reactivation experiment. Only observed RGB is consumed.
Both camera-identity and point-cloud connections use the same predictions.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'MemNavData')]
from run_gem_reactivation_probe import (CONFIG, REPO, WEIGHTS, GCTStream, sha, save,
    load_and_preprocess_images, pose_encoding_to_extri_intri, transform_points)
from NavDP.baselines.memnav.gem.connected import ConnectedEpisodeMemory, pose_matrices
from NavDP.baselines.memnav.gem.reactivation import fit_similarity


def prepare(out, parent):
    parent = parent.resolve()
    baseline = json.loads((parent/'manifest.json').read_text())
    assert json.loads((parent/'completion.json').read_text())['completed']
    files = [Path(__file__), ROOT/'MemNavData/score_gem_connected_probe.py',
        ROOT/'MemNavData/test_gem_connected.py',
        ROOT/'NavDP/baselines/memnav/gem/connected.py',
        ROOT/'NavDP/baselines/memnav/gem/bindings.py']
    sources = dict(baseline['sources_sha256'])
    assert all(sha(p)==v for p,v in sources.items())
    sources.update({str(p):sha(p) for p in files})
    out.mkdir(parents=True,exist_ok=False)
    manifest=dict(schema='gem_connected_episode_probe_v1',parent=str(parent),
        parent_manifest_sha256=sha(parent/'manifest.json'),cases=baseline['cases'],
        constructor=CONFIG,checkpoint=str(WEIGHTS),checkpoint_sha256=sha(WEIGHTS),
        sources_sha256=sources,overlap_views=8,new_views=8,keyframe_interval=7,
        preprocessing='native 518 pad',camera_head='native FP32 wrapper',
        aggregator_weights_dtype='bfloat16',no_goal_or_truth_in_inference=True,
        source_keyframe_schedule_unchanged=True,
        native_baseline_sha256={c['id']:sha(parent/c['id']/'baseline.npz') for c in baseline['cases']},
        primary='camera orientation, shared-depth scale, camera-center translation',
        comparison='unconstrained point-cloud Sim3 on exactly the same shared pixels and network outputs',
        scope='Fixed development histories, no new navigation results',created_at=time.time())
    save(out/'manifest.json',manifest)
    return manifest


def points_from_depth(poses9, depth, uv):
    _, k = pose_encoding_to_extri_intri(torch.from_numpy(poses9).float().unsqueeze(0),
        image_size_hw=(518,518))
    rays=np.concatenate([uv,np.ones((len(uv),1))],axis=1)
    return np.einsum('nij,pj->npi',np.linalg.inv(k[0].numpy().astype(np.float64)),rays)*depth[...,None]


def point_alignment(overlap, uv, valid_pixels):
    old_local=points_from_depth(overlap['old_pose9'],overlap['old_depth'],uv)
    new_local=points_from_depth(overlap['new_pose9'],overlap['new_depth'],uv)
    xs,ys=[],[]
    for i in range(len(old_local)):
        valid=(valid_pixels & np.isfinite(old_local[i]).all(1) & np.isfinite(new_local[i]).all(1)
            & (overlap['old_depth'][i]>0) & (overlap['new_depth'][i]>0))
        ys.append(transform_points(overlap['old_poses'][i],old_local[i,valid]))
        xs.append(transform_points(overlap['new_poses'][i],new_local[i,valid]))
    return fit_similarity(np.concatenate(xs),np.concatenate(ys))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        manifest=prepare(args.out,args.parent)
        print(json.dumps(dict(prepared=True,cases=len(manifest['cases']))),flush=True)
        return
    manifest=json.loads((args.out/'manifest.json').read_text())
    parent=Path(manifest['parent'])
    assert all(sha(p)==v for p,v in manifest['sources_sha256'].items())
    assert sha(WEIGHTS)==manifest['checkpoint_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model=GCTStream(**manifest['constructor'])
    checkpoint=torch.load(WEIGHTS,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint.get('model',checkpoint),strict=True)
    del checkpoint
    gc.collect()
    model=model.to('cuda').eval().requires_grad_(False)
    model.aggregator.to(dtype=torch.bfloat16)
    yy,xx=np.meshgrid(np.arange(8,518,16),np.arange(8,518,16),indexing='ij')
    uv=np.stack([xx.ravel(),yy.ravel()],-1)
    for case in manifest['cases']:
        out=args.out/case['id']
        out.mkdir(exist_ok=False)
        (out/'overlaps').mkdir()
        (out/'depths').mkdir()
        assert sha(parent/case['id']/'baseline.npz')==manifest['native_baseline_sha256'][case['id']]
        assert all(sha(p)==v for p,v in zip(case['rgb_paths'],case['rgb_sha256']))
        torch.manual_seed(case['seed'])
        np.random.seed(case['seed'])
        images=load_and_preprocess_images(case['rgb_paths'],mode='pad',image_size=518,patch_size=14)
        with Image.open(case['rgb_paths'][0]) as image:
            width,height=image.size
        factor=518/max(width,height)
        nw,nh=round(width*factor/14)*14,round(height*factor/14)*14
        left,top=(518-nw)//2,(518-nh)//2
        valid=(uv[:,0]>=left)&(uv[:,0]<left+nw)&(uv[:,1]>=top)&(uv[:,1]<top+nh)
        memory=ConnectedEpisodeMemory(model,uv,valid)
        arrays={k:[] for k in ('camera_pose9','point_pose9','local_pose9','scale','point_scale','episode','keys','sample_depth')}
        timing=[]
        point_world=np.eye(4)
        last_episode=0
        captured={}

        def dino(module,args,output):
            captured['keys']=output['x_norm_clstoken'].detach().float().cpu().numpy()

        handle=model.aggregator.patch_embed.register_forward_hook(dino)
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        point_relations=[]
        try:
            for frame,image in enumerate(images):
                torch.cuda.synchronize()
                tick=time.perf_counter()
                result=memory.write(frame,image)
                torch.cuda.synchronize()
                write_ms=(time.perf_counter()-tick)*1000
                if result is None:
                    continue
                if result.episode!=last_episode:
                    overlap=memory.last_overlap
                    transform,audit=point_alignment(overlap,uv,valid)
                    point_world=point_world@transform
                    point_relations.append(dict(episode=result.episode,old_from_new=transform.tolist(),
                        world_from_episode=point_world.tolist(),**audit))
                    np.savez_compressed(out/'overlaps'/f'{result.episode:04d}.npz',**overlap)
                    last_episode=result.episode
                poses=pose_matrices(result.local_pose9)
                point_poses=point_world@poses
                point_scale=float(np.cbrt(np.linalg.det(point_world[:3,:3])))
                point_pose9=result.local_pose9.astype(np.float64).copy()
                point_pose9[:,:3]=point_poses[:,:3,3]
                point_pose9[:,3:7]=Rotation.from_matrix(point_poses[:,:3,:3]/point_scale).as_quat()
                depth=result.predictions['depth'][0].float().cpu().numpy()[...,0]
                confidence=result.predictions['depth_conf'][0].float().cpu().numpy()
                for j,identity in enumerate(result.frames):
                    np.savez_compressed(out/'depths'/f'{identity:06d}.npz',depth=depth[j].astype(np.float16),
                        confidence=confidence[j].astype(np.float16),frame=np.array(identity),
                        world_scale=np.array(result.world_scale),point_scale=np.array(point_scale))
                arrays['camera_pose9'].append(result.world_pose9)
                arrays['point_pose9'].append(point_pose9)
                arrays['local_pose9'].append(result.local_pose9)
                arrays['scale'].extend([result.world_scale]*len(result.frames))
                arrays['point_scale'].extend([point_scale]*len(result.frames))
                arrays['episode'].extend([result.episode]*len(result.frames))
                arrays['keys'].append(captured.pop('keys').reshape(len(result.frames),-1))
                arrays['sample_depth'].append(depth[:,uv[:,1],uv[:,0]])
                timing.append(dict(frame=frame,write_ms=write_ms,
                    with_archive_ms=(time.perf_counter()-tick)*1000,episode=result.episode,
                    live_committed=int(model.aggregator.total_frames_processed),
                    gpu_allocated_bytes=torch.cuda.memory_allocated(),gpu_reserved_bytes=torch.cuda.memory_reserved()))
                del result,depth,confidence
                if (frame+1)%128==0:
                    print(json.dumps(dict(stage='connected_write',case=case['id'],frames=frame+1,
                        total=len(images),episodes=memory.episode+1,scale=memory.relations[-1]['scale'] if memory.relations else 1.,
                        world_scale=float(np.cbrt(np.linalg.det(memory.world_from_episode[:3,:3]))),
                        seconds=time.perf_counter()-started)),flush=True)
        finally:
            handle.remove()
        arrays={k:np.asarray(v) if k in ('scale','point_scale','episode') else np.concatenate(v) for k,v in arrays.items()}
        assert len(arrays['camera_pose9'])==case['frame_count']
        for value in arrays.values():
            assert np.isfinite(value).all()
        # Both writers have exactly the same keys and native state until the
        # first rebase at raw frame 64; compare actual untransformed predictions.
        with np.load(parent/case['id']/'baseline.npz') as base:
            parity=dict(pose_enc=bool(np.array_equal(arrays['local_pose9'][:64],base['pose_enc'][:64])),
                depth=bool(np.array_equal(arrays['sample_depth'][:64],base['depth'][:64])),
                keys=bool(np.array_equal(arrays['keys'][:64],base['keys'][:64])))
        assert all(parity.values()),parity
        np.savez_compressed(out/'geometry.npz',**arrays,uv=uv,valid_pixels=valid)
        save(out/'camera_relations.json',memory.relations)
        save(out/'point_relations.json',point_relations)
        save(out/'timing.json',timing)
        save(out/'completion.json',dict(completed=True,frames=case['frame_count'],
            native_first64_exact=parity,episodes=memory.episode+1,
            max_committed=max(x['live_committed'] for x in timing),
            native_write_ms_median=float(np.median([x['write_ms'] for x in timing[1:]])),
            native_write_ms_p95=float(np.percentile([x['write_ms'] for x in timing[1:]],95)),
            writer_with_archive_ms_median=float(np.median([x['with_archive_ms'] for x in timing[1:]])),
            wall_seconds=time.perf_counter()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            files_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps(dict(stage='case_complete',case=case['id'])),flush=True)
        del memory,images,arrays
        model.clean_kv_cache()
        gc.collect()
        torch.cuda.empty_cache()
    assert all(sha(p)==v for p,v in manifest['sources_sha256'].items())
    save(args.out/'completion.json',dict(completed=True,cases=len(manifest['cases']),
        manifest_sha256=sha(args.out/'manifest.json'),sources_unchanged=True))


if __name__=='__main__':main()
