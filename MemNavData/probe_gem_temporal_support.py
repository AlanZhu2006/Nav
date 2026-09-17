"""Reestimate all nine histories from a fixed64-view temporal support.

One unchanged LingBot: initial8 actual RGBs plus56 uniformly spaced actual
observations ending at the current frame. Reindex only this new inference
sequence to0..63; every selected view is committed with native FP32 camera.
No target images, truth, retrieval, parameter search or online state updates.
This tests information in a temporally spanning memory consolidation, not a
delivered runtime architecture or a navigation improvement.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from run_gem_reactivation_probe import (CONFIG,WEIGHTS,GCTStream,forward_block,
    load_and_preprocess_images,sha,save)


def load(path): return json.loads(Path(path).read_text())


def prepare(parent,out):
    source=load(parent/'manifest.json')
    assert load(parent/'completion.json')['completed']
    cases=[]
    for original in source['cases']:
        count=original['frame_count']
        frames=np.r_[np.arange(8),np.rint(np.linspace(8,count-1,56)).astype(int)].tolist()
        assert len(frames)==len(set(frames))==64 and frames[:9]==list(range(9)) and frames[-1]==count-1
        assert all(b>a for a,b in zip(frames,frames[1:]))
        cases.append(dict(**original,support_frames=frames))
    sources=dict(source['sources_sha256'])
    assert all(sha(p)==h for p,h in sources.items())
    sources[str(Path(__file__).resolve())]=sha(__file__)
    out.mkdir(parents=True,exist_ok=False)
    save(out/'manifest.json',dict(schema='gem_temporal_support_information_v1',
        parent=str(parent),parent_manifest_sha256=sha(parent/'manifest.json'),cases=cases,
        constructor=CONFIG,checkpoint_sha256=source['checkpoint_sha256'],sources_sha256=sources,
        support_size=64,initial_views=8,remaining='56 uniform source-frame indices including the last observation',
        inference_keyframe_interval=1,committed_temporal_positions='0..63',
        baseline_sha256={c['id']:sha(parent/c['id']/'baseline.npz') for c in cases},
        no_goals_or_truth_in_inference=True,training=False,
        scope='All9 complete development histories; geometry information-gain only, no online implementation or SR claim',
        created_at=time.time()))


def run(out):
    manifest=load(out/'manifest.json')
    assert all(sha(p)==h for p,h in manifest['sources_sha256'].items())
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
    parent=Path(manifest['parent'])
    for case in manifest['cases']:
        folder=out/case['id']
        (folder/'depths').mkdir(parents=True,exist_ok=False)
        frames=case['support_frames']
        paths=[case['rgb_paths'][i] for i in frames]
        assert all(sha(p)==case['rgb_sha256'][i] for i,p in zip(frames,paths))
        assert sha(parent/case['id']/'baseline.npz')==manifest['baseline_sha256'][case['id']]
        with np.load(parent/case['id']/'baseline.npz') as data:
            expected_pose=data['pose_enc'][:9]
            expected_keys=data['keys'][frames]
        images=load_and_preprocess_images(paths,mode='pad',image_size=518,patch_size=14)
        model.clean_kv_cache()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(case['seed'])
        captured=[]
        precision=[]
        def camera(module,args,kwargs):
            assert all(t.dtype==torch.float32 for t in args[0])
            assert not torch.is_autocast_enabled('cuda')
            precision.append(True)
        handles=[model.aggregator.patch_embed.register_forward_hook(
            lambda m,i,o:captured.append(o['x_norm_clstoken'].detach().float().cpu().numpy())),
            model.camera_head.register_forward_pre_hook(camera,with_kwargs=True)]
        poses,keys,timings=[],[],[]
        started=time.perf_counter()
        try:
            for start in [0]+list(range(8,64)):
                count=8 if start==0 else 1
                torch.cuda.synchronize()
                tick=time.perf_counter()
                pred=forward_block(model,images[start:start+count],start,interval=1)
                pose=pred['pose_enc'][0].float().cpu().numpy()
                key=captured.pop().reshape(count,-1)
                assert not captured and np.array_equal(key,expected_keys[start:start+count])
                if start<9: assert np.array_equal(pose,expected_pose[start:start+count])
                poses.append(pose);keys.append(key)
                depth=pred['depth'][0,...,0].float().cpu().numpy()
                confidence=pred['depth_conf'][0].float().cpu().numpy()
                for j in range(count):
                    np.savez_compressed(folder/'depths'/f'{frames[start+j]:06d}.npz',
                        frame=np.array(frames[start+j]),depth=depth[j],confidence=confidence[j])
                torch.cuda.synchronize()
                timings.append(dict(support_index=start+count-1,source_frame=frames[start+count-1],
                    committed=model.aggregator.total_frames_processed,with_archive_ms=1000*(time.perf_counter()-tick)))
                del pred,depth,confidence
        finally:
            for handle in handles:handle.remove()
        assert model.aggregator.total_frames_processed==64 and len(precision)==57
        np.savez_compressed(folder/'geometry.npz',source_frames=np.asarray(frames),
            pose_enc=np.concatenate(poses),keys=np.concatenate(keys))
        save(folder/'timing.json',timings)
        save(folder/'completion.json',dict(completed=True,source_frames=case['frame_count'],
            inferred_support_frames=64,native_committed=64,first9_pose_parity=True,all_support_dino_exact=True,
            fp32_camera_calls=len(precision),seconds=time.perf_counter()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            files_sha256={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}))
        print(json.dumps(dict(case=case['id'],source_frames=case['frame_count'],support=64,completed=True)),flush=True)
        model.clean_kv_cache()
        del images,poses,keys
        gc.collect();torch.cuda.empty_cache()
    assert all(sha(p)==h for p,h in manifest['sources_sha256'].items())
    save(out/'completion.json',dict(completed=True,cases=len(manifest['cases']),
        support_frames=len(manifest['cases'])*64,sources_unchanged=True,manifest_sha256=sha(out/'manifest.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args.parent.resolve(),args.out.resolve())
    else: run(args.out.resolve())
