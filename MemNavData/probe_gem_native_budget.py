"""One fixed compact native LingBot control, with persistent camera history.

This is an existing upstream retention configuration, not a novel mechanism.
The inference process consumes observed RGB only. Goal/GT scoring is separate.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
from run_gem_reactivation_probe import (CONFIG, REPO, WEIGHTS, GCTStream,
    forward_block, decode, load_and_preprocess_images, sha, save)


def prepare(parent, out):
    source = json.loads((parent / 'manifest.json').read_text())
    assert json.loads((parent / 'completion.json').read_text())['completed']
    hashes = dict(source['sources_sha256'])
    assert all(sha(p) == h for p, h in hashes.items())
    hashes[str(Path(__file__).resolve())] = sha(__file__)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'manifest.json', dict(schema='gem_fixed_native_budget_v1',
        parent=str(parent), parent_sha256=sha(parent / 'manifest.json'),
        cases=source['cases'], sources_sha256=hashes,
        constructor={**CONFIG, 'kv_cache_sliding_window': 16},
        checkpoint_sha256=source['checkpoint_sha256'], keyframe_interval=7,
        camera_head='native FP32; retain all committed camera-head KV',
        aggregator='BF16; initial8 and recent16 full views; old special tokens retained',
        preprocessing='native 518 pad', depth_archive_dtype='float32',
        no_goal_or_truth_in_inference=True, no_context_reset_inside_history=True,
        baseline_sha256={c['id']:sha(parent/c['id']/'baseline.npz') for c in source['cases']},
        scope='Fixed development control; no novel retention claim, no window sweep',
        created_at=time.time()))


def run(out):
    manifest = json.loads((out/'manifest.json').read_text())
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
    assert next(model.camera_head.parameters()).dtype==torch.float32
    parent=Path(manifest['parent'])
    for case in manifest['cases']:
        target=out/case['id']
        (target/'depths').mkdir(parents=True,exist_ok=False)
        assert all(sha(p)==h for p,h in zip(case['rgb_paths'],case['rgb_sha256']))
        assert sha(parent/case['id']/'baseline.npz')==manifest['baseline_sha256'][case['id']]
        with np.load(parent/case['id']/'baseline.npz') as data:
            expected={k:data[k] for k in ('pose_enc','keys','uv','depth','confidence')}
        uv=expected['uv'].astype(int)
        images=load_and_preprocess_images(case['rgb_paths'],mode='pad',image_size=518,patch_size=14)
        model.clean_kv_cache()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(case['seed'])
        np.random.seed(case['seed'])
        captured=[]
        precision=[]
        def camera(module,args,kwargs):
            assert all(t.dtype==torch.float32 for t in args[0])
            assert not torch.is_autocast_enabled('cuda')
            precision.append(True)
        handles=[model.aggregator.patch_embed.register_forward_hook(
            lambda m,i,o:captured.append(o['x_norm_clstoken'].detach().float().cpu().numpy())),
            model.camera_head.register_forward_pre_hook(camera,with_kwargs=True)]
        arrays={k:[] for k in ('poses','intrinsics','xyz','depth','confidence','pose_enc','keys')}
        timing=[]
        started=time.perf_counter()
        try:
            for start in [0]+list(range(8,len(images))):
                count=8 if start==0 else 1
                torch.cuda.synchronize()
                tick=time.perf_counter()
                predictions=forward_block(model,images[start:start+count],start)
                torch.cuda.synchronize()
                inference_ms=1000*(time.perf_counter()-tick)
                values=decode(predictions,uv)
                assert len(captured)==1
                values['keys']=captured.pop().reshape(count,-1)
                assert np.array_equal(values['keys'],expected['keys'][start:start+count])
                # No eviction difference exists until the W16 dense cache has
                # reached initial8 + recent16 committed views (raw frame113).
                if start<114:
                    for key in ('pose_enc','depth','confidence'):
                        assert np.array_equal(values[key],expected[key][start:start+count]),(case['id'],start,key)
                for key,value in values.items(): arrays[key].append(value)
                depth=predictions['depth'][0,...,0].float().cpu().numpy()
                confidence=predictions['depth_conf'][0].float().cpu().numpy()
                for j in range(count):
                    np.savez_compressed(target/'depths'/f'{start+j:06d}.npz',
                        frame=np.array(start+j),depth=depth[j],confidence=confidence[j],world_scale=np.array(1.))
                timing.append(dict(frame=start+count-1,inference_ms=inference_ms,
                    with_archive_ms=1000*(time.perf_counter()-tick),
                    committed=model.aggregator.total_frames_processed,
                    gpu_allocated_bytes=torch.cuda.memory_allocated(),
                    gpu_reserved_bytes=torch.cuda.memory_reserved()))
                del predictions,depth,confidence,values
                if (start+count)%128==0:
                    print(json.dumps(dict(case=case['id'],frames=start+count,total=len(images),
                        seconds=time.perf_counter()-started)),flush=True)
        finally:
            for handle in handles: handle.remove()
        np.savez_compressed(target/'geometry.npz',uv=uv,
            **{key:np.concatenate(value) for key,value in arrays.items()})
        save(target/'timing.json',timing)
        save(target/'completion.json',dict(completed=True,frames=len(images),
            committed=model.aggregator.total_frames_processed,fp32_camera_calls=len(precision),
            dino_exact_native64=True,first114_frames_exact_native64=True,
            seconds=time.perf_counter()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            files_sha256={str(p.relative_to(target)):sha(p) for p in target.rglob('*') if p.is_file()}))
        print(json.dumps(dict(case=case['id'],completed=True)),flush=True)
        model.clean_kv_cache()
        del images,expected,arrays
        gc.collect()
        torch.cuda.empty_cache()
    assert all(sha(p)==h for p,h in manifest['sources_sha256'].items())
    save(out/'completion.json',dict(completed=True,cases=len(manifest['cases']),
        sources_unchanged=True,manifest_sha256=sha(out/'manifest.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args.parent.resolve(),args.out.resolve())
    else: run(args.out.resolve())
