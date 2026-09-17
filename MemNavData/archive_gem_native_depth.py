"""Archive native depth for a paired, unchanged SP/LightGlue/PnP evaluation.

The completed native geometry probe supplies the fixed RGB manifest. Every
prediction is checked against that probe; navigation goals are never loaded.
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
from run_gem_reactivation_probe import (
    CONFIG, WEIGHTS, GCTStream, forward_block, load_and_preprocess_images, sha, save)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    parent = args.parent.resolve()
    source = json.loads((parent / 'manifest.json').read_text())
    assert json.loads((parent / 'completion.json').read_text())['completed']
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {str(Path(__file__)): sha(__file__), **source['sources_sha256']}
    assert all(sha(p) == h for p, h in hashes.items())
    save(args.out / 'manifest.json', dict(parent=str(parent),
        parent_sha256=sha(parent / 'manifest.json'), cases=source['cases'],
        sources_sha256=hashes, archive_dtype='float16 on both comparison arms',
        no_goal_or_truth_in_inference=True, constructor=CONFIG))
    torch.set_num_threads(4)
    model = GCTStream(**CONFIG)
    assert sha(WEIGHTS) == source['checkpoint_sha256']
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    del checkpoint
    gc.collect()
    model = model.to('cuda').eval().requires_grad_(False)
    model.aggregator.to(dtype=torch.bfloat16)
    for case in source['cases']:
        out = args.out / case['id']
        (out / 'depths').mkdir(parents=True)
        assert all(sha(p) == h for p, h in zip(case['rgb_paths'], case['rgb_sha256']))
        images = load_and_preprocess_images(case['rgb_paths'], mode='pad', image_size=518, patch_size=14)
        with np.load(parent / case['id'] / 'baseline.npz') as a:
            expected = {k: a[k] for k in ('pose_enc', 'depth', 'confidence', 'keys', 'uv')}
        uv = expected['uv'].astype(int)
        model.clean_kv_cache()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(case['seed'])
        np.random.seed(case['seed'])
        captured = []
        handle = model.aggregator.patch_embed.register_forward_hook(
            lambda m, i, o: captured.append(o['x_norm_clstoken'].detach().float().cpu().numpy()))
        timings = []
        started = time.perf_counter()
        try:
            for start in [0] + list(range(8, len(images))):
                count = 8 if start == 0 else 1
                torch.cuda.synchronize()
                tick = time.perf_counter()
                pred = forward_block(model, images[start:start+count], start)
                torch.cuda.synchronize()
                inference_ms = 1000 * (time.perf_counter() - tick)
                depth = pred['depth'][0, ..., 0].float().cpu().numpy()
                confidence = pred['depth_conf'][0].float().cpu().numpy()
                actual = dict(pose_enc=pred['pose_enc'][0].float().cpu().numpy(),
                    depth=depth[:, uv[:, 1], uv[:, 0]],
                    confidence=confidence[:, uv[:, 1], uv[:, 0]],
                    keys=captured.pop().reshape(count, -1))
                assert not captured
                for k, v in actual.items():
                    assert np.array_equal(v, expected[k][start:start+count]), (case['id'], start, k)
                for j in range(count):
                    np.savez_compressed(out / 'depths' / f'{start+j:06d}.npz',
                        depth=depth[j].astype(np.float16), confidence=confidence[j].astype(np.float16),
                        frame=np.array(start+j), world_scale=np.array(1.))
                timings.append(dict(frame=start+count-1, inference_ms=inference_ms,
                    with_archive_ms=1000*(time.perf_counter()-tick),
                    gpu_allocated_bytes=torch.cuda.memory_allocated(),
                    gpu_reserved_bytes=torch.cuda.memory_reserved()))
                del pred, depth, confidence, actual
                if (start+count) % 256 == 0:
                    print(json.dumps(dict(case=case['id'], frames=start+count,
                        total=len(images), seconds=time.perf_counter()-started)), flush=True)
        finally:
            handle.remove()
        save(out / 'timing.json', timings)
        save(out / 'completion.json', dict(completed=True, frames=len(images),
            original_native_predictions_exact=True, seconds=time.perf_counter()-started,
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            files_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}))
        print(json.dumps(dict(case=case['id'], completed=True)), flush=True)
        model.clean_kv_cache()
        del images, expected
        gc.collect()
        torch.cuda.empty_cache()
    assert all(sha(p) == h for p, h in hashes.items())
    save(args.out / 'completion.json', dict(completed=True, cases=len(source['cases']),
        sources_unchanged=True, manifest_sha256=sha(args.out / 'manifest.json')))


if __name__ == '__main__':
    main()
