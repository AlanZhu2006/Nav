"""Fixed observation-only LingBot information-gain experiment.

Inference never loads goal images or evaluator poses. See the sealed manifest
and separate score_gem_reactivation_probe.py for geometry-only evaluation.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('LINGBOT_REPO', '/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map'))
WEIGHTS = REPO / 'weights/lingbot-map-long.pt'
sys.path[:0] = [str(ROOT), str(REPO)]
from NavDP.baselines.memnav.gem.reactivation import (
    LingBotReactivation, fit_similarity, retrieve_packet)
from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

CONFIG = dict(img_size=518, patch_size=14, enable_3d_rope=True,
    max_frame_num=4096, kv_cache_sliding_window=64, kv_cache_scale_frames=8,
    kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
    use_sdpa=True, camera_num_iterations=4)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def prepare(out):
    out.mkdir(exist_ok=False, parents=True)
    long = ROOT / '.diagnostics/gem_observed_runtime_20260913/run_001/manifest.json'
    short = ROOT / '.diagnostics/gem_direct_memory_20260912/run_001/manifest.json'
    cases, evaluation = [], []
    for parent in (long, short):
        for case in json.loads(parent.read_text())['cases']:
            paths = []
            for i, digest in enumerate(case['rgb_sha256']):
                path = Path(case['rgb_dir']) / f'{i}.jpg'
                if not path.exists():
                    path = Path(case['rgb_dir']) / f'{i:06d}.jpg'
                assert path.is_file() and sha(path) == digest, path
                paths.append(str(path))
            cases.append(dict(id=case['id'], seed=case['seed'],
                frame_count=case['frame_count'], rgb_paths=paths,
                rgb_sha256=case['rgb_sha256'], checkpoints=case['checkpoints'],
                cohort=case['cohort'], source_manifest=str(parent), source_sha256=sha(parent)))
            evaluation.append(dict(id=case['id'], trace=case['trace'],
                trace_sha256=sha(case['trace']), queries=case['queries']))
    files = [Path(__file__), ROOT / 'MemNavData/score_gem_reactivation_probe.py',
             ROOT / 'NavDP/baselines/memnav/gem/reactivation.py',
             ROOT / 'MemNavData/test_gem_reactivation.py']
    files += sorted((REPO / 'lingbot_map').rglob('*.py'))
    save(out / 'manifest.json', dict(schema='gem_lingbot_reactivation_probe_v1',
        cases=cases, constructor=CONFIG, preprocess='pad', keyframe_interval=7,
        packet_size=8, exclude_recent=64, candidate_budget=1,
        retrieval='Current observation DINO cosine; oldest index wins ties; no score threshold',
        scoring='All preregistered prefixes; raw relative pose primary, same-pixel Sim3 binding secondary',
        no_goal_or_truth_in_inference=True, training=False, development_cases=True,
        baseline='Native forward sequence, initial 8, skip_append every non-7th view; FP32 camera',
        checkpoint=str(WEIGHTS), checkpoint_sha256=sha(WEIGHTS),
        sources_sha256={str(p): sha(p) for p in files},
        upstream_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        torch=torch.__version__, python=sys.version, created_at=time.time()))
    save(out / 'evaluation_inputs.json', dict(cases=evaluation,
        scope='Accessible to post-inference scorer only; excluded from inference manifest'))
    return json.loads((out / 'manifest.json').read_text())


def decode(predictions, uv):
    poses, intrinsics = pose_encoding_to_extri_intri(
        predictions['pose_enc'].float().cpu(), image_size_hw=(518, 518))
    poses, intrinsics = poses[0].double().cpu().numpy(), intrinsics[0].double().cpu().numpy()
    homogeneous = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    homogeneous[:, :3] = poses
    y, x = uv[:, 1].astype(int), uv[:, 0].astype(int)
    depth = predictions['depth'][0, :, y, x, 0].float().cpu().numpy()
    confidence = predictions['depth_conf'][0, :, y, x].float().cpu().numpy()
    rays = np.concatenate([uv, np.ones((len(uv), 1))], axis=1)
    xyz = np.einsum('nij,pj->npi', np.linalg.inv(intrinsics), rays) * depth[..., None]
    return dict(poses=homogeneous, intrinsics=intrinsics, xyz=xyz,
                depth=depth, confidence=confidence,
                pose_enc=predictions['pose_enc'][0].float().cpu().numpy())


def forward_block(model, images, start, interval=7):
    count = len(images)
    commit = start == 0 or (start - 8) % interval == 0
    before = model.aggregator.total_frames_processed
    model._set_skip_append(not commit)
    torch.compiler.cudagraph_mark_step_begin()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        predictions = model(images.unsqueeze(0).to('cuda'), num_frame_for_scale=8,
            num_frame_per_block=count, causal_inference=True)
    model._set_skip_append(False)
    assert model.aggregator.total_frames_processed == before + (count if commit else 0)
    assert before + count - 1 < 320
    return predictions


def baseline(model, images, uv, out, *, compare_only=False):
    model.clean_kv_cache()
    arrays = {k: [] for k in ('poses', 'intrinsics', 'xyz', 'depth', 'confidence', 'pose_enc', 'keys')}
    captured = []
    precision = []

    def dino(module, args, output):
        captured.append(output['x_norm_clstoken'].detach().float().cpu().numpy())

    def camera(module, args, kwargs):
        tokens = args[0]
        assert all(t.dtype == torch.float32 for t in tokens)
        assert not torch.is_autocast_enabled('cuda')
        precision.append(True)

    handles = [model.aggregator.patch_embed.register_forward_hook(dino),
        model.camera_head.register_forward_pre_hook(camera, with_kwargs=True)]
    timing, commits = [], []
    start_time = time.perf_counter()
    try:
        for start in [0] + list(range(8, len(images))):
            count = 8 if start == 0 else 1
            torch.cuda.synchronize()
            tick = time.perf_counter()
            predictions = forward_block(model, images[start:start + count], start)
            torch.cuda.synchronize()
            timing.append((time.perf_counter() - tick) * 1000)
            values = decode(predictions, uv)
            assert len(captured) == 1
            values['keys'] = captured.pop().reshape(count, -1)
            for key, value in values.items():
                arrays[key].append(value)
            commits.append(int(model.aggregator.total_frames_processed))
            del predictions
            if not compare_only and (start + count) % 128 == 0:
                print(json.dumps(dict(stage='baseline', case=out.name, frames=start + count,
                    total=len(images), committed=commits[-1], seconds=time.perf_counter()-start_time)), flush=True)
    finally:
        for handle in handles:
            handle.remove()
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    result['uv'] = uv
    result['write_ms'] = np.array(timing)
    result['committed'] = np.array(commits)
    if not compare_only:
        np.savez_compressed(out / 'baseline.npz', **result)
        save(out / 'baseline_runtime.json', dict(completed=True, frames=len(images),
            committed=commits[-1], fp32_camera_calls=len(precision),
            write_ms_median=float(np.median(timing[1:])), write_ms_p95=float(np.percentile(timing[1:],95)),
            wall_seconds=time.perf_counter()-start_time,
            gpu_allocated_bytes=torch.cuda.memory_allocated(),
            gpu_reserved_bytes=torch.cuda.memory_reserved(),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    return result


def state_contract(model, images, uv, out):
    # Forty observed frames establish real live KV; the next frame is a held
    # continuation. These are inference-contract tests, not accuracy samples.
    short = images[:40]
    control = baseline(model, short, uv, out, compare_only=True)
    expected = decode(forward_block(model, images[40:41], 40), uv)
    baseline(model, short, uv, out, compare_only=True)
    request = retrieve_packet(control['keys'], 39, exclude_recent=16)
    snapshots = (model.aggregator.kv_cache, model.camera_head.kv_cache,
                 model.aggregator.total_frames_processed, model.camera_head.frame_idx)
    reestimated = LingBotReactivation(model).estimate(images[list(request.frames)], request)
    assert model.aggregator.kv_cache is snapshots[0]
    assert model.camera_head.kv_cache is snapshots[1]
    assert model.aggregator.total_frames_processed == snapshots[2]
    assert model.camera_head.frame_idx == snapshots[3]
    actual = decode(forward_block(model, images[40:41], 40), uv)
    exact = {key: bool(np.array_equal(value, actual[key])) for key, value in expected.items()}
    assert all(exact.values()), exact
    del reestimated
    with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
        official = model.inference_streaming(short, num_scale_frames=8,
            keyframe_interval=7, output_device=torch.device('cpu'))
    official = decode(official, uv)
    parity = {key: bool(np.array_equal(value, control[key])) for key, value in official.items()}
    assert all(parity.values()), parity
    save(out / 'state_contract.json', dict(passed=True, official_entrypoint_parity=parity,
        next_live_frame_exact_after_reactivation=exact,
        same_live_container_identity=True, request=asdict(request)))


def transform_points(transform, points):
    return points @ transform[:3, :3].T + transform[:3, 3]


def aligned_relation(base, estimate, request, valid_pixels):
    transforms, audits = [], []
    for packet, anchor in ((request.historical_frames, request.reference),
                           (request.recent_frames, request.current)):
        xs, ys = [], []
        for frame in packet:
            index = request.frames.index(frame)
            old_xyz, new_xyz = base['xyz'][frame], estimate['xyz'][index]
            valid = (valid_pixels & np.isfinite(old_xyz).all(1) & np.isfinite(new_xyz).all(1)
                & (base['depth'][frame] > 0) & (estimate['depth'][index] > 0)
                & np.isfinite(base['confidence'][frame]) & np.isfinite(estimate['confidence'][index]))
            local = np.linalg.solve(base['poses'][anchor], base['poses'][frame])
            xs.append(transform_points(local, old_xyz[valid]))
            ys.append(transform_points(estimate['poses'][index], new_xyz[valid]))
        transform, audit = fit_similarity(np.concatenate(xs), np.concatenate(ys))
        transforms.append(transform)
        audits.append(audit)
    return np.linalg.solve(transforms[0], transforms[1]), audits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        manifest = prepare(args.out)
        print(json.dumps(dict(prepared=True, cases=len(manifest['cases']),
            prefixes=sum(len(c['checkpoints']) for c in manifest['cases']))), flush=True)
        return
    manifest = json.loads((args.out/'manifest.json').read_text())
    assert all(sha(p)==v for p,v in manifest['sources_sha256'].items())
    assert sha(WEIGHTS)==manifest['checkpoint_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = GCTStream(**manifest['constructor'])
    checkpoint = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('model', checkpoint), strict=True)
    del checkpoint
    gc.collect()
    model = model.to('cuda').eval().requires_grad_(False)
    model.aggregator.to(dtype=torch.bfloat16)
    yy, xx = np.meshgrid(np.arange(8,518,16), np.arange(8,518,16), indexing='ij')
    uv = np.stack([xx.ravel(), yy.ravel()], -1)
    estimator = LingBotReactivation(model)
    for case_index, case in enumerate(manifest['cases']):
        out = args.out/case['id']
        out.mkdir(exist_ok=False)
        torch.manual_seed(case['seed'])
        np.random.seed(case['seed'])
        assert all(sha(p)==v for p,v in zip(case['rgb_paths'],case['rgb_sha256']))
        images = load_and_preprocess_images(case['rgb_paths'], mode='pad', image_size=518, patch_size=14)
        if case_index == 0:
            state_contract(model, images, uv, args.out)
            print('Actual GPU state and native-entrypoint contracts passed', flush=True)
        torch.cuda.reset_peak_memory_stats()
        base = baseline(model, images, uv, out)
        with Image.open(case['rgb_paths'][0]) as original:
            width, height = original.size
        factor = 518 / max(width, height)
        nw, nh = round(width * factor / 14)*14, round(height * factor / 14)*14
        left, top = (518-nw)//2, (518-nh)//2
        valid_pixels = ((uv[:,0] >= left) & (uv[:,0] < left+nw)
                        & (uv[:,1] >= top) & (uv[:,1] < top+nh))
        save(out/'sampling.json',dict(valid_pixels=valid_pixels.tolist(), pixel_grid=uv.tolist()))
        rows=[]
        for prefix in case['checkpoints']:
            request = retrieve_packet(base['keys'], prefix-1,
                packet_size=manifest['packet_size'], exclude_recent=manifest['exclude_recent'])
            row=dict(prefix=prefix, request=asdict(request) if request is not None else None)
            if request is None:
                row.update(status='insufficient_history')
            else:
                torch.cuda.synchronize()
                tick=time.perf_counter()
                try:
                    predictions=estimator.estimate(images[list(request.frames)],request)
                    estimates=decode(predictions,uv)
                    del predictions
                    torch.cuda.synchronize()
                    row['inference_ms']=(time.perf_counter()-tick)*1000
                    arrays=dict(**estimates, frame_ids=np.array(request.frames))
                    h=request.frames.index(request.reference)
                    arrays['raw_relation']=np.linalg.solve(estimates['poses'][h],estimates['poses'][-1])
                    arrays['baseline_relation']=np.linalg.solve(base['poses'][request.reference],base['poses'][request.current])
                    try:
                        arrays['aligned_relation'],row['alignment']=aligned_relation(base,estimates,request,valid_pixels)
                        row['alignment_status']='valid'
                    except (ValueError,np.linalg.LinAlgError) as error:
                        row['alignment_status']='invalid'
                        row['alignment_error']=str(error)
                    np.savez_compressed(out/f'packet_{prefix}.npz',**arrays)
                    row.update(status='complete', gpu_allocated_bytes=torch.cuda.memory_allocated(),
                        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                    row.update(status='failed',error_type=type(error).__name__,error=str(error))
                save(out/f'packet_{prefix}.json',row)
            rows.append(row)
            print(json.dumps(dict(stage='packet',case=case['id'],**row)),flush=True)
        save(out/'completion.json',dict(completed=True,rows=rows,
            manifest_sha256=sha(args.out/'manifest.json'),
            files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
        del images,base
        model.clean_kv_cache()
        gc.collect()
        torch.cuda.empty_cache()
    assert all(sha(p)==v for p,v in manifest['sources_sha256'].items())
    save(args.out/'completion.json',dict(completed=True,cases=len(manifest['cases']),
        sources_unchanged=True,manifest_sha256=sha(args.out/'manifest.json')))


if __name__=='__main__':
    main()
