"""Causal spatial-feature probe on four completed, policy-executed A/B/A traces.

freeze reads source metadata; extract and match never read evaluator poses or
goal images. No controller, simulator, map updater, or fallback is invoked.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'MemNavData')]


def load(path):
    return json.loads(Path(path).read_text())


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


def freeze(out):
    import numpy as np
    from PIL import Image
    from MemNavData.run_gem_reactivation_probe import CONFIG, REPO, WEIGHTS
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT/'.diagnostics/gem_connected_memory_20260913/continuous_002/tasks'
    cases, evaluation = [], []
    for task in (1, 4, 7, 10):
        directory = source/str(task)
        manifest = load(directory/'manifest.json')
        assert manifest['mode'] == 'native_interval7'
        chain = load(directory/'evaluation/chain_summary.json')
        assert chain['completed'] and len(chain['legs']) == 3
        pose_log = [json.loads(s) for s in (directory/'lingbot_pose_readout.jsonl').read_text().splitlines()]
        images, digests, original_poses = [], [], []
        for i, row in enumerate(pose_log):
            assert row['frame_idx'] == i
            image = directory/'buffer/ep_0001'/f'{i}.jpg'
            assert sha(image) == row['image_sha256']
            images.append(str(image))
            digests.append(row['image_sha256'])
            original_poses.append(row['camera_pose9'])
        queries = []
        for leg in chain['legs']:
            first, last = leg['first_memory_index'], leg['last_memory_index']
            for phase, current in zip(('start', 'middle', 'end'), (first, (first+last)//2, last)):
                queries.append(dict(current=current, leg=leg['leg_index'], phase=phase))
        keys = list(range(8, len(images), 7))
        keep = sorted(set(keys) | {q['current'] for q in queries})
        cell = manifest['cell']
        case = dict(id=f'task_{task}', scene=cell['scene'], source_directory=str(directory),
            source_manifest_sha256=sha(directory/'manifest.json'),
            pose_log_sha256=sha(directory/'lingbot_pose_readout.jsonl'),
            history_frames=chain['history_frames'], frames=len(images), rgb_paths=images,
            rgb_sha256=digests, original_pose9=original_poses, raw_wh=Image.open(images[0]).size,
            queries=queries, reference_keyframes=keys, feature_frames=keep)
        cases.append(case)
        roles_path = Path(cell['benchmark'])/cell['scene']/cell['episode']/'role_pairs.json'
        roles = load(roles_path)
        trace_path = Path(roles['online_a_episode'])/'online_a_trace.json'
        assert sha(trace_path) == roles['online_a_trace_sha256']
        truth = load(trace_path)['poses']
        assert len(truth) == case['history_frames']
        truth = [dict(frame=i, **{k:r[k] for k in ('x','y','z','yaw')}) for i,r in enumerate(truth)]
        provenance = {str(trace_path):sha(trace_path)}
        for leg in chain['legs']:
            path = directory/f'evaluation/leg_{leg["leg_index"]}/actual_trace.json'
            records = load(path)['poses']
            first, last = leg['first_memory_index'], leg['last_memory_index']
            assert len(records) == last-first+1
            for j, row in enumerate(records):
                frame = first+j
                assert row['jpg_sha256'] == digests[frame]
                assert frame == len(truth)
                truth.append(dict(frame=frame, **{k:row[k] for k in ('x','y','z','yaw')}))
            provenance[str(path)] = sha(path)
        assert len(truth) == case['frames']
        evaluation.append(dict(id=case['id'], rows=truth, source_sha256=provenance))
    sources = [Path(__file__), ROOT/'MemNavData/gem_spatial_features.py',
        ROOT/'MemNavData/test_gem_spatial_features.py', ROOT/'MemNavData/run_gem_reactivation_probe.py',
        ROOT/'MemNavData/lingbot_pnp_localization.py', ROOT/'MemNavData/lingbot_colored_registration.py',
        ROOT/'MemNavData/certified_relocalization_runtime.py',
        ROOT/'MemNavData/certified_relocalization_contract.py',
        *sorted((REPO/'lingbot_map').rglob('*.py'))]
    dependency = ROOT/'.diagnostics/dependencies/LightGlue'
    sources += sorted((dependency/'lightglue').glob('*.py'))
    weights = Path.home()/'.cache/torch/hub/checkpoints'
    sources += [weights/'superpoint_v1.pth', weights/'superpoint_lightglue_v0-1_arxiv.pth']
    plan = dict(schema='gem_spatial_revisit_probe_v1_20260915', cases=cases,
        constructor=CONFIG, checkpoint=str(WEIGHTS), checkpoint_sha256=sha(WEIGHTS),
        layers_zero_based=[17,23], feature_dtype='bfloat16',
        descriptor_types=['dino_patch','gca_18','gca_24'],
        reference_protocol='Native committed views 8+7k over full observed prefix, <= current-64; same pool for every arm',
        queries='First/middle/last actual observation of each of 12 completed legs; all retained',
        candidate_top_k=8, exclude_recent=64, retrieval='Original DINO CLS cosine; stable lower-frame tie order; no GT or score cutoff',
        feature_matching='Unthresholded mutual cosine nearest neighbours at valid 14px patch centres; not actual attention',
        pose_protocol='Original correspondence_pnp_localize and strict certificate; identical reference depth and intrinsics convention for all arms',
        epipolar_threshold_px=1.5, pnp_reprojection_px=3.0,
        scope='Four consumed short/mid-range policy-executed A/B/A histories; correspondence feasibility, not long-range navigation or generalization',
        source_sha256={str(p):sha(p) for p in sources},
        task_goals_in_inference=False, evaluator_pose_in_inference=False,
        extra_simulator_renders=False, navigation_executed=False)
    save(out/'plan.json',plan)
    save(out/'evaluation_inputs.json',dict(cases=evaluation, inference_access=False))
    save(out/'inventory.json',dict(cases=len(cases), frames=sum(c['frames'] for c in cases),
        queries=sum(len(c['queries']) for c in cases), feature_frames=sum(len(c['feature_frames']) for c in cases),
        original_features_present=False, all_rgb_bytes_verified=True))
    print((out/'inventory.json').read_text(),flush=True)


def verify_sources(plan):
    for path,digest in plan['source_sha256'].items():
        assert sha(path) == digest, path


def extract(out):
    import numpy as np
    import torch
    from MemNavData.run_gem_reactivation_probe import GCTStream, forward_block, load_and_preprocess_images
    from MemNavData.gem_spatial_features import SpatialCapture, patch_coordinates
    plan = load(out/'plan.json')
    verify_sources(plan)
    assert sha(plan['checkpoint']) == plan['checkpoint_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = GCTStream(**plan['constructor'])
    state = torch.load(plan['checkpoint'], map_location='cpu', weights_only=False)
    model.load_state_dict(state.get('model',state),strict=True)
    del state
    gc.collect()
    model = model.to('cuda').eval().requires_grad_(False)
    model.aggregator.to(dtype=torch.bfloat16)
    capture = SpatialCapture(model,tuple(plan['layers_zero_based']))
    torch.cuda.reset_peak_memory_stats()
    receipts = []
    try:
        for case in plan['cases']:
            folder = out/case['id']
            folder.mkdir(exist_ok=False)
            (folder/'features').mkdir()
            model.clean_kv_cache()
            ids,uv,raw_uv = patch_coordinates(case['raw_wh'])
            features = set(case['feature_frames'])
            keys, poses, errors, metadata = [],[],[],[]
            started=time.perf_counter()
            # Load only each next block, keeping the complete original order.
            for start in [0]+list(range(8,case['frames'])):
                count=8 if start==0 else 1
                paths=case['rgb_paths'][start:start+count]
                assert all(sha(p)==d for p,d in zip(paths,case['rgb_sha256'][start:start+count]))
                images=load_and_preprocess_images(paths,mode='pad',image_size=518,patch_size=14)
                assert images.shape == (count,3,518,518), images.shape
                capture.enabled=any(f in features for f in range(start,start+count))
                before=time.perf_counter()
                prediction=forward_block(model,images,start)
                current_pose=prediction['pose_enc'][0].float().cpu().numpy()
                keys.extend(capture.cls.numpy())
                poses.extend(current_pose)
                captured=capture.pop(count,ids) if capture.enabled else None
                for j,frame in enumerate(range(start,start+count)):
                    old=case['original_pose9'][frame]
                    if old is not None:
                        old=np.asarray(old,dtype=float).copy()
                        new=current_pose[j].astype(float).copy()
                        # EpisodicGEM logs unit quaternions; pose_enc carries
                        # the unnormalized camera-head output. Compare the
                        # same rotation representation, without changing KV
                        # or the predictions saved in stream.npz.
                        old[3:7]/=np.linalg.norm(old[3:7])
                        new[3:7]/=np.linalg.norm(new[3:7])
                        if old[3:7]@new[3:7]<0:new[3:7]*=-1
                        errors.append(float(np.max(np.abs(old-new))))
                        if errors[-1] >= 1e-4:
                            save(folder/'parity_failure.json',dict(frame=frame,old=old.tolist(),new=new.tolist(),difference=errors[-1]))
                            raise RuntimeError('Original streamed pose changed before spatial matching')
                    if frame in features:
                        path=folder/'features'/f'{frame:06d}.pt'
                        with path.open('xb') as stream:
                            torch.save(dict(frame=frame,rgb_sha256=case['rgb_sha256'][frame],
                                uv=torch.from_numpy(uv), raw_uv=torch.from_numpy(raw_uv),
                                **{k:v[j].clone() for k,v in captured.items()}),stream)
                        metadata.append(dict(frame=frame,file=str(path.relative_to(out)),sha256=sha(path),bytes=path.stat().st_size))
                del prediction,images,captured
                if start%64==0 or start+count==case['frames']:
                    print(json.dumps(dict(stage='extract',case=case['id'],frames=start+count,total=case['frames'],
                        saved=len(metadata),max_pose9_difference=max(errors,default=0),seconds=time.perf_counter()-started)),flush=True)
            arrays=folder/'stream.npz'
            np.savez_compressed(arrays,pose9=np.asarray(poses),keys=np.asarray(keys))
            receipt=dict(complete=True,frames=len(poses),features=metadata,
                original_pose9_max_abs_difference=max(errors),
                framewise_pose_comparison_count=len(errors),
                stream_sha256=sha(arrays),wall_seconds=time.perf_counter()-started,
                feature_bytes=sum(x['bytes'] for x in metadata),
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            save(folder/'extraction.json',receipt)
            receipts.append(dict(case=case['id'],extraction_sha256=sha(folder/'extraction.json')))
            del keys,poses
            gc.collect()
        verify_sources(plan)
        save(out/'extraction_complete.json',dict(complete=True,cases=receipts,plan_sha256=sha(out/'plan.json'),
            gpu=torch.cuda.get_device_name(),torch=torch.__version__))
    finally:
        capture.close()


def match(out):
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from MemNavData.gem_spatial_features import mutual_cosine
    from MemNavData.lingbot_pnp_localization import LightGluePointMatcher, correspondence_pnp_localize, SiftPnPConfig, jsonable_pnp
    from MemNavData.certified_relocalization_runtime import fundamental_support, fundamental_can_reach_certificate, certificate_decision
    plan=load(out/'plan.json')
    verify_sources(plan)
    assert load(out/'extraction_complete.json')['complete']
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    pairs=[]
    arrays={}
    for case in plan['cases']:
        receipt=load(out/case['id']/'extraction.json')
        # Original predictions are recorded for audit; substantial differences
        # require investigation before mixing them with archived geometry.
        assert receipt['original_pose9_max_abs_difference']<1e-4, receipt
        path=out/case['id']/'stream.npz'
        assert sha(path)==receipt['stream_sha256']
        with np.load(path) as data: arrays[case['id']]={k:data[k] for k in data.files}
        keys=torch.from_numpy(arrays[case['id']]['keys']).to('cuda')
        for query in case['queries']:
            current=query['current']
            eligible=[i for i in case['reference_keyframes'] if i<=current-plan['exclude_recent']]
            # Original cosine operation on the complete descriptor prefix.
            scores=F.cosine_similarity(keys[current][None,None],keys[:current+1][None],dim=-1)[0].cpu().numpy()
            selected=sorted(eligible,key=lambda i:(-float(scores[i]),i))[:plan['candidate_top_k']]
            for rank,ref in enumerate(selected):
                pairs.append(dict(case=case['id'],**query,reference=ref,rank=rank,score=float(scores[ref])))
    save(out/'selected_pairs.json',dict(pairs=pairs,selection='Image-only DINO on fixed native keyframe pool'))
    directory=out/'matches'
    directory.mkdir(exist_ok=False)
    matcher=LightGluePointMatcher(ROOT/'.diagnostics/dependencies/LightGlue',
        dependency_root=ROOT/'.diagnostics/dependencies/python',device='cuda:0',max_keypoints=2048,reference_cache_size=8)
    lookup={c['id']:c for c in plan['cases']}
    rows=[]
    for n,pair in enumerate(pairs):
        case=lookup[pair['case']]
        h,t=pair['reference'],pair['current']
        folder=out/case['id']
        saved={r['frame']:r for r in load(folder/'extraction.json')['features']}
        feature_data=[]
        for frame in (h,t):
            p=out/saved[frame]['file'];assert sha(p)==saved[frame]['sha256']
            feature_data.append(torch.load(p,map_location='cuda',weights_only=True))
        reference,current=feature_data
        geometrical=Path(case['source_directory'])/'buffer/ep_0001/geometry'/f'{h:06d}.npz'
        with np.load(geometrical) as geo:
            depth=geo['depth']*np.float32(geo['world_scale']);confidence=geo['confidence']
        pose=arrays[case['id']]['pose9'][h]
        for arm in ['superpoint_lightglue',*plan['descriptor_types']]:
            began=time.perf_counter()
            if arm=='superpoint_lightglue':
                matches=matcher.match_paths(Path(case['rgb_paths'][h]),Path(case['rgb_paths'][t]),
                    target_height=518,target_width=518,patch_size=14)
            else:
                i,j,affinity=mutual_cosine(reference[arm],current[arm])
                matches=dict(reference_points=reference['uv'].cpu().numpy()[i],
                    query_points=current['uv'].cpu().numpy()[j],
                    reference_raw_points=reference['raw_uv'].cpu().numpy()[i],
                    query_raw_points=current['raw_uv'].cpu().numpy()[j],
                    scores=(affinity+1)/2, cosine_affinity=affinity,
                    reference_raw_hw=np.asarray(case['raw_wh'][::-1]),query_raw_hw=np.asarray(case['raw_wh'][::-1]))
            torch.cuda.synchronize()
            match_ms=1000*(time.perf_counter()-began)
            cv2.setRNGSeed(0)
            evidence=fundamental_support(matches['reference_raw_points'],matches['query_raw_points'],matches['scores'],
                tuple(matches['reference_raw_hw']),tuple(matches['query_raw_hw']),threshold_px=plan['epipolar_threshold_px'])
            passed,reason=fundamental_can_reach_certificate(evidence)
            began=time.perf_counter()
            pnp=jsonable_pnp(correspondence_pnp_localize(matches['reference_points'],matches['query_points'],depth,confidence,pose,
                config=SiftPnPConfig(),match_scores=matches['scores'],epipolar_threshold_px=plan['epipolar_threshold_px']))
            certificate=certificate_decision(pnp)
            file=directory/f'{n:04d}_{arm}.npz'
            with file.open('xb') as stream:np.savez_compressed(stream,**{k:np.asarray(v) for k,v in matches.items()})
            row=dict(**pair,arm=arm,evidence=evidence,f_precheck=passed,f_reason=reason,pnp=pnp,certificate=certificate,
                original_frontend_pass=bool(passed and certificate['accepted']),
                match_ms=match_ms,pnp_and_storage_ms=1000*(time.perf_counter()-began),
                match_file=str(file.relative_to(out)),match_sha256=sha(file),reference_geometry_sha256=sha(geometrical))
            rows.append(row)
            with (out/'matching_progress.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        if n%8==7:
            print(json.dumps(dict(stage='match',pairs=n+1,total=len(pairs))),flush=True)
    verify_sources(plan)
    save(out/'inference.json',dict(complete=True,rows=rows,pairs=len(pairs),plan_sha256=sha(out/'plan.json'),
        selected_pairs_sha256=sha(out/'selected_pairs.json'),navigation_executed=False,updated_map=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','extract','match'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.action](args.out.resolve())
