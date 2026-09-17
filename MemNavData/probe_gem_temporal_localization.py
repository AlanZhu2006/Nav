"""Frozen native-motion multiview probe; no ground-truth inference inputs."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from MemNavData.probe_gem_loop_observer import load,save,sha,verify,MAST,UPSTREAM
from MemNavData.gem_loop_observer import PairRaster,official_loop_support,sample_current_grid

PREVIOUS=ROOT/'.diagnostics/gem_loop_observer_20260915'
OLD=PREVIOUS/'attempt_002'


def freeze(out):
    oldplan=load(OLD/'plan.json')
    verify(oldplan)
    receipt=load(PREVIOUS/'completion.json')
    for name,digest in receipt['files_sha256'].items():
        assert sha(name)==digest,name
    previous=load(OLD/'inference.json')
    lookup={(r['case'],r['current'],r['reference']):r for r in previous['rows'] if r['arm']=='mast'}
    queries=[]
    for q in load(OLD/'selections.json')['queries']:
        chosen=next((lookup[q['case'],q['current'],h] for h in q['dino'] if lookup[q['case'],q['current'],h]['accepted']),None)
        queries.append(dict(case=q['case'],current=q['current'],reference=chosen['reference'] if chosen else None,
            views=[q['current']-14,q['current']-7,q['current']],leg=q['leg'],phase=q['phase']))
    assert sum(q['reference'] is not None for q in queries)==22
    files=[Path(__file__),ROOT/'MemNavData/gem_temporal_localization.py',
        ROOT/'MemNavData/test_gem_temporal_localization.py',ROOT/'MemNavData/score_gem_temporal_localization.py',
        ROOT/'MemNavData/GEM_TEMPORAL_LOCALIZATION_PROTOCOL_20260915.md',PREVIOUS/'completion.json']
    files += [OLD/name for name in ('plan.json','inference.json','mast_inference.json','selections.json')]
    cases={c['id']:c for c in oldplan['cases']}
    for q in queries:
        if q['reference'] is None:
            continue
        c=cases[q['case']];t,h=q['current'],q['reference']
        assert 0<=h<=t-64 and min(q['views'])>h
        geometry=Path(c['source_directory'])/'buffer/ep_0001/geometry'/f'{h:06d}.npz'
        assert sha(geometry)==lookup[q['case'],t,h]['reference_geometry_sha256']
        files.append(geometry)
        for i in q['views']+[h]:
            assert sha(c['rgb_paths'][i])==c['rgb_sha256'][i]
    import scipy
    out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(queries=queries,cases=oldplan['cases'],weights=oldplan['weights'],
        sources={**oldplan['sources'],**{str(p):sha(p) for p in files}},
        point_budget=2048,robust_px=3.,max_nfev=100,scipy_version=scipy.__version__,
        arms=['current_rigid','shared_rigid','shared_similarity'],primary_arm='shared_similarity',
        selected_queries=22,original_queries=36,expected_new_pairs=44,ground_truth_in_inference=False,
        selection='Previous DINO+MASt3R first-pass reference, frozen before new evidence'))
    print(dict(frozen=str(out),selected_queries=22,new_pairs=44),flush=True)


def match(out):
    plan=load(out/'plan.json');verify(plan)
    sys.path[:0]=[str(UPSTREAM),str(UPSTREAM/'thirdparty/mast3r')]
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from unittest.mock import patch
    from mast3r_slam.config import config,load_config
    from mast3r_slam.mast3r_utils import load_mast3r,resize_img,mast3r_match_symmetric
    from mast3r_slam.global_opt import FactorGraph
    import mast3r_slam.global_opt as official_graph
    torch.set_num_threads(4);torch.manual_seed(0)
    load_config(str(UPSTREAM/'config/base.yaml'))
    assert config['local_opt']['Q_conf']==1.5 and config['local_opt']['min_match_frac']==.1
    assert config['dataset']['img_downsample']==1
    model=load_mast3r(plan['weights'][0]['file'],device='cuda:0').eval()
    torch.cuda.reset_peak_memory_stats()
    old=load(OLD/'mast_inference.json')
    old_features={(r['case'],r['frame']):r for r in old['encodings']}
    old_pairs={(r['case'],r['current'],r['reference']):r for r in old['rows']}
    cases={c['id']:c for c in plan['cases']}
    cache={};features=[];rows=[]
    (out/'features').mkdir();(out/'matches').mkdir()
    began=time.perf_counter()

    def frame(cid,i,t):
        assert i<=t
        key=(cid,i)
        if key not in cache:
            case=cases[cid]
            assert sha(case['rgb_paths'][i])==case['rgb_sha256'][i]
            reuse=old_features.get(key)
            if reuse:
                file=OLD/reuse['file'];assert sha(file)==reuse['sha256']
                payload=torch.load(file,map_location='cpu',weights_only=True)
                elapsed=None
            else:
                rgb=np.asarray(Image.open(case['rgb_paths'][i]).convert('RGB'))
                resized,transform=resize_img(rgb.astype(np.float32)/255.,512,return_transformation=True)
                torch.cuda.synchronize();tic=time.perf_counter()
                with torch.inference_mode():
                    feat,pos,_=model._encode_image(resized['img'].to('cuda:0'),resized['true_shape'])
                torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tic)
                raster=PairRaster(*resized['img'].shape[-2:],*transform,rgb.shape[1],rgb.shape[0])
                payload=dict(feat=feat.cpu(),pos=pos.cpu(),shape=torch.from_numpy(resized['true_shape']),raster=asdict(raster))
                file=out/'features'/f'{cid}_{i:06d}.pt'
                with file.open('xb') as stream:torch.save(payload,stream)
            cache[key]=payload
            features.append(dict(case=cid,frame=i,observed_prefix=t,reused=bool(reuse),
                file=str(file),sha256=sha(file),bytes=file.stat().st_size,encoder_ms=elapsed))
        p=cache[key]
        return SimpleNamespace(feat=p['feat'].to('cuda:0'),pos=p['pos'].to('cuda:0'),
            img_true_shape=p['shape'].to('cuda:0')),PairRaster(**p['raster'])

    for q in plan['queries']:
        if q['reference'] is None:continue
        cid,t,h=q['case'],q['current'],q['reference']
        original=old_pairs[cid,t,h]
        assert sha(OLD/original['file'])==original['sha256']
        rows.append(dict(original,file=str(OLD/original['file']),reused=True,anchor_current=t))
        historical,hr=frame(cid,h,t)
        for j in q['views'][:-1]:
            current,jr=frame(cid,j,t)
            torch.cuda.synchronize();tic=time.perf_counter()
            with torch.inference_mode():
                raw=mast3r_match_symmetric(model,historical.feat,historical.pos,current.feat,current.pos,
                    [historical.img_true_shape],[current.img_true_shape])
                support=official_loop_support(raw,confidence_threshold=1.5,minimum_fraction=.1)
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-tic)
            graph=FactorGraph(model,{0:historical,2:current},None,device='cuda:0')
            with patch.object(official_graph,'mast3r_match_symmetric',return_value=raw):
                passed=bool(graph.add_factors([0],[2],.1))
            assert passed==support['accepted']
            idx=raw[0][0].cpu().numpy().astype(np.int32)
            mask=support['current_valid'][0,:,0].cpu().numpy()
            sampled=sample_current_grid(mask,limit=2048)
            Q=support['current_Q'][0,:,0].cpu().numpy()
            hp,jp=hr.pixels(idx[sampled]),jr.pixels(sampled)
            file=out/'matches'/f'{cid}_{j}_{h}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream,reference_index_per_current=idx,
                    sampled_current_indices=sampled,scores=Q[sampled],Q_current=Q,
                    geometry_valid_current=raw[2][0,:,0].cpu().numpy(),
                    reference_points=hr.lingbot(hp),query_points=jr.lingbot(jp),
                    reference_raw_points=hr.raw(hp),query_raw_points=jr.raw(jp))
            row=dict(case=cid,current=j,reference=h,anchor_current=t,reused=False,
                official_pair_pass=passed,official_decision_exact=True,
                reference_fraction=support['reference_fraction'],current_fraction=support['current_fraction'],
                reference_raster=asdict(hr),current_raster=asdict(jr),pair_ms=elapsed,
                file=str(file),sha256=sha(file))
            rows.append(row)
            with (out/'matching_progress.jsonl').open('a') as stream:
                import json
                stream.write(json.dumps(row)+'\n')
            del current,raw,support,graph
        del historical
        print(dict(case=cid,current=t,completed_new_pairs=sum(not r['reused'] for r in rows)),flush=True)
    assert sum(not r['reused'] for r in rows)==44
    verify(plan)
    save(out/'matching.json',dict(complete=True,rows=rows,features=features,
        wall_seconds=time.perf_counter()-began,torch_peak_bytes=torch.cuda.max_memory_allocated(),
        plan_sha256=sha(out/'plan.json')))


def solve(out):
    import cv2
    from MemNavData.gem_reference_readout import pose_matrix
    from MemNavData.gem_temporal_localization import TemporalView,shared_transform
    from MemNavData.lingbot_pnp_localization import lift_reference_keypoints,intrinsics_from_pose9
    cv2.setNumThreads(1)
    plan=load(out/'plan.json');verify(plan)
    matching=load(out/'matching.json');assert matching['complete']
    pairs={(r['case'],r['anchor_current'],r['current']):r for r in matching['rows']}
    old={(r['case'],r['current'],r['reference']):r for r in load(OLD/'inference.json')['rows'] if r['arm']=='mast'}
    cases={c['id']:c for c in plan['cases']}
    poses={}
    for cid,c in cases.items():
        with np.load(c['native_stream']) as d:poses[cid]=d['pose9'].copy()
    (out/'view_inputs').mkdir()
    outputs=[]
    for q in plan['queries']:
        if q['reference'] is None:continue
        cid,t,h=q['case'],q['current'],q['reference']
        p=poses[cid]
        Th,Tt=pose_matrix(p[h]),pose_matrix(p[t])
        initial=np.linalg.solve(Th,Tt)
        geometry=Path(cases[cid]['source_directory'])/'buffer/ep_0001/geometry'/f'{h:06d}.npz'
        assert sha(geometry)==old[cid,t,h]['reference_geometry_sha256']
        with np.load(geometry) as d:
            depth=d['depth']*np.float32(d['world_scale']);confidence=d['confidence'].copy()
        local_pose=p[h].copy();local_pose[:7]=[0,0,0,0,0,0,1]
        views=[];evidence=[]
        for j in q['views']:
            record=pairs[cid,t,j];assert sha(record['file'])==record['sha256']
            L=np.eye(4) if j==t else np.linalg.solve(Tt,pose_matrix(p[j]))
            K=intrinsics_from_pose9(p[j],518,518)
            with np.load(record['file']) as d:
                ref=d['reference_points'].copy();pixels=d['query_points'].copy()
            raw_count=len(ref)
            F_mask=np.zeros(raw_count,dtype=bool)
            if record['official_pair_pass'] and raw_count>=8:
                cv2.setRNGSeed(0)
                _,mask=cv2.findFundamentalMat(ref.astype(np.float32),pixels.astype(np.float32),
                    cv2.USAC_MAGSAC,1.5,.999,10000)
                if mask is not None:F_mask=np.asarray(mask).reshape(-1).astype(bool)
            indices=np.flatnonzero(F_mask)
            X,valid=lift_reference_keypoints(ref[indices],depth,confidence,local_pose,confidence_quantile=0.)
            indices=indices[valid];X=X[valid];pixels=pixels[indices]
            if j==t:
                assert len(X)==old[cid,t,h]['pnp']['depth_valid_matches']
            views.append(TemporalView(j,X,pixels,K,L))
            file=out/'view_inputs'/f'{cid}_{t}_{j}_{h}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream,reference_points=X,pixels=pixels,intrinsic=K,
                    current_from_camera=L,original_match_indices=indices)
            evidence.append(dict(frame=j,official_pair_pass=record['official_pair_pass'],
                raw_count=raw_count,epipolar_count=int(F_mask.sum()),depth_valid_count=len(X),
                file=str(file),sha256=sha(file),match_file=record['file'],match_sha256=record['sha256']))
        arms={}
        for arm in plan['arms']:
            chosen=views[-1:] if arm=='current_rigid' else views
            arms[arm]=shared_transform(chosen,current=t,reference_from_current=initial,
                estimate_scale=arm=='shared_similarity',point_budget=plan['point_budget'],
                robust_px=plan['robust_px'],max_nfev=plan['max_nfev'])
        row=dict(case=cid,current=t,reference=h,views=evidence,arms=arms,
            native_relation=initial.tolist(),previous_pair_pose9=old[cid,t,h]['pnp']['pose9'],
            reference_geometry_sha256=sha(geometry))
        outputs.append(row)
        with (out/'solve_progress.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(dict(case=cid,current=t,usable_views=[len(v.pixels) for v in views],
            outputs={k:(v['status'],v.get('numerical_rank'),v.get('scale')) for k,v in arms.items()}),flush=True)
    verify(plan)
    save(out/'inference.json',dict(complete=True,rows=outputs,queries=plan['queries'],
        matching_sha256=sha(out/'matching.json'),plan_sha256=sha(out/'plan.json'),
        ground_truth_consumed=False,online_coordinate_update=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','match','solve'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    {'freeze':freeze,'match':match,'solve':solve}[args.action](args.out.resolve())
