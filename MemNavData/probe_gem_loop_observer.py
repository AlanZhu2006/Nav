"""Freeze -> MASt3R reference evidence -> native-map transfer, without GT."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'MemNavData')]
from MemNavData.probe_gem_spatial_revisit import load, save, sha
from MemNavData.gem_loop_observer import CausalKeyframePool, PairRaster, official_loop_support, sample_current_grid

SOURCE = ROOT/'.diagnostics/gem_spatial_revisit_20260915'
MAST = ROOT/'.diagnostics/gem_mast3r_slam_20260913'
UPSTREAM = MAST/'upstream'


def verify(plan):
    for name, digest in plan['sources'].items():
        assert sha(name) == digest, name


def freeze(out, reuse_features_from=None):
    original = load(SOURCE/'attempt_003/plan.json')
    for name, digest in load(SOURCE/'completion.json')['files_sha256'].items():
        assert sha(name) == digest, name
    reuse = []
    reuse_sources = []
    if reuse_features_from is not None:
        previous = load(reuse_features_from/'plan.json')
        receipt = load(reuse_features_from/'failure_receipt.json')
        assert receipt['completed_pair_results'] == 0
        assert sha(reuse_features_from/'plan.json') == receipt['plan_sha256']
        snapshot = reuse_features_from/'probe_source_at_failure.py'
        assert sha(snapshot) == receipt['source_snapshot_sha256'] == previous['sources'][str(Path(__file__))]
        assert sha(reuse_features_from/'mast.log') == receipt['log_sha256']
        for name, digest in previous['sources'].items():
            if name != str(Path(__file__)):
                assert sha(name) == digest, name
        reuse = receipt['features']
        case_lookup = {c['id']:c for c in original['cases']}
        assert len({(r['case'], r['frame']) for r in reuse}) == len(reuse)
        for record in reuse:
            assert sha(record['file']) == record['sha256']
            assert case_lookup[record['case']]['rgb_sha256'][record['frame']] == record['rgb_sha256']
        reuse_sources = [reuse_features_from/p for p in
            ('plan.json', 'failure_receipt.json', 'probe_source_at_failure.py', 'mast.log')]
        reuse_sources += [Path(r['file']) for r in reuse]
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT/'MemNavData/gem_loop_observer.py',
        ROOT/'MemNavData/GEM_LOOP_OBSERVER_PROTOCOL_20260915.md',
        ROOT/'MemNavData/score_gem_loop_observer.py',
        ROOT/'MemNavData/lingbot_pnp_localization.py', ROOT/'MemNavData/certified_relocalization_runtime.py',
        SOURCE/'attempt_003/plan.json', SOURCE/'attempt_003/selected_pairs.json',
        SOURCE/'attempt_003/inference.json', SOURCE/'current_intrinsics_001/inference.json',
        UPSTREAM/'config/base.yaml', UPSTREAM/'main.py']
    files += sorted((UPSTREAM/'mast3r_slam').glob('*.py'))
    files += sorted((UPSTREAM/'thirdparty/mast3r/mast3r').rglob('*.py'))
    files += sorted((UPSTREAM/'thirdparty/mast3r/dust3r/dust3r').rglob('*.py'))
    files += [Path(p) for p in load(MAST/'ENVIRONMENT.json')['compiled_modules']]
    files += reuse_sources
    cases = []
    for c in original['cases']:
        case = {k:c[k] for k in ('id', 'scene', 'source_directory', 'raw_wh', 'rgb_paths',
                                 'rgb_sha256', 'reference_keyframes', 'queries')}
        case['queries'] = sorted(case['queries'], key=lambda q:q['current'])
        assert len({q['current'] for q in case['queries']}) == len(case['queries'])
        stream = SOURCE/'attempt_003'/c['id']/'stream.npz'
        files.append(stream)
        case['native_stream'] = str(stream)
        cases.append(case)
    weights = load(MAST/'CHECKPOINTS.json')
    for item in weights:
        assert sha(item['file']) == item['sha256'], item['file']
    save(out/'plan.json', dict(cases=cases, candidate_budget=3, exclude_recent=64,
        transfer_point_cap=2048, known_failure_pairs=[['task_7', 327, 204], ['task_7', 328, 218]],
        sources={str(p):sha(p) for p in files}, weights=weights,
        reuse_features=reuse,
        interface_correction='Decoder true_shape is a device Tensor, as in official Frame; encoder artifacts unchanged',
        retrieval='Official ASMK k3 score>0.005 and original DINO first3, on identical causal keyframe pool',
        pairing='Union of both retrieved lists plus two separately labeled prior failures',
        selection='First accepted pair in original retriever order; no cross-verifier fallback',
        original_results_reused=True, native_geometry_changed=False, ground_truth_in_inference=False,
        scope='Consumed development traces; no online coordinate update or navigation'))
    print(json.dumps(dict(frozen=str(out), queries=sum(len(c['queries']) for c in cases))), flush=True)


def mast(out):
    plan = load(out/'plan.json')
    verify(plan)
    sys.path[:0] = [str(UPSTREAM), str(UPSTREAM/'thirdparty/mast3r')]
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from unittest.mock import patch
    from mast3r_slam.config import config, load_config
    from mast3r_slam.mast3r_utils import load_mast3r, load_retriever, resize_img, mast3r_match_symmetric
    from mast3r_slam.global_opt import FactorGraph
    import mast3r_slam.global_opt as official_graph
    torch.set_num_threads(4)
    torch.manual_seed(0)
    load_config(str(UPSTREAM/'config/base.yaml'))
    assert config['retrieval'] == {'k':3, 'min_thresh':.005}
    assert config['local_opt']['Q_conf'] == 1.5 and config['local_opt']['min_match_frac'] == .1
    model = load_mast3r(plan['weights'][0]['file'], device='cuda:0').eval()
    torch.cuda.reset_peak_memory_stats()
    old_pairs = load(SOURCE/'attempt_003/selected_pairs.json')['pairs']
    selections, rows, encodings, retrieval_times = [], [], [], []
    reusable = {(r['case'], r['frame']):r for r in plan.get('reuse_features', [])}
    folder = out/'mast_matches'
    folder.mkdir(exist_ok=False)
    began = time.perf_counter()
    for case in plan['cases']:
        cid = case['id']
        feature_dir = out/cid/'features'
        feature_dir.mkdir(parents=True, exist_ok=False)
        database = load_retriever(model, plan['weights'][1]['file'], device='cuda:0')
        database.model.eval()
        pool = CausalKeyframePool(case['reference_keyframes'], exclude_recent=plan['exclude_recent'])
        cache = {}

        def frame(index, now):
            if index > now:
                raise ValueError('Attempted to encode a future observation')
            if index not in cache:
                path = Path(case['rgb_paths'][index])
                assert sha(path) == case['rgb_sha256'][index]
                feature_file = feature_dir/f'{index:06d}.pt'
                reused = reusable.get((cid,index))
                if reused is not None:
                    assert sha(reused['file']) == reused['sha256']
                    assert reused['rgb_sha256'] == case['rgb_sha256'][index]
                    payload = torch.load(reused['file'], map_location='cpu', weights_only=True)
                    raster = PairRaster(**payload['raster'])
                    elapsed = None
                    os.link(reused['file'], feature_file)
                else:
                    rgb = np.asarray(Image.open(path).convert('RGB'))
                    result, transform = resize_img(rgb.astype(np.float32)/255., 512, return_transformation=True)
                    torch.cuda.synchronize()
                    tic = time.perf_counter()
                    with torch.inference_mode():
                        feat, pos, _ = model._encode_image(result['img'].to('cuda:0'), result['true_shape'])
                    torch.cuda.synchronize()
                    elapsed = 1000*(time.perf_counter()-tic)
                    raster = PairRaster(*result['img'].shape[-2:], *transform, rgb.shape[1], rgb.shape[0])
                    payload = dict(feat=feat.detach().cpu(), pos=pos.detach().cpu(),
                        shape=torch.from_numpy(result['true_shape']), raster=asdict(raster))
                    with feature_file.open('xb') as stream:
                        torch.save(payload, stream)
                assert isinstance(payload['shape'], torch.Tensor) and tuple(payload['shape'].shape) == (1,2)
                cache[index] = payload
                encodings.append(dict(case=cid, frame=index, observed_prefix=now,
                    rgb_sha256=case['rgb_sha256'][index], encoder_ms=elapsed,
                    reused=reused is not None, reused_from=reused['file'] if reused else None,
                    file=str(feature_file.relative_to(out)), sha256=sha(feature_file),
                    bytes=feature_file.stat().st_size, raster=asdict(raster)))
            p = cache[index]
            return SimpleNamespace(feat=p['feat'].to('cuda:0'), pos=p['pos'].to('cuda:0'),
                                   img_true_shape=p['shape'].to('cuda:0')), PairRaster(**p['raster'])

        for query in case['queries']:
            t = query['current']
            # ASMK receives only eligible old observations; current is query-only.
            added = pool.advance(t)
            for h in added:
                historical, _ = frame(h, t)
                with torch.inference_mode():
                    feature = database.prep_features(historical.feat)[0].cpu().numpy()
                ids = np.full(len(feature), database.kf_counter, dtype=np.int64)
                database.add_to_database(feature, ids, None)
                del historical
            assert database.kf_counter == len(pool.indexed)
            current, current_raster = frame(t, t)
            captured_query = []
            original_query = database.query

            def record_query(*args, **kwargs):
                result = original_query(*args, **kwargs)
                captured_query.append(result)
                return result

            torch.cuda.synchronize()
            tic = time.perf_counter()
            with torch.inference_mode(), patch.object(database, 'query', record_query):
                retrieved = database.update(current, add_after_query=False,
                    k=config['retrieval']['k'], min_thresh=config['retrieval']['min_thresh'])
            torch.cuda.synchronize()
            elapsed = 1000*(time.perf_counter()-tic)
            asmk = pool.resolve(retrieved)
            ranks, scores, _ = captured_query[0]
            score_map = dict(zip(ranks[0].tolist(), scores[0].tolist()))
            dino_rows = sorted([p for p in old_pairs if p['case']==cid and p['current']==t], key=lambda p:p['rank'])[:3]
            dino = [p['reference'] for p in dino_rows]
            assert all(h in pool.indexed for h in dino+asmk)
            selected = dict(case=cid, **query, dino=dino, asmk=asmk,
                asmk_scores=[float(score_map[i]) for i in retrieved], dino_scores=[p['score'] for p in dino_rows],
                eligible=list(pool.indexed), indexed_now=added, asmk_query_ms=elapsed)
            selections.append(selected)
            retrieval_times.append(elapsed)
            extras = [h for c,tt,h in plan['known_failure_pairs'] if c==cid and tt==t]
            for h in sorted(set(dino+asmk+extras)):
                historical, reference_raster = frame(h, t)
                torch.cuda.synchronize()
                tic = time.perf_counter()
                with torch.inference_mode():
                    raw = mast3r_match_symmetric(model, historical.feat, historical.pos,
                        current.feat, current.pos, [historical.img_true_shape], [current.img_true_shape])
                    support = official_loop_support(raw, confidence_threshold=1.5, minimum_fraction=.1)
                torch.cuda.synchronize()
                pair_ms = 1000*(time.perf_counter()-tic)
                # Exact upstream acceptance check, reusing decoded tensors.
                graph = FactorGraph(model, {0:historical, 2:current}, None, device='cuda:0')
                with patch.object(official_graph, 'mast3r_match_symmetric', return_value=raw):
                    official_pass = bool(graph.add_factors([0], [2], .1))
                assert official_pass == support['accepted']
                idx_i2j, idx_j2i, valid_j, valid_i, *_ = raw
                index_h = idx_i2j[0].detach().cpu().numpy().astype(np.int32)
                index_t = idx_j2i[0].detach().cpu().numpy().astype(np.int32)
                mask = support['current_valid'][0, :, 0].detach().cpu().numpy()
                samples = sample_current_grid(mask, limit=plan['transfer_point_cap'])
                ph = reference_raster.pixels(index_h[samples])
                pt = current_raster.pixels(samples)
                Q = support['current_Q'][0, :, 0].detach().cpu().numpy()
                filename = folder/f'{cid}_{t}_{h}.npz'
                with filename.open('xb') as stream:
                    np.savez_compressed(stream, reference_index_per_current=index_h,
                        current_index_per_reference=index_t,
                        geometry_valid_current=valid_j[0,:,0].detach().cpu().numpy(),
                        geometry_valid_reference=valid_i[0,:,0].detach().cpu().numpy(),
                        Q_current=Q, Q_reference=support['reference_Q'][0,:,0].detach().cpu().numpy(),
                        sampled_current_indices=samples, scores=Q[samples],
                        reference_raw_points=reference_raster.raw(ph), query_raw_points=current_raster.raw(pt),
                        reference_points=reference_raster.lingbot(ph), query_points=current_raster.lingbot(pt),
                        reference_raw_hw=np.array(case['raw_wh'][::-1]), query_raw_hw=np.array(case['raw_wh'][::-1]))
                record = dict(case=cid, current=t, reference=h,
                    official_pair_pass=official_pass, reference_fraction=support['reference_fraction'],
                    current_fraction=support['current_fraction'], official_decision_exact=True,
                    pair_ms=pair_ms, pnp_samples=len(samples), reference_raster=asdict(reference_raster),
                    current_raster=asdict(current_raster), file=str(filename.relative_to(out)), sha256=sha(filename),
                    extra_diagnostic_only=h not in dino+asmk)
                rows.append(record)
                with (out/'mast_progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                del historical, raw, support, graph
            print(json.dumps(dict(case=cid,current=t,asmk=asmk,dino=dino,unique_pairs=len(rows))), flush=True)
            del current
        del database, cache
    verify(plan)
    save(out/'selections.json', dict(queries=selections, plan_sha256=sha(out/'plan.json')))
    save(out/'mast_inference.json', dict(complete=True, rows=rows, encodings=encodings,
        wall_seconds=time.perf_counter()-began, torch_peak_bytes=torch.cuda.max_memory_allocated(),
        encoder_count=len(encodings), pair_count=len(rows), asmk_query_ms=retrieval_times,
        new_encoder_count=sum(not r['reused'] for r in encodings),
        reused_encoder_count=sum(r['reused'] for r in encodings),
        sources_verified=True, plan_sha256=sha(out/'plan.json'), selections_sha256=sha(out/'selections.json')))


def transfer(out):
    plan = load(out/'plan.json')
    verify(plan)
    import cv2
    from MemNavData.lingbot_pnp_localization import (LightGluePointMatcher, correspondence_pnp_localize,
        SiftPnPConfig, jsonable_pnp, intrinsics_from_pose9)
    from MemNavData.certified_relocalization_runtime import (fundamental_support,
        fundamental_can_reach_certificate, certificate_decision)
    cv2.setNumThreads(1)
    cases = {c['id']:c for c in plan['cases']}
    poses = {}
    for cid,c in cases.items():
        with np.load(c['native_stream']) as data:
            poses[cid] = data['pose9'].copy()
    old = {(r['case'],r['current'],r['reference']):r for r in
        load(SOURCE/'current_intrinsics_001/inference.json')['rows'] if r['arm']=='superpoint_lightglue'}
    master = load(out/'mast_inference.json')
    assert master['complete']
    rows, new_sp, reused_sp = [], 0, 0
    matcher = None
    (out/'sp_matches').mkdir(exist_ok=False)
    for counter,record in enumerate(master['rows']):
        cid,t,h = record['case'],record['current'],record['reference']
        case = cases[cid]
        geometry = Path(case['source_directory'])/'buffer/ep_0001/geometry'/f'{h:06d}.npz'
        with np.load(geometry) as data:
            depth = data['depth']*np.float32(data['world_scale'])
            confidence = data['confidence'].copy()
        for arm in ('sp','mast'):
            key = (cid,t,h)
            if arm=='sp' and key in old:
                original = old[key]
                assert original['reference_geometry_sha256'] == sha(geometry)
                # Original control stores matches in its inherited match folder.
                matchfile = SOURCE/'current_intrinsics_001'/original['match_file']
                assert sha(matchfile) == original['match_sha256']
                row = dict(case=cid,current=t,reference=h,arm=arm,reused=True,
                    pnp=original['pnp'], certificate=original['certificate'],
                    frontend_pass=original['original_frontend_pass'], accepted=original['original_frontend_pass'],
                    f_precheck=original['f_precheck'], official_pair_pass=None,
                    match_file=str(matchfile),match_sha256=sha(matchfile),reference_geometry_sha256=sha(geometry))
                rows.append(row)
                reused_sp += 1
                continue
            if arm=='sp':
                if matcher is None:
                    import torch
                    torch.set_num_threads(4)
                    matcher = LightGluePointMatcher(ROOT/'.diagnostics/dependencies/LightGlue',
                        dependency_root=ROOT/'.diagnostics/dependencies/python', device='cuda:0',
                        max_keypoints=2048, reference_cache_size=8)
                for idx in (h,t):
                    assert sha(case['rgb_paths'][idx]) == case['rgb_sha256'][idx]
                tic = time.perf_counter()
                matches = matcher.match_paths(Path(case['rgb_paths'][h]),Path(case['rgb_paths'][t]),
                    target_height=518,target_width=518,patch_size=14)
                torch.cuda.synchronize()
                match_ms = 1000*(time.perf_counter()-tic)
                matchfile = out/'sp_matches'/f'{cid}_{t}_{h}.npz'
                with matchfile.open('xb') as stream:
                    np.savez_compressed(stream,**{k:np.asarray(v) for k,v in matches.items()})
                new_sp += 1
            else:
                matchfile = out/record['file']
                assert sha(matchfile) == record['sha256']
                with np.load(matchfile) as data:
                    matches = {k:data[k].copy() for k in ('reference_points','query_points',
                        'reference_raw_points','query_raw_points','scores','reference_raw_hw','query_raw_hw')}
                match_ms = record['pair_ms']
            cv2.setRNGSeed(0)
            evidence = fundamental_support(matches['reference_raw_points'],matches['query_raw_points'],matches['scores'],
                tuple(matches['reference_raw_hw']),tuple(matches['query_raw_hw']),threshold_px=1.5)
            passed,reason = fundamental_can_reach_certificate(evidence)
            tic = time.perf_counter()
            result = jsonable_pnp(correspondence_pnp_localize(matches['reference_points'],matches['query_points'],
                depth,confidence,poses[cid][h],config=SiftPnPConfig(),match_scores=matches['scores'],
                epipolar_threshold_px=1.5,query_intrinsic=intrinsics_from_pose9(poses[cid][t],518,518)))
            certificate = certificate_decision(result)
            frontend = bool(passed and certificate['accepted'])
            accepted = frontend and (arm=='sp' or record['official_pair_pass'])
            row = dict(case=cid,current=t,reference=h,arm=arm,reused=False,pnp=result,
                certificate=certificate, frontend_pass=frontend,accepted=accepted,f_precheck=passed,
                f_reason=reason,evidence=evidence, official_pair_pass=record['official_pair_pass'] if arm=='mast' else None,
                match_file=str(matchfile),match_sha256=sha(matchfile),reference_geometry_sha256=sha(geometry),
                match_ms=match_ms,pnp_ms=1000*(time.perf_counter()-tic))
            rows.append(row)
        with (out/'transfer_progress.jsonl').open('a') as stream:
            stream.write(json.dumps(rows[-2:],allow_nan=False)+'\n')
        if counter%12==11:
            print(json.dumps(dict(pairs=counter+1,total=len(master['rows']),reused_sp=reused_sp,new_sp=new_sp)),flush=True)
    verify(plan)
    save(out/'inference.json',dict(complete=True,rows=rows,new_sp_pairs=new_sp,reused_sp_pairs=reused_sp,
        plan_sha256=sha(out/'plan.json'),mast_sha256=sha(out/'mast_inference.json'),
        native_geometry_changed=False,ground_truth_consumed=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','mast','transfer'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reuse-features-from',type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.out, args.reuse_features_from)
    else:
        assert args.reuse_features_from is None, 'Feature reuse must be declared at freeze time'
        {'mast':mast,'transfer':transfer}[args.action](args.out)
