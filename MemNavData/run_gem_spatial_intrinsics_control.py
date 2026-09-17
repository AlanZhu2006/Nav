"""One post-hoc CPU control: use the already predicted current camera K.

Reuses all sealed pairs, correspondences, reference geometry and thresholds.
No new matching, layers, oracle calibration, feature inference or navigation.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from MemNavData.probe_gem_spatial_revisit import load,save,sha,verify_sources


def freeze(parent,out):
    assert load(parent/'inference.json')['complete']
    plan=load(parent/'plan.json')
    verify_sources(plan)
    out.mkdir(exist_ok=False)
    plan.update(schema='gem_spatial_current_intrinsics_control_v1',
        parent=str(parent),parent_inference_sha256=sha(parent/'inference.json'),
        control='Post-hoc single control: original query_intrinsic=None replaced by current first-write LingBot K; all pairs and matching unchanged',
        pose_protocol='Original correspondence_pnp_localize and strict certificate; same reference geometry, current LingBot predicted query intrinsics')
    plan['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    save(out/'plan.json',plan)
    for name in ['evaluation_inputs.json','selected_pairs.json','matches',*[c['id'] for c in plan['cases']]]:
        (out/name).symlink_to(parent/name)
    from MemNavData.score_gem_spatial_revisit import freeze as freeze_scoring
    freeze_scoring(out)


def run(out):
    import numpy as np
    import cv2
    from MemNavData.lingbot_pnp_localization import (intrinsics_from_pose9,
        correspondence_pnp_localize,SiftPnPConfig,jsonable_pnp)
    from MemNavData.certified_relocalization_runtime import certificate_decision
    plan=load(out/'plan.json');verify_sources(plan)
    parent=Path(plan['parent'])
    assert sha(parent/'inference.json')==plan['parent_inference_sha256']
    original=load(parent/'inference.json')
    poses={}
    cases={c['id']:c for c in plan['cases']}
    for cid in cases:
        with np.load(parent/cid/'stream.npz') as d:poses[cid]=d['pose9']
    cv2.setNumThreads(1)
    rows=[];previous_geometry=None;depth=confidence=None
    began=time.perf_counter()
    for r in original['rows']:
        cid,h,t=r['case'],r['reference'],r['current']
        geometry=Path(cases[cid]['source_directory'])/'buffer/ep_0001/geometry'/f'{h:06d}.npz'
        if geometry!=previous_geometry:
            assert sha(geometry)==r['reference_geometry_sha256']
            with np.load(geometry) as d:depth=d['depth']*np.float32(d['world_scale']);confidence=d['confidence']
            previous_geometry=geometry
        file=parent/r['match_file'];assert sha(file)==r['match_sha256']
        intrinsic=intrinsics_from_pose9(poses[cid][t],518,518)
        before=time.perf_counter()
        with np.load(file) as m:
            pnp=jsonable_pnp(correspondence_pnp_localize(m['reference_points'],m['query_points'],
                depth,confidence,poses[cid][h],config=SiftPnPConfig(),match_scores=m['scores'],
                epipolar_threshold_px=plan['epipolar_threshold_px'],query_intrinsic=intrinsic))
        certificate=certificate_decision(pnp)
        row=deepcopy(r)
        row.update(pnp=pnp,certificate=certificate,
            original_frontend_pass=bool(r['f_precheck'] and certificate['accepted']),
            query_intrinsic=intrinsic.tolist(),query_intrinsic_source='current_first_write_LingBot_pose9',
            pnp_and_storage_ms=None,pnp_cpu_ms=1000*(time.perf_counter()-before),
            matching_rerun=False)
        rows.append(row)
    verify_sources(plan)
    save(out/'inference.json',dict(complete=True,rows=rows,pairs=original['pairs'],
        plan_sha256=sha(out/'plan.json'),selected_pairs_sha256=sha(out/'selected_pairs.json'),
        navigation_executed=False,updated_map=False,feature_or_match_rerun=False,
        cpu_wall_seconds=time.perf_counter()-began,parent_inference_sha256=sha(parent/'inference.json')))
    print(dict(complete=True,rows=len(rows),seconds=time.perf_counter()-began),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.parent.resolve(),args.out.resolve())
    else:run(args.out.resolve())
