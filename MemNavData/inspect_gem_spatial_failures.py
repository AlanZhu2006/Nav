"""Post-hoc CPU forensics of saved correspondences, not a new method trial.

Recover the original hidden RANSAC inlier identities for five explicit cases;
check returned pose parity before attributing properties to those inliers.
Evaluator pose/calibration are used only for analysis, never optimization.
"""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from MemNavData.probe_gem_spatial_revisit import load,save,sha,verify_sources
from MemNavData.gem_reference_readout import pose_matrix
from MemNavData.score_gem_reactivation_probe import truth
from MemNavData.lingbot_pnp_localization import (lift_reference_keypoints,
    solve_camera_pose_pnp,intrinsics_from_pose9,SiftPnPConfig,map_raw_intrinsic_to_lingbot_pad)


def stats(values):
    v=np.asarray(values,dtype=float)
    v=v[np.isfinite(v)]
    return dict(n=len(v),median=float(np.median(v)) if len(v) else None,
                p95=float(np.percentile(v,95)) if len(v) else None,
                max=float(np.max(v)) if len(v) else None)


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])


def sampson(points0,points1,F):
    a=np.c_[points0,np.ones(len(points0))]
    b=np.c_[points1,np.ones(len(points1))]
    fa=a@F.T;fb=b@F
    return abs(np.sum(b*fa,axis=1))/np.sqrt(np.sum(fa[:,:2]**2+fb[:,:2]**2,axis=1))


def main():
    root=ROOT/'.diagnostics/gem_spatial_revisit_20260915'
    parent=root/'attempt_003'
    out=root/'failure_forensics_001'
    out.mkdir(exist_ok=False)
    plan=load(parent/'plan.json');verify_sources(plan)
    raw=load(parent/'inference.json')['rows']
    evaluation=load(parent/'evaluation.json')['rows']
    scored={(r['case'],r['current'],r['reference'],r['arm']):r for r in evaluation}
    cases={c['id']:c for c in plan['cases']}
    poses={}
    for cid in cases:
        with np.load(parent/cid/'stream.npz') as d:poses[cid]=d['pose9']
    gt={c['id']:[truth(p) for p in c['rows']] for c in load(parent/'evaluation_inputs.json')['cases']}
    selected=[('task_1',543,22,'gca_18'),('task_1',543,22,'superpoint_lightglue'),
              ('task_1',543,15,'gca_18'),('task_7',327,204,'superpoint_lightglue'),
              ('task_7',328,218,'superpoint_lightglue')]
    calibration_source=ROOT/'MemNavData/HABITAT_MINIMAL_EXECUTION_AUDIT_20260908.md'
    K_raw=np.array([[355.8146095,0,240],[0,355.8146060,135],[0,0,1.]])
    K_real=map_raw_intrinsic_to_lingbot_pad(K_raw,raw_height=270,raw_width=480,
        target_height=518,target_width=518,patch_size=14)
    save(out/'plan.json',dict(posthoc=True,parent_inference_sha256=sha(parent/'inference.json'),
        selected=selected,selection_reason='Previously reported near-zero-motion GCA estimate with same-pair SP control, another collapsed GCA example, and both consecutive catastrophic SP estimates',
        geometry_optimizer_uses_gt=False,calibration_for_evaluation_only=K_raw.tolist(),
        calibration_source=str(calibration_source),calibration_source_sha256=sha(calibration_source),
        source_sha256={str(Path(__file__)):sha(__file__)},
        model_forward=False,new_matching=False,new_rendering=False,thresholds_changed=False))
    cv2.setNumThreads(1)
    cfg=SiftPnPConfig()
    full=[];details=[]
    for record in raw:
        cid,t,h,arm=record['case'],record['current'],record['reference'],record['arm']
        item=scored[cid,t,h,arm]
        if record['original_frontend_pass']:
            Tp,Tn,Th=[pose_matrix(p) for p in (record['pnp']['pose9'],poses[cid][t],poses[cid][h])]
            n=float(np.linalg.norm(Tn[:3,3]-Th[:3,3]));p=float(np.linalg.norm(Tp[:3,3]-Th[:3,3]))
            full.append(dict(case=cid,current=t,reference=h,arm=arm,
                direction_scored=item['observed']['bearing_deg'] is not None,
                native_baseline_raw=n,observed_baseline_raw=p,baseline_ratio=p/n if n>1e-12 else None,
                bearing_deg=item['observed']['bearing_deg'],cheirality=record['pnp']['cheirality_fraction']))
        if (cid,t,h,arm) not in selected:continue
        matchfile=parent/record['match_file'];assert sha(matchfile)==record['match_sha256']
        with np.load(matchfile) as m:
            p0=m['reference_points'].copy();p1=m['query_points'].copy()
        geometry=Path(cases[cid]['source_directory'])/'buffer/ep_0001/geometry'/f'{h:06d}.npz'
        assert sha(geometry)==record['reference_geometry_sha256']
        with np.load(geometry) as d:
            depth=d['depth']*np.float32(d['world_scale']);confidence=d['confidence']
        # The only repeated solve recovers the discarded original mask.
        cv2.setRNGSeed(0)
        F,mask=cv2.findFundamentalMat(p0.astype(np.float32),p1.astype(np.float32),cv2.USAC_MAGSAC,1.5,.999,10000)
        ids=np.flatnonzero(mask.reshape(-1).astype(bool))
        points,valid=lift_reference_keypoints(p0[ids],depth,confidence,poses[cid][h],confidence_quantile=0.)
        ids=ids[valid];points=points[valid]
        K=intrinsics_from_pose9(poses[cid][h],518,518)
        solved=solve_camera_pose_pnp(points,p1[ids],K,config=cfg,fov_pose9=poses[cid][h])
        old=np.asarray(record['pnp']['pose9']);new=solved['pose9'].copy()
        if old[3:7]@new[3:7]<0:new[3:7]*=-1
        delta=float(np.max(abs(old-new)))
        assert delta<1e-10 and solved['inliers']==record['pnp']['inliers']
        inliers=np.asarray(solved['inlier_indices']);original_ids=ids[inliers]
        u0,u1=p0[original_ids],p1[original_ids]
        gt_h_from_t=np.linalg.solve(gt[cid][h],gt[cid][t]);gt_t_from_h=np.linalg.inv(gt_h_from_t)
        F_gt=np.linalg.inv(K_real).T@skew(gt_t_from_h[:3,3])@gt_t_from_h[:3,:3]@np.linalg.inv(K_real)
        gt_epipolar=sampson(u0,u1,F_gt)
        displacement=np.linalg.norm(u1-u0,axis=1)
        # Triangulation uses GT cameras only as an evaluator-side consistency
        # check; it is not substitute depth and is not fed to PnP.
        q0=(np.c_[u0,np.ones(len(u0))]@np.linalg.inv(K_real).T)[:,:2]
        q1=(np.c_[u1,np.ones(len(u1))]@np.linalg.inv(K_real).T)[:,:2]
        xyz4=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],gt_t_from_h[:3],q0.T,q1.T)
        xyz=(xyz4[:3]/xyz4[3]).T
        xyz_t=xyz@gt_t_from_h[:3,:3].T+gt_t_from_h[:3,3]
        positive=(xyz[:,2]>0)&(xyz_t[:,2]>0)
        singular=np.linalg.svd(points[inliers]-points[inliers].mean(0),compute_uv=False)
        details.append(dict(case=cid,current=t,reference=h,arm=arm,accepted=record['original_frontend_pass'],
            pose_reproduction_max_abs_error=delta,inliers=len(inliers),
            identical_pixel_inliers=int(np.count_nonzero(displacement<1e-6)),
            same_row_inliers=int(np.count_nonzero(abs(u0[:,1]-u1[:,1])<1e-6)),
            inlier_displacement_px=stats(displacement),
            saved_reprojection_rmse_px=record['pnp']['reprojection_rmse_px'],
            predicted_pointcloud_singular_values=singular.tolist(),
            predicted_pointcloud_smallest_over_largest=float(singular[-1]/singular[0]),
            gt_epipolar_sampson_px=stats(gt_epipolar),
            gt_epipolar_under1_5_px=int(np.count_nonzero(gt_epipolar<=1.5)),
            gt_triangulation_positive_in_both=int(positive.sum()),
            original_record=record,evaluation=item))
        np.savez_compressed(out/f'{cid}_{t}_{h}_{arm}.npz',reference_points=u0,query_points=u1,
            original_match_indices=original_ids,world_points=points[inliers],
            gt_sampson=gt_epipolar,gt_triangulated_reference_xyz=xyz,
            gt_triangulated_current_xyz=xyz_t)
    summary={}
    for arm in ('superpoint_lightglue','dino_patch','gca_18','gca_24'):
        rows=[r for r in full if r['arm']==arm and r['direction_scored']]
        ratios=[r['baseline_ratio'] for r in rows if r['baseline_ratio'] is not None]
        summary[arm]=dict(scored_accepted=len(rows),baseline_ratio=stats(ratios),
            ratio_under_1e_4=sum(x<1e-4 for x in ratios),ratio_under_0_1=sum(x<.1 for x in ratios),
            all_accepted_points_in_front=all(r['cheirality']==1 for r in full if r['arm']==arm))
    save(out/'analysis.json',dict(complete=True,posthoc=True,summary=summary,all_accepted=full,details=details,
        plan_sha256=sha(out/'plan.json'),source_sha256=sha(__file__),
        caution='GT epipolar residual rules out inconsistent correspondences but low residual alone does not prove identity. This is selected failure diagnosis, not a held-out population or new threshold.'))
    print(summary)
    for r in details:print({k:v for k,v in r.items() if k not in ('original_record','evaluation')})


if __name__=='__main__':main()
