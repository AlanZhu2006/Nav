"""Recheck saved PnP bearings and score every query role independently."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(values):
    return dict(n=len(values), median=float(np.median(values)) if values else None,
        mean=float(np.mean(values)) if values else None,
        p95=float(np.quantile(values,.95)) if values else None)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    manifest, completion=load(root/'manifest.json'),load(root/'completion.json')
    assert completion['completed'] and completion['sources_unchanged']
    assert sha(root/'inference.json')==completion['inference_sha256']
    inference=load(root/'inference.json')['rows']
    assert len(inference)==completion['queries']
    native, connected, reciprocal=[Path(manifest[k]) for k in ('native','connected','reciprocal')]
    parent=Path(load(native/'manifest.json')['parent'])
    cases={c['id']:c for c in manifest['cases']}
    evaluation={c['id']:c for c in load(root/'evaluation_inputs.json')['cases']}
    previous={(r['case'],r['prefix'],r['query']):r for r in load(root/'evaluation.json')['rows']}
    geometry, traces={},{}
    for identity,case in cases.items():
        with np.load(parent/identity/'baseline.npz') as a:
            native_pose=a['pose_enc']
        with np.load(connected/identity/'geometry.npz') as a:
            camera_pose=a['camera_pose9']
        with np.load(reciprocal/identity/'geometry.npz') as a:
            reciprocal_pose=a['pose9']
        geometry[identity]=dict(native=native_pose,camera=camera_pose,reciprocal=reciprocal_pose)
        ev=evaluation[identity]
        assert sha(ev['trace'])==ev['trace_sha256']
        traces[identity]={r['step']:r for r in load(ev['trace'])['poses']}
    expected={(c['id'],p,q['id']) for c in manifest['cases'] for p in c['checkpoints'] for q in manifest['queries'][c['id']]}
    seen, rows=set(),[]
    for record in inference:
        identity,prefix,qid=record['case'],record['prefix'],record['query']
        key=identity,prefix,qid
        assert key in expected and key not in seen
        seen.add(key)
        case=cases[identity]
        query=next(q for q in evaluation[identity]['queries'] if q['id']==qid)
        current=traces[identity][prefix-1]
        dx=query['goal_position_reporting_only'][0]-current['x']
        dz=query['goal_position_reporting_only'][2]-current['z']
        yaw=current['yaw']
        truth=np.array([-math.sin(yaw)*dx-math.cos(yaw)*dz,
                        -math.cos(yaw)*dx+math.sin(yaw)*dz])
        scored=dict(case=identity,prefix=prefix,query=qid,role=query['analysis_role'],
            complete_history=prefix==case['frame_count'],horizontal_goal_distance_m=float(np.linalg.norm(truth)),
            accepted={},bearing_deg={})
        ranking=None
        for mode,result in record['arms'].items():
            assert mode in geometry[identity] and result['frame_idx']==prefix-1
            anchors=[r['anchor'] for r in result['ranked_candidates']]
            assert all(8<=h<=prefix-2 for h in anchors)
            if ranking is None:ranking=result['ranked_candidates']
            assert result['ranked_candidates']==ranking
            accepted=bool(result['accepted'])
            scored['accepted'][mode]=accepted
            angle=None
            if accepted:
                assert result['selected_anchor'] in anchors
                assert result['selected_anchor_image_sha256']==case['rgb_sha256'][result['selected_anchor']]
                pnp=result['pnp']
                assert pnp['status']=='ok' and pnp['inliers']>=16 and pnp['reprojection_rmse_px']<=2.
                assert min(pnp['reference_inlier_coverage'],pnp['query_inlier_coverage'])>=.05
                # Runtime's existing bearing boundary casts current pose to
                # FP32; reproduce that representation, not the writer precision.
                pose=geometry[identity][mode][prefix-1].astype(np.float32).astype(float)
                local=Rotation.from_quat(pose[3:7]).inv().apply(np.asarray(pnp['pose9'])[:3]-pose[:3])
                prediction=np.array([local[2],-local[0]])
                np.testing.assert_allclose(prediction,result['aux_pose'],atol=2e-9,rtol=2e-9)
                if np.linalg.norm(truth)>=.5 and np.linalg.norm(prediction)>1e-10:
                    delta=math.atan2(prediction[1],prediction[0])-math.atan2(truth[1],truth[0])
                    angle=abs(math.degrees(math.atan2(math.sin(delta),math.cos(delta))))
            scored['bearing_deg'][mode]=angle
            old=previous[key]['errors'][mode]
            assert (old is None)==(angle is None)
            if angle is not None:
                assert abs(old-angle)<3e-6,(key,mode,old,angle)
        assert scored['accepted']==previous[key]['accepted']
        rows.append(scored)
    assert seen==expected
    summary={}
    for role in sorted({r['role'] for r in rows}):
        summary[role]={}
        for scope in ('complete_history','all_prefixes'):
            selected=[r for r in rows if r['role']==role and (scope=='all_prefixes' or r['complete_history'])]
            summary[role][scope]={mode:dict(queries=len(selected),accepted=sum(r['accepted'][mode] for r in selected),
                bearing_deg=describe([r['bearing_deg'][mode] for r in selected if r['bearing_deg'][mode] is not None]))
                for mode in ('native','camera','reciprocal')}
    result=dict(verified=True,queries=len(rows),summary=summary,rows=rows,
        scope='All fixed development queries, including rerendered history diagnostics; not navigation SR or independent prefix samples',
        pose_to_bearing_verified=True,all_candidate_rankings_shared=True,
        source_sha256={str(root/name):sha(root/name) for name in ('manifest.json','completion.json','inference.json','evaluation.json','evaluation_inputs.json')},
        auditor_sha256=sha(__file__))
    (root/'independent_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(verified=True,queries=len(rows),role_counts={r:summary[r]['all_prefixes']['native']['queries'] for r in summary})))


if __name__=='__main__':main()
