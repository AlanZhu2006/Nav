"""Development-only coordinate counterfactual: preserve the last shared camera.

Uses sealed local predictions. It does not change the running memory or its
frozen navigation experiment. Cached PnP witnesses are re-expressed through
their historical reference camera; no correspondence or acceptance changes.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from NavDP.baselines.memnav.gem.connected import pose_matrices
from NavDP.baselines.memnav.gem.reciprocal import reciprocal_camera_transport
from score_gem_reactivation_probe import errors,sha,truth


def save(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--connected',type=Path,required=True)
    parser.add_argument('--reciprocal',type=Path,required=True)
    parser.add_argument('--readout',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    source=json.loads((a.connected/'manifest.json').read_text())
    parent=Path(source['parent'])
    save(a.out/'manifest.json',dict(source=str(a.connected.resolve()),
        source_sha256=sha(a.connected/'manifest.json'),method='Last shared camera rigid pose; unchanged symmetric log-depth scale',
        source_script_sha256=sha(__file__),no_new_inference=True,posthoc_development=True,
        counterfactual_query='Re-express the same accepted PnP witness through its original reference camera',
        selection='All sealed cases and prefixes; no per-case switch or acceptance threshold'))
    geometries={}
    for case in source['cases']:
        folder=a.connected/case['id']
        with np.load(folder/'geometry.npz') as x:
            local,episodes,valid=x['local_pose9'],x['episode'],x['valid_pixels']
        transforms=[np.eye(4)]
        audits=[]
        for path in sorted((folder/'overlaps').glob('*.npz')):
            with np.load(path) as x:
                original,info=reciprocal_camera_transport(x['old_poses'],x['new_poses'],x['old_depth'],x['new_depth'],valid)
                rotation=x['old_poses'][-1,:3,:3]@x['new_poses'][-1,:3,:3].T
                scale=info['scale']
                translation=x['old_poses'][-1,:3,3]-scale*rotation@x['new_poses'][-1,:3,3]
                relation=np.eye(4);relation[:3,:3]=scale*rotation;relation[:3,3]=translation
                audits.append(dict(episode=int(path.stem),shared_reference=int(x['shared_frames'][-1]),
                    mean_transport_latest_position_residual=float(np.linalg.norm(
                        x['old_poses'][-1,:3,3]-(original[:3,:3]@x['new_poses'][-1,:3,3]+original[:3,3]))),
                    old_from_new=relation.tolist()))
                transforms.append(transforms[-1]@relation)
        transforms=np.array(transforms)
        world=transforms[episodes]@pose_matrices(local)
        scales=np.cbrt(np.linalg.det(transforms[episodes,:3,:3]))
        pose9=local.astype(np.float64).copy();pose9[:,:3]=world[:,:3,3]
        pose9[:,3:7]=Rotation.from_matrix(world[:,:3,:3]/scales[:,None,None]).as_quat()
        out=a.out/case['id'];out.mkdir()
        np.savez_compressed(out/'geometry.npz',pose9=pose9,scale=scales)
        save(out/'relations.json',audits)
        with np.load(a.reciprocal/case['id']/'geometry.npz') as x:
            np.testing.assert_allclose(scales,x['scale'],rtol=1e-12,atol=1e-12)
            geometries[case['id']]=dict(handoff=pose_matrices(pose9),reciprocal=pose_matrices(x['pose9']))
        with np.load(parent/case['id']/'baseline.npz') as x:
            geometries[case['id']]['native']=pose_matrices(x['pose_enc'])
    rows=[]
    inputs=json.loads((a.readout/'inference.json').read_text())['rows']
    for row in inputs:
        result=row['arms']['reciprocal']
        transformed=None
        if result['accepted']:
            h=int(result['selected_anchor'])
            geometry=geometries[row['case']]
            reference_change=geometry['handoff'][h]@np.linalg.inv(geometry['reciprocal'][h])
            goal_pose=pose_matrices(np.asarray(result['pnp']['pose9'])[None])[0]
            transformed=reference_change@goal_pose
        rows.append(dict(case=row['case'],prefix=row['prefix'],query=row['query'],
            accepted=result['accepted'],goal_pose=None if transformed is None else transformed.tolist()))
    save(a.out/'inference.json',dict(rows=rows))
    # Evaluator-only inputs start here.
    evaluation={c['id']:c for c in json.loads((parent/'evaluation_inputs.json').read_text())['cases']}
    scores=[]
    for row,original in zip(rows,inputs):
        case=next(c for c in source['cases'] if c['id']==row['case'])
        ev=evaluation[row['case']]
        assert sha(ev['trace'])==ev['trace_sha256']
        gt={r['step']:truth(r) for r in json.loads(Path(ev['trace']).read_text())['poses']}
        q=next(q for q in ev['queries'] if q['id']==row['query'])
        t=row['prefix']-1
        local=gt[t][:3,:3].T@(np.asarray(q['goal_position_reporting_only'])-gt[t][:3,3])
        expected=np.array([local[2],-local[0]])
        score=dict(case=row['case'],prefix=row['prefix'],query=row['query'],role=q['analysis_role'],
            complete_history=row['prefix']==case['frame_count'],accepted=row['accepted'])
        for arm in ('native','reciprocal','handoff'):
            prediction=original['arms'][arm].get('aux_pose') if arm!='handoff' else None
            if arm=='handoff' and row['accepted']:
                current=geometries[row['case']]['handoff'][t]
                relative=current[:3,:3].T@(np.asarray(row['goal_pose'])[:3,3]-current[:3,3])
                prediction=np.array([relative[2],-relative[0]])
            score[arm]=None
            if prediction is not None and np.linalg.norm(expected)>=.5 and np.linalg.norm(prediction)>1e-10:
                prediction=np.asarray(prediction).reshape(-1)
                score[arm]=float(np.degrees(np.arccos(np.clip(
                    prediction@expected/np.linalg.norm(prediction)/np.linalg.norm(expected),-1,1))))
        scores.append(score)
    summary={}
    for role in sorted({s['role'] for s in scores}):
        summary[role]={}
        for scope in ('complete_history','all_prefixes'):
            selected=[s for s in scores if s['role']==role and (scope=='all_prefixes' or s['complete_history'])]
            summary[role][scope]={}
            for arm in ('native','reciprocal','handoff'):
                values=[s[arm] for s in selected if s[arm] is not None]
                summary[role][scope][arm]=dict(n=len(values),median=float(np.median(values)) if values else None,
                    mean=float(np.mean(values)) if values else None,p95=float(np.percentile(values,95)) if values else None)
    save(a.out/'evaluation.json',dict(rows=scores,summary=summary,
        scope='Coordinate counterfactual of saved accepted goal witnesses; not new PnP or navigation'))
    print(json.dumps(summary['revisit'],indent=2))


if __name__=='__main__':main()
