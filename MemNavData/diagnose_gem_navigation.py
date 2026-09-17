"""Posthoc first-query bearing diagnosis; no model or controller execution.

Initial queries share the executed start pose across arms. Later trajectories
can diverge, so this diagnostic does not pool their steering errors as paired
recall trials. Ground truth is used here only after execution.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def diagnose(root,allow_partial=False):
    root=Path(root).resolve()
    plan=load(root/'plan.json')
    rows=[]
    for cell in plan['cells']:
        for j,mode in enumerate(plan['modes']):
            folder=root/'tasks'/str(cell['index']*len(plan['modes'])+j)
            audit_path=folder/'independent_verification.json'
            if not audit_path.exists():
                assert allow_partial, f'Missing verified task: {folder.name}'
                continue
            audit=load(audit_path)
            assert audit['verified'] and audit['cell']==cell and audit['mode']==mode
            assert audit['plan_sha256']==sha(root/'plan.json')
            assert audit['manifest_sha256']==sha(folder/'manifest.json')
            assert audit['summary_sha256']==sha(folder/'summary.json')
            for record in audit['records']:
                target=folder/'evaluation'/record['role']
                for name in ('query_plans.json','full_plan_outputs.jsonl'):
                    p=target/name
                    assert sha(p)==audit['artifact_sha256'][str(p)]
                query=load(target/'query_plans.json')['query_leg'][0]
                with (target/'full_plan_outputs.jsonl').open() as stream:
                    executed=json.loads(next(stream))
                assert query['step']==executed['next_action_index']==0
                assert executed['plan_index']==0
                goal=record['goal_xz']
                position=executed['position']
                dx,dz=goal[0]-position[0],goal[1]-position[2]
                yaw=executed['yaw']
                forward=-math.sin(yaw)*dx-math.cos(yaw)*dz
                left=-math.cos(yaw)*dx+math.sin(yaw)*dz
                distance=math.hypot(forward,left)
                truth=math.atan2(left,forward)
                bearing=query['memory_bearing_unit']
                accepted=query['certified_relocalization_accepted'] is True
                takeover=query['revisit_adapter_takeover'] is True
                error=None
                if takeover:
                    assert accepted and bearing is not None and len(bearing)==2
                    assert math.isclose(math.hypot(*bearing),1,abs_tol=1e-5)
                    if distance>=.5:
                        predicted=math.atan2(bearing[1],bearing[0])
                        delta=predicted-truth
                        error=abs(math.degrees(math.atan2(math.sin(delta),math.cos(delta))))
                pnp=query.get('certified_relocalization_pnp') or {}
                rows.append(dict(index=cell['index'],dataset=cell['dataset'],scene=cell['scene'],
                    episode=cell['episode'],mode=mode,role=record['role'],
                    candidate_design_scene=cell['candidate_design_scene'],
                    reached=record['reached'],steps=record['steps'],spl=record['spl'],
                    actual_path_m=record['actual_path_m'],takeover_plans=record['takeover_plans'],
                    start_position=position,start_yaw=yaw,goal_xz=goal,
                    first_query_accepted=accepted,first_query_takeover=takeover,
                    first_query_anchor=query['anchor'],first_query_frame=query['frame_idx'],
                    first_query_bearing=bearing,first_query_bearing_error_deg=error,
                    initial_planar_goal_distance_m=distance,
                    pnp_inliers=pnp.get('inliers'),pnp_rmse_px=pnp.get('reprojection_rmse_px'),
                    depth_median_m=query['monocular_depth_receipt']['depth_nonzero_median_m'],
                    task_verification_sha256=sha(audit_path)))
    complete=len(rows)==plan['total_rollouts']
    indexed={(r['index'],r['role'],r['mode']):r for r in rows}
    discordances=[]
    for cell in plan['cells']:
        for role in ('novel','revisit'):
            keys=[(cell['index'],role,m) for m in plan['modes']]
            if not all(k in indexed for k in keys):continue
            arms={k[2]:indexed[k] for k in keys}
            for r in arms.values():
                assert all(r[n]==arms['legacy'][n] for n in ('start_position','start_yaw','goal_xz'))
            if len({r['reached'] for r in arms.values()})>1:
                discordances.append(dict(index=cell['index'],role=role,arms=arms))
    result=dict(complete=complete,rows=len(rows),expected_rows=plan['total_rollouts'],
        plan_sha256=sha(root/'plan.json'),records=rows,success_discordances=discordances,
        interpretation='Posthoc diagnosis of the verified fixed population' if complete else
            'INCOMPLETE posthoc diagnosis; not a population result',
        scope='Ground-truth first-query bearings only; no estimator changes or reruns',
        diagnostic_sha256=sha(__file__))
    path=root/'first_query_diagnosis.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete','rows','expected_rows')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    diagnose(args.root,args.allow_partial)
