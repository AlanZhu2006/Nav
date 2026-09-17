"""Posthoc all-decision bearing diagnosis of audited continuous goal cycles."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
    return digest.hexdigest()


def load(path): return json.loads(Path(path).read_text())


def run(root):
    reduction=load(root/'independent_reduction.json')
    assert reduction['complete'] and reduction['verified_episodes']==12
    rows=[]
    receipts={}
    for folder in sorted((root/'tasks').iterdir(),key=lambda p:int(p.name)):
        audit_path=folder/'independent_verification.json'
        audit=load(audit_path)
        assert audit['verified']
        receipts[str(audit_path)]=sha(audit_path)
        for name,digest in audit['artifact_sha256'].items(): assert sha(name)==digest,name
        for target in sorted((folder/'evaluation').glob('leg_*')):
            if not target.is_dir(): continue
            query=load(target/'query.json')
            continuity=load(target/'continuity.json')
            actual=load(target/'actual_trace.json')
            poses={p['step']:p for p in actual['poses']}
            plans=[]
            for plan in actual['plans']:
                pose=poses[plan['step']]
                dx=query['floor_position'][0]-pose['x']
                dz=query['floor_position'][2]-pose['z']
                yaw=pose['yaw']
                forward=-math.sin(yaw)*dx-math.cos(yaw)*dz
                left=-math.cos(yaw)*dx+math.sin(yaw)*dz
                distance=math.hypot(forward,left)
                bearing=plan.get('memory_bearing_unit')
                error=None
                if bearing is not None and distance>=.5:
                    delta=math.atan2(bearing[1],bearing[0])-math.atan2(left,forward)
                    error=abs(math.degrees(math.atan2(math.sin(delta),math.cos(delta))))
                pnp=plan.get('certified_relocalization_pnp') or {}
                aux=plan.get('aux_pose')
                plans.append(dict(step=plan['step'],frame=plan.get('frame_idx'),
                    xz=[pose['x'],pose['z']],yaw=yaw,distance_m=distance,
                    bearing_error_deg=error,bearing=bearing,aux_pose=aux,
                    raw_goal_norm=math.hypot(*aux) if aux is not None else None,
                    accepted=plan.get('certified_relocalization_accepted'),
                    takeover=plan.get('revisit_adapter_takeover'),
                    cached=plan.get('certified_relocalization_cached'),
                    anchor=plan.get('anchor'),pnp_inliers=pnp.get('inliers'),
                    pnp_rmse_px=pnp.get('reprojection_rmse_px')))
            defined=[p['bearing_error_deg'] for p in plans if p['bearing_error_deg'] is not None]
            rows.append(dict(task=int(folder.name),scene=audit['scene'],mode=audit['mode'],
                leg=continuity['leg_index'],goal=query['name'],
                reached=actual['reached'],steps=actual['steps'],
                final_distance_m=actual['final_goal_dist_m'],termination=actual['termination_reason'],
                first_memory_index=continuity['first_memory_index'],
                goal_xz=[query['floor_position'][0],query['floor_position'][2]],
                first_error_deg=plans[0]['bearing_error_deg'],
                max_error_deg=max(defined) if defined else None,
                all_accepted=all(p['accepted'] is True for p in plans),
                plans=plans))
    assert len(rows)==reduction['actual_goal_legs']==35
    out=dict(complete=True,actual_legs=len(rows),rows=rows,
        verification_sha256=receipts,reduction_sha256=sha(root/'independent_reduction.json'),
        diagnostic_sha256=sha(__file__),
        scope='Evaluator-only posthoc diagnostic; all attempted legs and decisions retained; later paths differ across modes, so steering errors are not paired trials; raw goal norm is not a calibrated distance')
    (root/'continuous_diagnosis.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:v for k,v in row.items() if k!='plans'} for row in rows if not row['reached']],indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    run(parser.parse_args().root.resolve())
