"""Score connected memory against native streaming and point alignment."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parent))
from score_gem_reactivation_probe import errors,sha,truth


def matrices(poses):
    out=np.broadcast_to(np.eye(4),(len(poses),4,4)).copy()
    out[:,:3,:3]=Rotation.from_quat(poses[:,3:7]).as_matrix()
    out[:,:3,3]=poses[:,:3]
    return out


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((args.out/'manifest.json').read_text())
    parent=Path(manifest['parent'])
    evaluation={c['id']:c for c in json.loads((parent/'evaluation_inputs.json').read_text())['cases']}
    rows=[]
    all_frame=[]
    for case in manifest['cases']:
        folder=args.out/case['id']
        if not (folder/'completion.json').exists():
            if args.partial:continue
            raise RuntimeError('Missing completed case '+case['id'])
        receipt=json.loads((folder/'completion.json').read_text())
        assert all(sha(folder/p)==v for p,v in receipt['files_sha256'].items())
        data=evaluation[case['id']]
        assert sha(data['trace'])==data['trace_sha256']
        trace={x['step']:x for x in json.loads(Path(data['trace']).read_text())['poses']}
        gt=np.stack([truth(trace[i]) for i in range(case['frame_count'])])
        with np.load(parent/case['id']/'baseline.npz') as a:
            native=matrices(a['pose_enc'])
        with np.load(folder/'geometry.npz') as a:
            controls=dict(native=native,camera=matrices(a['camera_pose9']),point=matrices(a['point_pose9']))
            scales={arm:dict(min=float(a[key].min()),max=float(a[key].max()),last=float(a[key][-1]))
                for arm,key in [('camera','scale'),('point','point_scale')]}
        # Fixed reference 8 is chosen by protocol, not by successful retrieval.
        # Also report exactly the observation-selected references from stage A.
        for prefix in case['checkpoints']:
            request=json.loads((parent/case['id']/f'packet_{prefix}.json').read_text())['request']
            for selection,h in [('fixed_reference_8',8),('stage_a_dino_reference',request['reference'])]:
                t=prefix-1
                expected=np.linalg.solve(gt[h],gt[t])
                row=dict(case=case['id'],prefix=prefix,selection=selection,reference=h)
                for arm,poses in controls.items():
                    row[arm]=errors(np.linalg.solve(poses[h],poses[t]),expected)
                rows.append(row)
        per_case=dict(case=case['id'],scale=scales,frames=case['frame_count'],resources={
            k:v for k,v in receipt.items() if k!='files_sha256'})
        for arm,poses in controls.items():
            expected=gt[8,:3,:3].T@gt[8:,:3,:3]
            predicted=poses[8,:3,:3].T@poses[8:,:3,:3]
            rot=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',expected,predicted)-1)/2,-1,1)))
            per_case[arm]=dict(rotation_from_frame8_mean=float(rot.mean()),
                rotation_from_frame8_median=float(np.median(rot)),rotation_from_frame8_p95=float(np.percentile(rot,95)))
        all_frame.append(per_case)
    metrics={}
    for selection in ('fixed_reference_8','stage_a_dino_reference'):
        metrics[selection]={}
        for metric in ('rotation_deg','translation_direction_deg','bearing_deg'):
            metrics[selection][metric]={}
            for arm in ('native','camera','point'):
                values=[r[arm][metric] for r in rows if r['selection']==selection and r[arm][metric] is not None]
                metrics[selection][metric][arm]=dict(n=len(values),
                    mean=float(np.mean(values)) if values else None,median=float(np.median(values)) if values else None,
                    p95=float(np.percentile(values,95)) if values else None)
    report=dict(rows=rows,all_frame=all_frame,metrics=metrics,complete_cases=len(all_frame),
        expected_cases=len(manifest['cases']),scope='Development camera geometry; not goal PnP or navigation SR')
    path=args.out/('partial_evaluation.json' if args.partial else 'evaluation.json')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','all_frame')},indent=2))


if __name__=='__main__':main()
