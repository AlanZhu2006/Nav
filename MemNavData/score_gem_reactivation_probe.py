"""Post-inference relative geometry scoring; no model or query modification."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(row):
    c,s=np.cos(row['yaw']),np.sin(row['yaw'])
    transform=np.eye(4)
    transform[:3,:3]=[[c,0,-s],[0,-1,0],[-s,0,-c]]
    transform[:3,3]=[row[k] for k in ('x','y','z')]
    return transform


def angle(a,b):
    if np.linalg.norm(a)<1e-10 or np.linalg.norm(b)<1e-10:
        return None
    return float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))


def errors(prediction, expected):
    scale=float(np.cbrt(np.linalg.det(prediction[:3,:3])))
    if scale<=0 or not np.isfinite(prediction).all():
        raise ValueError('Invalid transform')
    rotation=prediction[:3,:3]/scale
    rotation_error=float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@expected[:3,:3])-1)/2,-1,1))))
    back=np.linalg.inv(prediction)[:3,3][[2,0]]*np.array([1,-1])
    expected_back=np.linalg.inv(expected)[:3,3][[2,0]]*np.array([1,-1])
    distance=float(np.linalg.norm(expected_back))
    return dict(rotation_deg=rotation_error,
        translation_direction_deg=angle(prediction[:3,3],expected[:3,3]),
        bearing_deg=angle(back,expected_back) if distance>=.5 else None,
        reference_distance_m=distance, scale=scale)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((args.out/'manifest.json').read_text())
    evaluation=json.loads((args.out/'evaluation_inputs.json').read_text())
    records=[]
    for case in evaluation['cases']:
        directory=args.out/case['id']
        if not (directory/'completion.json').exists():
            if args.partial: continue
            raise RuntimeError('Missing completed case '+case['id'])
        assert sha(case['trace'])==case['trace_sha256']
        trace={r['step']:r for r in json.loads(Path(case['trace']).read_text())['poses']}
        completion=json.loads((directory/'completion.json').read_text())
        assert all(sha(directory/p)==v for p,v in completion['files_sha256'].items())
        for row in completion['rows']:
            record=dict(case=case['id'],prefix=row['prefix'],status=row['status'])
            if row['status']=='complete':
                request=row['request']
                h,t=request['reference'],request['current']
                expected=np.linalg.solve(truth(trace[h]),truth(trace[t]))
                record.update(reference=h,similarity=request['similarity'],inference_ms=row['inference_ms'])
                with np.load(directory/f"packet_{row['prefix']}.npz") as arrays:
                    for name in ('baseline','raw','aligned'):
                        if name+'_relation' in arrays:
                            record[name]=errors(arrays[name+'_relation'],expected)
            records.append(record)
    metrics={}
    for metric in ('rotation_deg','translation_direction_deg','bearing_deg'):
        metrics[metric]={}
        for arm in ('baseline','raw','aligned'):
            values=[r[arm][metric] for r in records if arm in r and r[arm][metric] is not None]
            metrics[metric][arm]=dict(n=len(values),median=float(np.median(values)) if values else None,
                p95=float(np.percentile(values,95)) if values else None,
                mean=float(np.mean(values)) if values else None)
            if arm!='baseline':
                pairs=[(r['baseline'][metric],r[arm][metric]) for r in records
                    if arm in r and r[arm][metric] is not None and r['baseline'][metric] is not None]
                metrics[metric][arm].update(improved=sum(b<a for a,b in pairs),
                    worsened=sum(b>a for a,b in pairs),
                    worsened_over_5deg=sum(b>a+5 for a,b in pairs))
    report=dict(rows=records,metrics=metrics,complete_cases=len({r['case'] for r in records}),
        expected_cases=len(manifest['cases']),expected_prefixes=sum(len(c['checkpoints']) for c in manifest['cases']),
        observed_prefixes=len(records),scope='Development relative camera geometry, not navigation SR or independent episodes',
        manifest_sha256=sha(args.out/'manifest.json'))
    name='partial_evaluation.json' if args.partial else 'evaluation.json'
    (args.out/name).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__': main()
