"""Evaluator-only fixed-reference geometry for a compact native control."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from score_gem_reactivation_probe import errors,truth,sha
from score_gem_connected_probe import matrices


def load(path): return json.loads(Path(path).read_text())


def describe(values):
    return dict(n=len(values),mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None,p95=float(np.percentile(values,95)) if values else None)


def run(root,partial=False):
    manifest=load(root/'manifest.json')
    parent=Path(manifest['parent'])
    assert sha(parent/'manifest.json')==manifest['parent_sha256']
    evaluation={c['id']:c for c in load(parent/'evaluation_inputs.json')['cases']}
    connected=root.parent/'reciprocal_001'
    rows,resources=[],[]
    for case in manifest['cases']:
        folder=root/case['id']
        if not (folder/'completion.json').exists():
            assert partial,case['id']
            continue
        receipt=load(folder/'completion.json')
        assert receipt['completed']
        assert all(sha(folder/p)==h for p,h in receipt['files_sha256'].items())
        ev=evaluation[case['id']]
        assert sha(ev['trace'])==ev['trace_sha256']
        trace={p['step']:p for p in load(ev['trace'])['poses']}
        gt=np.stack([truth(trace[i]) for i in range(case['frame_count'])])
        with np.load(parent/case['id']/'baseline.npz') as data: native=matrices(data['pose_enc'])
        with np.load(folder/'geometry.npz') as data: budget=matrices(data['pose_enc'])
        with np.load(connected/case['id']/'geometry.npz') as data: reciprocal=matrices(data['pose9'])
        controls=dict(native64=native,native16=budget,connected=reciprocal)
        for prefix in case['checkpoints']:
            request=load(parent/case['id']/f'packet_{prefix}.json')['request']
            for selection,anchor in (('fixed_reference_8',8),('fixed_dino_reference',request['reference'])):
                current=prefix-1
                expected=np.linalg.solve(gt[anchor],gt[current])
                rows.append(dict(case=case['id'],prefix=prefix,reference=anchor,selection=selection,
                    complete_history=prefix==case['frame_count'],
                    errors={arm:errors(np.linalg.solve(p[anchor],p[current]),expected) for arm,p in controls.items()}))
        resources.append(dict(case=case['id'],frames=case['frame_count'],
            wall_seconds=receipt['seconds'],gpu_peak_allocated_bytes=receipt['gpu_peak_allocated_bytes'],
            scope='One probe run including archival; not a repeated or exclusive resource benchmark'))
    summary={}
    for selection in ('fixed_reference_8','fixed_dino_reference'):
        summary[selection]={}
        for scope in ('all_prefixes','complete_history'):
            selected=[r for r in rows if r['selection']==selection and (scope=='all_prefixes' or r['complete_history'])]
            summary[selection][scope]={metric:{arm:describe([r['errors'][arm][metric] for r in selected
                if r['errors'][arm][metric] is not None]) for arm in ('native64','native16','connected')}
                for metric in ('rotation_deg','translation_direction_deg','bearing_deg')}
    report=dict(complete=len(resources)==len(manifest['cases']),cases=len(resources),expected_cases=len(manifest['cases']),
        rows=rows,summary=summary,resources=resources,scorer_sha256=sha(__file__),
        scope='Development camera geometry, fixed old references; not goal PnP or navigation SR')
    target=root/('partial_geometry_evaluation.json' if partial else 'geometry_evaluation.json')
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=report['complete'],cases=len(resources),summary=summary),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    run(args.root.resolve(),args.partial)
