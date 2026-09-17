"""Score full-history current-to-frame8 geometry after support reestimation."""
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


def run(out,partial=False):
    manifest=load(out/'manifest.json')
    parent=Path(manifest['parent'])
    assert sha(parent/'manifest.json')==manifest['parent_manifest_sha256']
    evaluation={c['id']:c for c in load(parent/'evaluation_inputs.json')['cases']}
    rows=[]
    for case in manifest['cases']:
        folder=out/case['id']
        if not (folder/'completion.json').exists():
            assert partial
            continue
        receipt=load(folder/'completion.json')
        assert receipt['completed'] and receipt['native_committed']==64
        assert all(sha(folder/p)==h for p,h in receipt['files_sha256'].items())
        ev=evaluation[case['id']]
        assert sha(ev['trace'])==ev['trace_sha256']
        trace={p['step']:truth(p) for p in load(ev['trace'])['poses']}
        with np.load(folder/'geometry.npz') as data:
            support=matrices(data['pose_enc'])
            assert data['source_frames'].tolist()==case['support_frames']
        with np.load(parent/case['id']/'baseline.npz') as data: native=matrices(data['pose_enc'])
        with np.load(out.parent/'native_budget16_001'/case['id']/'geometry.npz') as data: budget=matrices(data['pose_enc'])
        with np.load(out.parent/'reciprocal_001'/case['id']/'geometry.npz') as data: connected=matrices(data['pose9'])
        h,t=8,case['frame_count']-1
        expected=np.linalg.solve(trace[h],trace[t])
        measurements=dict(native64=errors(np.linalg.solve(native[h],native[t]),expected),
            native16=errors(np.linalg.solve(budget[h],budget[t]),expected),
            connected=errors(np.linalg.solve(connected[h],connected[t]),expected),
            temporal_support=errors(np.linalg.solve(support[8],support[-1]),expected))
        rows.append(dict(case=case['id'],frames=case['frame_count'],reference=h,current=t,errors=measurements,
            resources={k:receipt[k] for k in ('seconds','gpu_peak_allocated_bytes')},
            receipt_sha256=sha(folder/'completion.json')))
    summary={}
    for cohort in ('all','long_over1000'):
        selected=[r for r in rows if cohort=='all' or r['frames']>1000]
        summary[cohort]={metric:{arm:describe([r['errors'][arm][metric] for r in selected if r['errors'][arm][metric] is not None])
            for arm in ('native64','native16','connected','temporal_support')}
            for metric in ('rotation_deg','translation_direction_deg','bearing_deg')}
    result=dict(complete=len(rows)==len(manifest['cases']),cases=len(rows),expected_cases=len(manifest['cases']),
        rows=rows,summary=summary,scorer_sha256=sha(__file__),
        scope='One full-history endpoint per development history; current-to-original-frame8 geometry only; no target readout, no online maintenance, no SR or production latency claim')
    path=out/('partial_evaluation.json' if partial else 'evaluation.json')
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(complete=result['complete'],cases=len(rows),rows=rows),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    run(args.root.resolve(),args.partial)
