"""Evaluator-only information-gain score for actual observed frame pairs."""
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
        median=float(np.median(values)) if values else None,p95=float(np.percentile(values,95)) if values else None,
        max=float(np.max(values)) if values else None)


def run(out):
    manifest,receipt=load(out/'manifest.json'),load(out/'completion.json')
    assert receipt['completed'] and receipt['inference_sha256']==sha(out/'inference.json')
    assert receipt['manifest_sha256']==sha(out/'manifest.json')
    assert all(sha(p)==h for p,h in receipt['geometry_receipt_sha256'].items())
    parent=Path(manifest['parent'])
    root=Path(manifest['source_root'])
    evaluations={c['id']:c for c in load(parent/'evaluation_inputs.json')['cases']}
    geometries,traces,episodes={},{},{}
    for case in manifest['cases']:
        identity=case['id']
        with np.load(parent/identity/'baseline.npz') as data: native=matrices(data['pose_enc'])
        with np.load(root/'reciprocal_001'/identity/'geometry.npz') as data: connected=matrices(data['pose9'])
        with np.load(root/'probe_001'/identity/'geometry.npz') as data: episodes[identity]=data['episode']
        geometries[identity]=dict(native64=native,connected=connected)
        ev=evaluations[identity]
        assert sha(ev['trace'])==ev['trace_sha256']
        traces[identity]={p['step']:truth(p) for p in load(ev['trace'])['poses']}
    rows=[]
    for record in load(out/'inference.json')['rows']:
        identity,h,t=record['case'],record['reference'],record['current']
        expected=np.linalg.solve(traces[identity][h],traces[identity][t])
        current_to_ref=np.linalg.inv(expected)[:3,3]
        distance=float(np.linalg.norm(current_to_ref[[0,2]]))
        row=dict(case=identity,prefix=t+1,reference=h,age_frames=t-h,
            interface_eligible=record['interface_eligible'],
            context_gap=int(episodes[identity][t]-episodes[identity][h]),
            horizontal_reference_distance_m=distance,arms={})
        for arm,result in record['arms'].items():
            poses=geometries[identity][arm]
            prior=errors(np.linalg.solve(poses[h],poses[t]),expected)
            observed=None
            if result['accepted']:
                pose=matrices(np.asarray(result['pnp']['pose9'])[None])[0]
                observed=errors(np.linalg.solve(poses[h],pose),expected)
                assert result['selected_anchor']==h
                assert result['pnp']['status']=='ok' and result['pnp']['inliers']>=16
                assert result['pnp']['reprojection_rmse_px']<=2.
                assert min(result['pnp']['reference_inlier_coverage'],result['pnp']['query_inlier_coverage'])>=.05
            if distance<.5:
                prior['bearing_deg']=None
                if observed is not None: observed['bearing_deg']=None
            if np.linalg.norm(expected[:3,3])<.5:
                prior['translation_direction_deg']=None
                if observed is not None: observed['translation_direction_deg']=None
            row['arms'][arm]=dict(accepted=result['accepted'],reason=result['reason'],
                prior=prior,observed=observed,pnp_inliers=result.get('pnp',{}).get('inliers'),
                pnp_rmse_px=result.get('pnp',{}).get('reprojection_rmse_px'))
        rows.append(row)
    assert len(rows)==receipt['queries']==38
    summary={}
    for scope in ('all','nonadjacent_contexts'):
        selected=[r for r in rows if scope=='all' or r['context_gap']>=2]
        summary[scope]={}
        for arm in ('native64','connected'):
            accepted=[r['arms'][arm] for r in selected if r['arms'][arm]['accepted']]
            metrics={}
            for metric in ('rotation_deg','translation_direction_deg','bearing_deg'):
                common=[r for r in accepted if r['prior'][metric] is not None and r['observed'][metric] is not None]
                metrics[metric]=dict(prior=describe([r['prior'][metric] for r in common]),
                    observed=describe([r['observed'][metric] for r in common]),
                    improved=sum(r['observed'][metric]<r['prior'][metric] for r in common),
                    worsened=sum(r['observed'][metric]>r['prior'][metric] for r in common))
            summary[scope][arm]=dict(queries=len(selected),
                interface_eligible=sum(r['interface_eligible'] for r in selected),
                unsupported_initial_scale_anchor=sum(not r['interface_eligible'] for r in selected),
                accepted=len(accepted),metrics=metrics)
    result=dict(complete=True,rows=rows,summary=summary,inference_sha256=sha(out/'inference.json'),
        scorer_sha256=sha(__file__),scope='Diagnostic of available observation constraints, not a graph optimizer or causal navigation improvement; paired errors conditioned on accepted local certificates; nearby-reference direction scores undefined below0.5m')
    (out/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    run(parser.parse_args().root.resolve())
