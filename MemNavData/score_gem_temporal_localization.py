"""Post-inference GT scoring; every finite output, without accuracy gates."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from MemNavData.probe_gem_temporal_localization import load,save,sha,verify,PREVIOUS,OLD
from MemNavData.score_gem_reactivation_probe import truth,errors
from MemNavData.gem_reference_readout import pose_matrix
from MemNavData.score_gem_spatial_revisit import describe


def main(out):
    plan=load(out/'plan.json');verify(plan)
    inference=load(out/'inference.json');assert inference['complete']
    assert sha(out/'matching.json')==inference['matching_sha256']
    evaluation_source=ROOT/'.diagnostics/gem_spatial_revisit_20260915/attempt_003/evaluation_inputs.json'
    gt={c['id']:[truth(r) for r in c['rows']] for c in load(evaluation_source)['cases']}
    poses={}
    for c in plan['cases']:
        with np.load(c['native_stream']) as d:poses[c['id']]=d['pose9'].copy()
    rows=[]
    for record in inference['rows']:
        cid,t,h=record['case'],record['current'],record['reference']
        expected=np.linalg.solve(gt[cid][h],gt[cid][t])
        native=np.array(record['native_relation'])
        pair=np.linalg.solve(pose_matrix(poses[cid][h]),pose_matrix(record['previous_pair_pose9']))
        e=dict(case=cid,current=t,reference=h,native=errors(native,expected),pair=errors(pair,expected),arms={})
        for arm,output in record['arms'].items():
            metric=errors(np.array(output['transform']),expected) if output['transform'] is not None else None
            e['arms'][arm]=dict(output,error=metric)
        # Scale GT below is evaluator-only diagnostic of native recent motion.
        spans=[]
        for j in [v['frame'] for v in record['views'] if v['depth_valid_count']]:
            if j==t:continue
            native_span=np.linalg.norm(poses[cid][j,:3]-poses[cid][t,:3])
            true_span=np.linalg.norm(gt[cid][j][:3,3]-gt[cid][t][:3,3])
            spans.append(dict(frame=j,native_span=float(native_span),true_span_m=float(true_span)))
        e['recent_motion']=spans
        rows.append(e)
    summaries={}
    for arm in ['native','pair']+plan['arms']:
        metrics=[r[arm] if arm in ('native','pair') else r['arms'][arm]['error'] for r in rows]
        scored=[(r,m) for r,m in zip(rows,metrics) if m is not None and m['bearing_deg'] is not None]
        outputs=[] if arm in ('native','pair') else [r['arms'][arm] for r in rows]
        summaries[arm]=dict(selected_queries=len(rows),finite_outputs=sum(m is not None for m in metrics),
            direction_scored=len(scored),bearing_deg=describe([m['bearing_deg'] for _,m in scored]),
            rotation_deg=describe([m['rotation_deg'] for m in metrics if m is not None]),
            over15=sum(m['bearing_deg']>15 for _,m in scored),over90=sum(m['bearing_deg']>90 for _,m in scored),
            improved_over_pair=sum(m['bearing_deg']<r['pair']['bearing_deg'] for r,m in scored),
            worsened_over_pair5=sum(m['bearing_deg']>r['pair']['bearing_deg']+5 for r,m in scored),
            improved_over_native=sum(m['bearing_deg']<r['native']['bearing_deg'] for r,m in scored),
            worsened_over_native5=sum(m['bearing_deg']>r['native']['bearing_deg']+5 for r,m in scored),
            improved_over_current_fit=sum(m['bearing_deg']<r['arms']['current_rigid']['error']['bearing_deg']
                for r,m in scored if r['arms']['current_rigid']['error'] is not None),
            worsened_over_current_fit5=sum(m['bearing_deg']>r['arms']['current_rigid']['error']['bearing_deg']+5
                for r,m in scored if r['arms']['current_rigid']['error'] is not None),
            full_rank=sum(o['identified'] for o in outputs),converged=sum(o.get('optimizer_success',False) for o in outputs),
            full_rank_converged=sum(o['identified'] and o.get('optimizer_success',False) for o in outputs),
            full_rank_converged_bearing_deg=describe([m['bearing_deg'] for o,m in zip(outputs,metrics)
                if o['identified'] and o.get('optimizer_success',False) and m is not None]),
            solve_ms=describe([o['solve_ms'] for o in outputs]),scale=describe([o.get('scale') for o in outputs]))
    by_case={}
    for cid in sorted({r['case'] for r in rows}):
        by_case[cid]={}
        for arm in ['native','pair']+plan['arms']:
            metrics=[r[arm] if arm in ('native','pair') else r['arms'][arm]['error'] for r in rows if r['case']==cid]
            by_case[cid][arm]=describe([m['bearing_deg'] for m in metrics if m is not None])
    matching=load(out/'matching.json')
    costs=dict(new_encoded_frames=sum(not r['reused'] for r in matching['features']),
        reused_encoded_frames=sum(r['reused'] for r in matching['features']),
        new_pairs=sum(not r['reused'] for r in matching['rows']),reused_pairs=sum(r['reused'] for r in matching['rows']),
        encoder_ms=describe([r['encoder_ms'] for r in matching['features'] if not r['reused']]),
        pair_ms=describe([r['pair_ms'] for r in matching['rows'] if not r['reused']]),
        new_feature_bytes=sum(r['bytes'] for r in matching['features'] if not r['reused']),
        torch_peak_bytes=matching['torch_peak_bytes'])
    save(out/'evaluation.json',dict(complete=True,rows=rows,summaries=summaries,by_case=by_case,costs=costs,
        inference_sha256=sha(out/'inference.json'),evaluator_sha256=sha(evaluation_source),
        scope='22 preselected references of 36 consumed-development queries, 18 bearing-scored, no new coverage or navigation claim'))
    for arm,s in summaries.items():print(arm,s)
    print('costs',costs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
