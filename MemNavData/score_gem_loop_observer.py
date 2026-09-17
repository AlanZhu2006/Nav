"""Evaluator-only scoring of a completed fixed loop-observer experiment."""
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'MemNavData')]
from MemNavData.probe_gem_loop_observer import SOURCE, load, save, sha, verify
from MemNavData.score_gem_reactivation_probe import truth, errors
from MemNavData.gem_reference_readout import pose_matrix
from MemNavData.score_gem_spatial_revisit import describe


def summarize(rows, query_count):
    accepted = [r for r in rows if r['accepted']]
    direction = [r for r in accepted if r['observed'] is not None and r['observed']['bearing_deg'] is not None]
    return dict(queries=query_count, candidate_pairs=len(rows), accepted_pairs=len(accepted),
        queries_with_any_accepted=len({(r['case'],r['current']) for r in accepted}),
        direction_scored=len(direction), native_bearing_deg=describe([r['prior']['bearing_deg'] for r in direction]),
        observed_bearing_deg=describe([r['observed']['bearing_deg'] for r in direction]),
        observed_rotation_deg=describe([r['observed']['rotation_deg'] for r in accepted if r['observed'] is not None]),
        over15=sum(r['observed']['bearing_deg']>15 for r in direction),
        over90=sum(r['observed']['bearing_deg']>90 for r in direction),
        worsened_over5=sum(r['observed']['bearing_deg']>r['prior']['bearing_deg']+5 for r in direction),
        estimated_baseline_ratio=describe([r['baseline_ratio'] for r in direction]),
        official_pair_pass=sum(r['official_pair_pass'] is True for r in rows),
        native_frontend_pass=sum(r['frontend_pass'] for r in rows))


def main(out):
    plan = load(out/'plan.json')
    verify(plan)
    inference = load(out/'inference.json')
    assert inference['complete']
    master = load(out/'mast_inference.json')
    assert sha(out/'mast_inference.json') == inference['mast_sha256']
    assert sha(out/'selections.json') == master['selections_sha256']
    queries = load(out/'selections.json')['queries']
    evaluator = SOURCE/'attempt_003/evaluation_inputs.json'
    gt = {c['id']:[truth(row) for row in c['rows']] for c in load(evaluator)['cases']}
    poses = {}
    for case in plan['cases']:
        with np.load(case['native_stream']) as data:
            poses[case['id']] = [pose_matrix(p) for p in data['pose9']]
    rows, lookup = [], {}
    for r in inference['rows']:
        cid,t,h = r['case'],r['current'],r['reference']
        assert h <= t-plan['exclude_recent']
        assert sha(r['match_file']) == r['match_sha256']
        expected = np.linalg.solve(gt[cid][h], gt[cid][t])
        native = np.linalg.solve(poses[cid][h], poses[cid][t])
        row = dict(r,prior=errors(native,expected),observed=None,baseline_ratio=None)
        if r['pnp'].get('pose9') is not None:
            predicted = np.linalg.solve(poses[cid][h],pose_matrix(r['pnp']['pose9']))
            row['observed'] = errors(predicted,expected)
            norm = float(np.linalg.norm(native[:3,3]))
            row['baseline_ratio'] = float(np.linalg.norm(predicted[:3,3])/norm) if norm>1e-12 else None
        assert not row['accepted'] or row['observed'] is not None
        key = (cid,t,h,r['arm'])
        assert key not in lookup
        lookup[key] = row
        rows.append(row)
    assert len(rows) == 2*master['pair_count']
    summaries, decisions = {}, []
    for retriever in ('dino','asmk'):
        for arm in ('sp','mast'):
            top1, all_pairs, first = [], [], []
            for q in queries:
                ordered = [lookup[q['case'],q['current'],h,arm] for h in q[retriever]]
                assert len(ordered)<=plan['candidate_budget']
                all_pairs.extend(ordered)
                top1.extend(ordered[:1])
                chosen = next((r for r in ordered if r['accepted']),None)
                if chosen is not None:
                    first.append(chosen)
                decisions.append(dict(case=q['case'],current=q['current'],retriever=retriever,verifier=arm,
                    candidates=q[retriever],reference=chosen['reference'] if chosen else None,
                    prior=chosen['prior'] if chosen else None,observed=chosen['observed'] if chosen else None))
            summaries[retriever+'_'+arm] = dict(top1=summarize(top1,len(queries)),
                all_candidates=summarize(all_pairs,len(queries)),first_passing=summarize(first,len(queries)),
                empty_retrieval_queries=sum(not q[retriever] for q in queries))
    failure_rows = [lookup[c,t,h,arm] for c,t,h in plan['known_failure_pairs'] for arm in ('sp','mast')]
    report = dict(complete=True,summaries=summaries,decisions=decisions,rows=rows,known_failures=failure_rows,
        evaluator_file=str(evaluator),evaluator_sha256=sha(evaluator),
        inference_sha256=sha(out/'inference.json'),plan_sha256=sha(out/'plan.json'),
        scope='Four consumed development traces, 36 query times; historical-reference bearing, not goal/navigation success',
        resource_scope='MASt3R components are additional; no native LingBot model was rerun',
        costs=dict(encoder_count=master['encoder_count'],pairs=master['pair_count'],
            encoder_ms=describe([r['encoder_ms'] for r in master['encodings']]),
            pair_ms=describe([r['pair_ms'] for r in master['rows']]),
            asmk_query_ms=describe(master['asmk_query_ms']),
            archived_feature_bytes=sum(r['bytes'] for r in master['encodings']),
            torch_peak_bytes=master['torch_peak_bytes'], new_sp_pairs=inference['new_sp_pairs'],
            reused_sp_pairs=inference['reused_sp_pairs']))
    save(out/'evaluation.json',report)
    for key,value in summaries.items():
        print(key,value['first_passing'])
    print('costs',report['costs'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
