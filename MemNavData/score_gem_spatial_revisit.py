"""CPU-only scoring of sealed causal correspondences; no inference selection.

Camera poses are camera-to-world. Direction refers to the historical reference,
not the task goal. Every fixed pair and every failed arm remains in the report.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'MemNavData')]
from MemNavData.probe_gem_spatial_revisit import load, save, sha, verify_sources
from MemNavData.score_gem_reactivation_probe import truth, errors
from MemNavData.gem_reference_readout import pose_matrix
from MemNavData.certified_relocalization_runtime import certificate_decision

ARMS = ('superpoint_lightglue', 'dino_patch', 'gca_18', 'gca_24')
METRICS = ('rotation_deg', 'translation_direction_deg', 'bearing_deg')


def describe(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return dict(n=0, mean=None, median=None, p95=None, max=None)
    assert np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p95=float(np.percentile(a, 95)), max=float(a.max()))


def paired(rows):
    summary = {}
    for metric in METRICS:
        pairs = [(r['prior'][metric], r['observed'][metric]) for r in rows
                 if r['observed'] is not None and r['prior'][metric] is not None
                 and r['observed'][metric] is not None]
        summary[metric] = dict(prior=describe([a for a,b in pairs]),
            observed=describe([b for a,b in pairs]),
            improved=sum(b<a for a,b in pairs), worsened=sum(b>a for a,b in pairs),
            worsened_over_5deg=sum(b>a+5 for a,b in pairs),
            delta_observed_minus_prior=describe([b-a for a,b in pairs]))
    return summary


def aggregate(rows):
    result = {}
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        accepted = [r for r in group if r['accepted']]
        poses = [r for r in group if r['observed'] is not None]
        counts = {}
        for threshold in (5, 15, 30):
            # Thresholds describe error, never change acceptance or selection.
            counts[str(threshold)] = {
                metric: dict(valid=sum(r['observed'][metric] is not None for r in accepted),
                    over=sum(r['observed'][metric] is not None and r['observed'][metric]>threshold
                             for r in accepted)) for metric in METRICS}
        result[arm] = dict(pairs=len(group), f_precheck=sum(r['f_precheck'] for r in group),
            pnp_pose_returned=len(poses), accepted=len(accepted),
            queries_with_any_accepted=len({(r['case'],r['current']) for r in accepted}),
            queries=len({(r['case'],r['current']) for r in group}),
            correspondence_count=describe([r['matches'] for r in group]),
            match_ms=describe([r['match_ms'] for r in group]),
            failure_reasons=dict(Counter(r['reason'] for r in group if not r['accepted'])),
            accepted_errors=paired(accepted), all_returned_pose_errors=paired(poses),
            accepted_error_tail_counts=counts)
    return result


def freeze(out):
    assert not (out/'inference.json').exists()
    assert not (out/'matching_progress.jsonl').exists()
    sources = [Path(__file__), ROOT/'MemNavData/score_gem_reactivation_probe.py',
               ROOT/'MemNavData/gem_reference_readout.py']
    save(out/'scoring_plan.json', dict(schema='gem_spatial_revisit_score_v1',
        plan_sha256=sha(out/'plan.json'), evaluation_inputs_sha256=sha(out/'evaluation_inputs.json'),
        source_sha256={str(p):sha(p) for p in sources},
        metrics=list(METRICS), minimum_direction_distance_m=0.5,
        error_tail_thresholds_deg=[5,15,30],
        primary_selection='Original DINO top1; no truth or acceptance-based reselection',
        diagnostic_selection='All eight fixed pairs, and first passing rank; query coverage reported separately',
        native_pair_definition='inverse(predicted_history_camera) @ predicted_current_camera',
        observed_pair_definition='inverse(predicted_history_camera) @ PnP_current_camera',
        truth_pair_definition='inverse(GT_history_camera) @ GT_current_camera',
        bearing_definition='Horizontal current-to-historical-reference bearing, not task goal',
        unavailable=['Continuous unlocalizable interval from only three queries per leg',
                     'Current-to-task-goal correction: full original goal PnP is absent from these rollout receipts',
                     'Online fusion, Sim3 optimization, closed-loop navigation, long-range generalization'],
        thresholds_used_for_inference=False))


def run(out):
    plan, scoring = load(out/'plan.json'), load(out/'scoring_plan.json')
    inference, selected = load(out/'inference.json'), load(out/'selected_pairs.json')
    assert inference['complete'] and not inference['navigation_executed'] and not inference['updated_map']
    assert sha(out/'plan.json') == scoring['plan_sha256'] == inference['plan_sha256']
    assert sha(out/'selected_pairs.json') == inference['selected_pairs_sha256']
    assert sha(out/'evaluation_inputs.json') == scoring['evaluation_inputs_sha256']
    for path,digest in scoring['source_sha256'].items(): assert sha(path) == digest, path
    verify_sources(plan)
    evaluations = load(out/'evaluation_inputs.json')['cases']
    truth_by_case, poses, inventory = {}, {}, []
    cases = {c['id']:c for c in plan['cases']}
    for case in evaluations:
        for path,digest in case['source_sha256'].items(): assert sha(path) == digest, path
        truth_by_case[case['id']] = [truth(p) for p in case['rows']]
    for case in plan['cases']:
        folder = out/case['id']
        receipt = load(folder/'extraction.json')
        assert receipt['complete'] and receipt['frames'] == case['frames']
        assert receipt['original_pose9_max_abs_difference'] < 1e-4
        assert sha(folder/'stream.npz') == receipt['stream_sha256']
        with np.load(folder/'stream.npz') as data:
            poses[case['id']] = np.stack([pose_matrix(p) for p in data['pose9']])
        inventory.append(dict(case=case['id'],frames=case['frames'],
            feature_frames=len(receipt['features']), feature_bytes=receipt['feature_bytes'],
            wall_seconds=receipt['wall_seconds'],
            original_pose9_max_abs_difference=receipt['original_pose9_max_abs_difference']))
    expected = {(p['case'],p['current'],p['reference'],p['rank']) for p in selected['pairs']}
    assert len(expected) == inference['pairs'] == len(selected['pairs'])
    assert len(inference['rows']) == len(expected)*len(ARMS)
    rows, identities = [], set()
    geometry_verified = {}
    for r in inference['rows']:
        cid,h,t,rank,arm = r['case'],r['reference'],r['current'],r['rank'],r['arm']
        identity = (cid,t,h,rank)
        assert identity in expected and arm in ARMS and (*identity,arm) not in identities
        identities.add((*identity,arm))
        case = cases[cid]
        assert h in case['reference_keyframes'] and h <= t-plan['exclude_recent']
        assert any(q['current']==t and q['leg']==r['leg'] and q['phase']==r['phase'] for q in case['queries'])
        path = out/r['match_file']
        assert sha(path) == r['match_sha256']
        with np.load(path) as data:
            n = len(data['reference_points'])
            assert data['reference_points'].shape == data['query_points'].shape == (n,2)
            assert len(data['scores']) == n
            assert all(np.isfinite(data[k]).all() for k in (
                'reference_points','query_points','reference_raw_points','query_raw_points','scores'))
            if 'coordinate_source' in data:
                assert str(data['coordinate_source']) == 'native_rgb_to_lingbot_pad'
        geo = Path(case['source_directory'])/'buffer/ep_0001/geometry'/f'{h:06d}.npz'
        if str(geo) not in geometry_verified: geometry_verified[str(geo)] = sha(geo)
        assert geometry_verified[str(geo)] == r['reference_geometry_sha256']
        assert certificate_decision(r['pnp']) == r['certificate']
        accepted = bool(r['f_precheck'] and r['certificate']['accepted'])
        assert accepted == r['original_frontend_pass']
        gt = truth_by_case[cid]
        expected_relation = np.linalg.solve(gt[h],gt[t])
        prior = errors(np.linalg.solve(poses[cid][h],poses[cid][t]),expected_relation)
        observed = None
        if 'pose9' in r['pnp']:
            observed = errors(np.linalg.solve(poses[cid][h],pose_matrix(r['pnp']['pose9'])),expected_relation)
        if np.linalg.norm(expected_relation[:3,3]) < scoring['minimum_direction_distance_m']:
            prior['translation_direction_deg'] = None
            if observed is not None: observed['translation_direction_deg'] = None
        assert not accepted or observed is not None
        row = dict(case=cid,current=t,reference=h,rank=rank,arm=arm,leg=r['leg'],phase=r['phase'],
            temporal_gap_frames=t-h,
            reference_source='initial_survey' if h<case['history_frames'] else 'earlier_executed_return',
            relative_gt_rotation_deg=errors(np.eye(4),expected_relation)['rotation_deg'],
            accepted=accepted,f_precheck=r['f_precheck'],pnp_status=r['pnp']['status'],
            reason=r['f_reason'] if not r['f_precheck'] else r['certificate']['reason'],
            matches=n,match_ms=r['match_ms'],prior=prior,observed=observed)
        rows.append(row)
    primary = [r for r in rows if r['rank']==0]
    first_passing = []
    for case in plan['cases']:
        for query in case['queries']:
            for arm in ARMS:
                group = [r for r in rows if r['case']==case['id'] and r['current']==query['current'] and r['arm']==arm]
                assert sorted(r['rank'] for r in group) == list(range(plan['candidate_top_k']))
                passing = [r for r in group if r['accepted']]
                first_passing.append(min(passing or group,key=lambda r:r['rank']))
    summary = dict(top1=aggregate(primary), all_pairs=aggregate(rows),
        first_passing_rank_diagnostic=aggregate(first_passing),
        per_case={case['id']:aggregate([r for r in rows if r['case']==case['id']]) for case in plan['cases']},
        reference_source={source:aggregate([r for r in rows if r['reference_source']==source])
                          for source in ('initial_survey','earlier_executed_return')})
    save(out/'evaluation.json',dict(complete=True,rows=rows,summary=summary,inventory=inventory,
        inference_sha256=sha(out/'inference.json'),scoring_plan_sha256=sha(out/'scoring_plan.json'),
        scorer_sha256=sha(__file__),scope=plan['scope'],limitations=scoring['unavailable'],
        gpu_used_for_scoring=False,all_failures_retained=True))
    print(json.dumps({k:{a:{f:v[f] for f in ('pairs','f_precheck','pnp_pose_returned','accepted','queries_with_any_accepted')}
                        for a,v in summary[k].items()} for k in ('top1','all_pairs','first_passing_rank_diagnostic')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','score'))
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    (freeze if args.action=='freeze' else run)(args.out.resolve())
