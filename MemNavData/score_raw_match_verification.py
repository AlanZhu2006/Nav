#!/usr/bin/env python3
"""Describe fixed-pair matching evidence; no tuned threshold or inferred SR."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from MemNavData.run_raw_match_verification import sha_file, statistics, write_json
from MemNavData.score_low_covisibility_witnesses import angle, endpoint_bearing


def auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if not n1 or not n0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum()-n1*(n1+1)/2)/(n1*n0))


def group_for(row):
    if row['analysis_role'] == 'novel':
        return 'original_novel'
    q = row['q_frame8']
    for lo, hi in ((.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.000001)):
        if lo <= q < hi:
            return f'{lo:.1f}-{min(hi,1):.1f}'
    return 'outside_supported_bins'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    labels_path = args.inputs/'evaluation_only.json'
    source = json.loads(labels_path.read_text())
    labels = {r['task']: r for r in source['rows']}
    manifest_hash = sha_file(args.inputs/'pairs.json')
    methods = {}
    for path in args.results:
        result = json.loads((path/'summary.json').read_text())
        if (not result['completed'] or result['runtime']['pair_manifest_sha256'] != manifest_hash
                or {r['task'] for r in result['rows']} != set(labels)):
            raise ValueError('Incomplete/mismatched fixed-pair readout')
        rows = []
        for row in result['rows']:
            label = labels[row['task']]
            if row['raw_anchor'] != label['raw_anchor']:
                raise ValueError('Raw anchor changed')
            e = row['evidence']
            error = angle(label['raw_bearing'], endpoint_bearing(label['state'], label['goal_floor_position']))
            rows.append({'task': row['task'], 'history': label['history'], 'scene': label['scene'],
                'role': label['analysis_role'], 'group': group_for(label), 'q_raw_anchor': label['q_raw_anchor'],
                'raw_error_deg': error, 'matches': e['matches'], 'F_inliers': e['fundamental_inliers'],
                'F_inlier_ratio': e['fundamental_inlier_ratio'],
                'matched_confidence': row['matched_confidence']['median'],
                'query_dense_confidence': row['details'].get('dense_query_confidence', {}).get('median'),
                'min_F_hull': min(e['fundamental_query_hull_coverage'], e['fundamental_reference_hull_coverage']),
                'match_total_s': row['timings'][0]['match_total_s']})
        groups = {}
        for group in sorted({r['group'] for r in rows}):
            rs = [r for r in rows if r['group'] == group]
            valid = [r['raw_error_deg'] for r in rs if r['raw_error_deg'] is not None]
            groups[group] = {'queries': len(rs), 'histories': len({r['history'] for r in rs}),
                'scenes': len({r['scene'] for r in rs}), 'raw_bearing_available': len(valid),
                'raw_within_30deg': sum(e <= 30 for e in valid),
                **{k: statistics([r[k] for r in rs]) for k in
                   ('matches', 'F_inliers', 'matched_confidence', 'min_F_hull', 'match_total_s')}}
        aucs = {}
        for key in ('matches', 'F_inliers', 'F_inlier_ratio', 'matched_confidence',
                    'query_dense_confidence', 'min_F_hull'):
            valid = [r for r in rows if r[key] is not None]
            directional = [r for r in valid if r['raw_error_deg'] is not None]
            aucs[key] = {'support_role_auc': auc([r['role'] == 'revisit' for r in valid], [r[key] for r in valid]),
                'raw_direction_30deg_auc': auc([r['raw_error_deg'] <= 30 for r in directional],
                                              [r[key] for r in directional])}
        methods[result['runtime']['matcher']] = {'runtime': result['runtime'], 'groups': groups,
            'descriptive_auc_not_independent_confirmatory_tests': aucs, 'rows': rows}
    write_json(args.out, {'completed': True, 'evaluation_sha256': sha_file(labels_path),
        'new_navigation_rollouts': 0, 'new_SR': None, 'methods': methods,
        'scope': 'Consumed-data fixed-raw-pair readout. Roles and directional usefulness are separate. '
                 'No acceptance threshold fitted; correlated queries remain grouped by history/scene.'})
    print(json.dumps({'methods': list(methods), 'queries': len(labels), 'new_SR': None}))


if __name__ == '__main__':
    main()
