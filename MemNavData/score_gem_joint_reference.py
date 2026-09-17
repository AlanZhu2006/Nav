#!/usr/bin/env python3
"""Scoring-only recovery of three missing query labels from a pinned GT archive.

The frozen inference script and its 41 outputs are not modified or re-run.
Never uses archived model predictions, errors, fitted Sim(3), or accept decisions.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from MemNavData.run_gem_joint_reference import ROOT, INPUTS, TABLE, sha, save, aggregate
from MemNavData.diag_m2p_s1_gct_query import (
    _matrix, _resolve_generated_mount, _navdp_ground_truth_relative,
    _lingbot_relative_direction, direction_error_degrees)

ARCHIVE = ROOT/'.diagnostics/learned_relocalizer_20260817/pi3x_train40_hpc_925cd1ceb7ea9771'


def verified_archived_goals():
    binding = json.loads((ARCHIVE/'pi3x_train40_summary_bundled.json').read_text())['inputs']
    path = ARCHIVE/'pi3x_full.jsonl'
    assert sha(TABLE) == binding['rows_csv_sha256']
    assert sha(path) == binding['shadow_jsonl_sha256']
    rows = list(csv.DictReader(TABLE.open()))
    archived = [json.loads(line) for line in path.open() if line.strip()]
    assert len(archived) == len(rows) == 3840
    goals, indices, checks, max_error = {}, defaultdict(list), 0, 0.
    for record in archived:
        row = rows[record['row_index']]
        assert (record['scene'],record['episode'],record['anchor_frame']) == (
            row['scene'],row['episode'],int(row['candidate_frame']))
        # diag_pi3x_multiview_consistency._goal_center explicitly adds [0,0,.5].
        goal = np.asarray(record['true_goal_center_reporting_only'])-np.array([0.,0.,.5])
        sid = row['session_id']
        if sid in goals:
            np.testing.assert_allclose(goal, goals[sid], atol=0, rtol=0)
        goals[sid] = goal
        indices[sid].append(record['row_index'])
        meta_path = INPUTS/'supervision_only'/row['scene']/row['goal_episode']/'meta/gen_meta.json'
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            target = next(g['pos'] for g in meta['goals'] if g['name']==row['goal_role'])
            error = float(np.max(abs(goal-target)))
            assert error < 1e-10
            max_error = max(error,max_error)
            checks += 1
    return goals, indices, {'archive_sha256':sha(path), 'table_sha256':sha(TABLE),
                           'archive_records':len(archived), 'crosschecked_original_metadata_rows':checks,
                           'max_metadata_delta_m':max_error,
                           'only_field_used':'true_goal_center_reporting_only minus documented [0,0,0.5]',
                           'no_model_prediction_used':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    out = args.out
    manifest = json.loads((out/'manifest.json').read_text())
    complete = json.loads((out/'solve_receipt.json').read_text())
    assert complete['completed'] and complete['sessions']==len(manifest['sessions'])
    assert sha(out/'manifest.json')==complete['manifest_sha256']
    archived, source_indices, audit = verified_archived_goals()
    labels = defaultdict(list)
    for row in csv.DictReader(TABLE.open()):
        labels[row['session_id']].append(float(row['covisibility']))
    cache = {}
    def history_gt(key):
        if key not in cache:
            root = INPUTS/'supervision_only'/key
            meta = json.loads((root/'meta/gen_meta.json').read_text())
            frame = pd.read_parquet(root/'data/chunk-000/episode_000000.parquet',
                                    columns=['action','observation.camera_extrinsic'])
            poses = np.stack([_matrix(v,'pose') for v in frame['action']])
            mount = _resolve_generated_mount(_matrix(frame.iloc[0]['observation.camera_extrinsic'],'mount'),
                                              meta.get('frame_convention',''))
            cache[key] = poses,mount
        return cache[key]
    scored, repaired = [], []
    for index, session in enumerate(manifest['sessions']):
        result = json.loads((out/'sessions'/f'{index:03d}.json').read_text())
        assert session['id']==result['id']
        query = Path(session['query_relative_path'])
        meta_path = INPUTS/'supervision_only'/Path(*query.parts[:2])/'meta/gen_meta.json'
        if meta_path.exists():
            goal_pos = np.asarray(json.loads(meta_path.read_text())['goals'][int(query.stem.split('_')[-1])-1]['pos'])
            source = 'original_meta'
        else:
            goal_pos = archived[session['id']]
            source = 'pinned_archived_GT_field'
            repaired.append({'id':session['id'], 'query_sha256':session['query_sha256'],
                              'original_missing_path':str(meta_path),
                              'archive_row_indices':source_indices[session['id']],
                              'goal_pos_data':goal_pos.tolist()})
        np.testing.assert_allclose(goal_pos,archived[session['id']],atol=1e-10)
        poses,mount = history_gt(session['history'])
        pivot = result['pivot']
        goal = np.eye(4); goal[:3,3] = goal_pos
        target = _navdp_ground_truth_relative(poses[pivot],goal,mount)
        history = manifest['histories'][session['history']]
        pivot_pose = np.asarray(history['pose9'][pivot])
        def measure(pnp):
            if pnp.get('status')!='ok' or 'pose9' not in pnp:
                return {'valid':False,'error_m':None,'angle_deg':None,'within_05m':False}
            pred = _lingbot_relative_direction(pivot_pose,np.asarray(pnp['pose9']))*history['metric_scale']
            error = float(np.linalg.norm(pred-target))
            angle = direction_error_degrees(pred,target) if np.linalg.norm(target)>=.5 else None
            return {'valid':True,'xy_m':pred.tolist(),'error_m':error,'angle_deg':angle,'within_05m':error<=.5}
        singles = [(r['anchor'],measure(r['pnp'])) for r in result['all_singles']]
        valid = [(a,m) for a,m in singles if m['valid']]
        oracle = min(valid,key=lambda x:(x[1]['error_m'],x[0])) if valid else (None,measure({}))
        covis = max(labels[session['id']])
        support = 'no_strong_support' if covis<=.1 else ('partial' if covis<.5 else 'strong')
        scored.append({'id':session['id'],'scene':session['history'].split('/')[0],
                       'query_sha256':session['query_sha256'],'max_top8_covis':covis,'support':support,
                       'pivot':pivot,'target_xy_m':target.tolist(),'target_distance_m':float(np.linalg.norm(target)),
                       'single':measure(result['single']['pnp']),'oracle':oracle[1],'oracle_anchor':oracle[0],
                       'joint':measure(result['joint']),'all_single_errors':singles,
                       'current_certificate_accept':result['current_certificate_accept'],
                       'joint_reference_count':len(result['joint_inlier_reference_contribution']),
                       'goal_gt_source':source,'goal_pos_data':goal_pos.tolist()})
    save(out/'gt_label_recovery.json',audit|{'repaired_sessions':repaired})
    save(out/'scored_sessions.json',scored)
    summary = aggregate(scored)
    summary['label_recovery_sessions'] = len(repaired)
    save(out/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
