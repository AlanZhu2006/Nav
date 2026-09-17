#!/usr/bin/env python3
"""Re-read saved constraints and GT; verify local diagnostic without new GPU work."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from MemNavData.lingbot_pnp_localization import SiftPnPConfig, solve_camera_pose_pnp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    manifest = json.loads((out/'manifest.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    scores = json.loads((out/'scored_sessions.json').read_text())
    receipt = json.loads((out/'solve_receipt.json').read_text())
    recovery = json.loads((out/'gt_label_recovery.json').read_text())
    archive_path = Path(__file__).resolve().parents[1]/'.diagnostics/learned_relocalizer_20260817/pi3x_train40_hpc_925cd1ceb7ea9771/pi3x_full.jsonl'
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == recovery['archive_sha256']
    archive = {r['row_index']:r for r in (json.loads(line) for line in archive_path.open() if line.strip())}
    recovered = {r['id']:r for r in recovery['repaired_sessions']}
    assert receipt['completed'] and receipt['sessions'] == len(manifest['sessions'])
    assert receipt['manifest_sha256'] == hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest()
    base = Path(__file__).resolve().parents[1]/'.diagnostics/anchor_relation_geometry_20260906/inputs/unpacked/supervision_only'
    cache = {}
    def ground_truth(key):
        if key not in cache:
            root = base/key
            df = pd.read_parquet(root/'data/chunk-000/episode_000000.parquet',
                                 columns=['action', 'observation.camera_extrinsic'])
            meta = json.loads((root/'meta/gen_meta.json').read_text())
            pose = np.stack([np.asarray(np.asarray(v).tolist(), dtype=float).reshape(4,4)
                             for v in df['action']])
            mount = np.asarray(np.asarray(df.iloc[0]['observation.camera_extrinsic']).tolist(), dtype=float).reshape(4,4)[:3,:3]
            if str(meta.get('frame_convention','')).startswith('positions+parquet in data(Zup,M_W)'):
                expected = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=float)
                if np.allclose(mount, np.eye(3), atol=1e-6):
                    mount = expected
                else:
                    np.testing.assert_allclose(mount, expected, atol=1e-6)
            cache[key] = pose, mount, meta
        return cache[key]
    recomputed = []
    order_controls = []
    max_replay_delta = 0.
    for index, session in enumerate(manifest['sessions']):
        path = out/'sessions'/f'{index:03d}.json'
        result = json.loads(path.read_text())
        record = scores[index]
        assert session['id'] == result['id'] == record['id']
        assert len(result['all_singles']) == 8 and result['joint_authorized'] is False
        history = manifest['histories'][session['history']]
        frames = [candidate['anchor'] for candidate in session['candidates']]
        with np.load(path.with_suffix('.npz'), allow_pickle=False) as arrays:
            xyz = np.concatenate([arrays[f'{i}/xyz'] for i in frames])
            uv = np.concatenate([arrays[f'{i}/query'] for i in frames])
            indices = arrays['dedup_indices']
            assert len(indices) == len(np.unique(uv, axis=0)) == result['unique_query_correspondences']
            assert len(np.unique(uv[indices], axis=0)) == len(indices)
            assert len(xyz) == result['total_correspondences']
            rerun = solve_camera_pose_pnp(xyz[indices], uv[indices], np.asarray(result['query_intrinsic']),
                                          config=SiftPnPConfig(**result['pnp_config']),
                                          fov_pose9=history['pose9'][result['pivot']])
            sorted_singles = []
            for anchor in frames:
                points = arrays[f'{anchor}/query']
                order = np.lexsort((points[:,1], points[:,0]))
                control = solve_camera_pose_pnp(arrays[f'{anchor}/xyz'][order], points[order],
                                                np.asarray(result['query_intrinsic']),
                                                config=SiftPnPConfig(**result['pnp_config']),
                                                fov_pose9=history['pose9'][result['pivot']])
                sorted_singles.append((anchor,control))
        assert rerun['status'] == result['joint']['status']
        if 'pose9' in rerun:
            delta = float(np.max(np.abs(rerun['pose9']-np.asarray(result['joint']['pose9']))))
            max_replay_delta = max(max_replay_delta, delta)
            assert delta < 1e-8
        poses, mount, _ = ground_truth(session['history'])
        qpath = Path(session['query_relative_path'])
        meta_path = base/Path(*qpath.parts[:2])/'meta/gen_meta.json'
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            goal = np.asarray(metadata['goals'][int(qpath.stem.split('_')[-1])-1]['pos'])
        else:
            source = recovered[session['id']]
            values = [np.asarray(archive[i]['true_goal_center_reporting_only'])-np.array([0,0,.5])
                      for i in source['archive_row_indices']]
            goal = values[0]
            for value in values:
                np.testing.assert_array_equal(goal,value)
            np.testing.assert_array_equal(goal,source['goal_pos_data'])
        pivot = result['pivot']
        base_rotation = poses[pivot,:3,:3] @ mount.T
        displacement = goal-poses[pivot,:3,3]
        target = np.array([base_rotation[:,1] @ displacement, -base_rotation[:,0] @ displacement])
        np.testing.assert_allclose(record['target_xy_m'], target, atol=1e-10)
        pose9 = np.asarray(history['pose9'][pivot])
        rotation = Rotation.from_quat(pose9[3:7]).as_matrix()
        def metric(pnp):
            if pnp.get('status') != 'ok' or 'pose9' not in pnp:
                return None
            local = rotation.T @ (np.asarray(pnp['pose9'])[:3]-pose9[:3])
            prediction = local[[2,0]] * np.array([1,-1]) * history['metric_scale']
            return float(np.linalg.norm(prediction-target))
        errors = [(single['anchor'], metric(single['pnp'])) for single in result['all_singles']]
        finite = [(anchor,error) for anchor,error in errors if error is not None]
        best = min(finite, key=lambda x:(x[1],x[0])) if finite else (None,None)
        independent = {'single':metric(result['single']['pnp']), 'joint':metric(result['joint']), 'oracle':best[1]}
        sorted_errors = [(anchor,metric(control)) for anchor,control in sorted_singles]
        sorted_valid = [(anchor,error) for anchor,error in sorted_errors if error is not None]
        sorted_best = min(sorted_valid, key=lambda x:(x[1],x[0])) if sorted_valid else (None,None)
        sorted_top1 = next(error for anchor,error in sorted_errors if anchor == pivot)
        order_controls.append({'id':session['id'], 'sorted_top1_error_m':sorted_top1,
                               'sorted_oracle_error_m':sorted_best[1],
                               'sorted_oracle_anchor':sorted_best[0],
                               'joint_error_m':independent['joint']})
        assert best[0] == record['oracle_anchor']
        for arm, error in independent.items():
            recorded = record[arm]['error_m']
            assert (error is None) == (recorded is None)
            if error is not None:
                assert abs(error-recorded) < 1e-8
            assert record[arm]['within_05m'] == (error is not None and error <= .5)
        recomputed.append(independent)
    for arm in ['single','oracle','joint']:
        valid = [row[arm] for row in recomputed if row[arm] is not None]
        expected = summary['groups']['all'][arm]
        assert expected['valid'] == len(valid)
        assert expected['within_05m'] == sum(value <= .5 for value in valid)
        if valid:
            assert abs(expected['mean_error_m']-float(np.mean(valid))) < 1e-8
            assert abs(expected['median_error_m']-float(np.median(valid))) < 1e-8
    report = {'verified': True, 'sessions':len(scores),
              'scenes':len(set(record['scene'] for record in scores)),
              'unique_queries':len(set(record['query_sha256'] for record in scores)),
              'joint_solver_replayed_from_saved_correspondences':True,
              'joint_max_pose_delta':max_replay_delta,
              'errors_recomputed_from_original_GT':True,
              'raw_metadata_sessions':len(scores)-len(recovered),
              'pinned_archived_GT_sessions':len(recovered),
              'query_duplicates_counted_once':True,
              'navigation_executed':False,
              'order_control':{
                  'sorted_top1_within_05m':sum(r['sorted_top1_error_m'] is not None and r['sorted_top1_error_m']<=.5 for r in order_controls),
                  'sorted_oracle_within_05m':sum(r['sorted_oracle_error_m'] is not None and r['sorted_oracle_error_m']<=.5 for r in order_controls),
                  'joint_correct_where_sorted_oracle_incorrect':sum(r['joint_error_m'] is not None and r['joint_error_m']<=.5 and (r['sorted_oracle_error_m'] is None or r['sorted_oracle_error_m']>.5) for r in order_controls),
                  'joint_lower_error_than_sorted_oracle':sum(r['joint_error_m'] is not None and r['sorted_oracle_error_m'] is not None and r['joint_error_m']<r['sorted_oracle_error_m'] for r in order_controls),
                  'records':order_controls},
              'summary_sha256':hashlib.sha256((out/'summary.json').read_bytes()).hexdigest()}
    with (out/'independent_verification.json').open('x') as stream:
        json.dump(report,stream,indent=2)
        stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
