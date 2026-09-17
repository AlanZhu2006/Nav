"""Coordinate-only replay of sealed connected-memory predictions.

No model, RGB retrieval, navigation goal or ground truth enters transport.
Accuracy scoring follows completion and uses the separate evaluator inputs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from NavDP.baselines.memnav.gem.connected import pose_matrices
from NavDP.baselines.memnav.gem.reciprocal import reciprocal_camera_transport
from score_gem_reactivation_probe import errors,sha,truth


def save(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    parent=args.parent.resolve()
    source=json.loads((parent/'manifest.json').read_text())
    if not args.out.exists():
        args.out.mkdir(parents=True)
        files=[Path(__file__),ROOT/'NavDP/baselines/memnav/gem/reciprocal.py',
            ROOT/'MemNavData/test_gem_reciprocal.py']
        save(args.out/'manifest.json',dict(schema='gem_reciprocal_coordinate_replay_v1',
            source=str(parent),source_sha256=sha(parent/'manifest.json'),
            cases=[c['id'] for c in source['cases']],
            sources_sha256={str(p):sha(p) for p in files},created_at=time.time(),
            method='Camera rotation Procrustes, symmetric mean log depth ratio, mean camera-center translation',
            no_goal_or_truth_in_transport=True,posthoc_development=True,
            native_prediction_reuse='Global chart transform never enters LingBot; all local predictions are identical by construction',
            limitation='Needs actual online parity and navigation validation before integration claims'))
    manifest=json.loads((args.out/'manifest.json').read_text())
    assert manifest['source_sha256']==sha(parent/'manifest.json')
    assert all(sha(p)==v for p,v in manifest['sources_sha256'].items())
    complete=[]
    for case in source['cases']:
        old=parent/case['id']
        out=args.out/case['id']
        if not (old/'completion.json').exists():
            if args.partial:continue
            raise RuntimeError('Missing source case '+case['id'])
        if out.exists():
            assert (out/'completion.json').exists()
            complete.append(case)
            continue
        receipt=json.loads((old/'completion.json').read_text())
        assert all(sha(old/p)==v for p,v in receipt['files_sha256'].items())
        out.mkdir()
        with np.load(old/'geometry.npz') as arrays:
            local_pose9=arrays['local_pose9']
            episodes=arrays['episode']
            valid=arrays['valid_pixels']
        cumulative=[np.eye(4)]
        rows=[]
        for file in sorted((old/'overlaps').glob('*.npz')):
            with np.load(file) as a:
                ab,audit=reciprocal_camera_transport(a['old_poses'],a['new_poses'],a['old_depth'],a['new_depth'],valid)
                ba,_=reciprocal_camera_transport(a['new_poses'],a['old_poses'],a['new_depth'],a['old_depth'],valid)
                reverse_error=float(np.max(np.abs(ab@ba-np.eye(4))))
                assert reverse_error<1e-10
                rows.append(dict(episode=int(file.stem),shared_frames=a['shared_frames'].tolist(),
                    old_from_new=ab.tolist(),reciprocity_max_abs=reverse_error,**audit))
                cumulative.append(cumulative[-1]@ab)
        cumulative=np.asarray(cumulative)
        world=cumulative[episodes]@pose_matrices(local_pose9)
        scales=np.cbrt(np.linalg.det(cumulative[episodes,:3,:3]))
        pose9=local_pose9.astype(np.float64).copy()
        pose9[:,:3]=world[:,:3,3]
        pose9[:,3:7]=Rotation.from_matrix(world[:,:3,:3]/scales[:,None,None]).as_quat()
        np.savez_compressed(out/'geometry.npz',pose9=pose9,scale=scales,episode=episodes,
            world_from_episode=cumulative)
        save(out/'relations.json',rows)
        save(out/'completion.json',dict(completed=True,frames=len(pose9),relations=len(rows),
            reciprocity_max_abs=max(r['reciprocity_max_abs'] for r in rows),
            source_geometry_sha256=sha(old/'geometry.npz'),
            geometry_sha256=sha(out/'geometry.npz'),relations_sha256=sha(out/'relations.json')))
        complete.append(case)
    # The only accesses to evaluator data are below this scoring boundary.
    baseline_root=Path(source['parent'])
    evaluations={c['id']:c for c in json.loads((baseline_root/'evaluation_inputs.json').read_text())['cases']}
    scores=[]
    for case in complete:
        ev=evaluations[case['id']]
        assert sha(ev['trace'])==ev['trace_sha256']
        gt={r['step']:truth(r) for r in json.loads(Path(ev['trace']).read_text())['poses']}
        with np.load(baseline_root/case['id']/'baseline.npz') as a:
            arms=dict(native=pose_matrices(a['pose_enc']))
        with np.load(parent/case['id']/'geometry.npz') as a:
            arms.update(camera=pose_matrices(a['camera_pose9']),point=pose_matrices(a['point_pose9']))
        with np.load(args.out/case['id']/'geometry.npz') as a:
            arms['reciprocal']=pose_matrices(a['pose9'])
        for prefix in case['checkpoints']:
            request=json.loads((baseline_root/case['id']/f'packet_{prefix}.json').read_text())['request']
            for selection,h in [('fixed_reference_8',8),('stage_a_dino_reference',request['reference'])]:
                t=prefix-1
                expected=np.linalg.solve(gt[h],gt[t])
                row=dict(case=case['id'],prefix=prefix,selection=selection,reference=h)
                for name,poses in arms.items():
                    row[name]=errors(np.linalg.solve(poses[h],poses[t]),expected)
                scores.append(row)
    metrics={}
    for selection in ('fixed_reference_8','stage_a_dino_reference'):
        metrics[selection]={}
        for key in ('rotation_deg','translation_direction_deg','bearing_deg'):
            metrics[selection][key]={}
            for arm in ('native','camera','point','reciprocal'):
                values=[r[arm][key] for r in scores if r['selection']==selection and r[arm][key] is not None]
                metrics[selection][key][arm]=dict(n=len(values),median=float(np.median(values)) if values else None,
                    mean=float(np.mean(values)) if values else None,p95=float(np.percentile(values,95)) if values else None)
    report=dict(rows=scores,metrics=metrics,complete_cases=len(complete),expected_cases=len(source['cases']),
        scope='Coordinate-only development replay; no new neural inference, goal PnP or navigation SR')
    path=args.out/('partial_evaluation.json' if args.partial else 'evaluation.json')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
