#!/usr/bin/env python3
"""Independently verify the transferred RGB/geometry sources and raw GT labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    files=json.loads((args.input_dir/'FILES.json').read_text())
    for r in files:
        path=args.input_dir/r['path']
        if (path.stat().st_size!=r['bytes'] or
                hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']):
            raise ValueError(f"transferred input changed: {path}")
    manifest=json.loads((args.input_dir/'probe_manifest.json').read_text())
    info=json.loads((args.input_dir/'history_inputs.json').read_text())
    pose_cache={}
    transform=np.asarray([[1,0,0],[0,0,-1],[0,1,0]],dtype=np.float64)
    def load(key):
        if key not in pose_cache:
            root=args.input_dir/'supervision_only'/key
            df=pd.read_parquet(root/'data/chunk-000/episode_000000.parquet',
                               columns=['action','observation.camera_extrinsic'])
            metadata=json.loads((root/'meta/gen_meta.json').read_text())
            poses=np.stack([np.asarray(np.asarray(v).tolist(),dtype=np.float64).reshape(4,4)
                            for v in df['action']])
            mount=np.asarray(np.asarray(df.iloc[0]['observation.camera_extrinsic']).tolist(),
                             dtype=np.float64).reshape(4,4)[:3,:3]
            if str(metadata.get('frame_convention','')).startswith('positions+parquet in data(Zup,M_W)'):
                if np.allclose(mount,np.eye(3),atol=1e-6):
                    mount=transform.copy()
                elif not np.allclose(mount,transform,atol=1e-6):
                    raise ValueError(f'unknown fixed camera mounting convention: {key}')
            pose_cache[key]=(poses,mount,metadata)
        return pose_cache[key]
    checked=[]
    for pair in manifest['pairs']:
        history='/'.join(Path(pair['candidate_relative_path']).parts[:2])
        query_key='/'.join(Path(pair['query_relative_path']).parts[:2])
        poses,mount,meta=load(history)
        qp,qm,qmeta=load(query_key)
        name=Path(pair['query_relative_path']).stem
        if name=='goal_image':
            goal=np.asarray(qmeta['goals'][0]['pos'],dtype=np.float64)
        elif name.startswith('goal_'):
            goal=np.asarray(qmeta['goals'][int(name[5:])-1]['pos'],dtype=np.float64)
        else:
            goal=qp[int(name),:3,3]
        anchor=poses[pair['candidate_frame']]
        base_axes=anchor[:3,:3]@mount.T
        displacement=goal-anchor[:3,3]
        target=np.asarray([np.dot(base_axes[:,1],displacement),
                           -np.dot(base_axes[:,0],displacement)])
        delta=float(np.max(np.abs(target-np.asarray(pair['target_xy_m']))))
        if delta>1e-6:
            raise ValueError(f"old GT label differs from original parquet: {pair['pair_id']} {delta}")
        if not max(63,pair['candidate_frame'])<pair['decision_frame']:
            raise ValueError('geometry or height prefix exceeds this decision')
        scale=info['scales'][history]
        if scale['prefix_end_frame_exclusive']!=64 or not scale['valid']:
            raise ValueError('scale receipt is not the declared first64 receipt')
        checked.append({'pair_id':pair['pair_id'],'max_target_delta_m':delta,
                        'anchor':pair['candidate_frame'],'decision':pair['decision_frame']})
    result={'verified':True,'files':len(files),'pairs':len(checked),
            'scenes':len(set(p['scene'] for p in manifest['pairs'])),
            'raw_gt_independently_recomputed':True,
            'max_target_delta_m':max(p['max_target_delta_m'] for p in checked),
            'scope':'input/label integrity; no navigation result',
            'checks':checked}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':
    main()
