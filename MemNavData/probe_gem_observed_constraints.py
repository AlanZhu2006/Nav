"""Test actual-observation geometric constraints at the sealed38 prefixes.

Reuse exactly the DINO-selected historical/current frame pairs of the failed
LingBot reactivation experiment. No task goal or evaluator truth is consumed.
Run the existing SP/LG/PnP/certificate on those actual RGBs, with both saved
native64 and connected geometry. This only measures available information;
it is not an online graph, readout replacement, fallback or navigation result.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from run_gem_reactivation_probe import sha,save
from eval_gem_archived_readout import archived_agent,FrozenMatches
from eval_gem_native_budget import archive_encoder
from MemNavData.lingbot_pnp_localization import LightGluePointMatcher


def load(path): return json.loads(Path(path).read_text())


def prepare(root,out):
    native=root/'native_depth_001'
    parent=Path(load(native/'manifest.json')['parent'])
    source=load(parent/'manifest.json')
    assert load(parent/'completion.json')['completed']
    queries=[]
    for case in source['cases']:
        for prefix in case['checkpoints']:
            path=parent/case['id']/f'packet_{prefix}.json'
            request=load(path)['request']
            assert request is not None and request['current']==prefix-1
            queries.append(dict(case=case['id'],prefix=prefix,reference=request['reference'],
                current=request['current'],similarity=request['similarity'],selection_record=str(path),
                selection_record_sha256=sha(path)))
    sources=[Path(__file__),ROOT/'MemNavData/eval_gem_archived_readout.py',
        ROOT/'MemNavData/eval_gem_native_budget.py',ROOT/'MemNavData/lingbot_pnp_localization.py',
        ROOT/'MemNavData/certified_relocalization_runtime.py',ROOT/'NavDP/baselines/memnav/policy_agent.py',
        *sorted((ROOT/'NavDP/baselines/memnav/gem').glob('*.py'))]
    out.mkdir(parents=True,exist_ok=False)
    save(out/'manifest.json',dict(schema='gem_observation_constraint_information_v1',
        source_root=str(root),native=str(native),parent=str(parent),cases=source['cases'],queries=queries,
        sources_sha256={str(p):sha(p) for p in sources},
        parent_manifest_sha256=sha(parent/'manifest.json'),
        selection='Exactly the38 original observation-only DINO top1 pairs, no new pair selection',
        writer='No geometry rerun; immutable saved native64 and reciprocal-connected predictions',
        matcher='Actual unchanged SuperPoint/LightGlue; shared correspondence arrays for both arms',
        pnp='Existing default original strict PnP certificate; same camera-intrinsic convention as existing readout',
        task_goal_loaded=False,truth_in_inference=False,
        scope='Information-gain diagnostic only; no online map update, gate tuning or navigation claim'))


def run(out):
    manifest=load(out/'manifest.json')
    assert all(sha(p)==h for p,h in manifest['sources_sha256'].items())
    parent,native=Path(manifest['parent']),Path(manifest['native'])
    root=Path(manifest['source_root'])
    assert sha(parent/'manifest.json')==manifest['parent_manifest_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(0)
    matcher=FrozenMatches(LightGluePointMatcher(ROOT/'.diagnostics/dependencies/LightGlue',
        dependency_root=ROOT/'.diagnostics/dependencies/python',device='cuda:0',
        max_keypoints=2048,reference_cache_size=8),out/'matches')
    rows=[]
    geometry_receipts={}
    for case in manifest['cases']:
        identity=case['id']
        for folder in (native/identity,root/'probe_001'/identity):
            receipt=load(folder/'completion.json')
            assert receipt['completed']
            assert all(sha(folder/p)==h for p,h in receipt['files_sha256'].items())
            geometry_receipts[str(folder/'completion.json')]=sha(folder/'completion.json')
        with np.load(parent/identity/'baseline.npz') as data:
            native_pose,keys=data['pose_enc'],data['keys']
        with np.load(root/'reciprocal_001'/identity/'geometry.npz') as data:
            connected_pose,connected_scale=data['pose9'],data['scale']
        poses=dict(native64=native_pose,connected=connected_pose)
        agents={arm:archived_agent(out/'runtime'/identity/arm,archive_encoder(),matcher,case,
            pose,keys,(native if arm=='native64' else root/'probe_001')/identity/'depths',
            np.ones(len(pose)) if arm=='native64' else connected_scale) for arm,pose in poses.items()}
        for query in (q for q in manifest['queries'] if q['case']==identity):
            assert sha(query['selection_record'])==query['selection_record_sha256']
            current,reference=query['current'],query['reference']
            path=Path(case['rgb_paths'][current])
            assert sha(path)==case['rgb_sha256'][current]
            image=path.read_bytes()
            import hashlib
            key=hashlib.md5(image).hexdigest()
            candidates=[dict(anchor=reference,score=query['similarity'])]
            row=dict(**query,current_rgb_sha256=sha(path),reference_rgb_sha256=case['rgb_sha256'][reference],arms={})
            for arm,agent in agents.items():
                agent.memory.frame_count=current+1
                agent.memory.poses=[torch.from_numpy(p.copy()) for p in poses[arm][:current+1]]
                agent.memory.online_depths.count=current+1
                agent.memory.begin_goal(key)
                agent.memory.clear_goal(key)
                agent.memory.goal_start_frames[key]=current
                cv2.setRNGSeed(0)
                result=agent.certified_relocalize(image,candidates,reference_depth_source='online_history')
                # The original observation-only DINO selector also includes
                # initial scale frames0..7. Existing task readout excludes
                # those anchors. Keep all four such pairs as unsupported by
                # this interface, without substituting a different reference.
                if reference<8:
                    assert not result['ok'] and result['reason']=='invalid_candidate_contract'
                else:
                    assert result['ok'] and not any(r.get('error') for r in result['ranked_candidates'])
                assert result.get('pnp',{}).get('status')!='runtime_exception'
                if result['accepted']: assert result['selected_anchor']==reference
                row['arms'][arm]=result
            assert row['arms']['native64'].get('ranked_candidates')==row['arms']['connected'].get('ranked_candidates')
            row['interface_eligible']=reference>=8
            rows.append(row)
            with (out/'inference_progress.jsonl').open('a') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(dict(case=identity,prefix=current+1,reference=reference,
                accepted={a:r['accepted'] for a,r in row['arms'].items()})),flush=True)
    assert len(rows)==len(manifest['queries'])==38
    save(out/'inference.json',dict(rows=rows,actual_match_pairs=len(matcher.cache)))
    assert all(sha(p)==h for p,h in manifest['sources_sha256'].items())
    save(out/'completion.json',dict(completed=True,queries=len(rows),sources_unchanged=True,
        geometry_receipt_sha256=geometry_receipts,actual_match_pairs=len(matcher.cache),
        inference_sha256=sha(out/'inference.json'),manifest_sha256=sha(out/'manifest.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args.root.resolve(),args.out.resolve())
    else: run(args.out.resolve())
