"""Fixed SP/LG evidence and unchanged PnP for the native W16 control.

No GPU inference is needed: image correspondences and DINO shortlists are
sealed results of the earlier actual matcher. Native64 is recomputed as a
parity control. W16 is evaluated at both matched FP16 archival precision and
its production FP32 archival precision. These are readouts, not navigation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from run_gem_reactivation_probe import sha,save
from eval_gem_archived_readout import SavedDepth,archived_agent


def load(path): return json.loads(Path(path).read_text())


class EmptyKVState:
    """Empty reset state for a read-only archive; cannot predict geometry."""
    def __init__(self):
        self.kv_cache={}

    def clean_kv_cache(self):
        self.kv_cache.clear()


def archive_encoder():
    model=EmptyKVState()
    model.camera_head=EmptyKVState()
    return SimpleNamespace(patch_size=14,model=model)


class ExistingMatches:
    def __init__(self,directory,hashes):
        self.directory,self.hashes=directory,hashes
        self.cache={}

    def match_paths(self,reference,query,**kwargs):
        key=hashlib.sha256((sha(reference)+sha(query)+json.dumps(kwargs,sort_keys=True)).encode()).hexdigest()
        if key not in self.cache:
            path=self.directory/(key+'.npz')
            assert sha(path)==self.hashes[path.name]
            with np.load(path) as data: self.cache[key]={k:data[k] for k in data.files}
        return self.cache[key]


class QuantizedDepth(SavedDepth):
    def __getitem__(self,frame):
        depth,confidence=super().__getitem__(frame)
        return depth.astype(np.float16).astype(np.float32),confidence.astype(np.float16).astype(np.float32)


def prepare(source,budget,out):
    assert load(source/'independent_verification.json')['verified']
    source_manifest=load(source/'manifest.json')
    budget_manifest=load(budget/'manifest.json')
    assert source_manifest['cases']==budget_manifest['cases']
    sources=[Path(__file__),ROOT/'MemNavData/eval_gem_archived_readout.py',
        ROOT/'MemNavData/lingbot_pnp_localization.py',ROOT/'MemNavData/certified_relocalization_runtime.py',
        ROOT/'NavDP/baselines/memnav/policy_agent.py',*sorted((ROOT/'NavDP/baselines/memnav/gem').glob('*.py'))]
    out.mkdir(parents=True,exist_ok=False)
    save(out/'manifest.json',dict(schema='gem_native_budget_readout_v1',
        source=str(source),budget=str(budget),cases=source_manifest['cases'],
        queries=source_manifest['queries'],budget_manifest_sha256=sha(budget/'manifest.json'),
        source_files_sha256={str(source/p):sha(source/p) for p in
            ('manifest.json','inference.json','completion.json','evaluation.json','evaluation_inputs.json','independent_verification.json')},
        sources_sha256={str(p):sha(p) for p in sources},
        match_files_sha256={p.name:sha(p) for p in (source/'matches').glob('*.npz')},
        arms=['native64_fp16','native16_fp16','native16_fp32'],
        scope='All151 fixed development readouts; exact saved SP/LG correspondences and DINO shortlists; no new matching, thresholds or goal selection',
        no_goal_or_truth_in_writing=True,no_truth_in_query=True))


def describe(values):
    return dict(n=len(values),median=float(np.median(values)) if values else None,
        mean=float(np.mean(values)) if values else None,p95=float(np.percentile(values,95)) if values else None)


def run(out):
    manifest=load(out/'manifest.json')
    source,budget=Path(manifest['source']),Path(manifest['budget'])
    assert sha(budget/'manifest.json')==manifest['budget_manifest_sha256']
    assert load(budget/'completion.json')['completed']
    assert all(sha(p)==h for p,h in manifest['source_files_sha256'].items())
    assert all(sha(p)==h for p,h in manifest['sources_sha256'].items())
    native=Path(load(source/'manifest.json')['native'])
    parent=Path(load(native/'manifest.json')['parent'])
    old=load(source/'inference.json')['rows']
    matcher=ExistingMatches(source/'matches',manifest['match_files_sha256'])
    torch.set_num_threads(4)
    rows=[]
    geometry={}
    archived_hashes={}
    for case in manifest['cases']:
        identity=case['id']
        for folder in (native/identity,budget/identity):
            receipt=load(folder/'completion.json')
            assert receipt['completed']
            assert all(sha(folder/p)==h for p,h in receipt['files_sha256'].items())
            archived_hashes[str(folder/'completion.json')]=sha(folder/'completion.json')
        with np.load(parent/identity/'baseline.npz') as data:
            native_pose,keys=data['pose_enc'],data['keys']
        with np.load(budget/identity/'geometry.npz') as data:
            budget_pose=data['pose_enc']
            assert np.array_equal(keys,data['keys'])
        poses=dict(native64_fp16=native_pose,native16_fp16=budget_pose,native16_fp32=budget_pose)
        geometry[identity]=poses
        agents={}
        for arm,pose in poses.items():
            directory=(native if arm=='native64_fp16' else budget)/identity/'depths'
            agent=archived_agent(out/'runtime'/identity/arm,archive_encoder(),
                matcher,case,pose,keys,directory,np.ones(len(pose)))
            agent.device=torch.device('cpu')
            if arm=='native16_fp16':
                agent.memory.online_depths=QuantizedDepth(directory,np.ones(len(pose)),len(pose))
            agents[arm]=agent
        for previous in (r for r in old if r['case']==identity):
            prefix,qid=previous['prefix'],previous['query']
            query=next(q for q in manifest['queries'][identity] if q['id']==qid)
            assert sha(query['goal_path'])==query['goal_sha256']
            goal=Path(query['goal_path']).read_bytes()
            key=hashlib.md5(goal).hexdigest()
            ranking=previous['arms']['native']['ranked_candidates']
            ordered=sorted(ranking,key=lambda c:c['dino_rank'])
            assert [r['dino_rank'] for r in ordered]==list(range(1,len(ordered)+1))
            candidates=[dict(anchor=r['anchor'],score=r['dino_cosine']) for r in ordered]
            row=dict(case=identity,prefix=prefix,query=qid,arms={})
            for arm,agent in agents.items():
                agent.memory.frame_count=prefix
                agent.memory.poses=[torch.from_numpy(p.copy()) for p in poses[arm][:prefix]]
                agent.memory.online_depths.count=prefix
                agent.memory.begin_goal(key)
                agent.memory.clear_goal(key)
                agent.memory.goal_start_frames[key]=prefix-1
                cv2.setRNGSeed(0)
                result=agent.certified_relocalize(goal,candidates,reference_depth_source='online_history')
                assert result['ok'] and result['ranked_candidates']==ranking
                assert not any(r.get('error') for r in result['ranked_candidates'])
                assert result.get('pnp',{}).get('status')!='runtime_exception'
                cached=agent.certified_relocalize(goal,candidates,reference_depth_source='online_history')
                assert cached['cached'] and cached['accepted']==result['accepted']
                if result['accepted']:
                    np.testing.assert_allclose(cached['aux_pose'],result['aux_pose'],atol=1e-12)
                    pose=poses[arm][prefix-1].astype(np.float32).astype(float)
                    local=Rotation.from_quat(pose[3:7]).inv().apply(np.asarray(result['pnp']['pose9'])[:3]-pose[:3])
                    np.testing.assert_allclose([local[2],-local[0]],result['aux_pose'],atol=2e-9,rtol=2e-9)
                if arm=='native64_fp16':
                    expected=previous['arms']['native']
                    assert result['accepted']==expected['accepted'] and result['selected_anchor']==expected['selected_anchor']
                    if result['accepted']:
                        np.testing.assert_allclose(result['aux_pose'],expected['aux_pose'],atol=1e-9,rtol=1e-9)
                row['arms'][arm]=result
            rows.append(row)
        print(json.dumps(dict(case=identity,queries=sum(r['case']==identity for r in rows))),flush=True)
    assert len(rows)==len(old)==151
    save(out/'inference.json',dict(rows=rows,match_pairs_used=len(matcher.cache)))
    # Open evaluator positions only after every inference result is sealed.
    evaluation={c['id']:c for c in load(source/'evaluation_inputs.json')['cases']}
    cases={c['id']:c for c in manifest['cases']}
    scored=[]
    for row in rows:
        ev=evaluation[row['case']]
        assert sha(ev['trace'])==ev['trace_sha256']
        trace={p['step']:p for p in load(ev['trace'])['poses']}
        pose=trace[row['prefix']-1]
        query=next(q for q in ev['queries'] if q['id']==row['query'])
        dx=query['goal_position_reporting_only'][0]-pose['x']
        dz=query['goal_position_reporting_only'][2]-pose['z']
        yaw=pose['yaw']
        forward=-math.sin(yaw)*dx-math.cos(yaw)*dz
        left=-math.cos(yaw)*dx+math.sin(yaw)*dz
        errors={}
        for arm,result in row['arms'].items():
            value=None
            if result['accepted'] and math.hypot(forward,left)>=.5 and math.hypot(*result['aux_pose'])>1e-10:
                delta=math.atan2(result['aux_pose'][1],result['aux_pose'][0])-math.atan2(left,forward)
                value=abs(math.degrees(math.atan2(math.sin(delta),math.cos(delta))))
            errors[arm]=value
        scored.append(dict(case=row['case'],query=row['query'],prefix=row['prefix'],role=query['analysis_role'],
            complete_history=row['prefix']==cases[row['case']]['frame_count'],bearing_deg=errors,
            accepted={a:r['accepted'] for a,r in row['arms'].items()}))
    summary={}
    for role in sorted({r['role'] for r in scored}):
        summary[role]={}
        for scope in ('complete_history','all_prefixes'):
            selected=[r for r in scored if r['role']==role and (scope=='all_prefixes' or r['complete_history'])]
            summary[role][scope]={arm:dict(queries=len(selected),accepted=sum(r['accepted'][arm] for r in selected),
                bearing_deg=describe([r['bearing_deg'][arm] for r in selected if r['bearing_deg'][arm] is not None]))
                for arm in manifest['arms']}
    save(out/'evaluation.json',dict(rows=scored,summary=summary,scope=manifest['scope']))
    assert all(sha(p)==h for p,h in manifest['sources_sha256'].items())
    save(out/'completion.json',dict(completed=True,queries=len(rows),native64_readout_parity=True,
        pose_to_bearing_verified=True,all_rankings_exact=True,match_pairs_used=len(matcher.cache),
        archived_geometry_receipt_sha256=archived_hashes,
        files_sha256={name:sha(out/name) for name in ('manifest.json','inference.json','evaluation.json')}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path)
    parser.add_argument('--budget',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(args.source.resolve(),args.budget.resolve(),args.out.resolve())
    else: run(args.out.resolve())
