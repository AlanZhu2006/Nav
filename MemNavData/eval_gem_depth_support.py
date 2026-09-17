"""Read every original goal from the compressed geometric support archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'MemNavData')]
from gem_depth_support import FrameSupportArchive
from eval_gem_native_budget import ExistingMatches,archive_encoder
from eval_gem_archived_readout import archived_agent


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


class CheckedMatches:
    def __init__(self,matcher,archive):
        self.matcher,self.archive=matcher,archive
        self.checked=0

    def match_paths(self,reference,query,**kwargs):
        matched=self.matcher.match_paths(reference,query,**kwargs)
        frame=int(Path(reference).stem)
        assert sha(reference)==self.archive.records[frame]['rgb_sha256']
        self.archive.validate_matches(frame,matched['reference_points'])
        self.checked+=1
        return matched


def run(root,out):
    support=root/'depth_support_001'
    original=root/'readout_001'
    control=root/'native_budget16_readout_002'
    input_files=[support/'manifest.json',original/'manifest.json',original/'inference.json',
        control/'manifest.json',control/'inference.json',control/'completion.json']
    code=[Path(__file__),ROOT/'MemNavData/gem_depth_support.py',
        ROOT/'MemNavData/eval_gem_native_budget.py',ROOT/'MemNavData/eval_gem_archived_readout.py',
        ROOT/'MemNavData/lingbot_pnp_localization.py',ROOT/'MemNavData/certified_relocalization_runtime.py',
        ROOT/'NavDP/baselines/memnav/policy_agent.py',
        *sorted((ROOT/'NavDP/baselines/memnav/gem').glob('*.py'))]
    hashes={str(p):sha(p) for p in input_files+code}
    out.mkdir(parents=True,exist_ok=False)
    save(out/'manifest.json',dict(schema='gem_depth_support_readout_parity_v1',
        sources_sha256=hashes,arms=['native64_fp16','native16_fp32'],queries=151,
        support=str(support),wait_for=str(support/'completion.json'),pid=os.getpid(),
        scope='Unchanged full SP/LG/PnP/certificate/bearing on every original query; fixed matches are replayed and every accessed pixel must belong to independently written detector support; no truth',
        created_at=time.time()))
    try:
        while not (support/'completion.json').exists():
            if (support/'failure.json').exists():
                raise RuntimeError('Support writer failed; no readout substitution is permitted')
            time.sleep(5)
        assert all(sha(p)==h for p,h in hashes.items())
        written=load(support/'completion.json')
        assert written['completed'] and written['frames']==6028 and written['all_detector_lifts_bitwise_equal']
        assert written['manifest_sha256']==sha(support/'manifest.json')
        assert all(sha(p)==h for p,h in written['case_receipt_sha256'].items())
        source_manifest=load(original/'manifest.json')
        compare_manifest=load(control/'manifest.json')
        control_receipt=load(control/'completion.json')
        assert control_receipt['completed']
        assert all(sha(control/p)==h for p,h in control_receipt['files_sha256'].items())
        old=load(original/'inference.json')['rows']
        previous=load(control/'inference.json')['rows']
        assert len(old)==len(previous)==151
        assert all((a['case'],a['prefix'],a['query'])==(b['case'],b['prefix'],b['query']) for a,b in zip(old,previous))
        support_manifest=load(support/'manifest.json')
        parent=Path(support_manifest['parent']);budget=Path(support_manifest['budget'])
        shared=ExistingMatches(original/'matches',compare_manifest['match_files_sha256'])
        torch.set_num_threads(4)
        rows=[];checked=0
        fields=('accepted','reason','selected_anchor','selected_dino_rank','certificate','authority',
            'pnp','aux_pose','direction_vector','terminal_yaw_right_deg','terminal_pitch_up_deg','ranked_candidates')
        for case in source_manifest['cases']:
            identity=case['id'];receipt=load(support/identity/'completion.json')
            assert receipt['completed'] and receipt['frames']==case['frame_count']
            with np.load(parent/identity/'baseline.npz') as z:native_pose,keys=z['pose_enc'],z['keys']
            with np.load(budget/identity/'geometry.npz') as z:budget_pose=z['pose_enc']
            poses=dict(native64_fp16=native_pose,native16_fp32=budget_pose)
            agents={}
            for arm,pose in poses.items():
                geometry=support/identity/arm/'depths'
                record=receipt['archives'][arm]
                assert all(sha(geometry/f'{int(i):06d}.npz')==r['sha256'] for i,r in record['records'].items())
                archive=FrameSupportArchive.open(geometry,record['records'])
                guarded=CheckedMatches(shared,archive)
                agent=archived_agent(out/'runtime'/identity/arm,archive_encoder(),guarded,
                    case,pose,keys,geometry,np.ones(len(pose)))
                agent.memory.online_depths=archive
                agent.device=torch.device('cpu')
                agents[arm]=agent
            for record,expected in zip(old,previous):
                if record['case']!=identity:continue
                prefix,qid=record['prefix'],record['query']
                query=next(q for q in source_manifest['queries'][identity] if q['id']==qid)
                assert sha(query['goal_path'])==query['goal_sha256']
                goal=Path(query['goal_path']).read_bytes();key=hashlib.md5(goal).hexdigest()
                ranking=record['arms']['native']['ranked_candidates']
                ordered=sorted(ranking,key=lambda c:c['dino_rank'])
                candidates=[dict(anchor=r['anchor'],score=r['dino_cosine']) for r in ordered]
                row=dict(case=identity,prefix=prefix,query=qid,arms={})
                for arm,agent in agents.items():
                    agent.memory.frame_count=prefix
                    agent.memory.poses=[torch.from_numpy(p.copy()) for p in poses[arm][:prefix]]
                    agent.memory.online_depths.count=prefix
                    agent.memory.begin_goal(key);agent.memory.clear_goal(key)
                    agent.memory.goal_start_frames[key]=prefix-1
                    cv2.setRNGSeed(0)
                    result=agent.certified_relocalize(goal,candidates,reference_depth_source='online_history')
                    assert result['ok'] and result.get('pnp',{}).get('status')!='runtime_exception'
                    assert not any(r.get('error') for r in result['ranked_candidates'])
                    baseline=expected['arms'][arm]
                    for field in fields:
                        assert result.get(field)==baseline.get(field),(identity,prefix,qid,arm,field,result.get(field),baseline.get(field))
                    cached=agent.certified_relocalize(goal,candidates,reference_depth_source='online_history')
                    assert cached['cached']
                    for field in fields:
                        assert cached.get(field)==result.get(field),(identity,prefix,qid,arm,'cached',field)
                    row['arms'][arm]=dict(semantic_fields_exact=True,result=result)
                rows.append(row)
            checked+=sum(a.certified_relocalization_matcher.checked for a in agents.values())
            print(json.dumps(dict(case=identity,queries=sum(r['case']==identity for r in rows),all_fields_exact=True)),flush=True)
        assert len(rows)==151
        assert all(sha(p)==h for p,h in hashes.items())
        save(out/'inference.json',dict(rows=rows))
        save(out/'result.json',dict(passed=True,queries=151,arm_queries=302,semantic_fields_exact=list(fields),
            cached_fields_exact=True,matched_pixel_support_checks=checked,actual_match_pairs_reused=len(shared.cache),
            accepted={a:sum(r['arms'][a]['result']['accepted'] for r in rows) for a in poses},
            sources_sha256=hashes,support_completion_sha256=sha(support/'completion.json'),
            inference_sha256=sha(out/'inference.json'),
            scope='All original readouts reproduced from compressed predicted geometry; no navigation rerun, loop correction or whole-runtime cost claim'))
        save(out/'process_exit.json',dict(exit_code=0,completed_at=time.time()))
    except BaseException as error:
        save(out/'failure.json',dict(type=type(error).__name__,error=str(error),time=time.time()))
        save(out/'process_exit.json',dict(exit_code=1,completed_at=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.root.resolve(),args.out.resolve())
