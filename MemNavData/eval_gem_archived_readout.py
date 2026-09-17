"""Paired real SP/LightGlue/PnP on sealed native and episodic geometry.

All arms consume the same frozen matches and DINO shortlist. This isolates
memory geometry; it is not a measurement of cold matching runtime or nav SR.
Ground truth is opened only after every inference result has been saved.
"""
import argparse
from collections.abc import Mapping
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
from run_gem_reactivation_probe import CONFIG, WEIGHTS, GCTStream, load_and_preprocess_images, sha, save
from score_gem_reactivation_probe import truth
from NavDP.baselines.memnav.policy_agent import MemNavAgent
from NavDP.baselines.memnav.gem.episodic import EpisodicGEM
from NavDP.baselines.memnav.gem.memory import GeometricEpisodicMemory
from MemNavData.lingbot_pnp_localization import LightGluePointMatcher


class FrozenMatches:
    """Experiment-only shared correspondence evidence; no change to SP/LG."""
    def __init__(self, matcher, directory):
        self.matcher, self.directory = matcher, directory
        self.cache = {}
        directory.mkdir()

    def match_paths(self, reference, query, **kwargs):
        key = hashlib.sha256((sha(reference) + sha(query) + json.dumps(kwargs, sort_keys=True)).encode()).hexdigest()
        if key not in self.cache:
            value = self.matcher.match_paths(reference, query, **kwargs)
            self.cache[key] = value
            np.savez_compressed(self.directory / (key + '.npz'), **value)
        return self.cache[key]


class SavedDepth(Mapping):
    def __init__(self, directory, scale, count):
        self.directory, self.scale, self.count = directory, scale, count
        self.array_bytes = count * 518 * 518 * 4
        self.disk_bytes = sum(p.stat().st_size for p in directory.glob('*.npz'))

    def __len__(self):
        return self.count

    def __iter__(self):
        return iter(range(self.count))

    def __getitem__(self, frame):
        if not 0 <= frame < self.count:
            raise KeyError(frame)
        with np.load(self.directory / f'{frame:06d}.npz') as a:
            assert int(a['frame']) == frame
            return a['depth'].astype(np.float32) * np.float32(self.scale[frame]), a['confidence'].astype(np.float32)


class GoalEncoder:
    def __init__(self, model):
        self.model, self.agg, self.patch_size = model, model.aggregator, 14

    def load_images(self, paths):
        return load_and_preprocess_images(paths, mode='pad', image_size=518, patch_size=14)

    @torch.inference_mode()
    def dino(self, images):
        images = images.to('cuda')
        mean = self.agg._resnet_mean.reshape(1, 3, 1, 1).to(images.dtype)
        std = self.agg._resnet_std.reshape(1, 3, 1, 1).to(images.dtype)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out = self.agg.patch_embed.forward_features((images-mean)/std)
        return dict(cls=out['x_norm_clstoken'].float())


def archived_agent(root, encoder, matcher, case, poses, keys, depth, scale):
    agent = object.__new__(MemNavAgent)
    agent.S, agent.W, agent.flow_gate = 8, 64, 'off'
    agent.buffer_root, agent._episode_counter = str(root), -1
    agent.device = torch.device('cuda')
    agent.lb = encoder
    agent.certified_relocalization_matcher = matcher
    agent.certified_reference_depth_source = 'online_history'
    agent.memory = EpisodicGEM(agent)
    # Load sealed first-write records instead of running a second model. The
    # production readout methods are invoked unchanged on these real arrays.
    GeometricEpisodicMemory.reset(agent.memory, seed=case['seed'])
    for i, path in enumerate(case['rgb_paths']):
        (Path(agent.rgb_dir) / f'{i}.jpg').symlink_to(path)
    agent.memory.poses = [torch.from_numpy(p.copy()) for p in poses]
    agent.memory.descriptors = [torch.from_numpy(k.copy()) for k in keys]
    agent.memory.frame_count = len(poses)
    agent.memory.online_depths = SavedDepth(depth, scale, len(poses))
    return agent


def score(out, evaluations, cases, rows):
    scores = []
    for row in rows:
        ev = evaluations[row['case']]
        assert sha(ev['trace']) == ev['trace_sha256']
        trace = {r['step']: r for r in json.loads(Path(ev['trace']).read_text())['poses']}
        query = next(q for q in ev['queries'] if q['id'] == row['query'])
        current = truth(trace[row['prefix']-1])
        goal = np.asarray(query['goal_position_reporting_only'])
        local = current[:3, :3].T @ (goal-current[:3, 3])
        expected = np.array([local[2], -local[0]])
        errors = {}
        for arm, result in row['arms'].items():
            predicted = result.get('aux_pose')
            value = None
            if result.get('accepted') and predicted is not None and np.linalg.norm(expected) >= .5:
                predicted = np.asarray(predicted).reshape(-1)
                if len(predicted) == 2 and np.linalg.norm(predicted) > 1e-10:
                    value = float(np.degrees(np.arccos(np.clip(
                        predicted @ expected / np.linalg.norm(predicted) / np.linalg.norm(expected), -1, 1))))
            errors[arm] = value
        scores.append(dict(case=row['case'], query=row['query'], prefix=row['prefix'],
            role=query['analysis_role'], complete_history=row['prefix']==cases[row['case']]['frame_count'],
            horizontal_goal_distance_m=float(np.linalg.norm(expected)), errors=errors,
            accepted={a: bool(r.get('accepted')) for a, r in row['arms'].items()}))
    summary = {}
    for scope in ('complete_history', 'all_prefixes'):
        summary[scope] = {}
        for role in ('revisit', 'novel'):
            selected = [r for r in scores if r['role'] == role and (scope=='all_prefixes' or r['complete_history'])]
            summary[scope][role] = {}
            for arm in ('native', 'camera', 'reciprocal'):
                values = [r['errors'][arm] for r in selected if r['errors'][arm] is not None]
                joint = [r['errors'][arm] for r in selected if all(r['errors'][a] is not None for a in ('native','camera','reciprocal'))]
                describe = lambda v: dict(n=len(v), median=float(np.median(v)) if v else None,
                    mean=float(np.mean(v)) if v else None, p95=float(np.percentile(v, 95)) if v else None)
                summary[scope][role][arm] = dict(queries=len(selected),
                    accepted=sum(r['accepted'][arm] for r in selected),
                    bearing_deg=describe(values), common_accepted_bearing_deg=describe(joint))
    save(out/'evaluation.json', dict(rows=scores, summary=summary,
        scope='Fixed development goal readout, shared real matches, not navigation SR',
        prefix_caveat='A full-history Revisit goal need not be observed in an earlier prefix'))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--connected', type=Path, required=True)
    parser.add_argument('--reciprocal', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    native, connected, reciprocal = [p.resolve() for p in (args.native, args.connected, args.reciprocal)]
    args.out = args.out.resolve()
    parent = Path(json.loads((native/'manifest.json').read_text())['parent'])
    source = json.loads((parent/'manifest.json').read_text())
    for root in (native, connected):
        assert json.loads((root/'completion.json').read_text())['completed']
    # Strip evaluator-only labels and positions before constructing inference inputs.
    evaluation = json.loads((parent/'evaluation_inputs.json').read_text())
    queries = {c['id']: [{k:q[k] for k in ('id','goal_path','goal_sha256')} for q in c['queries']] for c in evaluation['cases']}
    args.out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), ROOT/'NavDP/baselines/memnav/policy_agent.py',
        ROOT/'MemNavData/lingbot_pnp_localization.py', ROOT/'MemNavData/certified_relocalization_runtime.py',
        *sorted((ROOT/'NavDP/baselines/memnav/gem').glob('*.py'))]
    hashes = {str(p):sha(p) for p in files}
    save(args.out/'manifest.json', dict(cases=source['cases'], queries=queries,
        native=str(native), connected=str(connected), reciprocal=str(reciprocal),
        sources_sha256=hashes, weights_sha256=source['checkpoint_sha256'],
        shared_frozen_matches=True, pnp='original strict certificate, default SiftPnPConfig',
        geometry_archive_dtype='float16 for all arms', no_goal_or_truth_in_writing=True,
        no_truth_in_query=True, created_at=time.time()))
    save(args.out/'evaluation_inputs.json', evaluation)
    del evaluation
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = GCTStream(**CONFIG)
    ckpt = torch.load(WEIGHTS, map_location='cpu', weights_only=False)
    assert sha(WEIGHTS) == source['checkpoint_sha256']
    model.load_state_dict(ckpt.get('model',ckpt), strict=True)
    del ckpt
    gc.collect()
    model = model.to('cuda').eval().requires_grad_(False)
    model.aggregator.to(dtype=torch.bfloat16)
    encoder = GoalEncoder(model)
    matcher = FrozenMatches(LightGluePointMatcher(ROOT/'.diagnostics/dependencies/LightGlue',
        dependency_root=ROOT/'.diagnostics/dependencies/python', device='cuda:0',
        max_keypoints=2048, reference_cache_size=8), args.out/'matches')
    rows = []
    for case in source['cases']:
        for root in (native, connected):
            receipt = json.loads((root/case['id']/'completion.json').read_text())
            assert all(sha(root/case['id']/p)==h for p,h in receipt['files_sha256'].items())
        with np.load(parent/case['id']/'baseline.npz') as a:
            poses = dict(native=a['pose_enc']); keys = a['keys']
        with np.load(connected/case['id']/'geometry.npz') as a:
            poses['camera']=a['camera_pose9']; camera_scale=a['scale']
            assert np.array_equal(keys, a['keys'])
        with np.load(reciprocal/case['id']/'geometry.npz') as a:
            poses['reciprocal']=a['pose9']; reciprocal_scale=a['scale']
        scales=dict(native=np.ones(len(keys)), camera=camera_scale, reciprocal=reciprocal_scale)
        agents={arm:archived_agent(args.out/'runtime'/case['id']/arm,encoder,matcher,case,p,keys,
            (native if arm=='native' else connected)/case['id']/'depths',scales[arm]) for arm,p in poses.items()}
        all_descriptors={arm:agent.memory.descriptors for arm,agent in agents.items()}
        for prefix in case['checkpoints']:
            for query in queries[case['id']]:
                assert sha(query['goal_path'])==query['goal_sha256']
                goal=Path(query['goal_path']).read_bytes()
                goal_key=hashlib.md5(goal).hexdigest()
                row=dict(case=case['id'],prefix=prefix,query=query['id'],arms={})
                frozen=None
                for arm,agent in agents.items():
                    agent.memory.frame_count=prefix
                    agent.memory.poses=[torch.from_numpy(p.copy()) for p in poses[arm][:prefix]]
                    agent.memory.descriptors=all_descriptors[arm][:prefix]
                    agent.memory.online_depths.count=prefix
                    agent.memory.begin_goal(goal_key)
                    agent.memory.clear_goal(goal_key)
                    agent.memory.goal_start_frames[goal_key]=prefix-1
                    candidates,_=agent.memory.retrieve(goal,goal_key,prefix-1,prefix-2)
                    if frozen is None:
                        frozen=candidates
                    assert candidates==frozen
                    cv2.setRNGSeed(0)
                    result=agent.certified_relocalize(goal,candidates,reference_depth_source='online_history')
                    if result.get('pnp',{}).get('status')=='runtime_exception':
                        raise RuntimeError(result['pnp'])
                    if any(r.get('error') for r in result.get('ranked_candidates',[])):
                        raise RuntimeError('Matcher exception in a controlled query')
                    repeated=agent.certified_relocalize(goal,candidates,reference_depth_source='online_history')
                    assert repeated.get('cached') is True
                    assert repeated['accepted']==result['accepted']
                    if result.get('accepted'):
                        np.testing.assert_allclose(repeated['aux_pose'],result['aux_pose'],atol=1e-12)
                    row['arms'][arm]=result
                rows.append(row)
                print(json.dumps(dict(case=case['id'],prefix=prefix,query=query['id'],
                    accepted={a:r['accepted'] for a,r in row['arms'].items()})),flush=True)
        del agents, poses, keys
    save(args.out/'inference.json',dict(rows=rows,shared_match_pairs=len(matcher.cache)))
    assert all(sha(p)==h for p,h in hashes.items())
    save(args.out/'completion.json',dict(completed=True,queries=len(rows),sources_unchanged=True,
        inference_sha256=sha(args.out/'inference.json')))
    evaluations={c['id']:c for c in json.loads((args.out/'evaluation_inputs.json').read_text())['cases']}
    score(args.out,evaluations,{c['id']:c for c in source['cases']},rows)


if __name__=='__main__':
    main()
