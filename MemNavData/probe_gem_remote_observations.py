"""Causal, distant observation constraints outside the native dense window.

This diagnostic differs from the earlier 38 top1 pairs: every original native
commit (and full-history endpoint) queries only observations older than the
64-view/interval7 dense context. It uses the existing eight-candidate temporal
NMS and unchanged SP/LG/PnP certificate. No target image or truth is loaded.
All nine original histories, including those without eligible old observations,
remain in the manifest. Nothing here updates a production map or navigates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
from run_gem_reactivation_probe import sha, save
from eval_gem_archived_readout import archived_agent, FrozenMatches
from eval_gem_native_budget import archive_encoder
from MemNavData.lingbot_pnp_localization import LightGluePointMatcher
from MemNavData.certified_relocalization_contract import (
    CERTIFIED_CANDIDATE_TOP_K, CERTIFIED_CANDIDATE_MIN_GAP,
    CERTIFIED_MINIMUM_ANCHOR)
from NavDP.baselines.memnav.router_candidates import temporal_nms_candidates


def load(path):
    return json.loads(Path(path).read_text())


def prepare(root, out):
    native = root / 'native_depth_001'
    parent = Path(load(native / 'manifest.json')['parent'])
    source = load(parent / 'manifest.json')
    assert load(parent / 'completion.json')['completed']
    assert source['keyframe_interval'] == 7
    assert source['constructor']['kv_cache_sliding_window'] == 64
    assert source['constructor']['kv_cache_scale_frames'] == 8
    interval = source['keyframe_interval']
    exclusion = source['constructor']['kv_cache_sliding_window'] * interval
    queries, geometry = [], {}
    for case in source['cases']:
        identity, count = case['id'], case['frame_count']
        path = parent / identity / 'baseline.npz'
        geometry[str(path)] = sha(path)
        geometry[str(root / 'reciprocal_001' / identity / 'geometry.npz')] = sha(
            root / 'reciprocal_001' / identity / 'geometry.npz')
        with np.load(path) as data:
            keys = data['keys'].astype(np.float64)
        assert keys.shape[0] == count and np.isfinite(keys).all()
        norms = np.linalg.norm(keys, axis=1)
        assert (norms > 0).all()
        keys /= norms[:, None]
        currents = sorted(set(range(8, count, interval)) | {count - 1})
        for current in currents:
            ceiling = current - exclusion
            eligible = np.zeros(current + 1, dtype=bool)
            if ceiling >= CERTIFIED_MINIMUM_ANCHOR:
                eligible[CERTIFIED_MINIMUM_ANCHOR:ceiling + 1] = True
            scores = keys[:current + 1] @ keys[current]
            candidates = temporal_nms_candidates(
                scores.tolist(), eligible.tolist(), top_k=CERTIFIED_CANDIDATE_TOP_K,
                min_frame_gap=CERTIFIED_CANDIDATE_MIN_GAP)
            assert all(8 <= c['anchor'] <= ceiling for c in candidates)
            queries.append(dict(case=identity, current=current, prefix=current + 1,
                history_endpoint=current == count - 1, candidate_ceiling=ceiling,
                candidates=candidates))
    sources = [Path(__file__), ROOT / 'MemNavData/score_gem_remote_observations.py',
        ROOT / 'MemNavData/eval_gem_archived_readout.py',
        ROOT / 'MemNavData/eval_gem_native_budget.py',
        ROOT / 'MemNavData/lingbot_pnp_localization.py',
        ROOT / 'MemNavData/certified_relocalization_contract.py',
        ROOT / 'MemNavData/certified_relocalization_runtime.py',
        ROOT / 'NavDP/baselines/memnav/policy_agent.py',
        ROOT / 'NavDP/baselines/memnav/router_candidates.py',
        *sorted((ROOT / 'NavDP/baselines/memnav/gem').glob('*.py'))]
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'manifest.json', dict(schema='gem_remote_observation_information_v1',
        source_root=str(root), parent=str(parent), native=str(native),
        cases=source['cases'], queries=queries, exclusion_raw_frames=exclusion,
        source_sha256={str(p): sha(p) for p in sources}, geometry_sha256=geometry,
        parent_manifest_sha256=sha(parent / 'manifest.json'),
        selection='All native interval7 commits plus final endpoints; DINO cosine float64, existing temporal NMS top8/gap4; only anchors at least448raw frames old; no similarity threshold',
        prediction='Unchanged native64 and connected archived geometry; actual observation RGB is the query; no LingBot rerun',
        readout='Unchanged geometry-first strict SP/LightGlue/PnP certificate, shared real matches between arms',
        task_goal_loaded=False, truth_in_inference=False,
        scope='Posthoc development information diagnostic, not held-out navigation, map optimization or candidate selection by evaluator error',
        created_at=time.time()))
    print(json.dumps(dict(cases=len(source['cases']), queries=len(queries),
        eligible=sum(bool(q['candidates']) for q in queries),
        candidate_pairs=sum(len(q['candidates']) for q in queries))), flush=True)


def run(out):
    manifest = load(out / 'manifest.json')
    assert all(sha(p) == h for p, h in manifest['source_sha256'].items())
    assert all(sha(p) == h for p, h in manifest['geometry_sha256'].items())
    parent, native = Path(manifest['parent']), Path(manifest['native'])
    root = Path(manifest['source_root'])
    assert sha(parent / 'manifest.json') == manifest['parent_manifest_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(0)
    matcher = FrozenMatches(LightGluePointMatcher(
        ROOT / '.diagnostics/dependencies/LightGlue',
        dependency_root=ROOT / '.diagnostics/dependencies/python', device='cuda:0',
        max_keypoints=2048, reference_cache_size=8), out / 'matches')
    rows, receipts = [], {}
    for case in manifest['cases']:
        identity = case['id']
        for folder in (native / identity, root / 'probe_001' / identity):
            receipt = load(folder / 'completion.json')
            assert receipt['completed']
            assert all(sha(folder / p) == h for p, h in receipt['files_sha256'].items())
            receipts[str(folder / 'completion.json')] = sha(folder / 'completion.json')
        with np.load(parent / identity / 'baseline.npz') as data:
            native_pose, keys = data['pose_enc'], data['keys']
        with np.load(root / 'reciprocal_001' / identity / 'geometry.npz') as data:
            connected_pose, connected_scale = data['pose9'], data['scale']
        poses = dict(native64=native_pose, connected=connected_pose)
        agents = {arm: archived_agent(out / 'runtime' / identity / arm,
            archive_encoder(), matcher, case, pose, keys,
            (native if arm == 'native64' else root / 'probe_001') / identity / 'depths',
            np.ones(len(pose)) if arm == 'native64' else connected_scale)
            for arm, pose in poses.items()}
        for query in (q for q in manifest['queries'] if q['case'] == identity):
            current = query['current']
            path = Path(case['rgb_paths'][current])
            assert sha(path) == case['rgb_sha256'][current]
            for candidate in query['candidates']:
                h = candidate['anchor']
                assert sha(case['rgb_paths'][h]) == case['rgb_sha256'][h]
            image = path.read_bytes()
            key = hashlib.md5(image).hexdigest()
            row = dict(**query, current_rgb_sha256=sha(path), arms={})
            for arm, agent in agents.items():
                agent.memory.frame_count = current + 1
                agent.memory.poses = [torch.from_numpy(p.copy()) for p in poses[arm][:current + 1]]
                agent.memory.online_depths.count = current + 1
                agent.memory.begin_goal(key)
                agent.memory.clear_goal(key)
                agent.memory.goal_start_frames[key] = current
                cv2.setRNGSeed(0)
                result = agent.certified_relocalize(image, query['candidates'],
                    reference_depth_source='online_history')
                assert result['ok'] and not any(
                    r.get('error') for r in result['ranked_candidates'])
                assert result.get('pnp', {}).get('status') != 'runtime_exception'
                if result['accepted']:
                    assert 8 <= result['selected_anchor'] <= query['candidate_ceiling']
                row['arms'][arm] = result
            assert row['arms']['native64']['ranked_candidates'] == row['arms']['connected']['ranked_candidates']
            rows.append(row)
            with (out / 'inference_progress.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            print(json.dumps(dict(case=identity, current=current,
                candidates=len(query['candidates']), accepted={a:r['accepted'] for a,r in row['arms'].items()})), flush=True)
        del agents
    assert len(rows) == len(manifest['queries'])
    save(out / 'inference.json', dict(rows=rows, actual_match_pairs=len(matcher.cache)))
    assert all(sha(p) == h for p, h in manifest['source_sha256'].items())
    save(out / 'completion.json', dict(completed=True, queries=len(rows),
        eligible_queries=sum(bool(q['candidates']) for q in manifest['queries']),
        geometry_receipt_sha256=receipts, actual_match_pairs=len(matcher.cache),
        match_sha256={str(p.relative_to(out)):sha(p) for p in sorted((out / 'matches').glob('*.npz'))},
        inference_sha256=sha(out / 'inference.json'), manifest_sha256=sha(out / 'manifest.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root.resolve(), args.out.resolve())
    else:
        run(args.out.resolve())
