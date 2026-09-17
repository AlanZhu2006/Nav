#!/usr/bin/env python3
"""Local train40 fixed-top8 geometry diagnostic, never a navigation controller.

prepare -> extract -> solve -> score. Only score reads GT or support labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / '.diagnostics/anchor_relation_geometry_20260906/inputs/unpacked'
TABLE = ROOT / '.diagnostics/certificate_distilled_compass_20260813/static_top8_480_lightglue_open_set_rows.csv'
IMAGES = ROOT / '.diagnostics/certificate_distilled_compass_20260813/certificate_images'
LB = Path('/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(jsonable(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def unique_query_correspondences(query, scores, anchors):
    """One contribution per identical cached SuperPoint query keypoint.

    No pixel-radius tuning: the same query is extracted once by the matcher,
    so repeated keypoint coordinates are exactly equal across reference pairs.
    """
    best = {}
    for i, (point, score, anchor) in enumerate(zip(query, scores, anchors)):
        key = tuple(map(float, point))
        priority = (-float(score), int(anchor), i)
        if key not in best or priority < best[key][0]:
            best[key] = (priority, i)
    return np.array([value[1] for _, value in sorted(best.items())], dtype=np.int64)


def prepare(out):
    info_path = INPUTS / 'history_inputs.json'
    info = json.loads(info_path.read_text())
    groups = defaultdict(list)
    with TABLE.open() as stream:
        for row in csv.DictReader(stream):
            if row['split_role'] != 'train':
                raise ValueError('only train40 is allowed')
            groups[row['session_id']].append(row)
    eligible = defaultdict(list)
    exclusions = Counter()
    for sid, rows in sorted(groups.items()):
        r = rows[0]
        key = r['scene'] + '/' + r['episode']
        history = info['histories'].get(key)
        anchors = [int(x['candidate_frame']) for x in rows]
        if history is None:
            exclusions['missing_history'] += 1
            continue
        if len(rows) != 8 or len(set(anchors)) != 8:
            raise ValueError('source must contain exactly eight distinct anchors')
        if max(anchors) >= history['frame_count'] or max(anchors) >= 384:
            exclusions['incomplete_or_over_budget_prefix'] += 1
            continue
        if min(anchors) < 8 or max(anchors) >= int(r['decision_frame']):
            exclusions['startup_or_noncausal_anchor'] += 1
            continue
        paths = [INPUTS / 'rgb' / x['candidate_relative_path'] for x in rows]
        query = IMAGES / r['query_relative_path']
        if not query.exists() or not all(path.exists() for path in paths):
            exclusions['missing_rgb'] += 1
            continue
        eligible[key].append((sid, rows))
    keys, scenes = [], set()
    for key in sorted(eligible):
        scene = key.split('/')[0]
        if len(eligible[key]) >= 4 and scene not in scenes:
            keys.append(key)
            scenes.add(scene)
        if len(keys) == 8:
            break
    if not keys:
        raise ValueError('no complete local histories')
    out.mkdir(parents=True, exist_ok=False)
    sessions, histories = [], {}
    files = {x['path']: x for x in json.loads((INPUTS / 'FILES.json').read_text())}
    for key in keys:
        required = set()
        for sid, rows in eligible[key]:
            r = rows[0]
            query = IMAGES / r['query_relative_path']
            if sha(query) != r['query_content_sha256']:
                raise ValueError('query identity mismatch')
            candidates = []
            for x in sorted(rows, key=lambda x: int(x['dino_rank'])):
                path = INPUTS / 'rgb' / x['candidate_relative_path']
                if sha(path) != x['candidate_rgb_content_sha256']:
                    raise ValueError('candidate identity mismatch')
                anchor = int(x['candidate_frame'])
                required.add(anchor)
                candidates.append({'anchor': anchor, 'path': str(path),
                                   'dino_cosine': float(x['dino_cosine']),
                                   'dino_rank': int(x['dino_rank']),
                                   'sha256': x['candidate_rgb_content_sha256']})
            sessions.append({'id': sid, 'history': key,
                             'decision_frame': int(r['decision_frame']),
                             'query_path': str(query),
                             'query_relative_path': r['query_relative_path'],
                             'query_sha256': r['query_content_sha256'],
                             'candidates': candidates})
        rgb = INPUTS / 'rgb' / key / 'videos/chunk-000/observation.images.rgb'
        prefix_hashes = []
        for frame in range(max(required) + 1):
            path = rgb / f'{frame}.jpg'
            expected = files[str(path.relative_to(INPUTS))]['sha256']
            actual = sha(path)
            if actual != expected:
                raise ValueError('causal prefix identity mismatch')
            prefix_hashes.append(actual)
        record = info['scales'][key]
        if (not record['valid'] or record['prefix_end_frame_exclusive'] != 64
                or min(s['decision_frame'] for s in sessions if s['history'] == key) < 64):
            raise ValueError('ineligible height receipt')
        histories[key] = {'anchors': sorted(required), 'rgb_dir': str(rgb),
                          'prefix_sha256': prefix_hashes,
                          'pose9': info['camera_predictions'][key]['cam_pose_enc'][:max(required)+1],
                          'metric_scale': record['metric_scale_m_per_raw'],
                          'scale_prefix_frames': 64}
    save(out / 'manifest.json', {
        'schema': 'gem_joint_reference_local_v1',
        'scope': 'train40 expert-prefix component diagnostic; no navigation',
        'selection': 'first eight lexicographic distinct scenes with >=4 complete sessions; <=384 prefix frames',
        'table_sha256': sha(TABLE), 'history_inputs_sha256': sha(info_path),
        'script_sha256': sha(__file__), 'protocol_sha256': sha(Path(__file__).with_name('GEM_JOINT_REFERENCE_PROTOCOL_20260907.md')),
        'qualification_exclusions': dict(exclusions), 'histories': histories,
        'sessions': sessions, 'created_unix': time.time(),
    })
    print(json.dumps({'prepared': str(out), 'histories': len(histories),
                      'sessions': len(sessions),
                      'unique_queries': len(set(s['query_sha256'] for s in sessions)),
                      'anchors': sum(len(h['anchors']) for h in histories.values()),
                      'frames': sum(len(h['prefix_sha256']) for h in histories.values())}), flush=True)


def extract(out, manifest):
    import torch
    from MemNavData.diag_m2p_s1_gct_query import _build_model
    from MemNavData.extract_anchor_relation_geometry import WEIGHTS_SHA
    config = argparse.Namespace(lingbot_repo=LB, weights=LB/'weights/lingbot-map-long.pt',
                                image_size=518, patch_size=14, window=32, num_scale=8,
                                max_frame_num=4096, camera_iterations=4, device='cuda')
    if sha(config.weights) != WEIGHTS_SHA:
        raise ValueError('LingBot weight identity changed')
    torch.set_num_threads(8)
    torch.manual_seed(11)
    model = _build_model(config)
    import lingbot_map.utils.load_fn as lf
    lf.tqdm = lambda iterable, **kwargs: iterable
    results = []
    for key, history in manifest['histories'].items():
        path = out / (key.replace('/', '__') + '.npz')
        if path.exists():
            raise FileExistsError(path)
        rgb = Path(history['rgb_dir'])
        last = max(history['anchors'])
        needed = set(history['anchors'])
        for index, expected in enumerate(history['prefix_sha256']):
            if sha(rgb/f'{index}.jpg') != expected:
                raise ValueError('RGB changed after manifest')
        model.clean_kv_cache()
        payload = {}
        start = time.monotonic()
        def consume(images, index, startup=False):
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                images = images[None].cuda()
                aggregated, psi = model._aggregate_features(
                    images, num_frame_for_scale=8,
                    num_frame_per_block=8 if startup else 1)
                if index in needed:
                    result = model._predict_depth(aggregated, images, psi)
                    payload[f'{index}/depth'] = result['depth'][0,-1,...,0].float().cpu().numpy()
                    payload[f'{index}/confidence'] = result['depth_conf'][0,-1].float().cpu().numpy()
        first = lf.load_and_preprocess_images([str(rgb/f'{i}.jpg') for i in range(8)],
                                              mode='pad', image_size=518, patch_size=14)
        consume(first, 7, True)
        del first
        for begin in range(8, last+1, 16):
            stop = min(begin+16, last+1)
            batch = lf.load_and_preprocess_images([str(rgb/f'{i}.jpg') for i in range(begin,stop)],
                                                  mode='pad', image_size=518, patch_size=14)
            for index in range(begin, stop):
                consume(batch[index-begin:index-begin+1], index)
            del batch
            if stop % 64 == 8 or stop == last+1:
                print(f'[geometry] {key} {stop}/{last+1} {time.monotonic()-start:.1f}s', flush=True)
        np.savez_compressed(path, **payload)
        results.append({'history': key, 'anchors': len(needed),
                        'elapsed_s': time.monotonic()-start, 'sha256': sha(path),
                        'bytes': path.stat().st_size})
    save(out/'geometry_receipt.json', {'histories': results, 'weights_sha256': WEIGHTS_SHA,
                                      'query_rgb_consumed': False, 'GT_consumed': False,
                                      'torch': torch.__version__, 'gpu': torch.cuda.get_device_name()})


def correspondences(match, depth, confidence, pose):
    from MemNavData.lingbot_pnp_localization import lift_reference_keypoints
    from MemNavData.certified_relocalization_runtime import CERTIFIED_EPIPOLAR_THRESHOLD_PX
    ref, query = match['reference_points'], match['query_points']
    if len(ref) < 8:
        return np.empty((0,3)), np.empty((0,2)), np.empty(0), np.empty((0,2))
    cv2.setRNGSeed(0)
    _, mask = cv2.findFundamentalMat(ref.astype(np.float32), query.astype(np.float32),
                                    cv2.USAC_MAGSAC, CERTIFIED_EPIPOLAR_THRESHOLD_PX, .999, 10000)
    keep = np.zeros(len(ref), dtype=bool) if mask is None else mask.reshape(-1).astype(bool)
    ref, query, scores = ref[keep], query[keep], match['scores'][keep]
    xyz, valid = lift_reference_keypoints(ref, depth, confidence, pose, confidence_quantile=0.)
    return xyz[valid], query[valid], scores[valid], ref[valid]


def solve(out, manifest):
    from MemNavData.lingbot_pnp_localization import (
        LightGluePointMatcher, SiftPnPConfig, correspondence_pnp_localize,
        intrinsics_from_pose9, solve_camera_pose_pnp, _coverage)
    from MemNavData.certified_relocalization_runtime import (
        fundamental_support, rank_candidates, certificate_decision,
        fundamental_can_reach_certificate, CERTIFIED_EPIPOLAR_THRESHOLD_PX)
    matcher = LightGluePointMatcher(ROOT/'.diagnostics/dependencies/LightGlue',
                                   dependency_root=ROOT/'.diagnostics/dependencies/python',
                                   device='cuda', max_keypoints=2048)
    config = SiftPnPConfig()
    geometry_receipt = json.loads((out/'geometry_receipt.json').read_text())
    expected_hash = {x['history']:x['sha256'] for x in geometry_receipt['histories']}
    for key in manifest['histories']:
        if sha(out/(key.replace('/', '__')+'.npz')) != expected_hash[key]:
            raise ValueError('geometry cache changed')
    (out/'sessions').mkdir(exist_ok=False)
    last_key, geometry = None, None
    for index, session in enumerate(manifest['sessions']):
        started = time.monotonic()
        key = session['history']
        history = manifest['histories'][key]
        poses = np.asarray(history['pose9'])
        if key != last_key:
            if geometry is not None:
                geometry.close()
            geometry = np.load(out/(key.replace('/', '__')+'.npz'))
            last_key = key
        matches, evidence = {}, []
        for item in session['candidates']:
            anchor = item['anchor']
            match = matcher.match_paths(Path(item['path']), Path(session['query_path']),
                                         target_height=518, target_width=518)
            matches[anchor] = match
            support = fundamental_support(match['reference_raw_points'], match['query_raw_points'],
                                           match['scores'], tuple(match['reference_raw_hw']),
                                           tuple(match['query_raw_hw']))
            evidence.append({**support, 'anchor': anchor,
                             'dino_cosine': item['dino_cosine']})
        match_s = time.monotonic()-started
        ranked = rank_candidates(evidence)
        pivot = ranked[0]['anchor']
        intrinsic = intrinsics_from_pose9(poses[pivot], 518, 518)
        singles, clouds, uv, confidence_list, sources = [], [], [], [], []
        correspondence_arrays = {}
        for item in session['candidates']:
            anchor = item['anchor']
            match = matches[anchor]
            depth, confidence = geometry[f'{anchor}/depth'], geometry[f'{anchor}/confidence']
            start = time.monotonic()
            result = correspondence_pnp_localize(
                match['reference_points'], match['query_points'], depth, confidence,
                poses[anchor], config=config, match_scores=match['scores'],
                epipolar_threshold_px=CERTIFIED_EPIPOLAR_THRESHOLD_PX, query_intrinsic=intrinsic)
            singles.append({'anchor': anchor, 'pnp': result,
                            'certificate': certificate_decision(result),
                            'solve_s': time.monotonic()-start})
            xyz, query, scores, ref = correspondences(match, depth, confidence, poses[anchor])
            clouds.append(xyz); uv.append(query); confidence_list.append(scores)
            sources.append(np.full(len(xyz), anchor))
            for name, value in [('xyz', xyz), ('query', query), ('scores', scores), ('reference', ref)]:
                correspondence_arrays[f'{anchor}/{name}'] = value
        xyz, query = np.concatenate(clouds), np.concatenate(uv)
        scores, anchors = np.concatenate(confidence_list), np.concatenate(sources)
        dedup = unique_query_correspondences(query, scores, anchors)
        start = time.monotonic()
        joint = solve_camera_pose_pnp(xyz[dedup], query[dedup], intrinsic,
                                      config=config, fov_pose9=poses[pivot])
        joint_s = time.monotonic()-start
        inliers = np.asarray(joint.get('inlier_indices', []), dtype=np.int64)
        contributions = Counter(map(int, anchors[dedup][inliers]))
        joint['query_inlier_coverage'] = _coverage(query[dedup][inliers], 518, 518)
        start_result = next(x for x in singles if x['anchor'] == pivot)
        possible, reason = fundamental_can_reach_certificate(ranked[0])
        path = out/'sessions'/f'{index:03d}.json'
        result = {'id': session['id'], 'history': key, 'pivot': pivot, 'ranked': ranked,
                  'single': start_result, 'all_singles': singles, 'joint': joint,
                  'current_certificate_accept': bool(possible and start_result['certificate']['accepted']),
                  'current_precheck_reason': reason,
                  'joint_authorized': False,
                  'total_correspondences': len(xyz), 'unique_query_correspondences': len(dedup),
                  'joint_inlier_reference_contribution': dict(contributions),
                  'matching_s': match_s, 'joint_pnp_s': joint_s,
                  'total_diagnostic_s': time.monotonic()-started,
                  'query_intrinsic': intrinsic, 'pnp_config': asdict(config)}
        save(path, result)
        correspondence_arrays['dedup_indices'] = dedup
        np.savez_compressed(path.with_suffix('.npz'), **correspondence_arrays)
        print(f"[solve {index+1}/{len(manifest['sessions'])}] {session['id']} "
              f"points={len(xyz)}->{len(dedup)} joint={joint['status']} "
              f"sources={len(contributions)} {result['total_diagnostic_s']:.2f}s", flush=True)
    save(out/'solve_receipt.json', {'completed': True, 'sessions': len(manifest['sessions']),
                                    'manifest_sha256': sha(out/'manifest.json'),
                                    'opencv': cv2.__version__, 'GT_consumed': False,
                                    'script_sha256': sha(__file__)})


def score(out, manifest):
    import pandas as pd
    from MemNavData.diag_m2p_s1_gct_query import (
        _matrix, _resolve_generated_mount, _goal_camera_to_world,
        _navdp_ground_truth_relative, _lingbot_relative_direction, direction_error_degrees)
    with TABLE.open() as stream:
        rows = list(csv.DictReader(stream))
    labels = defaultdict(list)
    for row in rows:
        labels[row['session_id']].append(float(row['covisibility']))
    cache = {}
    def load(key):
        if key not in cache:
            root = INPUTS/'supervision_only'/key
            meta = json.loads((root/'meta/gen_meta.json').read_text())
            frame = pd.read_parquet(root/'data/chunk-000/episode_000000.parquet',
                                    columns=['action','observation.camera_extrinsic'])
            poses = np.stack([_matrix(value, 'pose') for value in frame['action']])
            mount = _resolve_generated_mount(_matrix(frame.iloc[0]['observation.camera_extrinsic'], 'mount'),
                                              meta.get('frame_convention', ''))
            cache[key] = poses, mount, meta
        return cache[key]
    records = []
    for index, session in enumerate(manifest['sessions']):
        result = json.loads((out/'sessions'/f'{index:03d}.json').read_text())
        if result['id'] != session['id']:
            raise ValueError('paired session mismatch')
        key, pivot = session['history'], result['pivot']
        poses, mount, _ = load(key)
        query_rel = Path(session['query_relative_path'])
        _, _, qmeta = load('/'.join(query_rel.parts[:2]))
        goal = _goal_camera_to_world(qmeta['goals'][int(query_rel.stem.split('_')[-1])-1])
        target = _navdp_ground_truth_relative(poses[pivot], goal, mount)
        history = manifest['histories'][key]
        pivot_pose = np.asarray(history['pose9'][pivot])
        def measure(pnp):
            if pnp.get('status') != 'ok' or 'pose9' not in pnp:
                return {'valid': False, 'error_m': None, 'angle_deg': None, 'within_05m': False}
            pred = _lingbot_relative_direction(pivot_pose, np.asarray(pnp['pose9'])) * history['metric_scale']
            error = float(np.linalg.norm(pred-target))
            angle = direction_error_degrees(pred,target) if np.linalg.norm(target) >= .5 else None
            return {'valid': True, 'xy_m': pred.tolist(), 'error_m': error,
                    'angle_deg': angle, 'within_05m': error <= .5}
        all_single = [(x['anchor'], measure(x['pnp'])) for x in result['all_singles']]
        valid = [(a,m) for a,m in all_single if m['valid']]
        oracle = min(valid, key=lambda x:(x[1]['error_m'], x[0])) if valid else (None,measure({}))
        covis = max(labels[session['id']])
        support = 'no_strong_support' if covis <= .1 else ('partial' if covis < .5 else 'strong')
        records.append({'id':session['id'], 'scene': key.split('/')[0],
                        'query_sha256':session['query_sha256'], 'max_top8_covis':covis,
                        'support':support, 'pivot':pivot, 'target_xy_m':target.tolist(),
                        'target_distance_m':float(np.linalg.norm(target)),
                        'single': measure(result['single']['pnp']), 'oracle':oracle[1],
                        'oracle_anchor':oracle[0], 'joint': measure(result['joint']),
                        'current_certificate_accept':result['current_certificate_accept'],
                        'all_single_errors':all_single,
                        'joint_reference_count':len(result['joint_inlier_reference_contribution'])})
    summary = aggregate(records)
    save(out/'scored_sessions.json', records)
    save(out/'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def aggregate(records):
    summary = {'scope':'offline anchor-relative localization, NOT navigation SR',
               'sessions':len(records), 'scenes':len(set(r['scene'] for r in records)),
               'unique_query_hashes':len(set(r['query_sha256'] for r in records)),
               'support_counts':dict(Counter(r['support'] for r in records)), 'groups':{}}
    for label in ['all','no_strong_support','partial','strong']:
        selected = records if label == 'all' else [r for r in records if r['support']==label]
        arms = {}
        for arm in ['single','oracle','joint']:
            errors = [r[arm]['error_m'] for r in selected if r[arm]['valid']]
            angles = [r[arm]['angle_deg'] for r in selected if r[arm]['angle_deg'] is not None]
            arms[arm] = {'n':len(selected), 'valid':len(errors),
                         'within_05m':sum(r[arm]['within_05m'] for r in selected),
                         'median_error_m':float(np.median(errors)) if errors else None,
                         'mean_error_m':float(np.mean(errors)) if errors else None,
                         'p90_error_m':float(np.percentile(errors,90)) if errors else None,
                         'angle_n':len(angles), 'median_angle_deg':float(np.median(angles)) if angles else None}
        common = [r for r in selected if r['single']['valid'] and r['joint']['valid']]
        arms['paired'] = {
            'joint_gain_within_05m':sum(r['joint']['within_05m'] and not r['single']['within_05m'] for r in selected),
            'joint_loss_within_05m':sum(r['single']['within_05m'] and not r['joint']['within_05m'] for r in selected),
            'common_valid':len(common),
            'median_joint_minus_single_m':float(np.median([r['joint']['error_m']-r['single']['error_m'] for r in common])) if common else None,
            'joint_correct_where_all_singles_incorrect':sum(r['joint']['within_05m'] and not r['oracle']['within_05m'] for r in selected),
            'joint_lower_error_than_best_single':sum(r['joint']['valid'] and r['oracle']['valid'] and r['joint']['error_m']<r['oracle']['error_m'] for r in selected)}
        summary['groups'][label] = arms
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','extract','solve','score'])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.out)
        return
    manifest = json.loads((args.out/'manifest.json').read_text())
    if sha(TABLE) != manifest['table_sha256'] or sha(INPUTS/'history_inputs.json') != manifest['history_inputs_sha256']:
        raise ValueError('input artifacts changed')
    if sha(__file__) != manifest['script_sha256']:
        raise ValueError('experiment implementation changed after prepare')
    globals()[args.stage](args.out, manifest)


if __name__ == '__main__':
    main()
