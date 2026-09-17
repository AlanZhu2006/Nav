#!/usr/bin/env python3
"""Export fixed raw-first-decision image pairs from completed covisibility archives.

Standard-library-only, read-only on the archive host. The tar stream contains
RGB pairs and separate evaluation metadata, never simulator depth or a model
input derived from a role label. No navigation or inference runs here.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time


ROOT = Path('/scratch/yz11502/Research/Nav-axis-uturn-results/'
            'hm3d_covisibility_repaired_20260909/run_bfb8fc3d5a1bb081/'
            'evaluation_f284c23d7a98b0c2')
SUMMARY_SHA = '0d242ef3dabc35017e4f4bb0b4d7d66c80c1a7b3e8dc23d049d2e433f673bfca'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_checked(path, expected=None):
    data = Path(path).read_bytes()
    if expected is not None and sha(data) != expected:
        raise ValueError(f'Changed source: {path}')
    return data


def archived_plans(folder):
    receipt = json.loads(read_checked(folder / 'archive_receipt.json'))
    if not (receipt['completed'] and receipt['all_member_hashes_readback_verified']):
        raise ValueError(f'Incomplete source: {folder}')
    targets = {f'evaluation/{arm}/query_plans.json' for arm in ('raw_fixed', 'cec')}
    found, hashes, index = {}, {}, None
    with tarfile.open(receipt['archive'], 'r|gz') as archive:
        for member in archive:
            if member.name == 'task/artifact_index.json':
                index = {r['path']: r for r in json.load(archive.extractfile(member))['files']}
            relative = member.name.removeprefix('task/')
            if relative not in targets:
                continue
            if index is None or relative not in index:
                raise ValueError(f'Missing provenance: {relative}')
            data = archive.extractfile(member).read()
            entry = index[relative]
            if sha(data) != entry['sha256'] or len(data) != entry['bytes']:
                raise ValueError(f'Archive member mismatch: {relative}')
            found[relative.split('/')[1]] = json.loads(data)
            hashes[relative] = entry
            if len(found) == len(targets):
                break
    if set(found) != {'raw_fixed', 'cec'}:
        raise ValueError(f'Missing source plans: {folder}')
    return found, hashes, receipt['archive_sha256']


def first_state(plans):
    plan = plans['query_leg'][0]
    state = next(r for r in plans['rollout_traces']['query'] if r['step'] == plan['step'])
    return plan, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--expected-user', default='yz11502')
    args = parser.parse_args()
    if getpass.getuser() != args.expected_user:
        raise ValueError('Wrong archive account')
    summary = json.loads(read_checked(args.root / 'paired_summary.json', SUMMARY_SHA))
    records = sorted(summary['paired_records'], key=lambda r: r['task_index'])
    if len(records) != 159 or len({r['task_index'] for r in records}) != 159:
        raise ValueError('Frozen 159-query population changed')
    pairs, evaluation, files = [], [], []
    started = time.monotonic()
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz', compresslevel=1) as out:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            out.addfile(info, io.BytesIO(data))
            files.append({'path': name, 'bytes': len(data), 'sha256': sha(data)})

        for record in records:
            task = int(record['task_index'])
            folder = args.root / f'task_{task:03d}'
            manifest = json.loads(read_checked(folder / 'manifest.json'))
            construction_path = Path(manifest['task']['construction'])
            construction = json.loads(read_checked(
                construction_path, manifest['task']['construction_sha256']))
            query = next(q for q in construction['queries'] if q['query_id'] == record['query_id'])
            plans, members, archive_sha = archived_plans(folder)
            raw, state = first_state(plans['raw_fixed'])
            cec, other_state = first_state(plans['cec'])
            if any(state[k] != other_state[k] for k in ('step', 'x', 'y', 'z', 'yaw', 'jpg_sha256')):
                raise ValueError('Raw and GEM initial states differ')
            source = Path(construction['online_a_episode'])
            receipt = json.loads(read_checked(source / 'receipt.json', construction['online_a_receipt_sha256']))
            anchor = int(raw['anchor'])
            n = len(receipt['rgb_frame_hashes'])
            if not 0 <= anchor < n or int(raw['frame_idx']) != n:
                raise ValueError('Selected anchor is not in the causal A prefix')
            ref_data = read_checked(source / f'rgb/{anchor:06d}.jpg', receipt['rgb_frame_hashes'][anchor])
            goal_data = read_checked(construction_path.parent / query['goal_rgb'], query['goal_rgb_sha256'])
            prefix = f'pairs/task_{task:03d}'
            goal_path, ref_path = f'{prefix}/goal.jpg', f'{prefix}/reference.jpg'
            add(goal_path, goal_data)
            add(ref_path, ref_data)
            pairs.append({'task': task, 'goal': goal_path, 'reference': ref_path,
                          'goal_sha256': sha(goal_data), 'reference_sha256': sha(ref_data),
                          'raw_anchor': anchor, 'history_frames': n})
            curve = query['covis_curve']
            trials = cec.get('router_candidate_trials') or []
            same_candidate = next((r for r in trials if int(r['anchor']) == anchor), None)
            evaluation.append({
                'task': task, 'history': record['history_index'], 'scene': record['scene'],
                'query_id': record['query_id'], 'analysis_role': query['analysis_role'],
                'q_frame39': query['q_eligible'], 'q_frame8': max(curve[8:]),
                'q_raw_anchor': curve[anchor], 'geodesic_m': query['geodesic_from_a_end_m'],
                'state': state, 'goal_floor_position': query['floor_position'],
                'raw_bearing': raw.get('memory_bearing_unit'), 'raw_score': raw.get('raw_score'),
                'raw_anchor': anchor, 'gem_anchor': cec.get('router_selected_anchor'),
                'gem_accepted': cec.get('certified_relocalization_accepted'),
                'gem_matching_on_raw_anchor': same_candidate,
                'original_reached': {arm: values['reached'] for arm, values in record['arms'].items()},
                'provenance': {'construction': str(construction_path),
                    'construction_sha256': manifest['task']['construction_sha256'],
                    'archive_sha256': archive_sha, 'members': members,
                    'online_a_episode': str(source),
                    'online_a_receipt_sha256': construction['online_a_receipt_sha256']},
            })
            print(f'EXPORTED {len(pairs)}/159 task={task} elapsed={time.monotonic()-started:.1f}s',
                  file=sys.stderr, flush=True)
        common = {'source_summary_sha256': SUMMARY_SHA, 'source_root': str(args.root),
                  'selection': 'all 159 completed queries, no outcome selection',
                  'new_rollouts': 0, 'inference_calls': 0}
        add('pairs.json', json.dumps({**common, 'schema': 'fixed_raw_rgb_pairs_v1',
            'pairs': pairs}, allow_nan=False).encode())
        add('evaluation_only.json', json.dumps({**common, 'rows': evaluation}, allow_nan=False).encode())
        add('input_files.json', json.dumps(files, allow_nan=False).encode())
    print(f'EXPORT_COMPLETE queries={len(pairs)}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
