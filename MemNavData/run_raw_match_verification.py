#!/usr/bin/env python3
"""Read matching evidence for fixed raw image pairs; no pose or policy changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
from PIL import Image

from MemNavData.certified_relocalization_runtime import fundamental_support


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def unpack_bundle(source, out):
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, 'r:gz') as archive:
        entries = json.load(archive.extractfile('input_files.json'))
        if len({e['path'] for e in entries}) != len(entries):
            raise ValueError('Duplicate bundle path')
        for e in entries:
            relative = Path(e['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe bundle path')
            member = archive.getmember(e['path'])
            if not member.isfile():
                raise ValueError('Non-file bundle member')
            data = archive.extractfile(member).read()
            if len(data) != e['bytes'] or hashlib.sha256(data).hexdigest() != e['sha256']:
                raise ValueError(f"Changed bundle member {e['path']}")
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(data)
        write_json(out / 'input_files.json', entries)
    return {'verified': True, 'files': len(entries), 'bundle_sha256': sha_file(source)}


def raw_coordinates(points, original_wh, processed_hw, size=512):
    """Invert official DUSt3R resize + centred crop using pixel-centre coordinates."""
    width, height = original_wh
    resized_w, resized_h = (int(round(x * size / max(width, height))) for x in (width, height))
    ph, pw = map(int, processed_hw)
    left, top = resized_w // 2 - pw // 2, resized_h // 2 - ph // 2
    xy = np.asarray(points, dtype=np.float64).copy()
    return (xy + np.array([left, top]) + .5) / np.array(
        [resized_w / width, resized_h / height]) - .5


def statistics(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite matcher confidence')
    return {'count': len(values), 'mean': float(values.mean()) if len(values) else 0.,
            'p10': float(np.percentile(values, 10)) if len(values) else 0.,
            'median': float(np.median(values)) if len(values) else 0.,
            'p90': float(np.percentile(values, 90)) if len(values) else 0.}


class Mast3rPairs:
    def __init__(self, repository, checkpoint, device):
        sys.path.insert(0, str(repository))
        import torch
        from mast3r.model import AsymmetricMASt3R
        from mast3r.fast_nn import fast_reciprocal_NNs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
        self.torch, self.device = torch, device
        self.inference, self.load_images, self.nn = inference, load_images, fast_reciprocal_NNs
        self.model = AsymmetricMASt3R.from_pretrained(str(checkpoint)).eval().to(device)

    def match(self, reference, goal):
        views = self.load_images([str(reference), str(goal)], size=512, verbose=False)
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        result = self.inference([tuple(views)], self.model, self.device, batch_size=1, verbose=False)
        self.torch.cuda.synchronize()
        forward_s = time.perf_counter() - started
        preds = result['pred1'], result['pred2']
        desc0, desc1 = (p['desc'][0].detach().to(self.device) for p in preds)
        t = time.perf_counter()
        xy0, xy1 = self.nn(desc0, desc1, subsample_or_initxy1=8,
                          device=self.device, dist='dot', block_size=2**13)
        shapes = [tuple(map(int, view['true_shape'][0])) for view in views]
        keep = np.ones(len(xy0), dtype=bool)
        for xy, (h, w) in zip((xy0, xy1), shapes):
            keep &= (xy[:, 0] >= 3) & (xy[:, 0] < w-3) & (xy[:, 1] >= 3) & (xy[:, 1] < h-3)
        xy0, xy1 = xy0[keep], xy1[keep]
        conf = [p['desc_conf'][0].detach().cpu().numpy() for p in preds]
        values = np.sqrt(conf[0][xy0[:, 1], xy0[:, 0]] * conf[1][xy1[:, 1], xy1[:, 0]])
        self.torch.cuda.synchronize()
        match_s = time.perf_counter() - t
        original = [Image.open(path).size for path in (reference, goal)]
        return {'reference_points': raw_coordinates(xy0, original[0], shapes[0]),
                'query_points': raw_coordinates(xy1, original[1], shapes[1]),
                'scores': values, 'reference_shape': original[0][::-1], 'query_shape': original[1][::-1],
                'details': {'processed_shapes': shapes, 'forward_s': forward_s,
                            'reciprocal_matching_s': match_s,
                            'confidence_kind': 'sqrt(desc_conf_reference * desc_conf_goal)',
                            'dense_reference_confidence': statistics(conf[0]),
                            'dense_query_confidence': statistics(conf[1]),
                            'matching': 'official reciprocal-NN example; NOT SLAM valid_match'}}


class LightGluePairs:
    def __init__(self, repository, dependency_root, device):
        from MemNavData.lingbot_pnp_localization import LightGluePointMatcher
        self.matcher = LightGluePointMatcher(repository, dependency_root=dependency_root,
            device=device, max_keypoints=2048, reference_cache_size=0)

    def match(self, reference, goal):
        # All returned coordinates below are original pixels, not the auxiliary padded coordinates.
        m = self.matcher.match_paths(reference, goal, target_height=518, target_width=518)
        return {'reference_points': m['reference_raw_points'], 'query_points': m['query_raw_points'],
                'scores': m['scores'], 'reference_shape': m['reference_raw_hw'],
                'query_shape': m['query_raw_hw'], 'details': {'confidence_kind': 'LightGlue matching score'}}


def run(args):
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    args.out.mkdir(parents=True, exist_ok=False)
    manifest_path = args.inputs / 'pairs.json'
    manifest = json.loads(manifest_path.read_text())
    pairs = sorted(manifest['pairs'], key=lambda r: r['task'])
    if args.limit is not None:
        pairs = pairs[:args.limit]
    if len({r['task'] for r in pairs}) != len(pairs):
        raise ValueError('Duplicate raw pair')
    t = time.perf_counter()
    if args.matcher == 'mast3r':
        if args.checkpoint is None:
            raise ValueError('MASt3R checkpoint is required')
        model = Mast3rPairs(args.repository, args.checkpoint, args.device)
    else:
        model = LightGluePairs(args.repository, args.dependency_root, args.device)
    torch.cuda.synchronize()
    load_s = time.perf_counter()-t
    runtime = {'matcher': args.matcher, 'repository': str(args.repository),
        'repository_commit': subprocess.check_output(['git', '-C', str(args.repository), 'rev-parse', 'HEAD'], text=True).strip(),
        'checkpoint_sha256': sha_file(args.checkpoint) if args.checkpoint else None,
        'pair_manifest_sha256': sha_file(manifest_path), 'model_load_s': load_s,
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'precision': 'official model defaults; no explicit mixed precision override',
        'new_rollouts': 0, 'raw_proposal_changed': False, 'task_specific_training': False,
        'population': len(pairs), 'limit': args.limit}
    write_json(args.out / 'runtime.json', runtime)
    rows = []
    for index, pair in enumerate(pairs):
        ref, goal = args.inputs / pair['reference'], args.inputs / pair['goal']
        if sha_file(ref) != pair['reference_sha256'] or sha_file(goal) != pair['goal_sha256']:
            raise ValueError('Pair RGB changed')
        repeats = 3 if index == 0 else 1
        timings, values = [], []
        for repeat in range(repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            matched = model.match(ref, goal)
            torch.cuda.synchronize()
            model_s = time.perf_counter() - start
            fit_start = time.perf_counter()
            evidence = fundamental_support(matched['reference_points'], matched['query_points'],
                matched['scores'], matched['reference_shape'], matched['query_shape'])
            evidence['matches'] = evidence.pop('lightglue_matches')
            evidence['score_median'] = evidence.pop('lightglue_score_median')
            timings.append({'match_total_s': model_s, 'fundamental_s': time.perf_counter()-fit_start,
                'peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20})
            values.append(evidence)
            if repeat == 0:
                first = matched
        row = {'task': pair['task'], 'raw_anchor': pair['raw_anchor'],
               'goal_sha256': pair['goal_sha256'], 'reference_sha256': pair['reference_sha256'],
               'evidence': values[0], 'matched_confidence': statistics(first['scores']),
               'details': first['details'], 'timings': timings,
               'repeat_evidence_equal': all(v == values[0] for v in values),
               'accept': None, 'note': 'matching readout only; no navigation authority assigned'}
        np.savez_compressed(args.out / f"task_{pair['task']:03d}_matches.npz",
            reference_points=first['reference_points'], query_points=first['query_points'], scores=first['scores'])
        write_json(args.out / f"task_{pair['task']:03d}.json", row)
        rows.append(row)
        print(f"MATCHED {len(rows)}/{len(pairs)} task={pair['task']} "
              f"matches={row['evidence']['matches']} F={row['evidence']['fundamental_inliers']} "
              f"seconds={timings[0]['match_total_s']:.3f}", flush=True)
    write_json(args.out / 'summary.json', {'completed': True, 'runtime': runtime, 'rows': rows,
                                         'new_rollouts': 0, 'new_SR': None})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    unpack = sub.add_parser('unpack')
    unpack.add_argument('--archive', type=Path, required=True)
    unpack.add_argument('--out', type=Path, required=True)
    evaluate = sub.add_parser('run')
    evaluate.add_argument('--inputs', type=Path, required=True)
    evaluate.add_argument('--out', type=Path, required=True)
    evaluate.add_argument('--matcher', choices=('lightglue', 'mast3r'), required=True)
    evaluate.add_argument('--repository', type=Path, required=True)
    evaluate.add_argument('--checkpoint', type=Path)
    evaluate.add_argument('--dependency-root', type=Path)
    evaluate.add_argument('--device', default='cuda')
    evaluate.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.command == 'unpack':
        print(json.dumps(unpack_bundle(args.archive, args.out)))
    else:
        run(args)


if __name__ == '__main__':
    main()
