"""Stage only two consumed online histories and their fixed query pairs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


SOURCE = Path('/scratch/yz11502/Research/Nav-axis-uturn-results/hm3d_covisibility_repaired_20260909/run_bfb8fc3d5a1bb081/construction')
HISTORIES = (2, 12)
QUERIES = ('c10_30', 'natural_novel')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def package(out):
    manifest = {'scope': 'consumed targeted local navigation mechanism pilot', 'histories': [], 'files': []}
    with tarfile.open(out, 'x:gz', compresslevel=1) as archive:
        def add(name, path, expected=None):
            data = Path(path).read_bytes()
            if expected is not None and sha(data) != expected:
                raise ValueError(f'source hash changed: {path}')
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            archive.addfile(info, io.BytesIO(data))
            manifest['files'].append({'path': name, 'source': str(path), 'sha256': sha(data), 'bytes': len(data)})

        for index in HISTORIES:
            folder = SOURCE / f'history_{index:02d}'
            construction = json.loads((folder / 'construction.json').read_text())
            online = Path(construction['online_a_episode'])
            receipt = json.loads((online / 'receipt.json').read_text())
            prefix = f'history_{index:02d}'
            for name, expected in [('receipt.json', construction['online_a_receipt_sha256']),
                                   ('online_a_trace.json', construction['online_a_trace_sha256'])]:
                add(f'{prefix}/online/{name}', online / name, expected)
            for frame, expected in enumerate(receipt['rgb_frame_hashes']):
                add(f'{prefix}/online/rgb/{frame:06d}.jpg', online / f'rgb/{frame:06d}.jpg', expected)
            asset = Path(construction['source']['asset'])
            add(f'{prefix}/{asset.name}', asset, receipt['source_asset_sha256'])
            parquet = Path(receipt['source_episode']) / 'data/chunk-000/episode_000000.parquet'
            add(f'{prefix}/camera_source.parquet', parquet, receipt['source_parquet_sha256'])
            add(f'{prefix}/construction.json', folder / 'construction.json')
            selected = []
            for query_id in QUERIES:
                query = next(q for q in construction['queries'] if q['query_id'] == query_id)
                goal = f'{prefix}/{query_id}.jpg'
                add(goal, folder / query['goal_rgb'], query['goal_rgb_sha256'])
                selected.append(dict(query_id=query_id, goal=goal))
            manifest['histories'].append(dict(history=index, scene=construction['scene'],
                prefix=prefix, asset=f'{prefix}/{asset.name}', queries=selected))
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('manifest.json'); info.size = len(data); info.mode = 0o600
        archive.addfile(info, io.BytesIO(data))
    print(json.dumps({'bundle': str(out), 'bytes': out.stat().st_size, 'sha256': sha(out.read_bytes())}))


def unpack(bundle, out):
    out.mkdir(parents=True, exist_ok=False)
    with tarfile.open(bundle) as archive:
        for member in archive:
            path = Path(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('unexpected archive member')
            destination = out / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(archive.extractfile(member).read())
    manifest = json.loads((out / 'manifest.json').read_text())
    for item in manifest['files']:
        data = (out / item['path']).read_bytes()
        if len(data) != item['bytes'] or sha(data) != item['sha256']:
            raise ValueError(f"staged data changed: {item['path']}")
    print(json.dumps({'verified': True, 'files': len(manifest['files']), 'histories': HISTORIES}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('package', 'unpack'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    args = parser.parse_args()
    package(args.out) if args.mode == 'package' else unpack(args.bundle, args.out)
