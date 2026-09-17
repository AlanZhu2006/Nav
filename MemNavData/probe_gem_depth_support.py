"""Build goal-independent geometric support from every original history RGB.

Extracts the unchanged SuperPoint detector once per observation, without
LightGlue matching or loading any goal/ground truth. Existing W64 FP16 and
W16 FP32 first-write depth records are compressed independently. Every lifted
detector point is checked against its full original dense record.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'MemNavData')]
from gem_depth_support import FrameSupportArchive
from lingbot_pnp_localization import LightGluePointMatcher, map_raw_points_to_lingbot_pad, lift_reference_keypoints


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def prepare(root,out):
    budget = root/'native_budget16_001'
    manifest = load(budget/'manifest.json')
    native = root/'native_depth_001'
    parent = Path(manifest['parent'])
    assert load(budget/'completion.json')['completed']
    assert load(parent/'completion.json')['completed']
    assert manifest['cases']==load(native/'manifest.json')['cases']
    glue = ROOT/'.diagnostics/dependencies/LightGlue'
    weights = Path(torch.hub.get_dir())/'checkpoints/superpoint_v1.pth'
    assert weights.is_file()
    files = [Path(__file__),ROOT/'MemNavData/gem_depth_support.py',
        ROOT/'MemNavData/lingbot_pnp_localization.py',ROOT/'MemNavData/lingbot_colored_registration.py',
        *sorted((glue/'lightglue').glob('*.py')),weights]
    archives = {}
    for case in manifest['cases']:
        for directory in (native/case['id'],budget/case['id']):
            receipt = load(directory/'completion.json')
            assert receipt['completed']
            archives[str(directory/'completion.json')] = sha(directory/'completion.json')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'manifest.json',dict(schema='gem_frozen_detector_depth_support_v1',
        native=str(native),budget=str(budget),parent=str(parent),
        input_manifest=str(budget/'manifest.json'),input_manifest_sha256=sha(budget/'manifest.json'),
        source_sha256={str(p):sha(p) for p in files},archive_receipt_sha256=archives,
        cases=[dict(id=c['id'],frames=c['frame_count']) for c in manifest['cases']],
        arms=['native64_fp16','native16_fp32'],frame_count=sum(c['frame_count'] for c in manifest['cases']),
        detector='Unchanged pinned SuperPoint,2048maximum points,native RGB preprocessing',
        support='Union of every detector point four bilinear neighbours plus actual finite global minimum-confidence pixel',
        depth_filter_quantile=0.,goal_images_loaded=False,ground_truth_loaded=False,
        descriptor_storage='Detector descriptors are temporary and discarded; original downstream SP/LG is retained',
        scope='Full observed geometry archival and all-detector-point lift parity; extra SP extraction cost measured, not yet production navigation/whole-writer timing',
        created_at=time.time()))


def run(out):
    manifest = load(out/'manifest.json')
    assert all(sha(p)==h for p,h in manifest['source_sha256'].items())
    assert all(sha(p)==h for p,h in manifest['archive_receipt_sha256'].items())
    assert sha(manifest['input_manifest'])==manifest['input_manifest_sha256']
    cases = load(manifest['input_manifest'])['cases']
    native,budget,parent = map(Path,(manifest['native'],manifest['budget'],manifest['parent']))
    torch.set_num_threads(4);torch.manual_seed(0)
    save(out/'process.json',dict(pid=os.getpid(),manifest_sha256=sha(out/'manifest.json'),started_at=time.time()))
    matcher = LightGluePointMatcher(ROOT/'.diagnostics/dependencies/LightGlue',
        dependency_root=ROOT/'.diagnostics/dependencies/python',device='cuda:0',
        max_keypoints=2048,reference_cache_size=0)
    rows,case_receipts = [],{}
    started = time.monotonic()
    try:
        for case in cases:
            identity = case['id']
            with np.load(parent/identity/'baseline.npz') as z:native_pose=z['pose_enc'].astype(np.float32)
            with np.load(budget/identity/'geometry.npz') as z:budget_pose=z['pose_enc'].astype(np.float32)
            poses = dict(native64_fp16=native_pose,native16_fp32=budget_pose)
            archives = {a:FrameSupportArchive(out/identity/a/'depths') for a in poses}
            source_dirs = dict(native64_fp16=native/identity,native16_fp32=budget/identity)
            source_receipts = {a:load(d/'completion.json') for a,d in source_dirs.items()}
            case_rows = []
            for frame,path in enumerate(case['rgb_paths']):
                assert sha(path)==case['rgb_sha256'][frame]
                torch.cuda.synchronize();tick=time.perf_counter()
                height,width,features = matcher._reference_features(Path(path))
                points = matcher.rbd(features)['keypoints'].detach().cpu().numpy()
                torch.cuda.synchronize();detector_ms=1000*(time.perf_counter()-tick)
                mapped = map_raw_points_to_lingbot_pad(points,raw_height=height,raw_width=width,
                    target_height=518,target_width=518,patch_size=14)
                row = dict(case=identity,frame=frame,detector_points=len(points),detector_ms=detector_ms,arms={})
                for arm,archive in archives.items():
                    relative = f'depths/{frame:06d}.npz'
                    path = source_dirs[arm]/relative
                    assert sha(path)==source_receipts[arm]['files_sha256'][relative]
                    with np.load(path) as z:
                        assert int(z['frame'])==frame
                        depth,confidence=z['depth'].astype(np.float32),z['confidence'].astype(np.float32)
                        scale=float(z['world_scale'])
                    tick=time.perf_counter()
                    archive.append(frame,depth,confidence,scale,mapped,rgb_sha256=case['rgb_sha256'][frame])
                    archive_ms=1000*(time.perf_counter()-tick)
                    archive.validate_matches(frame,mapped)
                    restored_depth,restored_confidence=archive[frame]
                    full,valid = lift_reference_keypoints(mapped,depth*np.float32(scale),confidence,poses[arm][frame],confidence_quantile=0.)
                    restored,restored_valid = lift_reference_keypoints(mapped,restored_depth,restored_confidence,poses[arm][frame],confidence_quantile=0.)
                    np.testing.assert_array_equal(restored_valid,valid)
                    np.testing.assert_array_equal(restored,full)
                    item=archive.records[frame]
                    row['arms'][arm]=dict(dense_archive_bytes=path.stat().st_size,
                        support_archive_bytes=item['bytes'],support_pixels=item['support_pixels'],
                        valid_points=int(valid.sum()),lift_bitwise_equal=True,archive_write_ms=archive_ms)
                del features
                rows.append(row);case_rows.append(row)
                if frame%128==0 or frame==case['frame_count']-1:
                    print(json.dumps(dict(case=identity,frame=frame,detector_points=len(points),
                        support_pixels={a:r['support_pixels'] for a,r in row['arms'].items()})),flush=True)
            receipt = dict(completed=True,frames=case['frame_count'],all_detector_lifts_bitwise_equal=True,
                archives={a:dict(records=v.records,array_bytes=v.array_bytes,disk_bytes=v.disk_bytes) for a,v in archives.items()},
                rows=case_rows,source_receipt_sha256={str(source_dirs[a]/'completion.json'):sha(source_dirs[a]/'completion.json') for a in archives})
            p=out/identity/'completion.json';save(p,receipt);case_receipts[str(p)]=sha(p)
        assert len(rows)==manifest['frame_count']==6028
        assert all(sha(p)==h for p,h in manifest['source_sha256'].items())
        save(out/'completion.json',dict(completed=True,cases=len(cases),frames=len(rows),
            all_detector_lifts_bitwise_equal=True,case_receipt_sha256=case_receipts,
            manifest_sha256=sha(out/'manifest.json'),seconds=time.monotonic()-started,
            detector_seconds=sum(r['detector_ms'] for r in rows)/1000,
            storage={a:dict(dense_bytes=sum(r['arms'][a]['dense_archive_bytes'] for r in rows),
                support_bytes=sum(r['arms'][a]['support_archive_bytes'] for r in rows)) for a in manifest['arms']},
            scope=manifest['scope']))
        save(out/'process_exit.json',dict(exit_code=0,completed_at=time.time()))
    except BaseException as error:
        save(out/'failure.json',dict(type=type(error).__name__,error=str(error),time=time.time()))
        save(out/'process_exit.json',dict(exit_code=1,completed_at=time.time()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare:prepare(args.root.resolve(),args.out.resolve())
    else:run(args.out.resolve())
