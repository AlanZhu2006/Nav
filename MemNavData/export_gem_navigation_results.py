"""Export compact rows and audit metadata after the full navigation reduction."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    reduction=load(root/'independent_reduction.json')
    assert reduction['complete'] and reduction['verified_tasks']==reduction['expected_tasks']
    rows=[]
    members=[root/name for name in ('plan.json','submission.json','audit_submission.json','independent_reduction.json')]
    for task in sorted((root/'tasks').iterdir(),key=lambda p:int(p.name)):
        verified=load(task/'independent_verification.json')
        assert verified['verified']
        assert verified['manifest_sha256']==sha(task/'manifest.json')
        assert verified['summary_sha256']==sha(task/'summary.json')
        for r in verified['records']:
            rows.append({**{k:r[k] for k in ('index','dataset','scene','episode','mode','role','candidate_design_scene',
                'steps','reached','actual_path_m','spl','geodesic_m','wall_seconds','takeover_plans',
                'first_query_rgb_sha256','goal_sha256','history_trace_sha256')},
                'gpu_peak_allocated_gib':r['memory_resources']['gpu_peak_allocated_bytes']/2**30,
                'gpu_peak_reserved_gib':r['memory_resources']['gpu_peak_reserved_bytes']/2**30,
                'frames':r['memory_resources']['memory']['frames'],
                'depth_archive_gib':r['memory_resources']['memory'].get('depth_archive_bytes',0)/2**30,
                'verification_sha256':sha(task/'independent_verification.json')})
        members.extend(task/name for name in ('manifest.json','summary.json','independent_verification.json'))
    assert len(rows)==reduction['expected_rollouts']
    assert len({(r['index'],r['mode'],r['role']) for r in rows})==len(rows)
    (root/'navigation_rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (root/'navigation_rows.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    members.extend(root/name for name in ('navigation_rows.json','navigation_rows.csv'))
    with tarfile.open(root/'audit_metadata.tar.gz','w:gz') as archive:
        for member in members:
            archive.add(member,arcname=str(member.relative_to(root)),recursive=False)
    receipt=dict(rows=len(rows),source_reduction_sha256=sha(root/'independent_reduction.json'),
        artifact_sha256={str(p.relative_to(root)):sha(p) for p in members},
        archive_sha256=sha(root/'audit_metadata.tar.gz'),exporter_sha256=sha(__file__),
        raw_data_location=str(root/'tasks'),
        archive_scope='Metadata and independently verified rows; full trajectories, RGB/depth and raster evidence remain in raw task directories')
    (root/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),archive_bytes=(root/'audit_metadata.tar.gz').stat().st_size)))


if __name__=='__main__':main()
