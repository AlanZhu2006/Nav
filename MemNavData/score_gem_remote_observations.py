"""Score sealed distant observation constraints; truth is evaluation-only."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MemNavData')]
from score_gem_reactivation_probe import errors, truth, sha
from score_gem_connected_probe import matrices


def load(path):
    return json.loads(Path(path).read_text())


def describe(values):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,
        p95=float(np.quantile(a, .95)) if len(a) else None,
        max=float(a.max()) if len(a) else None)


def summarize(rows):
    result = {}
    for arm in ('native64', 'connected'):
        accepted = [r['arms'][arm] for r in rows if r['arms'][arm]['accepted']]
        metrics = {}
        for metric in ('rotation_deg', 'translation_direction_deg', 'bearing_deg'):
            common = [r for r in accepted if r['prior'][metric] is not None
                and r['observed'][metric] is not None]
            metrics[metric] = dict(
                prior=describe([r['prior'][metric] for r in common]),
                observed=describe([r['observed'][metric] for r in common]),
                difference=describe([r['observed'][metric] - r['prior'][metric] for r in common]),
                improved=sum(r['observed'][metric] < r['prior'][metric] for r in common),
                worsened=sum(r['observed'][metric] > r['prior'][metric] for r in common))
        result[arm] = dict(queries=len(rows), eligible=sum(r['eligible'] for r in rows),
            accepted=len(accepted), metrics=metrics,
            accepted_rotation_over_15deg=sum(r['observed']['rotation_deg'] > 15 for r in accepted))
    result['joint_accepted_same_reference'] = sum(
        all(r['arms'][a]['accepted'] for a in ('native64', 'connected'))
        and r['arms']['native64']['reference'] == r['arms']['connected']['reference'] for r in rows)
    return result


def run(out):
    manifest, receipt = load(out / 'manifest.json'), load(out / 'completion.json')
    assert receipt['completed'] and receipt['inference_sha256'] == sha(out / 'inference.json')
    assert receipt['manifest_sha256'] == sha(out / 'manifest.json')
    assert all(sha(p) == h for p, h in receipt['geometry_receipt_sha256'].items())
    assert all(sha(p) == h for p, h in manifest['geometry_sha256'].items())
    assert all(sha(out / p) == h for p, h in receipt['match_sha256'].items())
    assert sha(__file__) == manifest['source_sha256'][str(Path(__file__).resolve())]
    parent, root = Path(manifest['parent']), Path(manifest['source_root'])
    evaluations = {c['id']:c for c in load(parent / 'evaluation_inputs.json')['cases']}
    geometries, traces = {}, {}
    trace_hashes = {}
    for case in manifest['cases']:
        identity = case['id']
        with np.load(parent / identity / 'baseline.npz') as data:
            native = matrices(data['pose_enc'])
        with np.load(root / 'reciprocal_001' / identity / 'geometry.npz') as data:
            connected = matrices(data['pose9'])
        geometries[identity] = dict(native64=native, connected=connected)
        ev = evaluations[identity]
        assert sha(ev['trace']) == ev['trace_sha256']
        trace_hashes[ev['trace']] = ev['trace_sha256']
        traces[identity] = {p['step']:truth(p) for p in load(ev['trace'])['poses']}
    inference = load(out / 'inference.json')['rows']
    assert len(inference) == len(manifest['queries']) == receipt['queries']
    rows = []
    for record, query in zip(inference, manifest['queries']):
        assert all(record[k] == v for k, v in query.items())
        identity, current = record['case'], record['current']
        row = dict(case=identity, current=current, history_endpoint=record['history_endpoint'],
            eligible=bool(record['candidates']), candidates=len(record['candidates']), arms={})
        for arm, result in record['arms'].items():
            item = dict(accepted=result['accepted'], reason=result['reason'], reference=None,
                prior=None, observed=None, pnp_inliers=result.get('pnp', {}).get('inliers'),
                pnp_rmse_px=result.get('pnp', {}).get('reprojection_rmse_px'))
            if result['accepted']:
                h = int(result['selected_anchor'])
                assert h in {c['anchor'] for c in record['candidates']}
                assert current - h >= manifest['exclusion_raw_frames']
                poses = geometries[identity][arm]
                expected = np.linalg.solve(traces[identity][h], traces[identity][current])
                observed_pose = matrices(np.asarray(result['pnp']['pose9'])[None])[0]
                item.update(reference=h, age_frames=current - h,
                    prior=errors(np.linalg.solve(poses[h], poses[current]), expected),
                    observed=errors(np.linalg.solve(poses[h], observed_pose), expected))
                assert result['pnp']['status'] == 'ok' and result['pnp']['inliers'] >= 16
                assert result['pnp']['reprojection_rmse_px'] <= 2.
                assert min(result['pnp']['reference_inlier_coverage'], result['pnp']['query_inlier_coverage']) >= .05
                if np.linalg.norm(expected[:3, 3]) < .5:
                    item['prior']['translation_direction_deg'] = None
                    item['observed']['translation_direction_deg'] = None
            row['arms'][arm] = item
        rows.append(row)
    report = dict(complete=True, cases=len(manifest['cases']), queries=len(rows), rows=rows,
        summary=summarize(rows),
        by_case={c['id']:summarize([r for r in rows if r['case'] == c['id']]) for c in manifest['cases']},
        endpoints=summarize([r for r in rows if r['history_endpoint']]),
        trace_sha256=trace_hashes, inference_sha256=sha(out / 'inference.json'),
        scorer_sha256=sha(__file__),
        scope='Correlated development observations; accepted-relative error comparisons, rejection coverage and rotation tails, not navigation SR or a deployed optimizer; error thresholds are evaluator-only')
    path = out / 'evaluation.json'
    with path.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(cases=report['cases'], queries=len(rows), summary=report['summary']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    run(parser.parse_args().root.resolve())
