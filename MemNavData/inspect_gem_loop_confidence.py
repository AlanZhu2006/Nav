"""Read saved confidence at five previously recovered inlier sets; CPU only.

Post-hoc descriptive analysis, with no new matching, PnP, model or policy run.
Frame-relative confidence ranks exclude artificial padding and invalid depth.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from MemNavData.lingbot_depth_raster import lingbot_pad_raster


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sample(array, points):
    x, y = points.T
    assert np.all((x >= 0) & (x <= array.shape[1]-1))
    assert np.all((y >= 0) & (y <= array.shape[0]-1))
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, array.shape[1]-1), np.minimum(y0+1, array.shape[0]-1)
    wx, wy = x-x0, y-y0
    return ((1-wx)*(1-wy)*array[y0, x0] + wx*(1-wy)*array[y0, x1]
            + (1-wx)*wy*array[y1, x0] + wx*wy*array[y1, x1])


def main():
    base = ROOT/'.diagnostics/gem_spatial_revisit_20260915'
    out = base/'confidence_audit_001'
    out.mkdir(exist_ok=False)
    parent = base/'failure_forensics_001'
    previous = load(parent/'completion.json')
    for name, digest in previous['files_sha256'].items():
        assert sha(name) == digest, name
    plan = load(base/'attempt_003/plan.json')
    cases = {c['id']: c for c in plan['cases']}
    details = load(parent/'analysis.json')['details']
    save(out/'plan.json', dict(posthoc=True, selected_from=str(parent/'analysis.json'),
        selected_sha256=sha(parent/'analysis.json'), script_sha256=sha(__file__),
        selection='Exactly the five already recovered inlier sets; no new selection by confidence',
        model_forward=False, matching=False, pnp=False, rendering=False, navigation=False,
        primary_scope='Descriptive saved depth-confidence values, not calibration or a selection-policy experiment',
        normalization='Bilinear samples; percentile relative to finite positive-depth pixels in actual image crop',
        median_comparison='Descriptive count above image median; no threshold applied to solver'))
    rows, sources, visuals = [], {}, {}
    for r in details:
        case = cases[r['case']]
        stem = f"{r['case']}_{r['current']}_{r['reference']}_{r['arm']}"
        infile = parent/f'{stem}.npz'
        sources[str(infile)] = sha(infile)
        with np.load(infile) as d:
            points = {k: d[k].copy() for k in ('reference_points', 'query_points')}
            original_ids = d['original_match_indices'].copy()
        width, height = case['raw_wh']
        raster = lingbot_pad_raster((height, width))
        row = dict(case=r['case'], current=r['current'], reference=r['reference'], arm=r['arm'],
            inliers=r['inliers'], bearing_deg=r['evaluation']['observed']['bearing_deg'],
            gt_epipolar_under1_5_px=r['gt_epipolar_under1_5_px'],
            gt_triangulation_positive_in_both=r['gt_triangulation_positive_in_both'])
        samples = {}
        for side, frame, key in [('reference', r['reference'], 'reference_points'),
                                  ('current', r['current'], 'query_points')]:
            geometry = Path(case['source_directory'])/'buffer/ep_0001/geometry'/f'{frame:06d}.npz'
            sources[str(geometry)] = sha(geometry)
            if side == 'reference':
                assert sources[str(geometry)] == r['original_record']['reference_geometry_sha256']
            with np.load(geometry) as d:
                assert int(d['frame']) == frame
                confidence = d['confidence'].astype(float)
                depth = d['depth'].copy()
            assert confidence.shape == (518, 518) and depth.shape == confidence.shape
            valid = np.zeros(confidence.shape, bool)
            valid[raster.crop] = True
            valid &= np.isfinite(confidence) & np.isfinite(depth) & (depth > 1e-6)
            uv = points[key]
            assert np.all((uv[:, 0] >= raster.left) & (uv[:, 0] < raster.left+raster.resized_width))
            assert np.all((uv[:, 1] >= raster.top) & (uv[:, 1] < raster.top+raster.resized_height))
            population = np.sort(confidence[valid])
            values = sample(confidence, uv)
            rank = .5*(np.searchsorted(population, values, side='left')
                         + np.searchsorted(population, values, side='right'))/len(population)
            row[side] = dict(confidence_q10_median_q90=np.percentile(values, [10, 50, 90]).tolist(),
                visible_frame_median=float(np.median(population)), valid_image_pixels=len(population),
                inlier_median_visible_frame_percentile=float(np.median(rank)),
                above_visible_median=int((values > np.median(population)).sum()))
            samples[side+'_confidence'] = values
            samples[side+'_frame_percentile'] = rank
            if side == 'reference' and r['arm'] == 'superpoint_lightglue' and frame in (22, 204):
                rgb = Path(case['rgb_paths'][frame])
                assert sha(rgb) == case['rgb_sha256'][frame]
                sources[str(rgb)] = sha(rgb)
                visuals[frame] = (rgb, confidence, uv, population, raster)
        matchfile = base/'attempt_003'/r['original_record']['match_file']
        assert sha(matchfile) == r['original_record']['match_sha256']
        sources[str(matchfile)] = sha(matchfile)
        with np.load(matchfile) as d:
            scores = d['scores'][original_ids]
        row['matcher_score_median'] = float(np.median(scores))
        samples['original_match_scores'] = scores
        np.savez_compressed(out/f'{stem}.npz', **samples)
        rows.append(row)
    save(out/'analysis.json', dict(complete=True, posthoc=True, rows=rows,
        source_sha256=sources, previous_sealed_files_verified=len(previous['files_sha256']),
        limitations=['Five selected cases from two development scenes, not population-level confidence calibration',
                     'Different arms have distinct pixel support; their confidence medians cannot rank pose accuracy',
                     'Low geometric confidence does not establish incorrect image correspondence',
                     'No filtering, new pose estimate, threshold sweep, risk-coverage curve or runtime integration']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.7), layout='constrained')
    for i, frame in enumerate((22, 204)):
        rgb, confidence, uv, population, raster = visuals[frame]
        image = np.asarray(Image.open(rgb).convert('RGB'))
        raw = (uv-[raster.left, raster.top])*[image.shape[1]/raster.resized_width,
                                                    image.shape[0]/raster.resized_height]
        axes[i, 0].imshow(image)
        axes[i, 0].scatter(raw[:, 0], raw[:, 1], s=9, facecolors='none', edgecolors='#ff3864', linewidths=.6)
        crop = confidence[raster.crop]
        ranks = .5*(np.searchsorted(population, crop, side='left')
                     + np.searchsorted(population, crop, side='right'))/len(population)
        heat = axes[i, 1].imshow(ranks, vmin=0, vmax=1, cmap='viridis')
        axes[i, 1].scatter(uv[:, 0]-raster.left, uv[:, 1]-raster.top,
                           s=9, facecolors='none', edgecolors='#ff3864', linewidths=.6)
        title = 'Task 1, history 22: GT-consistent SP+LG correspondences' if frame == 22 else 'Task 7, history 204: GT-inconsistent SP+LG correspondences'
        axes[i, 0].set_title(title, fontsize=10)
        axes[i, 1].set_title('LingBot depth confidence: within-image percentile', fontsize=10)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(heat, ax=axes[:, 1], shrink=.8, label='Confidence percentile (not probability)')
    fig.suptitle('The original PnP inliers can have low or high depth confidence', fontsize=13)
    fig.savefig(out/'confidence_examples.png', dpi=180)
    plt.close(fig)
    print(json.dumps(dict(report=str(out/'analysis.json'), cases=len(rows), input_files=len(sources))))


if __name__ == '__main__':
    main()
