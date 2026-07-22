#!/usr/bin/env python3
"""Measure whether endpoint bearing is an adequate local-route condition.

For each fixed evaluator row, this compares the actual future path tangent with
three inference-available geometric cues: final endpoint bearing, direction to
the selected historical anchor, and recent observed motion.  It also evaluates
the tempting but usually wrong idea of replaying the historical trajectory in
reverse as a revisit plan.  Ground truth is used only for this offline diagnosis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve()
WORKTREE = SCRIPT.parents[3]


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not math.isfinite(denominator) or denominator <= 1e-9:
        return float('nan')
    cosine = float(np.dot(first, second) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def load_positions(path: Path) -> np.ndarray:
    frame = pd.read_parquet(path)
    extrinsics = np.asarray(
        [np.stack(value) for value in frame['action']], dtype=np.float64
    ).reshape(-1, 4, 4)
    # Generated MemNav parquet is Z-up; x/y are the navigable ground plane.
    return extrinsics[:, :2, 3]


def summarize(rows, recent_lags):
    result = {'rows': len(rows)}
    fields = ['endpoint_error_deg', 'anchor_error_deg', 'reverse_history_error_deg']
    fields.extend(f'recent_lag_{lag}_error_deg' for lag in recent_lags)
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result[field] = {
            'mean': float(values.mean()) if len(values) else None,
            'median': float(np.median(values)) if len(values) else None,
            'p90': float(np.percentile(values, 90)) if len(values) else None,
        }
    endpoint = np.asarray(
        [row['endpoint_error_deg'] for row in rows], dtype=np.float64
    )
    for lag in recent_lags:
        recent = np.asarray(
            [row[f'recent_lag_{lag}_error_deg'] for row in rows], dtype=np.float64
        )
        valid = np.isfinite(endpoint) & np.isfinite(recent)
        result[f'recent_lag_{lag}_better_than_endpoint'] = (
            float(np.mean(recent[valid] < endpoint[valid])) if valid.any() else None
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-report', required=True)
    parser.add_argument('--root-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--future-lookahead', type=int, default=16)
    parser.add_argument('--recent-lags', type=int, nargs='+', default=(2, 4, 8, 16))
    args = parser.parse_args()
    if args.future_lookahead < 1 or any(lag < 1 for lag in args.recent_lags):
        parser.error('lookahead and recent lags must be positive')
    output = Path(args.output).resolve()
    try:
        output.relative_to(WORKTREE.resolve())
    except ValueError as error:
        raise RuntimeError(f'output must stay inside {WORKTREE}') from error

    evaluation = json.loads(Path(args.evaluation_report).read_text())
    source_rows = evaluation.get('per_sample')
    if not source_rows:
        raise ValueError('evaluation report must contain per_sample records')
    root = Path(args.root_dir)
    trajectory_cache = {}
    rows = []
    for source in source_rows:
        episode = source['sample_identity'].split(':goal=')[0]
        if episode not in trajectory_cache:
            trajectory_cache[episode] = load_positions(
                root / episode / 'data/chunk-000/episode_000000.parquet'
            )
        positions = trajectory_cache[episode]
        current = int(source['cur_step'])
        goal = min(int(source['goal_step']), len(positions) - 1)
        anchor = min(max(int(source['match_index']), 0), current)
        future = min(current + args.future_lookahead, goal)
        desired = positions[future] - positions[current]
        endpoint = positions[goal] - positions[current]
        anchor_direction = positions[anchor] - positions[current]
        reverse_index = max(0, current - args.future_lookahead)
        row = {
            'sample_identity': source['sample_identity'],
            'goal_label': source['goal_label'],
            'is_revisit': bool(source['is_revisit']),
            'is_3leg': 'mp3d_3leg/' in episode,
            'current_step': current,
            'goal_step': goal,
            'anchor_step': anchor,
            'endpoint_error_deg': angle_degrees(desired, endpoint),
            'anchor_error_deg': angle_degrees(desired, anchor_direction),
            'reverse_history_error_deg': angle_degrees(
                desired, positions[reverse_index] - positions[current]
            ),
        }
        for lag in args.recent_lags:
            previous = max(0, current - lag)
            row[f'recent_lag_{lag}_error_deg'] = angle_degrees(
                desired, positions[current] - positions[previous]
            )
        rows.append(row)

    groups = {
        'all': rows,
        'revisit': [row for row in rows if row['is_revisit']],
        '2leg': [row for row in rows if not row['is_3leg']],
        '3leg': [row for row in rows if row['is_3leg']],
        '3leg_goal_c': [
            row for row in rows if row['is_3leg'] and row['goal_label'] == 'C'
        ],
        'endpoint_turn_ge_45': [
            row for row in rows
            if math.isfinite(row['endpoint_error_deg'])
            and row['endpoint_error_deg'] >= 45.0
        ],
    }
    result = {
        'metadata': {
            'evaluation_report': os.path.abspath(args.evaluation_report),
            'dataset_fingerprint': evaluation.get('dataset_fingerprint'),
            'future_lookahead': args.future_lookahead,
            'recent_lags': args.recent_lags,
            'ground_plane': 'generated Z-up parquet x/y',
            'gt_used_only_for_offline_diagnosis': True,
        },
        'summary': {
            name: summarize(group, args.recent_lags)
            for name, group in groups.items()
        },
        'records': rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(result['summary'], indent=2, sort_keys=True))
    print(f'REPORT_SAVED {output}')


if __name__ == '__main__':
    main()
