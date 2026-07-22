"""Strict paired comparison of two MemNav offline-evaluation JSON reports."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


_CONTRACT_KEYS = (
    'evaluation_type',
    'closed_loop_navigation',
    'git_commit',
    'root_dir',
    'feature_root',
    'cache_contract',
    'data_split',
    'validation_fraction',
    'split_seed',
    'sampling_mode',
    'sampling_seed',
    'random_seed',
    'dataset_fingerprint',
    'dataset_size',
    'subset_mode',
    'selection_indices',
    'eval_dataset_fingerprint',
    'evaluated_samples',
    'retrieval_anchor_mode',
    'original_anchor_margins',
    'anchor_margin_override',
    'oracle_positive',
    'full_diffusion_goal_shuffle',
    'diffusion_seed',
    'goal_shuffle_scope',
    'paired_diffusion_randomness',
)

_ROW_CONTRACT_KEYS = (
    'sample_index',
    'sample_identity',
    'cache_path',
    'cur_step',
    'goal_step',
    'goal_j',
    'goal_label',
    'has_covis',
    'leg_start',
    'memory_length',
    'remaining_path_span',
    'decision_route_angle_deg',
    'decision_curriculum_hard',
    'is_revisit',
    'num_candidates',
    'num_positive',
    'num_negative',
    'shuffled_goal_source_batch_index',
    'shuffled_goal_source_identity',
)

_METRICS = {
    'full_diffusion_action_mse': True,
    'full_diffusion_action_mse_x': True,
    'full_diffusion_action_mse_y': True,
    'full_diffusion_action_mse_theta': True,
    'full_diffusion_goal_sensitivity_mse': False,
    'full_diffusion_shuffled_goal_penalty': False,
    'action_mse': True,
    'rank_loss': True,
    'gate_bce': True,
    'gate_correct_at_0_5': False,
    'match_correct': False,
    'aux_direction_error_deg': True,
    'aux_range_code_abs_error': True,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', required=True)
    parser.add_argument('--treatment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap-resamples', type=int, default=100_000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    return parser.parse_args()


def _load_report(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'{path}: report root must be an object')
    return payload, path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_map(records, label):
    result = {}
    for record in records:
        identity = record.get('sample_identity')
        if not identity:
            raise ValueError(f'{label}: every row needs a sample_identity')
        if identity in result:
            raise ValueError(f'{label}: duplicate sample_identity {identity!r}')
        result[identity] = record
    return result


def validate_and_pair(control, treatment):
    """Fail closed on any evaluation or row-population mismatch."""
    mismatched = [
        key for key in _CONTRACT_KEYS
        if control.get(key) != treatment.get(key)
    ]
    if mismatched:
        raise ValueError(f'evaluation contract mismatch: {mismatched}')
    if not control.get('full_diffusion_goal_shuffle'):
        raise ValueError('paired comparison requires full-diffusion goal-shuffle reports')
    if not control.get('paired_diffusion_randomness'):
        raise ValueError('report does not assert paired diffusion randomness')

    control_rows = control.get('per_sample')
    treatment_rows = treatment.get('per_sample')
    if not isinstance(control_rows, list) or not isinstance(treatment_rows, list):
        raise ValueError('both reports must retain per_sample rows')
    control_map = _identity_map(control_rows, 'control')
    treatment_map = _identity_map(treatment_rows, 'treatment')
    if set(control_map) != set(treatment_map):
        only_control = sorted(set(control_map) - set(treatment_map))[:5]
        only_treatment = sorted(set(treatment_map) - set(control_map))[:5]
        raise ValueError(
            'sample population mismatch: '
            f'only_control={only_control} only_treatment={only_treatment}'
        )
    expected = control.get('evaluated_samples')
    if expected != len(control_rows):
        raise ValueError(
            f'evaluated_samples={expected} but control has {len(control_rows)} rows'
        )

    pairs = []
    for control_row in control_rows:
        identity = control_row['sample_identity']
        treatment_row = treatment_map[identity]
        row_mismatches = [
            key for key in _ROW_CONTRACT_KEYS
            if control_row.get(key) != treatment_row.get(key)
        ]
        if row_mismatches:
            raise ValueError(
                f'row contract mismatch for {identity!r}: {row_mismatches}'
            )
        for key in (
            'full_diffusion_action_mse',
            'full_diffusion_goal_sensitivity_mse',
            'full_diffusion_shuffled_goal_penalty',
        ):
            for label, row in (
                ('control', control_row), ('treatment', treatment_row)
            ):
                value = row.get(key)
                if value is None or not math.isfinite(float(value)):
                    raise ValueError(
                        f'{label} {identity!r} has invalid required metric {key}'
                    )
        pairs.append((control_row, treatment_row))
    return pairs


def _metric_value(record, key):
    if key == 'gate_correct_at_0_5':
        gate = record.get('gate')
        revisit = record.get('is_revisit')
        if gate is None or revisit is None:
            return None
        return float((float(gate) >= 0.5) == bool(revisit))
    value = record.get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _metric_summary(pairs, key, lower_is_better):
    values = []
    for control, treatment in pairs:
        control_value = _metric_value(control, key)
        treatment_value = _metric_value(treatment, key)
        if control_value is not None and treatment_value is not None:
            values.append((control_value, treatment_value))
    if not values:
        return None
    control_values = np.asarray([value[0] for value in values], dtype=np.float64)
    treatment_values = np.asarray([value[1] for value in values], dtype=np.float64)
    deltas = treatment_values - control_values
    if lower_is_better:
        improved = treatment_values < control_values
        worsened = treatment_values > control_values
    else:
        improved = treatment_values > control_values
        worsened = treatment_values < control_values
    control_mean = float(control_values.mean())
    treatment_mean = float(treatment_values.mean())
    delta = treatment_mean - control_mean
    return {
        'count': len(values),
        'lower_is_better': bool(lower_is_better),
        'control_mean': control_mean,
        'treatment_mean': treatment_mean,
        'mean_delta_treatment_minus_control': delta,
        'relative_delta_percent': (
            100.0 * delta / control_mean if control_mean != 0.0 else None
        ),
        'num_improved': int(improved.sum()),
        'num_worsened': int(worsened.sum()),
        'num_tied': int((~improved & ~worsened).sum()),
        '_deltas': deltas,
    }


def _bootstrap_mean_ci(deltas, resamples, seed):
    if resamples <= 0:
        return None
    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = []
    remaining = int(resamples)
    while remaining:
        batch = min(10_000, remaining)
        indices = rng.integers(0, deltas.size, size=(batch, deltas.size))
        means.append(deltas[indices].mean(axis=1))
        remaining -= batch
    samples = np.concatenate(means)
    return [float(value) for value in np.quantile(samples, (0.025, 0.975))]


def _groups(pairs):
    def identity(record):
        return str(record.get('sample_identity') or '')

    definitions = (
        ('all', lambda _: True),
        ('revisit', lambda row: bool(row.get('is_revisit'))),
        ('novel', lambda row: not bool(row.get('is_revisit'))),
        ('two_leg', lambda row: identity(row).startswith('mp3d_2leg/')),
        ('three_leg', lambda row: identity(row).startswith('mp3d_3leg/')),
        ('goal_A', lambda row: row.get('goal_label') == 'A'),
        ('goal_B', lambda row: row.get('goal_label') == 'B'),
        ('goal_C', lambda row: row.get('goal_label') == 'C'),
        (
            'three_leg_goal_C_revisit',
            lambda row: (
                identity(row).startswith('mp3d_3leg/')
                and row.get('goal_label') == 'C'
                and bool(row.get('is_revisit'))
            ),
        ),
        (
            'remaining_span_ge_256',
            lambda row: (
                row.get('remaining_path_span') is not None
                and int(row['remaining_path_span']) >= 256
            ),
        ),
        (
            'revisit_remaining_span_ge_256',
            lambda row: (
                bool(row.get('is_revisit'))
                and row.get('remaining_path_span') is not None
                and int(row['remaining_path_span']) >= 256
            ),
        ),
        (
            'hard_turn',
            lambda row: bool(row.get('decision_curriculum_hard')),
        ),
        (
            'easy_turn',
            lambda row: not bool(row.get('decision_curriculum_hard')),
        ),
    )
    return {
        name: [pair for pair in pairs if predicate(pair[0])]
        for name, predicate in definitions
    }


def compare_reports(control, treatment, bootstrap_resamples=100_000, bootstrap_seed=0):
    if bootstrap_resamples < 0:
        raise ValueError('bootstrap_resamples must be non-negative')
    pairs = validate_and_pair(control, treatment)
    result = {
        'paired_samples': len(pairs),
        'bootstrap_resamples': int(bootstrap_resamples),
        'bootstrap_seed': int(bootstrap_seed),
        'contract': {key: control.get(key) for key in _CONTRACT_KEYS},
        'groups': {},
    }
    for group_index, (name, selected) in enumerate(_groups(pairs).items()):
        group_result = {'count': len(selected), 'metrics': {}}
        for key, lower_is_better in _METRICS.items():
            summary = _metric_summary(selected, key, lower_is_better)
            if summary is None:
                continue
            deltas = summary.pop('_deltas')
            if key == 'full_diffusion_action_mse':
                summary['paired_bootstrap_95_ci'] = _bootstrap_mean_ci(
                    deltas,
                    bootstrap_resamples,
                    bootstrap_seed + 1009 * group_index,
                )
            group_result['metrics'][key] = summary
        result['groups'][name] = group_result
    return result


def _print_primary_table(result):
    print('group\tn\tcontrol\ttreatment\tdelta\tdelta_%\timproved')
    for name, group in result['groups'].items():
        metric = group['metrics'].get('full_diffusion_action_mse')
        if metric is None:
            continue
        print(
            f"{name}\t{metric['count']}\t{metric['control_mean']:.6f}\t"
            f"{metric['treatment_mean']:.6f}\t"
            f"{metric['mean_delta_treatment_minus_control']:+.6f}\t"
            f"{metric['relative_delta_percent']:+.2f}\t"
            f"{metric['num_improved']}/{metric['count']}"
        )


def main():
    args = parse_args()
    control, control_path = _load_report(args.control)
    treatment, treatment_path = _load_report(args.treatment)
    result = compare_reports(
        control,
        treatment,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result.update({
        'control_report': str(control_path.resolve()),
        'control_report_sha256': _sha256(control_path),
        'control_checkpoint': control.get('checkpoint'),
        'treatment_report': str(treatment_path.resolve()),
        'treatment_report_sha256': _sha256(treatment_path),
        'treatment_checkpoint': treatment.get('checkpoint'),
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    _print_primary_table(result)
    print(f'[compare] wrote {output}')


if __name__ == '__main__':
    main()
