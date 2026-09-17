"""Plot every preselected thousand-frame history, with matched real queries."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def plot(root):
    native = root / 'native_budget16_readout_002'
    original = root / 'readout_001'
    inputs = [native / 'evaluation.json', native / 'completion.json',
        native / 'inference.json', original / 'evaluation.json',
        original / 'independent_verification.json', original / 'manifest.json']
    receipt = load(native / 'completion.json')
    assert receipt['completed'] and receipt['native64_readout_parity']
    assert all(sha(native / p) == h for p, h in receipt['files_sha256'].items())
    audit = load(original / 'independent_verification.json')
    assert audit['verified'] and all(sha(p) == h for p, h in audit['source_sha256'].items())
    new, old = load(native / 'evaluation.json')['rows'], load(original / 'evaluation.json')['rows']
    assert len(new) == len(old) == 151
    key = lambda r: (r['case'], r['query'], r['prefix'])
    previous = {key(r): r for r in old}
    assert set(previous) == {key(r) for r in new}
    for row in new:
        a, b = row['bearing_deg']['native64_fp16'], previous[key(row)]['errors']['native']
        assert (a is None) == (b is None)
        if a is not None:
            assert abs(a - b) < 1e-6
    cases = [c for c in load(original / 'manifest.json')['cases'] if c['frame_count'] >= 1000]
    assert len(cases) == 3
    plt.rcParams.update({'font.size': 11, 'axes.spines.right': False, 'axes.spines.top': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.35), sharey=True)
    outputs = []
    for ax, case in zip(axes, cases):
        selected = sorted([r for r in new if r['case'] == case['id'] and r['role'] == 'revisit'],
            key=lambda r:r['prefix'])
        assert selected and selected[-1]['complete_history']
        prefixes = [r['prefix'] for r in selected]
        values = {
            'Native W64 + archive': [r['bearing_deg']['native64_fp16'] for r in selected],
            'Native W16 + archive': [r['bearing_deg']['native16_fp16'] for r in selected],
            'Connected + archive': [previous[key(r)]['errors']['reciprocal'] for r in selected]}
        for (label, values_arm), color, marker in zip(values.items(),
                ['#b27424', '#6f4e99', '#167296'], ['o', 's', '^']):
            # Missing/rejected directions are gaps, never zero error.
            y = [np.nan if v is None else v for v in values_arm]
            ax.plot(prefixes, y, color=color, marker=marker, markersize=4, linewidth=1.7, label=label)
        ax.set_title(f"{case['id'].split('_')[0]} ({case['frame_count']} RGBs)")
        ax.set_xlabel('Observed RGB frames')
        ax.set_ylim(-3, 184)
        ax.set_yticks([0, 45, 90, 135, 180])
        ticks = [x for x in [128, 512, 1024, 1536] if x < case['frame_count'] * .87]
        ax.set_xticks(ticks + [case['frame_count']])
        ax.grid(axis='y', alpha=.2)
        outputs.append(dict(case=case['id'], prefixes=prefixes, bearing_deg=values))
    axes[0].set_ylabel('Goal bearing error (degrees)')
    fig.suptitle('Fixed final Revisit goals over all three long development histories', y=.99)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=3, loc='lower center',
        bbox_to_anchor=(.5, .065), frameon=False)
    fig.text(.5, .025,
        'Shared SP/LG matches; FP16 archives for all shown arms. Correlated prefixes, not navigation success trials.',
        ha='center', fontsize=9.7)
    fig.subplots_adjust(top=.82, bottom=.27, left=.07, right=.985, wspace=.14)
    out = native / 'figures'
    out.mkdir(exist_ok=True)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(out / ('long_history_readout.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)
    (out / 'source.json').write_text(json.dumps(dict(rows=outputs,
        input_sha256={str(p):sha(p) for p in inputs}, script_sha256=sha(__file__),
        selection='Every history >=1000RGB in the original fixed nine-history manifest; every Revisit prefix; no error-based case selection'),
        indent=2, allow_nan=False) + '\n')
    print(str(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    plot(parser.parse_args().root.resolve())
