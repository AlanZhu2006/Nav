from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from MemNavData.table2_balanced_sampling import BINS
from MemNavData.table2_continuous_population import (
    SCHEMA, SEQUENCES, allocate_sequences, first_batch_order, task_at,
)
from MemNavData.table2_common_goals import choose_common
from MemNavData.summarize_table2_continuous import arm_counts
from MemNavData.table2_mixed_local import sha, dump


def population_records():
    return [dict(source_id=f"scene{i%18}/episode{i:03d}", scene=f"scene{i%18}",
                 source_index=i, a_distance_band=BINS[0 if i < 33 else 1 if i < 65 else 2])
            for i in range(97)]


def test_all_sources_retained_and_sequence_distance_balance():
    records = population_records()
    original = deepcopy(records)
    selected = first_batch_order(allocate_sequences(records))
    assert records == original
    assert len(selected) == len({r['source_id'] for r in selected}) == 97
    assert Counter(r['sequence'] for r in selected) == dict(NNN=25, NNR=24, NRN=24, NRR=24)
    assert Counter((r['sequence'],r['a_distance_band']) for r in selected[:12]) == {
        (s,b): 1 for s in SEQUENCES for b in BINS}
    assert len({r['scene'] for r in selected[:12]}) == 12
    assert [r['task_index'] for r in selected] == list(range(97))
    assert sum(r['first_batch'] for r in selected) == 12
    assert selected == first_batch_order(allocate_sequences(list(reversed(records))))


def test_success_fields_cannot_change_assignment():
    records = population_records()
    plain = first_batch_order(allocate_sequences(records))
    for i,r in enumerate(records):
        r['previous_reached'] = bool(i % 2)
        r['previous_spl'] = 1. if i % 3 else 0.
    augmented = first_batch_order(allocate_sequences(records))
    assert [(r['source_id'],r['sequence']) for r in plain] == [(r['source_id'],r['sequence']) for r in augmented]


def test_successor_preference_changes_no_eligibility():
    candidates = [dict(distance_band=b, goal_rgb_sha256=str(i)) for i,b in enumerate(BINS)]
    counts = dict(zip(('6_to_9','4_to_6','2_to_4'),range(3)))
    assert choose_common({'candidates': candidates}, 'B', counts)['distance_band'] == '6_to_9'
    assert choose_common({'candidates': candidates[:2]}, 'B', counts)['distance_band'] == '4_to_6'
    assert choose_common({'candidates': []}, 'B', counts) is None


def test_frozen_task_rejects_image_changes_and_negative_index(tmp_path):
    goal = tmp_path/'goal.jpg'
    goal.write_bytes(b'sealed-image')
    p=tmp_path/'population.json'
    dump(p,dict(schema=SCHEMA,formal_population=True,tasks=[dict(task_index=0,sequence='NNR',
        a_query=dict(goal_rgb='goal.jpg',goal_rgb_sha256=sha(goal)))]))
    assert task_at(p,0)['a_query']['goal_rgb'] == str(goal)
    with pytest.raises(ValueError): task_at(p,-1)
    goal.write_bytes(b'changed')
    with pytest.raises(ValueError): task_at(p,0)


def test_construction_censoring_is_not_navigation_failure():
    one = dict(stage='A',measurement=dict(reached=True))
    assert arm_counts(dict(status='task_construction_blocked',measurements=[one]))['cumulative'] == [1,None,None]
    failed = dict(stage='B',measurement=dict(reached=False))
    assert arm_counts(dict(status='lifecycle_finished',measurements=[one,failed]))['cumulative'] == [1,0,0]
    all_three = [dict(stage=s,measurement=dict(reached=True)) for s in 'ABC']
    assert arm_counts(dict(status='lifecycle_finished',measurements=all_three))['cumulative'] == [1,1,1]


def test_hpc_array_is_single_gpu_one_hour_and_does_not_release_remaining():
    root=Path(__file__).parent
    shell=(root/'slurm_table2_continuous_eval.sbatch').read_text()
    assert '#SBATCH --time=01:00:00' in shell
    assert '#SBATCH --partition=a100_tandon' in shell
    assert '#SBATCH --gres=gpu:1' in shell
    assert 'SLURM_ARRAY_TASK_ID' in shell and 'sbatch ' not in shell
    runner=(root/'run_table2_continuous_hpc.sh').read_text()
    assert 'claim_slurm_tcp_port_block table2_continuous 4' in runner
    assert '--prepare-only' in runner and 'covisibility_task_archive.py' in runner
    assert 'table2_full_reduce' not in runner
