"""Population completeness and paired-inference failure cases."""
import json
from pathlib import Path

import pytest

from MemNavData.verify_gem_memory_navigation import MODES, clustered_difference, reduce, save, sha


def test_scene_bootstrap_keeps_same_scene_histories_together():
    pairs = []
    for scene, delta in [('a', 1), ('a', 1), ('b', -1)]:
        pairs.append(dict(candidate=dict(dataset='d', scene=scene, reached=delta),
            control=dict(dataset='d', scene=scene, reached=0)))
    result = clustered_difference(pairs, 'candidate', 'control', 'reached')
    assert result['n'] == 3 and result['scenes'] == 2
    assert result['mean'] == pytest.approx(1/3)
    assert result['scene_bootstrap_ci95'] == [-1., 1.]
    assert clustered_difference(pairs[:2], 'candidate', 'control', 'reached')['scene_bootstrap_ci95'] is None


def test_incomplete_population_cannot_be_declared_final(tmp_path):
    plan = tmp_path/'plan.json'
    save(plan, dict(modes=MODES, total_rollouts=6, cells=[dict(index=0, dataset='d')]))
    tasks = tmp_path/'tasks'
    tasks.mkdir()
    with pytest.raises(RuntimeError, match='Full planned population'):
        reduce(plan, tasks, require_complete=True)
    result = json.loads((tmp_path/'independent_reduction.json').read_text())
    assert not result['complete'] and result['verified_tasks'] == result['paired_histories'] == 0
    assert len(result['states']) == 3 and all(r['status'] == 'not_started' for r in result['states'])


def test_duplicate_task_is_not_an_extra_repetition(tmp_path):
    plan, tasks = tmp_path/'plan.json', tmp_path/'tasks'
    cell = dict(index=0, dataset='d')
    save(plan, dict(modes=MODES, total_rollouts=6, cells=[cell]))
    for name in ('original', 'another_attempt'):
        folder = tasks/name
        folder.mkdir(parents=True)
        save(folder/'manifest.json', dict(cell=cell, mode='legacy', plan_sha256=sha(plan)))
        save(folder/'summary.json', dict(completed=False))
    with pytest.raises(AssertionError, match='duplicated'):
        reduce(plan, tasks)
