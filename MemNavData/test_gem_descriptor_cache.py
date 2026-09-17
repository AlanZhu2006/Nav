"""Descriptor data movement must preserve the existing public retrieval result."""
import gc
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch

from NavDP.baselines.memnav.gem.archive import FrameDepthArchive
from NavDP.baselines.memnav.gem.descriptor_cache import DescriptorReadCache
from NavDP.baselines.memnav.gem.episodic import EpisodicGEM
from NavDP.baselines.memnav.policy_agent import MemNavAgent


class AppendOnlyRows(list):
    """Reject rereading old descriptors, not just extra torch.stack calls."""
    readable_from = 0

    def __iter__(self):
        if self.readable_from:
            raise AssertionError('previously uploaded history was iterated')
        return super().__iter__()

    def __getitem__(self, key):
        start = key.start or 0 if isinstance(key, slice) else key
        if start < self.readable_from:
            raise AssertionError('previously uploaded history was read')
        return super().__getitem__(key)


def test_incremental_reads_keep_bits_and_do_not_read_old_rows():
    values = torch.randn(35, 1024, generator=torch.Generator().manual_seed(31))
    # Include edge FP32 encodings, including nonfinite payloads, in data movement.
    values[0, :6] = torch.tensor([0, -2147483648, 2139095040, -8388608,
        2143289345, 1], dtype=torch.int32).view(torch.float32)
    rows = AppendOnlyRows()
    cache = DescriptorReadCache()
    for count in (1, 8, 9, 16, 17, 35):
        rows.extend(values[len(rows):].unbind()[:count-len(rows)])
        current = cache.read(rows, 'cpu')
        assert current.is_contiguous() and current.shape == (1, count, 1024)
        assert torch.equal(current[0].view(torch.int32), values[:count].view(torch.int32))
        rows.readable_from = count
        repeated = cache.read(rows, 'cpu')
        assert repeated.data_ptr() == current.data_ptr()
        stats = cache.statistics()
        assert stats['uploaded_bytes'] == count * 1024 * 4
        assert count <= stats['capacity_frames'] < 2 * count
        assert stats['effective_bytes'] == count * 1024 * 4


def test_cache_releases_allocations_and_history_on_reset():
    cache = DescriptorReadCache()
    rows = AppendOnlyRows([torch.ones(1024)])
    cache.read(rows, 'cpu')
    source_ref, storage_ref = weakref.ref(rows), weakref.ref(cache._storage)
    del rows
    cache.reset()
    gc.collect()
    assert source_ref() is None and storage_ref() is None
    assert cache.statistics()['allocated_bytes'] == 0
    fresh = [torch.full((1024,), 3.)]
    assert torch.equal(cache.read(fresh, 'cpu')[0, 0], fresh[0])


@pytest.mark.parametrize('change', ['source', 'shorten', 'device', 'width', 'dtype'])
def test_invalid_history_or_device_cannot_silently_reuse_cached_values(change):
    cache = DescriptorReadCache()
    rows = [torch.ones(1024), torch.zeros(1024)]
    cache.read(rows, 'cpu')
    device = 'cpu'
    if change == 'source':
        rows = list(rows)
    elif change == 'shorten':
        rows.pop()
    elif change == 'device':
        device = 'meta'
    elif change == 'width':
        rows.append(torch.ones(8))
    else:
        rows.append(torch.ones(1024, dtype=torch.float64))
    before = cache.statistics()
    with pytest.raises(ValueError):
        cache.read(rows, device)
    assert cache.statistics() == before


def test_allocation_failure_requires_reset_without_publishing_partial_history(monkeypatch):
    cache = DescriptorReadCache()
    rows = [torch.ones(1024)]
    cache.read(rows, 'cpu')
    rows.append(torch.zeros(1024))
    with monkeypatch.context() as patch:
        patch.setattr(torch, 'empty', Mock(side_effect=RuntimeError('allocation failed')))
        with pytest.raises(RuntimeError, match='allocation failed'):
            cache.read(rows, 'cpu')
    assert cache.count == 1
    with pytest.raises(RuntimeError, match='reset is required'):
        cache.read(rows, 'cpu')
    cache.reset()
    assert cache.read(rows, 'cpu').shape == (1, 2, 1024)


def make_agent(directory, rows, goal, *, cached):
    directory.mkdir()
    agent = object.__new__(MemNavAgent)
    agent.memory_mechanism = 'native_interval7'
    agent.memory = EpisodicGEM(agent, bounded=False)
    agent.S, agent.W, agent.amargin = 8, 64, 8
    agent.device = torch.device('cpu')
    agent.certified_relocalization_matcher = object()
    agent._certified_route_live_depth_cache = {}
    agent.rgb_dir = str(directory)
    agent.dino_cls = list(rows)
    agent.n = len(rows)
    agent.lb = SimpleNamespace(
        load_images=Mock(return_value=torch.zeros(1, 3, 4, 4)),
        dino=Mock(return_value={'cls': goal}))
    agent.memory.online_depths = FrameDepthArchive(directory / 'geometry')
    if not cached:
        # Exact frozen implementation's input construction, for comparison.
        agent.memory.retrieval_descriptors = lambda data: torch.stack(data, 0)[None].to(agent.device)
    return agent


def semantic_plan(agent, goal, ceiling=None):
    result = agent.plan(goal, retrieval_only=True, candidate_ceiling_override=ceiling)
    # A new accounting field is observational; compare every navigation field.
    result.pop('memory_status')
    return result


def test_public_plan_matches_full_copy_through_new_frames_ceilings_and_goal_switches(tmp_path):
    rows = torch.randn(137, 1024, generator=torch.Generator().manual_seed(15))
    goal = rows[10:11].clone()
    candidate = make_agent(tmp_path / 'candidate', rows[:32], goal, cached=True)
    control = make_agent(tmp_path / 'control', rows[:32], goal, cached=False)
    selected = None
    for count, target, ceiling in [
            (32, b'a', None), (32, b'a', None), (40, b'a', None),
            (40, b'a', 15), (40, b'a', None), (72, b'b', None),
            (90, b'a', None), (137, b'a', None)]:
        for agent in (candidate, control):
            agent.dino_cls.extend(rows[agent.n:count].unbind())
            agent.n = count
        got, expected = semantic_plan(candidate, target, ceiling), semantic_plan(control, target, ceiling)
        assert got == expected
        if count == 32 and selected is None:
            selected = [dict(v) for v in got['certified_visual_candidates']]
            got['certified_visual_candidates'][0]['anchor'] = -99
        if count == 40 and ceiling is None:
            assert got['certified_visual_candidates'] == selected
    assert candidate.memory.goal_session_index == control.memory.goal_session_index == 3
    assert candidate.lb.dino.call_count == control.lb.dino.call_count == 3
    assert candidate.memory.descriptor_read_cache.statistics()['uploaded_bytes'] == 137 * 1024 * 4


def test_empty_shortlist_stays_frozen_while_current_similarity_updates(tmp_path):
    rows = torch.randn(32, 1024, generator=torch.Generator().manual_seed(71))
    goal = rows[20:21].clone()
    candidate = make_agent(tmp_path / 'candidate', rows[:8], goal, cached=True)
    control = make_agent(tmp_path / 'control', rows[:8], goal, cached=False)
    first = semantic_plan(candidate, b'early')
    assert first == semantic_plan(control, b'early')
    assert first['certified_visual_candidates'] == []
    for agent in (candidate, control):
        agent.dino_cls.extend(rows[8:].unbind())
        agent.n = len(rows)
    later = semantic_plan(candidate, b'early')
    assert later == semantic_plan(control, b'early')
    assert later['certified_visual_candidates'] == []
    assert later['current_goal_cos'] != first['current_goal_cos']


def test_history_readiness_guard_does_not_upload_an_incomplete_prefix(tmp_path):
    rows = torch.ones(9, 1024)
    agent = make_agent(tmp_path / 'candidate', rows[:8], rows[:1], cached=True)
    agent.n = 9
    assert agent.memory.retrieve(b'a', 'a', 8, 7) == ([], None)
    assert agent.memory.descriptor_read_cache.count == 0
    agent.lb.dino.assert_not_called()


def test_public_episode_reset_releases_descriptor_cache_with_the_old_history(tmp_path):
    rows = torch.ones(9, 1024)
    agent = make_agent(tmp_path / 'candidate', rows, rows[:1], cached=True)
    semantic_plan(agent, b'a')
    memory = agent.memory
    storage_ref = weakref.ref(memory.descriptor_read_cache._storage)
    agent.buffer_root = str(tmp_path / 'episodes')
    agent._episode_counter = 0
    agent.flow_gate = 'off'
    agent.certified_reference_depth_source = 'online_history'
    agent.certified_eager_depth_cache = agent.depth_observation_only = False
    agent.lb.model = SimpleNamespace(clean_kv_cache=Mock(), aggregator=Mock(), camera_head=Mock())
    agent.reset()
    assert agent.memory is memory and agent.n == 0
    assert memory.descriptor_read_cache.count == 0 and memory.descriptors == []
    assert storage_ref() is None
