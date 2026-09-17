"""Paged allocation/lifecycle contracts. CPU tests do not validate FA2 outputs."""
from types import SimpleNamespace
import os
import sys

import pytest
import torch


@pytest.fixture
def paged(monkeypatch):
    if os.environ.get('LINGBOT_REPO'):
        monkeypatch.syspath_prepend(os.environ['LINGBOT_REPO'])
    from NavDP.baselines.memnav.gem import paged
    return paged


@pytest.fixture
def manager(paged, monkeypatch):
    import flashinfer
    # Exercise the actual storage implementation on CPU, without pretending
    # to test CUDA kernels or adding a selectable production CPU backend.
    monkeypatch.setattr(flashinfer, 'BatchPrefillWithPagedKVCacheWrapper',
        lambda *a, **k: SimpleNamespace())
    return paged.PagedWorkingMemory(num_blocks=2, tokens_per_frame=1375,
        num_heads=1, head_dim=1, device='cpu')


def tensors(frame):
    return (torch.full((1375, 1, 1), frame % 128, dtype=torch.bfloat16),
            torch.full((1375, 1, 1), frame // 128, dtype=torch.bfloat16))


def decode(manager, layer):
    key, value = manager._gather_kv(layer)
    return (key.float() + 128 * value.float()).flatten().to(torch.int64)


def commit(manager, frame):
    key, value = tensors(frame)
    for layer in range(manager.num_blocks):
        manager.append_frame(layer, key, value)
        manager.evict_frames(layer, 8, 64)


def expected(count, temporary=None):
    full = list(range(min(count, 8))) + list(range(max(8, count - 64), count))
    special = list(range(count))
    if temporary is not None:
        full.append(temporary)
        special.append(temporary)
    return torch.tensor([i for i in full for _ in range(1369)] +
                        [i for i in special for _ in range(6)])


def test_native_visible_tokens_through_eviction_and_special_boundary(manager):
    for frame in range(320):
        commit(manager, frame)
        if frame in {7, 8, 71, 72, 227, 228, 285, 319}:
            for layer in range(2):
                assert torch.equal(decode(manager, layer), expected(frame + 1))
                visible = manager.build_visible_page_table(layer)
                assert len(visible) == len(set(visible))
            assert manager.build_visible_page_table(0) == manager.build_visible_page_table(1)
    assert all(len(t) == 75 for t in manager.kv_caches)
    assert manager.statistics()['effective_kv_bytes'] == 2 * (72 * 1369 + 320 * 6) * 4
    with pytest.raises(RuntimeError, match='limit'):
        manager.append_frame(0, *tensors(320))


@pytest.mark.parametrize('count', [8, 72, 228, 286])
def test_temporary_read_preserves_history_even_across_special_page(manager, monkeypatch, count):
    for frame in range(count):
        commit(manager, frame)
    original = [decode(manager, i).clone() for i in range(2)]
    manager._skip_append = True
    seen = []

    def read(layer, q):
        assert torch.equal(decode(manager, layer), expected(count, temporary=count))
        seen.append(layer)
        return q

    monkeypatch.setattr(manager, 'compute_attention', read)
    for layer in range(2):
        manager.attend(layer, torch.zeros(1375, 1, 1), *tensors(count))
        assert torch.equal(decode(manager, layer), original[layer])
        assert manager.frame_count[layer] == count
    assert seen == [0, 1]


def test_failed_temporary_attention_rolls_back_and_requires_reset(manager, monkeypatch):
    for frame in range(73):
        commit(manager, frame)
    before = decode(manager, 0)
    manager._skip_append = True

    def fail(*args):
        raise RuntimeError('injected kernel failure')

    monkeypatch.setattr(manager, 'compute_attention', fail)
    with pytest.raises(RuntimeError, match='injected kernel'):
        manager.attend(0, torch.zeros(1375, 1, 1), *tensors(73))
    assert torch.equal(decode(manager, 0), before)
    assert manager.frame_count == [73, 73]
    with pytest.raises(RuntimeError, match='reset'):
        manager.append_frame(1, *tensors(73))
    manager.reset()
    assert manager.failure is None and manager.frame_count == [0, 0]
    assert manager.statistics()['allocated_pool_bytes'] == 0
    commit(manager, 0)
    assert torch.equal(decode(manager, 1), expected(1))


def network(paged):
    result = torch.nn.Module()
    agg = paged.AggregatorStream.__new__(paged.AggregatorStream)
    torch.nn.Module.__init__(agg)
    agg.depth, agg.embed_dim = 24, 1024
    agg.use_sdpa, agg.use_flashinfer = True, False
    agg.kv_cache_manager = None
    agg.kv_cache = {'k_0': None, '_skip_append': False}
    agg.total_frames_processed, agg._cached_pos3d = 0, None
    agg.kv_cache_scale_frames, agg.kv_cache_sliding_window = 8, 64
    agg.kv_cache_cross_frame_special = agg.kv_cache_include_scale_frames = True
    agg.kv_cache_camera_only = False
    with torch.device('meta'):
        agg.global_blocks = torch.nn.ModuleList([
            paged.SDPABlock(1024, 16, qk_norm=True) for _ in range(24)])
        agg.frame_blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8)])
        result.camera_head = torch.nn.Linear(8, 8)
    result.aggregator = agg
    result.use_sdpa = True
    return result.eval()


def test_selection_keeps_weights_frame_attention_and_camera(paged):
    model = network(paged)
    weights = {k: id(v) for k, v in model.named_parameters()}
    frames, camera = model.aggregator.frame_blocks, model.camera_head
    receipt = paged.enable_paged_memory(model)
    assert paged.enable_paged_memory(model) == receipt
    assert {k: id(v) for k, v in model.named_parameters()} == weights
    assert model.aggregator.frame_blocks is frames and model.camera_head is camera
    assert model.aggregator.gem_paged_memory and not model.aggregator.use_sdpa
    model.aggregator.total_frames_processed = 1
    with pytest.raises(ValueError, match='first observation'):
        paged.enable_paged_memory(model)


def test_pool_precision_uses_attention_contract_not_residual_dtype(paged, monkeypatch):
    model = network(paged)
    paged.enable_paged_memory(model)
    monkeypatch.setattr(torch, 'is_autocast_enabled', lambda device: True)
    monkeypatch.setattr(torch, 'get_autocast_dtype', lambda device: torch.bfloat16)
    requests = []

    def pool(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(_skip_append=False)

    monkeypatch.setattr(paged, 'PagedWorkingMemory', pool)
    manager = model.aggregator._get_flashinfer_manager('cuda', torch.float32, 1375)
    assert model.aggregator.kv_cache_manager is manager
    assert requests[0]['tokens_per_frame'] == 1375
    with pytest.raises(ValueError, match='BF16 attention reads'):
        model.aggregator._get_flashinfer_manager('cuda', torch.float16, 1375)


@pytest.mark.parametrize('cleanup_error', [False, True])
def test_scale_isolation_restores_exact_live_pool_after_errors(paged, manager, cleanup_error):
    from NavDP.baselines.memnav.gem.reactivation import isolated_stream

    for frame in range(13):
        commit(manager, frame)
    before = decode(manager, 0).clone()
    agg = SimpleNamespace(use_sdpa=False, gem_paged_memory=True,
        kv_cache_manager=manager, kv_cache={'_skip_append': False},
        total_frames_processed=13, _cached_pos3d=object())
    camera = SimpleNamespace(kv_cache={'state': object()}, frame_idx=13, pos_cache=object())
    saved = (agg.kv_cache, agg._cached_pos3d, camera.kv_cache, camera.pos_cache)
    calls = []

    def clean():
        calls.append(1)
        if agg.kv_cache_manager is not None:
            agg.kv_cache_manager.reset()
        agg.kv_cache_manager = None
        agg.total_frames_processed = camera.frame_idx = 0
        if cleanup_error and len(calls) == 2:
            raise RuntimeError('temporary cleanup failure')

    model = SimpleNamespace(aggregator=agg, camera_head=camera, clean_kv_cache=clean)
    with pytest.raises(RuntimeError):
        with isolated_stream(model):
            assert agg.kv_cache_manager is None
            assert agg.kv_cache is not saved[0]
            assert camera.kv_cache is None
            agg.total_frames_processed = camera.frame_idx = 40
            raise RuntimeError('temporary inference failure')
    assert agg.kv_cache_manager is manager
    assert agg.total_frames_processed == camera.frame_idx == 13
    assert all(a is b for a, b in zip(saved,
        (agg.kv_cache, agg._cached_pos3d, camera.kv_cache, camera.pos_cache)))
    assert torch.equal(decode(manager, 0), before)
