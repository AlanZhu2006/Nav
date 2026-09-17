"""Selection, native temporal identity and isolation contracts; no GPU claim."""
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(os.environ['LINGBOT_REPO'])
    from NavDP.baselines.memnav.gem import lossless
    return lossless


def test_selection_preserves_weights_and_uses_one_scratch(adapter):
    from MemNavData.test_gem_paged import network
    model = network(adapter)
    parameters = {k: id(v) for k, v in model.named_parameters()}
    frame_blocks, camera = model.aggregator.frame_blocks, model.camera_head
    receipt = adapter.enable_lossless_storage(model)
    assert adapter.enable_lossless_storage(model) == receipt
    assert {k: id(v) for k, v in model.named_parameters()} == parameters
    assert model.aggregator.frame_blocks is frame_blocks and model.camera_head is camera
    assert {id(b.attn._gem_decode_workspace) for b in model.aggregator.global_blocks} == {
        id(model.aggregator.gem_decode_workspace)}
    assert model.aggregator.use_sdpa and model.aggregator.kv_cache_manager is None
    model.aggregator.total_frames_processed = 8
    with pytest.raises(ValueError, match='empty stream'):
        adapter.enable_lossless_storage(model)


@pytest.mark.parametrize('count,current', [(0, 8), (8, 1), (9, 1), (71, 1), (72, 1), (73, 1), (319, 1)])
def test_special_token_identity_matches_native(adapter, count, current):
    native = adapter.AggregatorStream.__new__(adapter.AggregatorStream)
    torch.nn.Module.__init__(native)
    native.num_frame_for_scale = 8
    native.num_special_tokens = 6
    native.kv_cache_manager = None
    native.kv_cache = {'k_0': None if not count else torch.empty(1, 1, min(count, 72), 1, 1)}
    native.camera_token = torch.nn.Parameter(torch.arange(8.).reshape(1, 2, 1, 4))
    native.register_token = torch.nn.Parameter(torch.arange(32.).reshape(1, 2, 4, 4))
    native.scale_token = torch.nn.Parameter(10 + torch.arange(8.).reshape(1, 2, 1, 4))
    expected = native._prepare_special_tokens(1, current, current, 4, 8)
    native.__class__ = adapter.LosslessAggregator
    native.kv_cache = {'_gem_lossless_0': None if not count else SimpleNamespace(count=count)}
    assert torch.equal(expected, native._prepare_special_tokens(1, current, current, 4, 8))


def test_isolated_scale_restores_state_container_on_error():
    from NavDP.baselines.memnav.gem.reactivation import isolated_stream
    layer = SimpleNamespace(count=72, failure=None, immutable_bits=object())
    scratch = SimpleNamespace(buffer=torch.zeros(8))
    aggregate = SimpleNamespace(use_sdpa=True, kv_cache_manager=None,
        kv_cache={'_gem_lossless_0': layer, '_skip_append': False},
        total_frames_processed=72, _cached_pos3d=object(), gem_decode_workspace=scratch)
    camera = SimpleNamespace(kv_cache={'a': torch.ones(2)}, frame_idx=72, pos_cache=object())
    old_cache, old_camera = aggregate.kv_cache, camera.kv_cache
    def clean():
        for key in aggregate.kv_cache:
            aggregate.kv_cache[key] = None
        aggregate.total_frames_processed = camera.frame_idx = 0
        camera.kv_cache = camera.pos_cache = None
    model = SimpleNamespace(aggregator=aggregate, camera_head=camera, clean_kv_cache=clean)
    with pytest.raises(RuntimeError, match='temporary failure'):
        with isolated_stream(model):
            assert aggregate.kv_cache['_gem_lossless_0'] is None
            scratch.buffer.fill_(2)
            aggregate.kv_cache['_gem_lossless_0'] = SimpleNamespace(count=8)
            raise RuntimeError('temporary failure')
    assert aggregate.kv_cache is old_cache and aggregate.kv_cache['_gem_lossless_0'] is layer
    assert camera.kv_cache is old_camera and camera.frame_idx == aggregate.total_frames_processed == 72
    assert aggregate.gem_decode_workspace is scratch and torch.all(scratch.buffer == 2)


def test_public_selection_and_failure_do_not_revert_to_native():
    from NavDP.baselines.memnav.policy_agent import MemNavAgent
    from NavDP.baselines.memnav.gem.episodic import EpisodicGEM
    agent = object.__new__(MemNavAgent)
    agent.memory_mechanism, agent.memory_kv_storage = 'native_interval7', 'lossless_bf16'
    assert agent.memory.kv_storage == 'lossless_bf16' and not agent.memory.bounded
    agent.S, agent.W = 8, 16
    agent.certified_reference_depth_source = 'online_history'
    agent.certified_eager_depth_cache = agent.depth_observation_only = False
    agent.lb = SimpleNamespace(model=SimpleNamespace(kv_cache_sliding_window=16))
    memory = EpisodicGEM(agent, bounded=False, kv_storage='lossless_bf16')
    with pytest.raises(ValueError, match='W64'):
        memory.reset()
    assert memory.failure is not None and memory.lossless_memory_receipt is None
    with pytest.raises(RuntimeError, match='reset'):
        memory.write(b'image')
