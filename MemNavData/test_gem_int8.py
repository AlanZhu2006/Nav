"""Selection, special-token time and isolated state for the INT8 GEM adapter."""
import os
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.syspath_prepend(os.environ['LINGBOT_REPO'])
    from NavDP.baselines.memnav.gem import int8
    return int8


def test_selection_shares_one_decoder_and_preserves_parameter_objects(adapter):
    from MemNavData.test_gem_paged import network
    model=network(adapter)
    original={k:id(v) for k,v in model.named_parameters()}
    frame_blocks=model.aggregator.frame_blocks
    camera=model.camera_head
    receipt=adapter.enable_int8_storage(model)
    assert adapter.enable_int8_storage(model)==receipt
    assert {k:id(v) for k,v in model.named_parameters()}==original
    assert model.aggregator.frame_blocks is frame_blocks and model.camera_head is camera
    assert {id(b.attn._gem_decode_workspace) for b in model.aggregator.global_blocks}=={
        id(model.aggregator.gem_decode_workspace)}
    assert model.aggregator.use_sdpa and model.aggregator.kv_cache_manager is None
    model.aggregator.total_frames_processed=8
    with pytest.raises(ValueError,match='empty stream'):adapter.enable_int8_storage(model)


@pytest.mark.parametrize('count,current',[(0,8),(8,1),(9,1),(64,1),(72,1),(73,1),(319,1)])
def test_special_token_identities_match_native_sdpa(adapter,count,current):
    native=adapter.AggregatorStream.__new__(adapter.AggregatorStream)
    torch.nn.Module.__init__(native)
    native.num_frame_for_scale=8;native.num_special_tokens=6
    native.kv_cache_manager=None
    native.kv_cache={'k_0':None if count==0 else torch.empty(1,1,min(count,72),1,1)}
    native.camera_token=torch.nn.Parameter(torch.arange(8.).reshape(1,2,1,4))
    native.register_token=torch.nn.Parameter(torch.arange(32.).reshape(1,2,4,4))
    native.scale_token=torch.nn.Parameter(10+torch.arange(8.).reshape(1,2,1,4))
    expected=native._prepare_special_tokens(1,current,current,4,8)
    native.__class__=adapter.Int8Aggregator
    native.kv_cache={'_gem_int8_0':None if count==0 else SimpleNamespace(count=count)}
    actual=native._prepare_special_tokens(1,current,current,4,8)
    assert torch.equal(expected,actual)


def test_scale_lease_keeps_semantic_int8_state_and_reuses_only_scratch(adapter):
    from NavDP.baselines.memnav.gem.reactivation import isolated_stream
    from NavDP.baselines.memnav.gem.int8_state import Int8LayerState
    layer=Int8LayerState(torch.ones(2,2,8,13,4,dtype=torch.bfloat16))
    layer.commit(torch.full((2,2,13,4),2.,dtype=torch.bfloat16))
    old=layer.codes.clone()
    scratch=SimpleNamespace(buffer=torch.zeros(8))
    aggregate=SimpleNamespace(use_sdpa=True,kv_cache_manager=None,
        kv_cache={'_gem_int8_0':layer,'_skip_append':False},
        total_frames_processed=9,_cached_pos3d=object(),gem_decode_workspace=scratch)
    camera=SimpleNamespace(kv_cache={'a':torch.ones(2)},frame_idx=9,pos_cache=object())
    original_cache=aggregate.kv_cache

    def clean():
        for key in aggregate.kv_cache:aggregate.kv_cache[key]=None
        aggregate.total_frames_processed=camera.frame_idx=0
        camera.kv_cache=None;camera.pos_cache=None

    model=SimpleNamespace(aggregator=aggregate,camera_head=camera,clean_kv_cache=clean)
    with pytest.raises(RuntimeError):
        with isolated_stream(model):
            assert aggregate.kv_cache['_gem_int8_0'] is None
            assert aggregate.gem_decode_workspace is scratch
            scratch.buffer.fill_(3.)
            aggregate.kv_cache['_gem_int8_0']=Int8LayerState(torch.zeros(2,2,8,13,4,dtype=torch.bfloat16))
            raise RuntimeError('temporary model failure')
    assert aggregate.kv_cache is original_cache and aggregate.kv_cache['_gem_int8_0'] is layer
    assert layer.count==9 and torch.equal(layer.codes,old)
    assert aggregate.total_frames_processed==camera.frame_idx==9
    assert torch.all(scratch.buffer==3.)  # Scratch data has no cross-frame meaning.
