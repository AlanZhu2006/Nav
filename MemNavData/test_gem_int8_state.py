"""INT8 role/temporal/format contracts. Fused decoding needs separate GPU proof."""
import numpy as np
import pytest
import torch

from NavDP.baselines.memnav.gem.int8_state import Int8LayerState, fixed_int8_encode, visible_layout


def make_initial():
    generator=torch.Generator().manual_seed(12)
    return torch.randn(2,2,8,13,4,generator=generator).to(torch.bfloat16)


def test_codec_has_declared_axes_actual_bf16_scales_and_signed_codes():
    generator=torch.Generator().manual_seed(4)
    k=(torch.randn(2,7,4,generator=generator)*torch.tensor([.01,1,10,100])).to(torch.bfloat16)
    v=(torch.randn(2,7,4,generator=generator)*torch.arange(1,8)[None,:,None]).to(torch.bfloat16)
    ki,vi,ks,vs=fixed_int8_encode(k,v)
    assert ki.dtype==vi.dtype==torch.int8
    assert ks.shape==(2,4) and vs.shape==(2,7,1)
    assert ks.dtype==vs.dtype==torch.bfloat16
    for h in range(2):
        for d in range(4):
            source=k[h,:,d].float().numpy()
            scale=torch.tensor(np.max(abs(source))/127).to(torch.bfloat16).float().item()
            np.testing.assert_array_equal(ki[h,:,d].numpy(),np.clip(np.rint(source/scale),-127,127))
        for token in range(7):
            source=v[h,token].float().numpy()
            scale=torch.tensor(np.max(abs(source))/127).to(torch.bfloat16).float().item()
            np.testing.assert_array_equal(vi[h,token].numpy(),np.clip(np.rint(source/scale),-127,127))
    zero=torch.zeros(2,7,4,dtype=torch.bfloat16)
    zk,zv,zks,zvs=fixed_int8_encode(zero,zero)
    assert not zk.any() and not zv.any() and torch.all(zks==1) and torch.all(zvs==1)


def test_all_initial_and_special_tokens_survive_ring_eviction():
    initial=make_initial();state=Int8LayerState(initial)
    protected=state.initial.clone()
    current=torch.zeros(2,2,13,4,dtype=torch.bfloat16)
    specials=[initial[:,:,i,:6].clone() for i in range(8)]
    for frame in range(8,320):
        current.fill_(frame)
        state.commit(current)
        specials.append(current[:,:,:6].clone())
    assert state.count==320 and state.capacity==64
    assert torch.equal(state.initial,protected)
    assert torch.equal(state.specials[:,:,:320],torch.stack(specials,dim=2))
    assert state.codes.shape==(2,64,2,7,4)
    assert state.statistics()['window_views']==64
    with pytest.raises(RuntimeError,match='limit'):state.commit(current)


def test_visible_sets_distinguish_commit_and_temporary_before_eviction():
    for count in [8,9,71,72,73,228,319]:
        for commit in [False,True]:
            visible,old,total=visible_layout(count,commit)
            historical=list(range(8,count))
            expected=historical[-(64-int(commit)):]
            assert len(expected)==visible
            assert old==len(historical)-len(expected)
            assert total==old*6+(8+visible+1)*1375
    assert visible_layout(72,True)==(63,1,99006)
    assert visible_layout(72,False)==(64,0,100375)
    with pytest.raises(ValueError):visible_layout(320,False)


def test_bad_features_are_explicit_failure_not_silent_int8_conversion():
    state=Int8LayerState(make_initial())
    current=torch.zeros(2,2,13,4,dtype=torch.bfloat16)
    current[0,1,8,2]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):state.commit(current)
    assert state.count==8 and state.failure is not None
    current.zero_()
    with pytest.raises(RuntimeError,match='reset'):state.commit(current)


def test_statistics_are_values_not_references_that_hold_old_caches():
    state=Int8LayerState(make_initial());snapshot=state.statistics()
    assert all(not isinstance(v,torch.Tensor) for v in snapshot.values())
    assert snapshot['effective_storage_bytes']<snapshot['allocated_storage_bytes']
    current=torch.ones(2,2,13,4,dtype=torch.bfloat16);state.commit(current)
    assert snapshot['committed_views']==8 and state.statistics()['committed_views']==9
