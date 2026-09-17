"""Incremental storage accounting against owned tensors; no model/navigation.

The CUDA cases use a tiny two-head stream to exercise real commit/eviction
operations. They verify accounting, not a new geometry or codec claim.
"""
import math
import weakref

import pytest
import torch

from NavDP.baselines.memnav.gem import lossless_state as implementation


@pytest.fixture
def state():
    if not torch.cuda.is_available():
        pytest.skip('Small CUDA state accounting check')
    initial = torch.zeros((2, 2, 8, 13, 4), dtype=torch.bfloat16, device='cuda')
    return implementation.LosslessLayerState(initial)


def tensor_inventory(state):
    blocks = [block for block in state.blocks if block is not None]
    compressed = sum(t.nbytes for block in blocks for t in block.tensors())
    return dict(
        committed_views=state.count,
        window_views=len(blocks),
        special_capacity=state.specials.shape[2],
        raw_window_bytes=sum(math.prod(block.shape) * 2 for block in blocks),
        compressed_window_bytes=compressed,
        effective_storage_bytes=(state.initial.nbytes + state.specials[:, :, :state.count].nbytes
                                 + state.pointers.nbytes + compressed),
        allocated_storage_bytes=sum(t.nbytes for t in state.tensors()),
        failure=state.failure,
    )


def test_incremental_counts_match_tensors_through_all_commits(state, monkeypatch):
    snapshot = state.statistics()
    assert snapshot == tensor_inventory(state)
    index = torch.arange(2 * 2 * 13 * 4, dtype=torch.int32, device=state.device)
    widths = set()
    for frame in range(8, 320):
        # Deliberately vary the encoded size so eviction must subtract the
        # actual outgoing block, rather than assume a fixed compression ratio.
        width = frame % 9
        bits = ((index % (1 << width)) << 7) | (index & 127)
        current = bits.to(torch.int16).view(torch.bfloat16).reshape(2, 2, 13, 4)
        previous = state.blocks[(state.count - 8) % 64]
        retired = None if previous is None else weakref.ref(previous)
        del previous
        state.commit(current)
        actual = state.statistics()
        assert actual == tensor_inventory(state), frame
        widths.add(state.blocks[(frame - 8) % 64].storage_bytes)
        assert retired is None or retired() is None, 'Accounting retained an evicted payload'
        assert all(isinstance(value, (int, str, type(None))) for value in actual.values())
    assert len(widths) > 1
    assert snapshot['committed_views'] == 8 and snapshot['compressed_window_bytes'] == 0
    assert state.count == 320
    before = state.statistics()
    with pytest.raises(RuntimeError, match='limit'):
        state.commit(current)
    assert state.statistics() == before

    class NoTraversal(list):
        def __iter__(self):
            raise AssertionError('A query traversed stored blocks')

    def no_tensors():
        raise AssertionError('A query inspected payload tensors')

    monkeypatch.setattr(state, 'blocks', NoTraversal(state.blocks))
    monkeypatch.setattr(state, 'tensors', no_tensors)
    allocated = torch.cuda.memory_allocated()
    assert state.statistics() == before
    assert torch.cuda.memory_allocated() == allocated
    returned = state.statistics()
    returned['compressed_window_bytes'] = -1
    assert state.statistics() == before
    state.failure = 'injected status failure'
    assert state.statistics()['failure'] == state.failure


@pytest.mark.parametrize('failure_stage', ['encode', 'address_after_growth'])
def test_failed_commit_reports_actual_owned_storage(state, monkeypatch, failure_stage):
    current = torch.zeros((2, 2, 13, 4), dtype=torch.bfloat16, device=state.device)
    before = state.statistics()

    def fail(*args, **kwargs):
        raise RuntimeError('injected write failure')

    if failure_stage == 'encode':
        monkeypatch.setattr(implementation, 'encode_bf16', fail)
    else:
        # Address construction occurs after the special-token allocation grows.
        monkeypatch.setattr(implementation.torch, 'tensor', fail)
    with pytest.raises(RuntimeError, match='injected write failure'):
        state.commit(current)
    assert state.statistics() == tensor_inventory(state)
    assert state.count == 8 and state.statistics()['compressed_window_bytes'] == 0
    if failure_stage == 'address_after_growth':
        assert state.statistics()['allocated_storage_bytes'] > before['allocated_storage_bytes']
        assert state.statistics()['effective_storage_bytes'] == before['effective_storage_bytes']
    with pytest.raises(RuntimeError, match='reset'):
        state.commit(current)
