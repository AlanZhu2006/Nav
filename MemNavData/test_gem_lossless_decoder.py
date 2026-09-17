"""Actual lossless decoder entry point: original-bit restore without a model."""
import pytest
import torch

from NavDP.baselines.memnav.gem.lossless_state import LosslessLayerState, LosslessDecodeWorkspace


@pytest.fixture(autouse=True)
def requires_cuda():
    if not torch.cuda.is_available():
        pytest.skip('Small CUDA decoder integration check')


def assert_restored(workspace, state, current, commit, expected):
    workspace.decode(state, current, commit=commit)
    for poison in (21930, -21931):
        workspace.buffer.view(torch.int16).fill_(poison)
        keys, values = workspace.decode(state, current, commit=commit)
        assert torch.equal(keys[0].view(torch.int16), expected[0].view(torch.int16))
        assert torch.equal(values[0].view(torch.int16), expected[1].view(torch.int16))


@pytest.mark.parametrize('commit', [False, True])
def test_public_decoder_preserves_window_and_evicted_specials(commit):
    def view(frame):
        return torch.arange(frame * 208, (frame + 1) * 208, dtype=torch.int32,
                            device='cuda').to(torch.int16).view(torch.bfloat16).reshape(2, 2, 13, 4)

    history = [view(frame) for frame in range(8)]
    state = LosslessLayerState(torch.stack(history, dim=2))
    workspace = LosslessDecodeWorkspace()
    for count in range(8, 75):
        current = view(count)
        if count in (8, 9, 72, 73, 74):
            first = max(8, count - (63 if commit else 64))
            expected = torch.cat([v[:, :, :6] for v in history[8:first]]
                                 + history[:8] + history[first:] + [current], dim=2)
            before = state.statistics()
            assert_restored(workspace, state, current, commit, expected)
            assert state.statistics() == before
        state.commit(current)
        history.append(current)


def test_public_decoder_restores_every_bf16_pattern():
    initial = torch.zeros((2, 2, 8, 1030, 16), dtype=torch.bfloat16, device='cuda')
    state = LosslessLayerState(initial)
    words = torch.arange(65536, dtype=torch.int32, device='cuda').to(torch.int16)
    specials = torch.zeros((2, 2, 6, 16), dtype=torch.bfloat16, device='cuda')
    patterns = torch.cat((specials, words.view(torch.bfloat16).reshape(2, 2, 1024, 16)), dim=2)
    state.commit(patterns)
    current = torch.zeros_like(patterns)
    expected = torch.cat([initial[:, :, frame] for frame in range(8)] + [patterns, current], dim=2)
    assert_restored(LosslessDecodeWorkspace(), state, current, False, expected)
