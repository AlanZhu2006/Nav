import pytest
import torch
import torch.nn as nn

from internnav.model.basemodel.memnav.gate_curriculum import (
    blend_decoder_gate,
    linear_teacher_ratio,
)
from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream


def test_linear_teacher_ratio_clamps_at_schedule_ends():
    assert linear_teacher_ratio(0, 1.0, 0.0, 1000) == pytest.approx(1.0)
    assert linear_teacher_ratio(250, 1.0, 0.0, 1000) == pytest.approx(0.75)
    assert linear_teacher_ratio(1000, 1.0, 0.0, 1000) == pytest.approx(0.0)
    assert linear_teacher_ratio(5000, 1.0, 0.0, 1000) == pytest.approx(0.0)
    assert linear_teacher_ratio(-10, 1.0, 0.0, 1000) == pytest.approx(1.0)


def test_zero_step_schedule_uses_end_value_as_explicit_off_switch():
    assert linear_teacher_ratio(0, 1.0, 0.0, 0) == pytest.approx(0.0)
    assert linear_teacher_ratio(99, 0.8, 0.2, 0) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(step=0, start=-0.1, end=0.0, decay_steps=1),
        dict(step=0, start=1.0, end=1.1, decay_steps=1),
        dict(step=0, start=1.0, end=0.0, decay_steps=-1),
    ],
)
def test_linear_teacher_ratio_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        linear_teacher_ratio(**kwargs)


def test_blend_decoder_gate_values_and_action_gradient():
    predicted = torch.tensor([0.2, 0.8], requires_grad=True)
    is_revisit = torch.tensor([1.0, 0.0])
    actual = blend_decoder_gate(predicted, is_revisit, 0.75, training=True)
    assert torch.allclose(actual, torch.tensor([0.8, 0.2]))

    actual.sum().backward()
    # The decoder/action gradient returns smoothly as teacher forcing decays.
    assert torch.allclose(predicted.grad, torch.full_like(predicted, 0.25))


def test_blend_decoder_gate_is_exact_original_path_at_eval_or_zero_ratio():
    predicted = torch.tensor([0.2, 0.8], requires_grad=True)
    target = torch.tensor([1.0, 0.0])
    assert blend_decoder_gate(predicted, target, 1.0, training=False) is predicted
    assert blend_decoder_gate(predicted, target, 0.0, training=True) is predicted


def test_frozen_lingbot_children_stay_eval_when_outer_policy_trains():
    # Construct a dependency-free shell rather than loading the multi-GB checkpoint.
    stream = LingBotStream.__new__(LingBotStream)
    nn.Module.__init__(stream)
    stream.model = nn.Sequential(nn.Dropout(p=0.5))
    stream.depth_feat_head = nn.Sequential(nn.Dropout(p=0.5))

    stream.train(True)
    assert stream.training is True
    assert stream.model.training is False
    assert stream.depth_feat_head.training is False

    stream.eval()
    assert stream.training is False
    assert stream.model.training is False
    assert stream.depth_feat_head.training is False
