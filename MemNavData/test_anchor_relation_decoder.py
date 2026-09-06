import numpy as np
import pytest
import torch

from MemNavData.anchor_relation_decoder import (
    AnchorRelationDecoder, position_direction_loss, transport_goal,
)
from MemNavData.train_anchor_relation_probe import prediction_metrics, split_scenes


def test_transport_updates_for_translation_and_rotation():
    anchor = torch.eye(4)
    current = torch.eye(4)
    current[0, 3] = 1
    assert torch.allclose(transport_goal(anchor, torch.tensor([2., 0., 3.]), current),
                          torch.tensor([1., 0., 3.]))
    current[:3, :3] = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    assert torch.allclose(transport_goal(anchor, torch.tensor([2., 0., 3.]), current),
                          torch.tensor([-3., 0., 1.]))


def test_transport_common_sim3_change():
    anchor = torch.eye(4, dtype=torch.float64)
    current = anchor.clone()
    current[:3, 3] = torch.tensor([1., 2., 3.])
    offset = torch.tensor([2., -1., 4.], dtype=torch.float64)
    original = transport_goal(anchor, offset, current)
    q = torch.tensor([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]], dtype=torch.float64)
    translation = torch.tensor([7., -3., 2.], dtype=torch.float64)
    scale = 3.7
    def change(pose):
        result = pose.clone()
        result[:3, :3] = q @ pose[:3, :3]
        result[:3, 3] = scale * q @ pose[:3, 3] + translation
        return result
    transformed = transport_goal(change(anchor), scale * offset, change(current))
    assert torch.allclose(transformed, scale * original, atol=1e-12)


def test_geometry_is_required_not_silently_filled():
    patches = torch.randn(2, 16, 32)
    model = AnchorRelationDecoder(feature_dim=32, width=16, layers=1, requires_geometry=True)
    with pytest.raises(ValueError, match="geometry input"):
        model(patches, patches)
    result = model(patches, patches, torch.randn(2, 16, 3))
    assert result.shape == (2, 2)
    result.sum().backward()
    assert model.geometry_encoding[0].weight.grad is not None


def test_visual_probe_does_not_accept_extra_geometry():
    patches = torch.randn(2, 16, 32)
    model = AnchorRelationDecoder(feature_dim=32, width=16, layers=1)
    with pytest.raises(ValueError, match="geometry input"):
        model(patches, patches, torch.zeros(2, 16, 3))


def test_loss_has_finite_gradient_at_zero_target():
    prediction = torch.zeros(3, 2, requires_grad=True)
    loss = position_direction_loss(prediction, torch.zeros(3, 2))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_scene_split_independent_of_input_order():
    scenes = [f"scene{i}" for i in range(40)]
    train, validation = split_scenes(scenes)
    assert len(train) == 32 and len(validation) == 8
    assert train.isdisjoint(validation)
    assert (train, validation) == split_scenes(scenes[::-1])


def test_invalid_pose_is_not_a_direction_success():
    pred = np.asarray([[0., 0.], [np.nan, np.nan], [2., 0.]])
    target = np.asarray([[1., 0.], [1., 0.], [2., 0.]])
    metrics = prediction_metrics(pred, target, ["a", "a", "b"])
    assert metrics["valid_predictions"] == 2
    assert metrics["anchor_local_direction_eligible"] == 3
    assert metrics["anchor_local_direction_valid"] == 1
    assert metrics["anchor_local_direction_within_30deg"] == 1
    assert metrics["navigation_SR"] is None
