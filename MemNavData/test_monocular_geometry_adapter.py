import pytest
import torch

from MemNavData.monocular_geometry_adapter import (
    GeometryAdapterConfig,
    GeometryTokenAdapter,
    adapter_parameter_receipt,
    geometry_distillation_losses,
)


def _inputs(batch=2, frames=8, lingbot_dim=64, depth_dim=16):
    torch.manual_seed(7)
    # Six specials + a 5x5 square patch grid; the adapter pools it to 4x4.
    window = torch.randn(batch, frames, 6 + 25, lingbot_dim)
    depth = torch.randn(batch, 49, depth_dim)
    scale = torch.tensor(
        [[-0.69, 1.2, 1.0, 0.5, 0.1, 0.0]] * batch,
        dtype=torch.float32,
    )
    return window, depth, scale


def _adapter():
    return GeometryTokenAdapter(
        GeometryAdapterConfig(
            lingbot_dim=64,
            depth_feature_dim=16,
            navdp_dim=32,
            navdp_tokens=12,
            recent_frames=8,
            special_tokens_per_frame=6,
            pooled_grid_side=4,
            scale_feature_dim=6,
            heads=4,
            layers=1,
            feedforward_multiplier=2,
        )
    )


def test_adapter_output_and_gradient_contract():
    adapter = _adapter()
    output = adapter(*_inputs())
    assert output.shape == (2, 12, 32)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in adapter.parameters())
    receipt = adapter_parameter_receipt(adapter)
    assert receipt["trainable_parameter_count"] > 0
    assert receipt["output_shape_without_batch"] == [12, 32]


def test_compact_input_is_exactly_equivalent():
    adapter = _adapter().eval()
    window, depth, scale = _inputs()
    with torch.no_grad():
        dense = adapter(window, depth, scale)
        compact = adapter.forward_compact(
            window[:, :, :6], window[:, -1, 6:], depth, scale
        )
    torch.testing.assert_close(dense, compact, rtol=0.0, atol=0.0)


def test_offline_pooled_input_is_exactly_equivalent():
    adapter = _adapter().eval()
    window, depth, scale = _inputs()
    current = window[:, -1, 6:]
    pooled_current = torch.nn.functional.adaptive_avg_pool2d(
        current.transpose(1, 2).reshape(2, 64, 5, 5), (4, 4)
    ).flatten(2).transpose(1, 2)
    pooled_depth = torch.nn.functional.adaptive_avg_pool2d(
        depth.transpose(1, 2).reshape(2, 16, 7, 7), (4, 4)
    ).flatten(2).transpose(1, 2)
    with torch.no_grad():
        online = adapter.forward_compact(
            window[:, :, :6], current, depth, scale
        )
        cached = adapter.forward_pooled_compact(
            window[:, :, :6], pooled_current, pooled_depth, scale
        )
    torch.testing.assert_close(online, cached, rtol=0.0, atol=0.0)


def test_short_prefix_is_right_aligned_and_masked():
    adapter = _adapter().eval()
    window, depth, scale = _inputs(batch=1, frames=3)
    mask = torch.tensor([[False, True, True]])
    changed = window.clone()
    changed[:, 0] = 10_000.0
    with torch.no_grad():
        first = adapter(window, depth, scale, mask)
        second = adapter(changed, depth, scale, mask)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_current_frame_cannot_be_masked():
    adapter = _adapter()
    window, depth, scale = _inputs(batch=1, frames=3)
    with pytest.raises(ValueError, match="current.*valid"):
        adapter(window, depth, scale, torch.tensor([[True, True, False]]))


def test_query_initialization_accepts_prefixed_checkpoint_key():
    adapter = _adapter()
    teacher = torch.randn_like(adapter.queries)
    key = "module.policy.rgbd_encoder.former_query.position_embedding.weight"
    matched = adapter.initialize_queries_from_navdp({key: teacher})
    assert matched == key
    torch.testing.assert_close(adapter.queries, teacher)


def test_non_square_spatial_tokens_fail_closed():
    adapter = _adapter()
    window, depth, scale = _inputs(batch=1)
    with pytest.raises(ValueError, match="not a square grid"):
        adapter(window[:, :, :-1], depth, scale)


def test_distillation_loss_is_zero_for_identical_functional_outputs():
    tokens = torch.randn(2, 12, 32, requires_grad=True)
    epsilon = torch.randn(2, 24, 3)
    critic = torch.randn(2, 4)
    losses = geometry_distillation_losses(
        tokens,
        tokens.detach(),
        student_epsilon=epsilon,
        teacher_epsilon=epsilon,
        student_critic=critic,
        teacher_critic=critic,
    )
    assert losses["token"].item() == pytest.approx(0.0, abs=1e-6)
    assert losses["denoise"].item() == pytest.approx(0.0, abs=1e-7)
    assert losses["critic"].item() == pytest.approx(0.0, abs=1e-7)
    # Rank loss is a soft teacher-weighted objective and is non-zero at finite
    # logits even when the two orderings agree exactly.
    assert losses["rank"].item() >= 0.0
