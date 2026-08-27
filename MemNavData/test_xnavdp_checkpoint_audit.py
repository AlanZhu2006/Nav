import torch

from MemNavData.xnavdp_checkpoint_audit import compare_state_dicts


def test_checkpoint_comparison_separates_frozen_and_new_modules():
    base = {
        "image_encoder.weight": torch.tensor([1.0, 2.0]),
        "point_encoder.weight": torch.tensor([3.0]),
        "decoder.weight": torch.tensor([4.0]),
        "rgbd_encoder.weight": torch.tensor([5.0]),
    }
    post = {
        "image_encoder.weight": torch.tensor([1.0, 2.0]),
        "point_encoder.weight": torch.tensor([3.0]),
        "decoder.weight": torch.tensor([4.0]),
        "rgbd_encoder.weight": torch.tensor([6.0]),
        "decoder_ft.weight": torch.tensor([7.0]),
        "q1_heads.weight": torch.tensor([8.0]),
        "q2_heads.weight": torch.tensor([9.0]),
    }

    report = compare_state_dicts(base, post)
    findings = report["findings"]
    assert findings["image_encoder_present_in_post"] is True
    assert findings["image_encoder_exactly_equal_to_base"] is True
    assert findings["point_encoder_exactly_equal_to_base"] is True
    assert findings["base_decoder_exactly_equal_to_base"] is True
    assert findings["rgbd_encoder_exactly_equal_to_base"] is False
    assert findings["fine_tuned_decoder_present"] is True
    assert findings["twin_q_heads_present"] is True
    assert report["modules"]["rgbd_encoder"]["changed"] == 1
    assert report["post_only_prefixes"] == [
        "decoder_ft", "q1_heads", "q2_heads"]


def test_shape_mismatch_is_not_counted_as_changed_value():
    report = compare_state_dicts(
        {"decoder.weight": torch.zeros(2)},
        {"decoder.weight": torch.zeros(3)},
    )
    assert report["shape_mismatch_count"] == 1
    assert report["changed_count"] == 0
    assert report["modules"]["decoder"]["shape_mismatch"] == 1
