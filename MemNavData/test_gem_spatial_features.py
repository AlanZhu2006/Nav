import numpy as np
import pytest
import torch

from MemNavData.gem_spatial_features import mutual_cosine, patch_coordinates
from MemNavData.lingbot_pnp_localization import map_raw_points_to_lingbot_pad


@pytest.mark.parametrize('wh', [(480, 270), (270, 480), (512, 512)])
def test_patch_addresses_roundtrip_original_preprocessing(wh):
    ids, pad, raw = patch_coordinates(wh)
    assert len(ids) == len(set(ids))
    assert ((raw >= 0) & (raw < np.asarray(wh))).all()
    restored = map_raw_points_to_lingbot_pad(raw, raw_height=wh[1], raw_width=wh[0],
        target_height=518, target_width=518, patch_size=14)
    np.testing.assert_allclose(restored, pad, rtol=0, atol=1e-5)
    if wh == (480, 270):
        assert len(ids) == 37*21


def test_permutation_preserves_correspondence_direction():
    reference = torch.eye(5)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    i, j, scores = mutual_cosine(reference, reference[permutation])
    np.testing.assert_array_equal(permutation[j], i)
    np.testing.assert_array_equal(scores, np.ones(5))


def test_ties_are_not_many_to_one_and_empty_is_retained():
    i, j, _ = mutual_cosine(torch.ones(3, 4), torch.ones(2, 4))
    np.testing.assert_array_equal(i, [0])
    np.testing.assert_array_equal(j, [0])
    assert len(mutual_cosine(torch.ones(0, 4), torch.ones(2, 4))[0]) == 0


def test_invalid_feature_is_an_error():
    with pytest.raises(ValueError):
        mutual_cosine(torch.zeros(2, 4), torch.ones(2, 4))
