import importlib.util

import numpy as np
from PIL import Image
import pytest

from MemNavData.audit_navdp_base_rgb import original_preprocess
from MemNavData.audit_navdp_depth_raster import depth_preprocess
from MemNavData.audit_navdp_padding import LOADER
from MemNavData.lingbot_depth_raster import lingbot_pad_raster, to_source_rgb_raster


@pytest.fixture(scope="module")
def loader():
    spec = importlib.util.spec_from_file_location("verified_lingbot_loader", LOADER)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.mark.parametrize("shape", [(270, 480), (360, 640), (480, 640), (640, 480), (518, 518)])
def test_mapping_matches_actual_lingbot_loader(tmp_path, loader, shape):
    path = tmp_path / "black.png"
    Image.fromarray(np.zeros((*shape, 3), np.uint8)).save(path)
    tensor = loader.load_and_preprocess_images([str(path)], mode="pad", image_size=518, patch_size=14)[0]
    actual_support = tensor[0].numpy() < .5
    raster = lingbot_pad_raster(shape)
    expected = np.zeros((518, 518), bool)
    expected[raster.crop] = True
    np.testing.assert_array_equal(actual_support, expected)


@pytest.mark.parametrize("shape", [(270, 480), (360, 640), (480, 640), (640, 480), (518, 518)])
def test_padding_never_reaches_actor_and_metric_values_unchanged(shape):
    raster = lingbot_pad_raster(shape)
    depth = np.full((518, 518), .8, np.float32)
    depth[raster.crop] = 2.
    source = to_source_rgb_raster(depth, raster)
    np.testing.assert_allclose(source, 2., atol=1e-7)
    assert source.shape == shape
    encoded = depth_preprocess(source)
    rgb = original_preprocess(np.full((1, *shape, 3), 255, np.uint8))[0, :, :, 0]
    np.testing.assert_allclose(encoded, rgb * 2., atol=1e-7)


def test_square_identity_does_not_change_values():
    depth = np.random.default_rng(3).uniform(.2, 4, (518, 518)).astype(np.float32)
    np.testing.assert_array_equal(to_source_rgb_raster(depth, lingbot_pad_raster((518, 518))), depth)


def test_pixel_coordinates_have_no_axis_flip():
    raster = lingbot_pad_raster((270, 480))
    depth = np.zeros((518, 518), np.float32)
    yy, xx = np.indices((raster.resized_height, raster.resized_width))
    depth[raster.crop] = 1 + yy / 1000 + xx / 10000
    result = to_source_rgb_raster(depth, raster)
    assert np.all(np.diff(result, axis=0) > 0)
    assert np.all(np.diff(result, axis=1) > 0)


def test_bad_shape_does_not_guess_another_transform():
    with pytest.raises(ValueError):
        to_source_rgb_raster(np.ones((270, 480), np.float32), lingbot_pad_raster((270, 480)))


def test_zero_remains_zero_not_a_depth_fallback():
    result = to_source_rgb_raster(np.zeros((518, 518), np.float32), lingbot_pad_raster((270, 480)))
    assert np.count_nonzero(result) == 0


def test_nonfinite_depth_is_not_silently_repaired():
    depth = np.ones((518, 518), np.float32)
    depth[0, 0] = np.nan
    with pytest.raises(ValueError):
        to_source_rgb_raster(depth, lingbot_pad_raster((270, 480)))
