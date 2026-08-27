import numpy as np
from PIL import Image
import pytest

from MemNavData.preflight_monocular_geometry_adapter import (
    _causal_scale_prefix,
    _teacher_inputs,
)


class _Agent:
    def process_image(self, images):
        return images.astype(np.float32)

    def process_depth(self, depths):
        self.depths = depths.copy()
        return depths


def test_saved_uint16_depth_is_decoded_from_metres_times_10000(tmp_path):
    rgb = tmp_path / "rgb"
    depth = tmp_path / "depth"
    rgb.mkdir()
    depth.mkdir()
    for frame in range(8):
        Image.fromarray(np.zeros((2, 3, 3), np.uint8)).save(rgb / f"{frame}.jpg")
    encoded = np.full((2, 3), 25000, np.uint16)
    Image.fromarray(encoded).save(depth / "7.png")
    agent = _Agent()
    _, decoded = _teacher_inputs(agent, rgb, depth, 7)
    assert decoded.shape == (1, 2, 3, 1)
    assert np.allclose(decoded, 2.5)


def test_causal_scale_prefix_never_reads_future_frames(tmp_path):
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    for index in range(45):
        (rgb_dir / f"{index}.jpg").touch()
    poses = np.arange(45 * 9, dtype=np.float32).reshape(45, 9)
    paths, prefix_poses = _causal_scale_prefix(rgb_dir, poses, prefix_frames=40)
    assert [path.name for path in paths] == [f"{index}.jpg" for index in range(40)]
    np.testing.assert_array_equal(prefix_poses, poses[:40])


def test_causal_scale_prefix_fails_closed_when_prefix_is_incomplete(tmp_path):
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    for index in range(39):
        (rgb_dir / f"{index}.jpg").touch()
    with pytest.raises(ValueError, match="camera poses"):
        _causal_scale_prefix(rgb_dir, np.zeros((39, 9)), prefix_frames=40)
