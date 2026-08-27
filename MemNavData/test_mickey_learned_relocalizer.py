import numpy as np
import pytest

from MemNavData.mickey_learned_relocalizer import scaled_intrinsic


def test_scaled_intrinsic_matches_independent_xy_resize():
    source = np.asarray([
        [400.0, 0.0, 200.0],
        [0.0, 300.0, 100.0],
        [0.0, 0.0, 1.0],
    ])
    output = scaled_intrinsic(source, (400, 200), (200, 50))
    assert np.allclose(output, [
        [200.0, 0.0, 100.0],
        [0.0, 75.0, 25.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.mark.parametrize("size", ((0, 10), (10, -1)))
def test_scaled_intrinsic_rejects_invalid_size(size):
    with pytest.raises(ValueError):
        scaled_intrinsic(np.eye(3), (10, 10), size)
