import numpy as np
from PIL import Image

from MemNavData.audit_monocular_geometry_teacher_depth_population import depth_receipt


def test_depth_receipt_distinguishes_valid_and_all_zero(tmp_path):
    valid = tmp_path / "valid.png"
    zero = tmp_path / "zero.png"
    Image.fromarray(np.full((4, 5), 25000, dtype=np.uint16)).save(valid)
    Image.fromarray(np.zeros((4, 5), dtype=np.uint16)).save(zero)
    good = depth_receipt(valid)
    bad = depth_receipt(zero)
    assert good["valid"] is True
    assert good["median_m"] == 2.5
    assert bad["valid"] is False
    assert bad["reason"] == "all_zero_depth"
