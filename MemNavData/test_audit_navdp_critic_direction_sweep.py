import math

from MemNavData.audit_navdp_critic_direction_sweep import (
    depth_png_bytes,
    format_optional_angle,
    jpg_bytes,
    poisson_binomial_upper_tail,
    summarize,
)


def test_wire_encoders_emit_images():
    import numpy as np
    from PIL import Image
    import io

    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth = np.full((4, 5), 1.25, dtype=np.float32)
    assert Image.open(io.BytesIO(jpg_bytes(rgb))).size == (5, 4)
    decoded = np.asarray(Image.open(io.BytesIO(depth_png_bytes(depth))))
    assert decoded.shape == (4, 5)
    assert int(decoded[0, 0]) == 12500


def test_poisson_binomial_tail_matches_binomial_special_case():
    # P(Binomial(2, .5) >= 1) = .75.
    assert math.isclose(poisson_binomial_upper_tail(1, [0.5, 0.5]), 0.75)
    assert math.isclose(poisson_binomial_upper_tail(2, [0.5, 0.5]), 0.25)


def test_summary_separates_request_choice_from_execution():
    rows = [{
        "scene": "a",
        "critic_request_error_deg": 10.0,
        "critic_executed_error_deg": 80.0,
        "execution_ceiling_error_deg": 20.0,
        "native_executed_error_deg": 70.0,
        "random_request_hit_probability": 0.125,
        "critic_score_margin": 0.2,
    }]
    report = summarize(rows, 30.0)
    assert report["critic_request_hits"] == 1
    assert report["critic_executed_hits"] == 0
    assert report["execution_ceiling_hits"] == 1
    assert report["native_executed_hits"] == 0


def test_optional_angle_formatter_preserves_missing_execution():
    assert format_optional_angle(None) == "none"
    assert format_optional_angle(12.34) == "12.3"
