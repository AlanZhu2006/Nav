import math

import numpy as np
import pytest

from MemNavData.audit_revisit_fresh_online_observability import (
    H,
    W,
    ONLINE_REVISIT_THRESHOLD,
    STRONG_SUPPORT_THRESHOLD,
    backproject,
    camera_to_world,
    covisibility_fraction,
    exact_mcnemar,
    paired_effect,
    run_contract_spec,
    support_band,
    summarize_observability,
    summarize_population,
    to_world,
    validate_episode_metadata,
)


def test_generator_equivalent_covisibility_identity_and_opposite_view():
    depth = np.full((H, W), 2.0, dtype=np.float32)
    transform = camera_to_world(np.asarray([0.0, 0.5, 0.0]), 0.0)
    points = to_world(backproject(depth), transform)
    assert covisibility_fraction(points, transform, depth) == pytest.approx(1.0)

    opposite = camera_to_world(np.asarray([0.0, 0.5, 0.0]), math.pi)
    assert covisibility_fraction(points, opposite, depth) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "no_support_lt_0p10"),
        (0.099999, "no_support_lt_0p10"),
        (0.10, "ambiguous_0p10_to_0p20"),
        (0.199999, "ambiguous_0p10_to_0p20"),
        (ONLINE_REVISIT_THRESHOLD, "supported_0p20_to_0p50"),
        (0.499999, "supported_0p20_to_0p50"),
        (STRONG_SUPPORT_THRESHOLD, "strong_support_ge_0p50"),
        (1.0, "strong_support_ge_0p50"),
    ],
)
def test_support_band_has_frozen_boundaries(value, expected):
    assert support_band(value) == expected


def test_metadata_threshold_contract_fails_closed():
    metadata = {
        "n_legs": 2,
        "goals": [{"name": "B", "kind": "revisit"}],
        "covis_pos_lo": 0.10,
        "covis_band": [0.20, 1.0],
        "covis_pos_hi": 0.50,
    }
    assert validate_episode_metadata(metadata) is metadata["goals"][0]
    changed = dict(metadata, covis_pos_hi=0.51)
    with pytest.raises(RuntimeError, match="positive co-visibility threshold"):
        validate_episode_metadata(changed)


def _outcome(reached_b):
    return {"reached_a": True, "reached_b": reached_b, "joint": reached_b}


def test_stratified_paired_effect_uses_only_supplied_episode_keys():
    keys = [("scene_a", "episode_0000"),
            ("scene_b", "episode_0000"),
            ("scene_b", "episode_0001")]
    outcomes = {
        "geometry_router": {
            keys[0]: _outcome(False),
            keys[1]: _outcome(False),
            keys[2]: _outcome(True),
        },
        "known_revisit_direct": {
            keys[0]: _outcome(True),
            keys[1]: _outcome(True),
            keys[2]: _outcome(False),
        },
        "native": {key: _outcome(False) for key in keys},
    }
    effect = paired_effect(
        keys[:2], "geometry_router", "known_revisit_direct", outcomes,
        seed=7, resamples=1000)
    assert effect["eligible"] == 2
    assert effect["left_successes"] == 0
    assert effect["right_successes"] == 2
    assert len(effect["gains"]) == 2
    assert len(effect["losses"]) == 0
    assert effect["risk_difference_right_minus_left"] == 1.0
    assert effect["mcnemar_exact_two_sided_p"] == 0.5
    assert effect["scene_cluster_bootstrap_risk_difference_95"] == [1.0, 1.0]

    summary = summarize_population(
        "supported", keys[:2], outcomes, seed=9, resamples=1000)
    assert summary["eligible"] == 2
    assert summary["arms"]["known_revisit_direct"]["success_rate"] == 1.0
    assert summary["arms"]["geometry_router"]["success_rate"] == 0.0


def test_observability_summary_separates_all_and_a_success_populations():
    common = {
        "scene": "scene",
        "goal_render_jpeg_hash_match": True,
        "rendered_trace_jpeg_hash_matches": 3,
        "rendered_trace_jpeg_hash_total": 3,
    }
    rows = [
        dict(common, episode="episode_0000", trace_reached_a=True,
             online_max_covis=0.60, online_revisit_supported=True,
             online_revisit_strong=True,
             online_support_band="strong_support_ge_0p50"),
        dict(common, episode="episode_0001", trace_reached_a=False,
             online_max_covis=0.30, online_revisit_supported=True,
             online_revisit_strong=False,
             online_support_band="supported_0p20_to_0p50"),
        dict(common, episode="episode_0002", trace_reached_a=True,
             online_max_covis=0.05, online_revisit_supported=False,
             online_revisit_strong=False,
             online_support_band="no_support_lt_0p10"),
    ]
    summary = summarize_observability(rows)
    assert summary["episodes"] == 3
    assert summary["shared_a_successes"] == 2
    assert summary["online_supported_ge_0p20_all"] == 2
    assert summary["online_supported_ge_0p20_given_a_success"] == 1
    assert summary["online_strong_ge_0p50_given_a_success"] == 1
    assert summary["trace_render_hash_matches"] == 9
    assert summary["trace_render_hash_total"] == 9


def test_exact_mcnemar_matches_fresh_primary_discordance():
    assert exact_mcnemar(20, 4) == pytest.approx(0.001543879508972168)


def test_certified_run_contract_freezes_four_arm_williams_design():
    spec = run_contract_spec("certified_relocalization")
    assert spec["arms"] == (
        "certified_relocalization",
        "known_revisit_direct",
        "geometry_router",
        "native",
    )
    assert spec["orders"][0] == (
        "certified_relocalization",
        "known_revisit_direct",
        "native",
        "geometry_router",
    )
    assert spec["contrasts"]["certified_minus_native"] == (
        "native", "certified_relocalization")
    assert spec["scene_contract_schema"] == (
        "certified_relocalization_closed_loop_v1")


def test_unknown_run_contract_fails_closed():
    with pytest.raises(ValueError, match="unsupported run contract"):
        run_contract_spec("unknown")
