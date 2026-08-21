import numpy as np
import pytest

from MemNavData.learned_relocalizer_contract import (
    LearnedPairPrediction,
    query_camera_to_world,
    scale_free_bearing_from_pairwise,
    shadow_contract,
)


IDENTITY_POSE9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def prediction(translation):
    return LearnedPairPrediction(
        model_id="unit-test",
        status="ok",
        rotation_reference_to_query=np.eye(3),
        translation_reference_to_query_m=translation,
        support_score=0.8,
        solver_support=42.0,
        latency_ms=1.0,
        reason="pose_estimated",
    )


def test_forward_query_center_becomes_forward_navdp_bearing():
    # A query camera centred 2 m in front of the reference obeys
    # X_query = X_reference + [0, 0, -2].
    output = scale_free_bearing_from_pairwise(
        IDENTITY_POSE9, IDENTITY_POSE9, prediction([0.0, 0.0, -2.0]),
        meters_per_history_unit=2.0)
    assert output == pytest.approx([1.0, 0.0])


def test_right_query_center_uses_negative_navdp_left_axis():
    output = scale_free_bearing_from_pairwise(
        IDENTITY_POSE9, IDENTITY_POSE9, prediction([-1.0, 0.0, 0.0]),
        meters_per_history_unit=1.0)
    assert output == pytest.approx([0.0, -1.0])


def test_reference_history_pose_and_metric_scale_are_composed():
    reference = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    query = query_camera_to_world(
        reference, np.eye(3), [0.0, 0.0, -2.0],
        meters_per_history_unit=2.0)
    assert query[:3, 3] == pytest.approx([0.0, 0.0, 4.0])


def test_abstention_cannot_leak_a_pose_or_bearing():
    abstain = LearnedPairPrediction(
        model_id="unit-test", status="abstain",
        rotation_reference_to_query=None,
        translation_reference_to_query_m=None,
        support_score=0.0, solver_support=0.0,
        latency_ms=1.0, reason="insufficient_support")
    abstain.validated()
    with pytest.raises(ValueError):
        scale_free_bearing_from_pairwise(
            IDENTITY_POSE9, IDENTITY_POSE9, abstain,
            meters_per_history_unit=1.0)


def test_malformed_rotation_fails_closed():
    malformed = LearnedPairPrediction(
        model_id="unit-test", status="ok",
        rotation_reference_to_query=np.ones((3, 3)),
        translation_reference_to_query_m=[0.0, 0.0, -1.0],
        support_score=1.0, solver_support=1.0,
        latency_ms=1.0, reason="pose_estimated")
    with pytest.raises(ValueError):
        malformed.validated()


def test_shadow_contract_never_reads_role_or_controls_navigation():
    contract = shadow_contract()
    assert contract["runtime_role_labels"] is False
    assert contract["runtime_ground_truth"] is False
    assert contract["shadow_only"] is True
    assert contract["candidate_universe"] == "frozen_dino_top8"
