from NavDP.baselines.memnav.policy_agent import MemNavAgent


def _valid_receipt():
    return {
        "schema": "mdtec_first40_scale_receipt_v1_20260819",
        "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
        "scale_prefix_frames": 40,
        "scale_prefix_first_frame": 0,
        "scale_prefix_last_frame": 39,
        "frozen_after_observation_count": 40,
        "active_from_frame_index": 40,
        "whole_episode_ground_cache_consumed": False,
        "camera_height_m": 0.42,
        "ground_h_est_raw": 0.15,
        "scale_valid": True,
        "scale_hat": 3.22,
        "valid_frame_ratio": 1.0,
        "relative_floor_iqr": 0.01,
        "scale_clamped": False,
        "freeze_error": None,
    }


def test_first40_local_pose_scale_reuses_valid_frozen_receipt():
    agent = object.__new__(MemNavAgent)
    agent.n = 90
    agent._first40_scale_receipt = _valid_receipt()

    result = agent._first40_local_pose_metric_scale()

    assert result["available"] is True
    assert result["metric_scale_m_per_raw"] == 3.22
    assert result["frame_count"] == 40
    assert len(result["scale_receipt_sha256"]) == 64


def test_first40_local_pose_scale_fails_closed_on_corrupt_receipt():
    agent = object.__new__(MemNavAgent)
    agent.n = 90
    agent._first40_scale_receipt = _valid_receipt()
    agent._first40_scale_receipt["whole_episode_ground_cache_consumed"] = True

    result = agent._first40_local_pose_metric_scale()

    assert result["available"] is False
    assert result["reason"] == "mdtec_first40_scale_invalid"
