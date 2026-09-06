from NavDP.baselines.memnav.policy_agent import MemNavAgent


def test_causal_metric_scale_receipt_is_read_only_composition():
    agent = object.__new__(MemNavAgent)
    agent.n = 64
    first40 = {"available": True, "metric_scale_m_per_raw": 2.1}
    first64 = {"available": True, "metric_scale_m_per_raw": 2.2}
    agent._first40_local_pose_metric_scale = lambda: first40
    agent._strict_arrival_metric_scale_preserving_stream = lambda: first64

    result = agent.causal_metric_scale_receipt()

    assert result == {
        "schema_version": "lingbot_causal_metric_scale_v1_20260902",
        "stream_observation_count": 64,
        "first40": first40,
        "first64": first64,
        "runtime_evaluator_pose_visible": False,
        "metric_depth_sensor_consumed": False,
    }
