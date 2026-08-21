import pathlib
import json

import numpy as np

import pytest

from MemNavData.goat_autonomous_multigoal_pilot import (
    ACTION_IDS,
    GOAT_ADAPTER_CONFIG,
    _arrival_query_with_intrinsic,
    _collision_recovery_action_sequence,
    _critic_fallback_action_sequence,
    _depth_clearance_recovery_turn,
    _finite_optional,
    _image_stop_is_authorized,
    _is_image_task,
    _is_navdp_motion_source,
    _navdp_observation,
    _nonstop_official_fallback,
    _official_observation,
    _queues_after_navdp_collision,
    _search_decision_json,
    _sticky_collision_recovery,
    _terminal_action_id,
    _terminal_hints,
)
from MemNavData.goat_navdp_camera_adapter import (
    NAVDP_CAMERA_HEIGHT,
    NAVDP_CAMERA_WIDTH,
    NAVDP_RGB_SENSOR_UUID,
)
from MemNavData.goat_navdp_discrete_adapter import (
    navdp_waypoints_to_goat_decision,
)
from MemNavData.goat_autonomous_stop import (
    SearchDecision,
    SearchDisposition,
)


def decision(disposition, action=None):
    return SearchDecision(
        disposition=disposition,
        action=action,
        phase="test",
        reason="test",
        stop_decision={"authorized_subtask_stop": disposition is SearchDisposition.STOP},
    )


def test_image_modality_is_read_from_observable_task_interface():
    assert _is_image_task(["chair", "image", "chair_1", 4])
    assert not _is_image_task(["chair", "object", None])
    assert not _is_image_task([])


def test_terminal_hint_requires_accepted_certificate():
    accepted = {
        "ok": True,
        "accepted": True,
        "terminal_yaw_right_deg": -73.5,
        "terminal_pitch_up_deg": 18.0,
    }
    assert _terminal_hints(accepted) == (-73.5, 18.0)
    assert _terminal_hints(dict(accepted, accepted=False)) == (None, None)
    assert _terminal_hints(None) == (None, None)


def test_nonfinite_alignment_is_fail_closed():
    assert _finite_optional(float("nan")) is None
    assert _finite_optional(float("inf")) is None
    assert _finite_optional(True) is None
    assert _finite_optional("31.25") == 31.25


def test_only_visual_stop_decision_maps_to_subtask_stop():
    stop = decision(SearchDisposition.STOP)
    assert _terminal_action_id(stop) == ACTION_IDS["subtask_stop"]
    motion = decision(SearchDisposition.MOTION, "turn_right")
    assert _terminal_action_id(motion) == ACTION_IDS["turn_right"]
    with pytest.raises(ValueError):
        _terminal_action_id(decision(SearchDisposition.REPLAN))


def test_official_fallback_can_never_bypass_image_stop_gate():
    assert _nonstop_official_fallback(ACTION_IDS["subtask_stop"]) is None
    assert _nonstop_official_fallback(ACTION_IDS["stop"]) is None
    assert _nonstop_official_fallback(
        ACTION_IDS["move_forward"]) == ACTION_IDS["move_forward"]
    assert _image_stop_is_authorized(
        ACTION_IDS["subtask_stop"], "official_goat_image_exact_fallback")
    assert _image_stop_is_authorized(
        ACTION_IDS["subtask_stop"], "autonomous_visual_subtask_stop")
    assert not _image_stop_is_authorized(
        ACTION_IDS["subtask_stop"], "navdp_motion_chunk")


def test_search_decision_has_json_safe_enum_projection():
    payload = _search_decision_json(
        decision(SearchDisposition.MOTION, "turn_left"))
    assert payload["disposition"] == "motion"
    assert payload["action"] == "turn_left"


def test_runner_reads_metrics_only_after_env_step():
    source = pathlib.Path(__file__).with_name(
        "goat_autonomous_multigoal_pilot.py").read_text()
    marker = "next_observation = env.step(ACTION_NAMES[int(chosen_action)])"
    audit = "metrics = _jsonable(env.get_metrics())"
    assert source.index(marker) < source.index(audit)
    assert '"ground_truth_used_by_decision": False' in source


def test_runner_does_not_confuse_critic_threshold_with_metric_stop_radius():
    source = pathlib.Path(__file__).with_name(
        "goat_autonomous_multigoal_pilot.py").read_text()
    assert "navdp_stop_threshold, request_timeout_s" in source
    assert "adapter.endpoint_stop_radius_m, request_timeout_s" not in source


def test_goat_action_quantum_requires_replanning_after_every_action():
    assert GOAT_ADAPTER_CONFIG.forward_step_m == 0.25
    assert GOAT_ADAPTER_CONFIG.turn_angle_deg == 30.0
    assert GOAT_ADAPTER_CONFIG.lookahead_distance_m == 0.70
    assert GOAT_ADAPTER_CONFIG.execution_horizon == 1


def test_collision_recovery_turns_toward_observed_free_space():
    depth = np.ones((6, 8), dtype=np.float32)
    depth[:, :4] = 4.0
    action, receipt = _depth_clearance_recovery_turn(depth)
    assert action == ACTION_IDS["turn_left"]
    assert receipt["left_clearance_m"] == 4.0
    assert receipt["right_clearance_m"] == 1.0
    assert receipt["forward_probe_after_turns"] is True

    action, receipt = _depth_clearance_recovery_turn(depth[:, ::-1])
    assert action == ACTION_IDS["turn_right"]
    assert receipt["right_clearance_m"] == 4.0


def test_collision_recovery_direction_stays_fixed_until_motion_succeeds():
    first_depth = np.ones((6, 8), dtype=np.float32)
    first_depth[:, :4] = 4.0
    action, first_schedule, sticky = _sticky_collision_recovery(
        first_depth, None)
    assert action == ACTION_IDS["turn_left"]
    assert "sticky_direction_reused" not in first_schedule

    opposite_depth = first_depth[:, ::-1]
    action, second_schedule, retained = _sticky_collision_recovery(
        opposite_depth, sticky)
    assert action == ACTION_IDS["turn_left"]
    assert second_schedule["sticky_direction_reused"] is True
    assert retained == sticky


def test_collision_recovery_probes_forward_after_ninety_degree_turn():
    actions = _collision_recovery_action_sequence(ACTION_IDS["turn_right"])
    assert actions == [
        ACTION_IDS["turn_right"],
        ACTION_IDS["turn_right"],
        ACTION_IDS["turn_right"],
        ACTION_IDS["move_forward"],
    ]


def test_lateral_critic_fallback_is_one_coarse_search_primitive():
    decision = navdp_waypoints_to_goat_decision(
        np.array([[0.0, 1.0, 0.0]]), GOAT_ADAPTER_CONFIG)
    assert _critic_fallback_action_sequence(decision) == [
        ACTION_IDS["turn_left"],
        ACTION_IDS["turn_left"],
        ACTION_IDS["turn_left"],
        ACTION_IDS["move_forward"],
    ]


def test_collision_recovery_does_not_abort_terminal_search():
    sentinel_search = object()
    origin = {"controller": "native_imagegoal_navdp", "plan_index": 3}
    queued, search, fallback, retained_origin = _queues_after_navdp_collision(
        [ACTION_IDS["move_forward"], ACTION_IDS["turn_left"]],
        sentinel_search,
        [ACTION_IDS["turn_right"]],
        origin,
    )
    assert queued == []
    assert search is sentinel_search
    assert fallback == [ACTION_IDS["turn_right"]]
    assert retained_origin == origin


def test_collision_override_is_scoped_to_navdp_motion():
    assert _is_navdp_motion_source("native_imagegoal_navdp_motion")
    assert _is_navdp_motion_source("cec_bearing_plus_navdp_motion")
    assert _is_navdp_motion_source("terminal_rejected_resampled_motion")
    assert not _is_navdp_motion_source("official_goat_image_exact_fallback")
    assert not _is_navdp_motion_source("terminal_rejected_official_motion")


def test_adapter_sensor_is_hidden_from_frozen_official_goat_policy():
    observation = {
        "rgb": np.zeros((4, 3, 3), dtype=np.uint8),
        NAVDP_RGB_SENSOR_UUID: np.ones((2, 2, 3), dtype=np.uint8),
        "goat_subtask_goal": np.zeros(2, dtype=np.float32),
    }
    filtered = _official_observation(observation)
    assert NAVDP_RGB_SENSOR_UUID not in filtered
    assert set(filtered) == {"rgb", "goat_subtask_goal"}


def test_navdp_observation_requires_aligned_canonical_rgbd():
    rgb = np.zeros(
        (NAVDP_CAMERA_HEIGHT, NAVDP_CAMERA_WIDTH, 3), dtype=np.uint8)
    depth = np.ones(
        (NAVDP_CAMERA_HEIGHT, NAVDP_CAMERA_WIDTH, 1), dtype=np.float32)
    observed_rgb, observed_depth = _navdp_observation({
        NAVDP_RGB_SENSOR_UUID: rgb,
        "depth": depth,
    })
    assert observed_rgb is rgb
    assert observed_depth is depth

    with pytest.raises(RuntimeError, match="RGB shape"):
        _navdp_observation({
            NAVDP_RGB_SENSOR_UUID: rgb[:100],
            "depth": depth,
        })


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.payload)


def test_arrival_query_sends_distinct_goal_intrinsic():
    session = _Session({
        "goal_camera_calibration": "explicit_distinct_intrinsic",
        "simulator_depth_consumed": False,
    })
    intrinsic = np.array([
        [250.0, 0.0, 128.0],
        [0.0, 250.0, 128.0],
        [0.0, 0.0, 1.0],
    ])
    goal = np.zeros((8, 8, 3), dtype=np.uint8)

    payload, latency = _arrival_query_with_intrinsic(
        session, "http://127.0.0.1:1234", goal, intrinsic, timeout_s=3.0)

    assert payload["goal_camera_calibration"] == "explicit_distinct_intrinsic"
    assert latency >= 0.0
    url, kwargs = session.calls[0]
    assert url.endswith("/arrival_query")
    np.testing.assert_allclose(
        json.loads(kwargs["data"]["goal_camera_intrinsic"]), intrinsic)


def test_arrival_query_fails_if_server_silently_uses_shared_intrinsic():
    session = _Session({
        "goal_camera_calibration": "legacy_shared_history_intrinsic",
        "simulator_depth_consumed": False,
    })
    with pytest.raises(RuntimeError, match="ignored GOAT goal intrinsic"):
        _arrival_query_with_intrinsic(
            session, "http://localhost", np.zeros((4, 4, 3), dtype=np.uint8),
            np.eye(3), timeout_s=1.0)
