import hashlib
import json

import pytest
import requests

from MemNavData.cec_controller_portability_hub import (
    CecControllerPortabilityRouter,
    PortabilityHubConfig,
    PortabilityHubError,
)
from MemNavData.monocular_depth_runtime import build_monocular_depth_payload


class FakeResponse:
    def __init__(self, payload=None, *, content=b"", headers=None, status=200):
        self.payload = payload
        self.content = content
        self.headers = dict(headers or {})
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if callable(response):
            return response(url, kwargs)
        if isinstance(response, Exception):
            raise response
        return response


def config(controller):
    return PortabilityHubConfig(
        controller=controller,
        memnav_url="http://mem",
        controller_url=(
            "http://fallback" if controller == "navdp" else "http://controller"),
        fallback_navdp_url="http://fallback",
        camera_height_m=0.5,
    )


def memory_reset():
    return FakeResponse({
        "algo": "memnav",
        "certified_relocalization": {"enabled": True},
    })


def fallback_reset():
    return FakeResponse({
        "algo": "navdp",
        "depth_source": "monocular_sidecar",
        "metric_depth_sensor_consumed_by_config": False,
        "monocular_depth_url_configured": True,
    })


def controller_reset(controller, adapter):
    return FakeResponse({
        "algo": controller,
        "portability_receipt": {
            "controller": controller,
            "comparison_protocol": "cec_proof_hybrid",
            "cec_accept_adapter": adapter,
        },
    })


def reset(router):
    return router.reset({
        "intrinsic": [
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 40.0],
            [0.0, 0.0, 1.0],
        ],
        "batch_size": 1,
        "stop_threshold": -2.0,
    })


def trajectory(**extra):
    return {
        "trajectory": [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]],
        "all_trajectory": [],
        "all_values": [],
        **extra,
    }


def accepted_certificate(anchor=12, direction=(3.0, 4.0), **extra):
    return FakeResponse({
        "certified_relocalization_schema_version": 2,
        "frame_idx": 40,
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": anchor,
        "direction_vector": list(direction),
        "aux_pose": list(direction),
        "pointgoal_units": "lingbot_raw_direction_only",
        "certificate": {"accepted": True},
        **extra,
    })


def echo_proof(_url, kwargs):
    return FakeResponse(trajectory(
        cec_proof_sha256=kwargs["data"]["cec_proof_sha256"]))


def depth_response(image):
    payload = build_monocular_depth_payload(
        relative_depth=None,
        depth_shape=(4, 5),
        image_sha256_value=hashlib.sha256(image).hexdigest(),
        frame_index=3,
        scale_receipt=None,
    )
    return FakeResponse(payload)


def test_reject_uses_shared_mono_navdp_and_rechecks_next_action():
    native = trajectory(
        depth_source="monocular_sidecar",
        metric_depth_sensor_consumed=False,
    )
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset("iplanner", "bearing_pointgoal"),
        FakeResponse({
            "frame_idx": 40, "certified_visual_candidates": [],
            "goal_session_index": 1, "goal_session_started": True,
            "long_term_memory_preserved": True,
            "goal_start_frame": 40, "candidate_ceiling": 39,
        }),
        FakeResponse({
            "ok": True, "accepted": False, "reason": "no_causal_candidate"}),
        FakeResponse(native),
        FakeResponse({
            "frame_idx": 41, "certified_visual_candidates": [],
            "goal_session_index": 1, "goal_session_started": False,
            "long_term_memory_preserved": True,
            "goal_start_frame": 40, "candidate_ceiling": 39,
        }),
        FakeResponse({
            "ok": True, "accepted": False, "reason": "still_unsupported"}),
        FakeResponse(native),
    ])
    router = CecControllerPortabilityRouter(
        config("iplanner"), session=session)
    reset(router)
    first = router.plan_imagegoal(image=b"i1", goal=b"g")
    second = router.plan_imagegoal(image=b"i2", goal=b"g")
    assert first["cec_action_state"] == "fallback"
    assert second["cec_action_state"] == "fallback"
    assert first["cec_decision_scope"] == "per_action"
    assert first["cec_controller_seed_consumed"] is True
    assert first["metric_depth_sensor_consumed"] is False
    assert first["cec_controller_ms"] >= 0.0
    assert first["cec_total_decision_ms"] >= first["cec_controller_ms"]
    assert first["cec_goal_session_expected_start"] is True
    assert first["cec_goal_session_started"] is True
    assert first["cec_goal_session_index"] == 1
    assert first["cec_goal_start_frame"] == 40
    assert first["cec_candidate_ceiling"] == 39
    assert first["memory_frame_idx"] == first["cec_frame_idx"] == 40
    assert first["cec_long_term_memory_preserved"] is True
    assert second["cec_goal_session_expected_start"] is False
    assert second["cec_goal_session_started"] is False
    assert [url for url, _ in session.calls].count(
        "http://mem/certified_relocalize") == 2
    assert [url for url, _ in session.calls[-3:]] == [
        "http://mem/retrieval_probe_step",
        "http://mem/certified_relocalize",
        "http://fallback/imagegoal_step",
    ]


def test_goal_session_replay_restores_boundary_without_inference():
    session = FakeSession([
        memory_reset(), fallback_reset(),
        FakeResponse({
            "goal_session_index": 1,
            "goal_session_started": True,
            "long_term_memory_preserved": True,
            "goal_start_frame": 80,
            "candidate_ceiling": 79,
            "frame_count": 80,
            "diffusion_sampled": False,
            "memory_appended": False,
        }),
    ])
    router = CecControllerPortabilityRouter(
        config("navdp"), session=session)
    reset(router)
    result = router.replay_goal_session(b"goal-c", 80)
    assert result["cec_goal_session_replayed"] is True
    assert result["cec_query_index"] == 1
    assert result["goal_start_frame"] == 80
    assert result["diffusion_sampled"] is False
    assert [url for url, _ in session.calls][-1] == (
        "http://mem/goal_session_replay")


def test_goal_session_replay_fails_closed_on_mutating_receipt():
    session = FakeSession([
        memory_reset(), fallback_reset(),
        FakeResponse({
            "goal_session_index": 1,
            "goal_session_started": True,
            "goal_start_frame": 80,
            "diffusion_sampled": False,
            "memory_appended": True,
        }),
    ])
    router = CecControllerPortabilityRouter(
        config("navdp"), session=session)
    reset(router)
    with pytest.raises(PortabilityHubError):
        router.replay_goal_session(b"goal-c", 80)
    assert router.reset_required is True


def test_iplanner_accept_uses_same_proof_bearing_and_lingbot_depth():
    image = b"current-rgb"
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset("iplanner", "bearing_pointgoal"),
        FakeResponse({
            "frame_idx": 40,
            "certified_visual_candidates": [{"anchor": 12, "score": 0.9}],
        }),
        accepted_certificate(),
        depth_response(image),
        echo_proof,
        FakeResponse({"algo": "navdp", "diffusion_sampled": False}),
    ])
    router = CecControllerPortabilityRouter(
        config("iplanner"), session=session)
    reset(router)
    result = router.plan_imagegoal(
        image=image, goal=b"goal", form={"diffusion_seed": "123"})
    assert result["cec_action_state"] == "takeover"
    assert result["cec_accept_controller"] == "iplanner"
    assert result["cec_fallback_context_shadowed"] is True
    assert result["cec_controller_seed_consumed"] is False
    assert result["metric_depth_sensor_consumed"] is False
    assert result["cec_depth_sidecar_ms"] >= 0.0
    assert result["cec_total_decision_ms"] >= result["cec_controller_ms"]
    assert result["diffusion_seed"] == 123
    assert result["cec_projected_goal"]["goal_x"] == [1.5]
    assert result["cec_projected_goal"]["goal_y"] == [2.0]
    assert result["cec_controller_portability_receipt"] is None
    assert result["monocular_depth_receipt"]["scale_state"] == (
        "bootstrap_zero_depth")
    url, kwargs = session.calls[-2]
    assert url == "http://controller/pointgoal_step"
    assert json.loads(kwargs["data"]["goal_data"]) == {
        "goal_x": [1.5], "goal_y": [2.0],
    }
    assert set(kwargs["files"]) == {"image", "depth"}


@pytest.mark.parametrize("controller", ["vint", "gnm", "nomad"])
def test_short_context_controller_receives_only_the_proof_bound_history_anchor(
        controller):
    anchor_image = b"certified-history-anchor"
    anchor_sha = hashlib.sha256(anchor_image).hexdigest()
    binary = FakeResponse(
        content=anchor_image,
        headers={
            "X-CEC-Anchor-Index": "12",
            "X-CEC-Anchor-SHA256": anchor_sha,
        },
    )
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset(controller, "verified_anchor_imagegoal"),
        FakeResponse({
            "frame_idx": 40,
            "certified_visual_candidates": [{"anchor": 12, "score": 0.9}],
        }),
        accepted_certificate(selected_anchor_image_sha256=anchor_sha),
        binary,
        echo_proof,
        FakeResponse({"algo": "navdp", "diffusion_sampled": False}),
    ])
    router = CecControllerPortabilityRouter(
        config(controller), session=session)
    reset(router)
    result = router.plan_imagegoal(image=b"current", goal=b"original-goal")
    assert result["cec_takeover"] is True
    url, kwargs = session.calls[-2]
    assert url == "http://controller/imagegoal_step"
    assert kwargs["files"]["goal"][1].read() == anchor_image
    assert "depth" not in kwargs["files"]
    assert kwargs["data"]["goal_source"] == "certified_history_anchor"


def test_takeover_can_safely_fall_back_on_the_next_action():
    image = b"current-rgb"
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset("iplanner", "bearing_pointgoal"),
        FakeResponse({"frame_idx": 40, "certified_visual_candidates": []}),
        accepted_certificate(), depth_response(image), echo_proof,
        FakeResponse({"algo": "navdp", "diffusion_sampled": False}),
        FakeResponse({"frame_idx": 41, "certified_visual_candidates": []}),
        FakeResponse({"ok": True, "accepted": False, "reason": "changed"}),
        FakeResponse(trajectory(
            depth_source="monocular_sidecar",
            metric_depth_sensor_consumed=False,
        )),
    ])
    router = CecControllerPortabilityRouter(
        config("iplanner"), session=session)
    reset(router)
    first = router.plan_imagegoal(image=image, goal=b"goal")
    second = router.plan_imagegoal(image=b"next", goal=b"goal")
    assert first["cec_action_state"] == "takeover"
    assert first["cec_fallback_context_shadowed"] is True
    assert second["cec_action_state"] == "fallback"
    assert router.reset_required is False
    assert session.calls[-1][0] == "http://fallback/imagegoal_step"


def test_new_goal_reopens_cec_without_resetting_episode_history():
    first_native = trajectory(
        depth_source="monocular_sidecar",
        metric_depth_sensor_consumed=False,
    )
    second_image = b"goal-b-current"
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset("iplanner", "bearing_pointgoal"),
        FakeResponse({"frame_idx": 40, "certified_visual_candidates": []}),
        FakeResponse({"ok": True, "accepted": False, "reason": "novel"}),
        FakeResponse(first_native),
        FakeResponse({
            "frame_idx": 41,
            "certified_visual_candidates": [{"anchor": 12, "score": 0.9}],
        }),
        accepted_certificate(), depth_response(second_image), echo_proof,
        FakeResponse({"algo": "navdp", "diffusion_sampled": False}),
    ])
    router = CecControllerPortabilityRouter(
        config("iplanner"), session=session)
    reset(router)
    first = router.plan_imagegoal(image=b"goal-a-current", goal=b"goal-a")
    second = router.plan_imagegoal(image=second_image, goal=b"goal-b")
    assert first["cec_query_index"] == 1
    assert first["cec_action_state"] == "fallback"
    assert second["cec_query_index"] == 2
    assert second["cec_action_state"] == "takeover"
    assert [url for url, _ in session.calls].count(
        "http://mem/certified_relocalize") == 2


@pytest.mark.parametrize("controller", ["vint", "gnm", "nomad"])
def test_short_context_controller_fallback_keeps_context_warm_without_action(
        controller):
    native = trajectory(
        depth_source="monocular_sidecar",
        metric_depth_sensor_consumed=False,
    )
    shadow = FakeResponse({
        "algo": controller,
        "observed": True,
        "portability_receipt": {
            "controller": controller,
            "endpoint": "observation_step",
        },
    })
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset(controller, "verified_anchor_imagegoal"),
        FakeResponse({"frame_idx": 40, "certified_visual_candidates": []}),
        FakeResponse({"ok": True, "accepted": False, "reason": "novel"}),
        FakeResponse(native), shadow,
    ])
    router = CecControllerPortabilityRouter(
        config(controller), session=session)
    reset(router)
    result = router.plan_imagegoal(image=b"current", goal=b"novel")
    assert result["cec_action_state"] == "fallback"
    assert session.calls[-1][0] == "http://controller/observation_step"
    assert set(session.calls[-1][1]["files"]) == {"image"}


def test_memory_step_replays_only_into_long_term_cec_history():
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset("iplanner", "bearing_pointgoal"),
        FakeResponse({"ok": True, "frame_idx": 17}),
    ])
    router = CecControllerPortabilityRouter(
        config("iplanner"), session=session)
    reset(router)
    result = router.memory_step(b"causal-history-rgb")
    assert result == {"ok": True, "frame_idx": 17}
    assert session.calls[-1][0] == "http://mem/memory_step"
    assert set(session.calls[-1][1]["files"]) == {"image"}


@pytest.mark.parametrize("controller", ["vint", "gnm", "nomad"])
def test_controller_replay_updates_fallback_and_short_context_without_sampling(
        controller):
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset(controller, "verified_anchor_imagegoal"),
        FakeResponse({
            "algo": "navdp",
            "diffusion_sampled": False,
            "history_frame_count": 8,
        }),
        FakeResponse({
            "algo": controller,
            "observed": True,
            "portability_receipt": {
                "controller": controller,
                "endpoint": "observation_step",
            },
        }),
    ])
    router = CecControllerPortabilityRouter(
        config(controller), session=session)
    reset(router)
    result = router.controller_memory_replay(b"decision-rgb")
    assert result["diffusion_sampled"] is False
    assert result["alternate_context_shadowed"] is True
    assert [url for url, _kwargs in session.calls[-2:]] == [
        "http://fallback/memory_replay_step",
        "http://controller/observation_step",
    ]


def test_short_reset_clears_anchor_cache_but_preserves_active_goal_session():
    session = FakeSession([
        memory_reset(), fallback_reset(),
        controller_reset("iplanner", "bearing_pointgoal"),
        FakeResponse({"algo": "navdp"}),
        FakeResponse({"algo": "iplanner"}),
    ])
    router = CecControllerPortabilityRouter(
        config("iplanner"), session=session)
    reset(router)
    router._goal_sha256 = hashlib.sha256(b"old-goal").hexdigest()
    router._anchor_jpeg = b"old-anchor"
    result = router.reset_short_context(0)
    assert result["long_term_cec_history_preserved"] is True
    assert result["active_goal_session_preserved"] is True
    assert router._goal_sha256 == hashlib.sha256(b"old-goal").hexdigest()
    assert router._anchor_jpeg is None
    assert router.last_action_state == "unresolved"
    assert all(url != "http://mem/navigator_reset"
               for url, _kwargs in session.calls[3:])


def forced_config(controller="navdp"):
    return PortabilityHubConfig(
        controller=controller,
        memnav_url="http://mem",
        controller_url="http://fallback",
        fallback_navdp_url="http://fallback",
        camera_height_m=0.5,
        force_reject_native=True,
    )


def test_forced_reject_native_never_takes_over_on_accept():
    native = trajectory(
        depth_source="monocular_sidecar",
        metric_depth_sensor_consumed=False,
    )
    session = FakeSession([
        memory_reset(), fallback_reset(),
        FakeResponse({
            "frame_idx": 40, "certified_visual_candidates": [{"anchor": 12}],
            "goal_session_index": 1, "goal_session_started": True,
            "long_term_memory_preserved": True,
            "goal_start_frame": 40, "candidate_ceiling": 39,
        }),
        accepted_certificate(),
        FakeResponse(native),
    ])
    router = CecControllerPortabilityRouter(forced_config(), session=session)
    receipt = reset(router)
    assert receipt["force_reject_native"] is True
    result = router.plan_imagegoal(image=b"i1", goal=b"g")
    # The certificate accepted, but authority stays with the shared fallback.
    assert result["cec_takeover"] is False
    assert result["cec_shadow_takeover"] is True
    assert result["cec_forced_reject_native"] is True
    assert result["cec_action_state"] == "forced_reject"
    # The action came from the shared mono-NavDP fallback endpoint.
    assert session.calls[-1][0] == "http://fallback/imagegoal_step"
    # Session receipts survive unchanged for the memory-recording audit.
    assert result["cec_goal_session_started"] is True
    assert result["cec_candidate_ceiling"] == 39


def test_forced_reject_native_plain_reject_stays_fallback():
    native = trajectory(
        depth_source="monocular_sidecar",
        metric_depth_sensor_consumed=False,
    )
    session = FakeSession([
        memory_reset(), fallback_reset(),
        FakeResponse({
            "frame_idx": 40, "certified_visual_candidates": [],
            "goal_session_index": 1, "goal_session_started": True,
            "long_term_memory_preserved": True,
            "goal_start_frame": 40, "candidate_ceiling": 39,
        }),
        FakeResponse({
            "ok": True, "accepted": False, "reason": "no_causal_candidate"}),
        FakeResponse(native),
    ])
    router = CecControllerPortabilityRouter(forced_config(), session=session)
    reset(router)
    result = router.plan_imagegoal(image=b"i1", goal=b"g")
    assert result["cec_takeover"] is False
    assert result["cec_shadow_takeover"] is False
    assert result["cec_action_state"] == "fallback"
    assert result["cec_forced_reject_native"] is True


def test_default_mode_reports_shadow_equal_to_takeover():
    native = trajectory(
        depth_source="monocular_sidecar",
        metric_depth_sensor_consumed=False,
    )
    session = FakeSession([
        memory_reset(), fallback_reset(),
        FakeResponse({
            "frame_idx": 40, "certified_visual_candidates": [],
            "goal_session_index": 1, "goal_session_started": True,
            "long_term_memory_preserved": True,
            "goal_start_frame": 40, "candidate_ceiling": 39,
        }),
        FakeResponse({
            "ok": True, "accepted": False, "reason": "no_causal_candidate"}),
        FakeResponse(native),
    ])
    router = CecControllerPortabilityRouter(config("navdp"), session=session)
    receipt = reset(router)
    assert receipt["force_reject_native"] is False
    result = router.plan_imagegoal(image=b"i1", goal=b"g")
    assert result["cec_shadow_takeover"] == result["cec_takeover"] is False
    assert result["cec_forced_reject_native"] is False
