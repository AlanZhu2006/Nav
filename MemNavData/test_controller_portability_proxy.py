from pathlib import Path
import hashlib

import pytest

from controller_portability_contract import (
    CEC_BEARING_EXECUTOR,
    CEC_PROOF_HYBRID,
    NATIVE_IMAGEGOAL,
    ComparisonPlan,
)
from controller_portability_proxy import (
    ControllerPortabilityProxy,
    ProxyConfig,
    create_app,
    parse_checkpoint_arguments,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return dict(self.payload)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return FakeResponse(self.responses.pop(0))


def trajectory_payload():
    trajectory = [[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]]
    return {
        "trajectory": trajectory,
        "all_trajectory": [trajectory],
        "all_values": [[0.5]],
    }


def checkpoint(tmp_path, name):
    path = tmp_path / name
    path.write_bytes((name + "-frozen").encode())
    return path


RGB_ONLY_CONTROLLERS = frozenset({"vint", "gnm", "nomad"})


def make_proxy(tmp_path, controller="iplanner", session=None):
    if controller in RGB_ONLY_CONTROLLERS:
        comparison = ComparisonPlan(
            controller=controller,
            protocol=NATIVE_IMAGEGOAL,
            depth_source="none",
            query_population="mixed_role",
            reject_policy="not_applicable",
        )
        checkpoints = {controller: checkpoint(tmp_path, f"{controller}.pth")}
    elif controller == "viplanner":
        comparison = ComparisonPlan(
            controller="viplanner",
            protocol=CEC_BEARING_EXECUTOR,
            depth_source="metric_sensor",
            query_population="revisit_only",
            reject_policy="score_uncovered",
        )
        checkpoints = {
            "planner": checkpoint(tmp_path, "viplanner.pt"),
            "mask2former": checkpoint(tmp_path, "mask2former.pth"),
        }
    else:
        comparison = ComparisonPlan(
            controller="iplanner",
            protocol=CEC_BEARING_EXECUTOR,
            depth_source="metric_sensor",
            query_population="revisit_only",
            reject_policy="score_uncovered",
        )
        checkpoints = {"iplanner": checkpoint(tmp_path, "iplanner.pth")}
    return ControllerPortabilityProxy(ProxyConfig(
        comparison=comparison,
        repo_root=ROOT,
        upstream_base="http://127.0.0.1:19999",
        checkpoints=checkpoints,
        timeout_s=3.0,
    ), session=session or FakeSession([]))


def make_hybrid_proxy(
        tmp_path, controller, session=None,
        reject_policy="shared_native_exact"):
    depth = "none" if controller in RGB_ONLY_CONTROLLERS else "metric_sensor"
    checkpoints = (
        {controller: checkpoint(tmp_path, f"{controller}-hybrid.pth")}
        if controller in RGB_ONLY_CONTROLLERS
        else {"iplanner": checkpoint(tmp_path, "iplanner-hybrid.pth")}
    )
    return ControllerPortabilityProxy(ProxyConfig(
        comparison=ComparisonPlan(
            controller=controller,
            protocol=CEC_PROOF_HYBRID,
            depth_source=depth,
            query_population="mixed_role",
            reject_policy=reject_policy,
            fallback_controller=(
                controller
                if reject_policy == "controller_native_exact" else "navdp"),
        ),
        repo_root=ROOT,
        upstream_base="http://127.0.0.1:19999",
        checkpoints=checkpoints,
        timeout_s=3.0,
    ), session=session or FakeSession([]))


def request_files(include_goal=False):
    files = {
        "image": ("image.jpg", b"rgb", "image/jpeg"),
        "depth": ("depth.png", b"depth", "image/png"),
    }
    if include_goal:
        files["goal"] = ("goal.jpg", b"goal", "image/jpeg")
    return files


def test_health_binds_source_checkpoint_and_non_privileged_contract(tmp_path):
    proxy = make_proxy(tmp_path)
    health = proxy.health()
    assert health["controller"] == "iplanner"
    assert health["checkpoint_sha256"]["iplanner"]
    assert health["local_source_tree_sha256"]
    assert health["role_label_visible"] is False
    assert health["uses_oracle_pose"] is False


def test_reset_checks_upstream_identity_and_adds_receipt(tmp_path):
    session = FakeSession([{"algo": "iplanner"}])
    proxy = make_proxy(tmp_path, session=session)
    result = proxy.reset({"intrinsic": [[1.0]], "batch_size": 1})
    assert result["portability_receipt"]["upstream_algo"] == "iplanner"
    assert session.calls[0][0].endswith("/navigator_reset")


def test_reset_rejects_runtime_role_and_wrong_upstream(tmp_path):
    proxy = make_proxy(tmp_path, session=FakeSession([{"algo": "wrong"}]))
    with pytest.raises(ValueError, match="role"):
        proxy.reset({"role": "revisit"})
    with pytest.raises(ValueError, match="expected 'iplanner'"):
        proxy.reset({"batch_size": 1})


def test_pointgoal_forwards_only_frozen_radius_and_valid_trajectory(tmp_path):
    session = FakeSession([trajectory_payload()])
    proxy = make_proxy(tmp_path, session=session)
    result = proxy.step(
        "pointgoal_step",
        files=request_files(),
        form={"goal_data": '{"goal_x":[1.5],"goal_y":[2.0]}'},
    )
    receipt = result["portability_receipt"]
    assert receipt["pointgoal_frame"] == "forward_left"
    assert receipt["pointgoal_radius_m"] == 2.5
    assert receipt["step_count"] == 1

    with pytest.raises(ValueError, match="frozen radius"):
        proxy.step(
            "pointgoal_step",
            files=request_files(),
            form={"goal_data": '{"goal_x":[1.0],"goal_y":[0.0]}'},
        )


def test_pointgoal_rejects_privileged_fields_and_malformed_output(tmp_path):
    proxy = make_proxy(tmp_path, session=FakeSession([{
        "trajectory": [[[float("nan"), 0.0, 0.0]]],
        "all_trajectory": [[[[0.0, 0.0, 0.0]]]],
        "all_values": [[0.0]],
    }]))
    with pytest.raises(ValueError, match="privileged"):
        proxy.step(
            "pointgoal_step",
            files=request_files(),
            form={
                "goal_data": '{"goal_x":[1.5],"goal_y":[2.0]}',
                "query_role": "revisit",
            },
        )
    with pytest.raises(ValueError, match="finite"):
        proxy.step(
            "pointgoal_step",
            files=request_files(),
            form={"goal_data": '{"goal_x":[1.5],"goal_y":[2.0]}'},
        )


@pytest.mark.parametrize("controller", ["vint", "gnm", "nomad"])
def test_rgb_only_native_imagegoal_is_allowed_but_pointgoal_is_not(
        tmp_path, controller):
    session = FakeSession([trajectory_payload()])
    proxy = make_proxy(tmp_path, controller=controller, session=session)
    result = proxy.step(
        "imagegoal_step",
        files=request_files(include_goal=True),
        form={},
    )
    assert result["portability_receipt"]["controller"] == controller
    assert result["portability_receipt"]["pointgoal_radius_m"] is None
    with pytest.raises(ValueError, match="does not support pointgoal"):
        proxy.step(
            "pointgoal_step",
            files=request_files(),
            form={"goal_data": '{"goal_x":[1.5],"goal_y":[2.0]}'},
        )


def test_checkpoint_labels_are_atomic_and_paths_are_real(tmp_path):
    one = checkpoint(tmp_path, "one.pt")
    parsed = parse_checkpoint_arguments([f"planner={one}"])
    assert parsed == {"planner": one.resolve()}
    with pytest.raises(ValueError, match="duplicate"):
        parse_checkpoint_arguments([f"planner={one}", f"planner={one}"])
    with pytest.raises(ValueError, match="LABEL"):
        parse_checkpoint_arguments([str(one)])


def test_flask_health_smoke_does_not_touch_upstream(tmp_path):
    proxy = make_proxy(tmp_path)
    client = create_app(proxy).test_client()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True


def cec_form(anchor=12):
    return {
        "cec_proof_sha256": "a" * 64,
        "cec_action_authorized": "1",
        "cec_selected_anchor": str(anchor),
    }


def test_cec_iplanner_proxy_requires_proof_and_action_authorization(tmp_path):
    proxy = make_hybrid_proxy(
        tmp_path, "iplanner", FakeSession([trajectory_payload()]))
    form = {
        **cec_form(),
        "goal_data": '{"goal_x":[1.5],"goal_y":[2.0]}',
    }
    result = proxy.step("pointgoal_step", files=request_files(), form=form)
    assert result["cec_proof_sha256"] == "a" * 64
    assert result["portability_receipt"]["fallback_controller"] == "navdp"
    assert result["portability_receipt"]["cec_accept_adapter"] == (
        "bearing_pointgoal")

    with pytest.raises(ValueError, match="authorization"):
        proxy.step(
            "pointgoal_step", files=request_files(),
            form={**form, "cec_action_authorized": "0"})


@pytest.mark.parametrize("controller", ["vint", "gnm", "nomad"])
def test_cec_rgb_only_proxy_accepts_only_hash_bound_history_anchor(
        tmp_path, controller):
    anchor = b"certified-anchor"
    anchor_sha = hashlib.sha256(anchor).hexdigest()
    files = request_files(include_goal=True)
    files.pop("depth")
    files["goal"] = ("anchor.jpg", anchor, "image/jpeg")
    form = {
        **cec_form(),
        "cec_anchor_sha256": anchor_sha,
        "goal_source": "certified_history_anchor",
    }
    proxy = make_hybrid_proxy(
        tmp_path, controller, FakeSession([trajectory_payload()]))
    result = proxy.step("imagegoal_step", files=files, form=form)
    assert result["portability_receipt"]["cec_accept_adapter"] == (
        "verified_anchor_imagegoal")

    with pytest.raises(ValueError, match="do not match"):
        proxy.step(
            "imagegoal_step", files=files,
            form={**form, "cec_anchor_sha256": "b" * 64})


@pytest.mark.parametrize("controller", ["vint", "gnm", "nomad"])
def test_cec_rgb_only_shadow_observation_advances_context_without_goal(
        tmp_path, controller):
    proxy = make_hybrid_proxy(
        tmp_path, controller,
        FakeSession([{"algo": controller, "observed": True}]))
    result = proxy.observe(
        files={"image": ("image.jpg", b"rgb", "image/jpeg")},
        form={},
    )
    assert result["observed"] is True
    assert result["portability_receipt"]["endpoint"] == "observation_step"
    assert result["portability_receipt"]["observation_count"] == 1
    with pytest.raises(ValueError, match="exactly one image"):
        proxy.observe(files=request_files(), form={})


def test_vint_controller_native_fallback_uses_original_goal_without_proof(
        tmp_path):
    session = FakeSession([trajectory_payload()])
    proxy = make_hybrid_proxy(
        tmp_path, "vint", session,
        reject_policy="controller_native_exact")
    files = request_files(include_goal=True)
    files.pop("depth")
    result = proxy.step("imagegoal_step", files=files, form={})
    receipt = result["portability_receipt"]
    assert receipt["controller"] == "vint"
    assert receipt["reject_policy"] == "controller_native_exact"
    assert receipt["fallback_controller"] == "vint"
    assert "cec_proof_sha256" not in result

    with pytest.raises(ValueError, match="authorization"):
        proxy.step(
            "imagegoal_step", files=files,
            form={"cec_action_authorized": "0"})


def test_cec_observation_step_is_refused_for_non_rgb_only_controllers(
        tmp_path):
    proxy = make_hybrid_proxy(
        tmp_path, "iplanner", FakeSession([]))
    with pytest.raises(ValueError, match="not available"):
        proxy.observe(
            files={"image": ("image.jpg", b"rgb", "image/jpeg")},
            form={},
        )
