import math
import hashlib
import json
import subprocess
import sys

import pytest

from MemNavData.hm3d_longrange_route_interface_attribution import (
    audit_controller_support,
    rotated_arm_order,
)


def plan(source, controller, *, projected=None):
    return {
        "geometry_stream_stop": False,
        "certified_relocalization_accepted": True,
        "revisit_adapter_takeover": True,
        "memory_bearing_unit": list(source),
        "local_tangent_unit_bearing": list(source),
        "memory_controller_pointgoal": list(controller),
        "navdp_support_projection_schema_version": (
            None if projected is None else 1),
        "memory_navdp_support_projection_applied": projected,
    }


def test_arm_order_is_pair_balanced():
    assert rotated_arm_order(0) == (
        "direct_pnp_canonical_bearing",
        "direct_pnp_support_projected_bearing",
    )
    assert rotated_arm_order(1) == tuple(reversed(rotated_arm_order(0)))


def test_canonical_bearing_preserves_rear_vector_before_navdp():
    receipt = audit_controller_support([
        plan((-0.8, 0.6), (-2.0, 1.5)),
    ], arm="direct_pnp_canonical_bearing")
    assert receipt["rear_source_bearing_plans"] == 1
    assert receipt["support_projection_plans"] == 0


def test_support_projection_preserves_fixed_norm_at_front_boundary():
    receipt = audit_controller_support([
        plan((-0.8, 0.6), (0.0, 2.5), projected=True),
        plan((0.6, -0.8), (1.5, -2.0), projected=False),
    ], arm="direct_pnp_support_projected_bearing")
    assert receipt["rear_source_bearing_plans"] == 1
    assert receipt["support_projection_plans"] == 1
    assert receipt["fixed_radius_verified"] is True


def test_support_projection_rejects_silent_token_shrink():
    with pytest.raises(RuntimeError, match="norm changed"):
        audit_controller_support([
            plan((-math.sqrt(0.99), 0.1), (0.0, 0.25), projected=True),
        ], arm="direct_pnp_support_projected_bearing")


def test_consumed_aggregate_and_independent_verifier(tmp_path):
    run_root = tmp_path / "run"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema_version": (
            "hm3d_longrange_route_interface_attribution_protocol_v1_20260903")
    }) + "\n")
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    for index in range(23):
        canonical = int(index < 5)
        projected = int(index < 7)
        row = {
            "schema_version": (
                "hm3d_longrange_route_interface_attribution_history_v1_20260903"),
            "history_index": index,
            "scene": f"scene_{index % 8}",
            "protocol_sha256": protocol_sha,
            "prefix_equality": True,
            "initial_cec_proof_equal": True,
            "arms": [
                "direct_pnp_canonical_bearing",
                "direct_pnp_support_projected_bearing",
            ],
            "formal_parent_route_tangent_outcome": int(index < 4),
            "outcomes": {
                "direct_pnp_canonical_bearing": canonical,
                "direct_pnp_support_projected_bearing": projected,
            },
            "geometry_stream_stop_plans": {
                "direct_pnp_canonical_bearing": int(index >= 20),
                "direct_pnp_support_projected_bearing": int(index >= 20),
            },
            "audits": {
                "direct_pnp_support_projected_bearing": {
                    "controller_support": {
                        "support_projection_plans": 2,
                        "rear_source_bearing_plans": 2,
                    }
                }
            },
        }
        root = run_root / "evaluation" / f"{index:03d}_case"
        root.mkdir(parents=True)
        path = root / "completion.json"
        path.write_text(json.dumps(row, sort_keys=True) + "\n")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (root / "completion.json.sha256").write_text(
            f"{digest}  completion.json\n")
    subprocess.run([
        sys.executable,
        "MemNavData/analyze_hm3d_longrange_route_interface_attribution.py",
        "--run-root", str(run_root),
        "--protocol", str(protocol),
        "--expected-protocol-sha256", protocol_sha,
    ], check=True)
    summary = json.loads((run_root / "result" / "summary.json").read_text())
    assert summary["successes"] == {
        "direct_pnp_canonical_bearing": 5,
        "direct_pnp_support_projected_bearing": 7,
    }
    assert summary["support_projected_vs_direct_canonical"] == {
        "gain": 2, "loss": 0, "net": 2}
    assert summary["selected_for_fresh_confirmation"] == (
        "fresh_confirm_direct_pnp_support_projected_bearing")
    verification = run_root / "result" / "independent_verification.json"
    subprocess.run([
        sys.executable,
        "MemNavData/independent_verify_hm3d_longrange_route_interface_attribution.py",
        "--run-root", str(run_root),
        "--expected-protocol-sha256", protocol_sha,
        "--summary", str(run_root / "result" / "summary.json"),
        "--out", str(verification),
    ], check=True)
    assert json.loads(verification.read_text())["verified"] is True
