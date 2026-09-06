import hashlib
import json
import subprocess
import sys

import pytest

from MemNavData.run_hm3d_longrange_route_motion_shadow_history import (
    local_se2_delta,
    pose_error,
)


def write_json(path, value):
    payload = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_local_pose_receipt_uses_reference_body_frame():
    reference = {"x": 1.0, "z": 2.0, "yaw": 0.0}
    query = {"x": 0.8, "z": 1.0, "yaw": 0.25}
    truth = local_se2_delta(reference, query)
    assert truth["forward_m"] == pytest.approx(1.0)
    assert truth["left_m"] == pytest.approx(0.2)
    assert truth["yaw_rad"] == pytest.approx(0.25)
    error = pose_error({
        "forward_m": 1.03,
        "left_m": 0.16,
        "yaw_rad": 0.30,
        "translation_m": (1.03 ** 2 + 0.16 ** 2) ** 0.5,
    }, truth)
    assert error["translation_vector_error_m"] == pytest.approx(0.05)
    assert error["yaw_error_deg"] == pytest.approx(2.86478897565)


def test_shadow_aggregate_requires_same_edge_reproduction_and_accuracy(
        tmp_path):
    protocol = {
        "schema_version": (
            "hm3d_longrange_route_motion_shadow_protocol_v1_20260903"),
        "selected_history_indices": list(range(14)),
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_sha = write_json(protocol_path, protocol)
    run_root = tmp_path / "run"
    for index in range(14):
        cell = {
            "schema_version": (
                "hm3d_longrange_route_motion_shadow_history_v1_20260903"),
            "history_index": index,
            "scene": f"scene_{index % 4}",
            "protocol_sha256": protocol_sha,
                "shadow_control_authority": False,
                "failure_edge": {
                    "primary_stop_reproduced": True,
                    "unfiltered_pnp_shadow": {
                    "local_motion_validity": {"accepted": index < 12},
                },
                "unfiltered_shadow_pose_error": {
                    "translation_vector_error_m": 0.04,
                    "yaw_error_deg": 2.0,
                },
            },
        }
        root = run_root / "evaluation" / f"{index:03d}_case"
        path = root / "completion.json"
        digest = write_json(path, cell)
        (root / "completion.json.sha256").write_text(
            f"{digest}  completion.json\n")

    subprocess.run([
        sys.executable,
        "MemNavData/analyze_hm3d_longrange_route_motion_shadow.py",
        "--run-root", str(run_root),
        "--protocol", str(protocol_path),
        "--expected-protocol-sha256", protocol_sha,
    ], check=True)
    result = json.loads((run_root / "result" / "summary.json").read_text())
    assert result["primary_stop_reproduced"] == 14
    assert result["unfiltered_shadow_valid"] == 12
    assert result["supports_epipolar_degeneracy"] is True
    assert result["navigation_claim_allowed"] is False

    verification = run_root / "result" / "independent_verification.json"
    subprocess.run([
        sys.executable,
        "MemNavData/independent_verify_hm3d_longrange_route_motion_shadow.py",
        "--run-root", str(run_root),
        "--protocol", str(protocol_path),
        "--expected-protocol-sha256", protocol_sha,
        "--summary", str(run_root / "result" / "summary.json"),
        "--out", str(verification),
    ], check=True)
    verified = json.loads(verification.read_text())
    assert verified["verified"] is True
    assert verified["supports_epipolar_degeneracy"] is True
    assert verified["navigation_claim_allowed"] is False
