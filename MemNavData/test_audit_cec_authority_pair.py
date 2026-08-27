import csv
import json
from pathlib import Path

import pytest

from MemNavData.audit_cec_authority_pair import audit_pair


PROOF = "a" * 64


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def make_arm(root: Path, scope: str, *, proof=PROOF, reached=1):
    result = root / scope / "result"
    result.mkdir(parents=True)
    write_json(result / "summary.json", {
        "server_backend": "cec_portability",
        "runtime_role_visibility": "none",
        "queries": 1,
    })
    row = {
        "scene": "scene0", "episode": "episode_0000",
        "pair_id": "pair_00", "query_id": "pair_00_revisit",
        "analysis_role": "revisit", "seed": "7",
        "shared_A_frames": "80", "shared_A_decision_frames": "10",
        "shared_A_hashes_ok": "1", "shared_A_diffusion_samples": "0",
        "metric_depth_sensor_consumed_any": "0",
        "runtime_failure_plans": "0", "reached": str(reached),
        "geodesic_m": "4.0", "final_goal_dist_m": "0.8" if reached else "2.2",
    }
    with (result / "metric.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    takeover = scope == "grant"
    action_state = "takeover" if takeover else "forced_reject"
    first = {
        "cec_portability_schema": "cec_controller_portability_hub_v2",
        "role_label_visible": False,
        "metric_depth_sensor_consumed": False,
        "cec_accept_controller": "vint",
        "cec_proof_sha256": proof,
        "cec_frame_idx": 80,
        "cec_goal_sha256": "b" * 64,
        "cec_goal_start_frame": 80,
        "cec_selected_anchor": 42,
        "cec_shadow_takeover": True,
        "cec_takeover": takeover,
        "cec_action_state": action_state,
        "cec_forced_reject_native": not takeover,
    }
    second = {
        **first,
        "cec_proof_sha256": ("c" * 64 if takeover else "d" * 64),
        "cec_takeover": takeover,
        "cec_action_state": action_state,
    }
    write_json(result / "episode_0000_pair_00_revisit_plans.json", {
        "replay": {"online_frames": 80, "all_rgb_hashes_verified": True},
        "legA": [{"step": 0}],
        "query_leg": [first, second],
    })
    write_json(root / scope / "compute_identity.json", {
        "host": "gpu-node", "gpu_uuid": "GPU-0",
        "memnav": {"pid": 10, "process_start_ticks": 100},
        "navdp": {"pid": 11, "process_start_ticks": 101},
        "accepted_controller": {"pid": 12, "process_start_ticks": 102},
        "controller_proxy": {"pid": 13, "process_start_ticks": 103},
        "cec_hub": {"pid": 20 if takeover else 21,
                    "process_start_ticks": 200 if takeover else 201},
    })


def pair(tmp_path):
    make_arm(tmp_path, "grant", reached=1)
    make_arm(tmp_path, "forced_reject_native", reached=0)
    return tmp_path


def test_authority_pair_requires_only_first_proof_identity(tmp_path):
    output = audit_pair(pair(tmp_path), "vint")
    assert output["verified"] is True
    assert output["same_process_pair"] is True
    assert output["paired_gain"] == 1
    assert output["paired_loss"] == 0
    assert output["post_handoff_proof_equality_required"] is False


def test_first_proof_mismatch_fails(tmp_path):
    make_arm(tmp_path, "grant")
    make_arm(tmp_path, "forced_reject_native", proof="e" * 64)
    with pytest.raises(RuntimeError, match="first CEC proof differs"):
        audit_pair(tmp_path, "vint")


def test_different_loaded_controller_process_fails(tmp_path):
    root = pair(tmp_path)
    path = root / "forced_reject_native/compute_identity.json"
    payload = json.loads(path.read_text())
    payload["accepted_controller"]["pid"] = 99
    write_json(path, payload)
    with pytest.raises(RuntimeError, match="accepted_controller process"):
        audit_pair(root, "vint")
