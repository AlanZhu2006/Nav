import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from MemNavData.audit_vint_controller_native_pair import audit_pair
from MemNavData.cec_handoff_contract import build_handoff_packet


PROOF = "a" * 64


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def handoff_packet(proof=PROOF, direction=(1.0, 0.0)):
    anchor = b"anchor"
    public = {
        "certified_relocalization_schema_version": 2,
        "frame_idx": 80,
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": 42,
        "selected_anchor_image_sha256": hashlib.sha256(anchor).hexdigest(),
        "direction_vector": list(direction),
        "pointgoal_units": "lingbot_raw_direction_only",
        "certificate": {"accepted": True},
    }
    return build_handoff_packet(
        public, current_rgb=b"current", goal_rgb=b"goal",
        anchor_jpeg=anchor, causal_history_sha256="c" * 64)


def make_plan(*, takeover, forced, proof=PROOF, direction=(1.0, 0.0)):
    packet = handoff_packet(proof, direction) if takeover else None
    if packet is not None:
        proof = packet["proof_sha256"]
    return {
        "cec_portability_schema": "cec_controller_portability_hub_v2",
        "role_label_visible": False,
        "metric_depth_sensor_consumed": False,
        "metric_depth_sensor_consumed_by_policy": False,
        "cec_accept_controller": "vint",
        "cec_reject_controller": "vint",
        "cec_reject_policy": "controller_native_exact",
        "cec_controller_portability_receipt": {
            "controller": "vint",
            "endpoint": "imagegoal_step",
            "reject_policy": "controller_native_exact",
            "fallback_controller": "vint",
        },
        "cec_proof_sha256": proof,
        "cec_frame_idx": 80,
        "cec_goal_sha256": "b" * 64,
        "cec_goal_start_frame": 80,
        "cec_selected_anchor": 42 if takeover else None,
        "cec_shadow_takeover": takeover,
        "cec_takeover": takeover and not forced,
        "cec_action_state": (
            "forced_reject" if takeover and forced
            else "takeover" if takeover else "fallback"),
        "cec_forced_reject_native": forced,
        "cec_reason": "certificate_accepted" if takeover else "no_candidate",
        "cec_handoff_packet": packet,
        "cec_handoff_packet_sha256": (
            None if packet is None else packet["packet_sha256"]),
    }


def bounded_trace(packet_sha: str):
    rows = []
    yaw = 0.0
    for index in range(3):
        next_yaw = yaw + math.radians(30.0)
        rows.append({
            "action_index": index,
            "packet_sha256": packet_sha,
            "observation_jpg_sha256": chr(ord("d") + index) * 64,
            "memory_frame_idx": 80 + index,
            "yaw_before_rad": yaw,
            "yaw_after_rad": next_yaw,
            "turn_delta_deg": 30.0,
            "remaining_after_deg": 60.0 - 30.0 * index,
            "translation_m": 0.0,
            "fresh_observation_required_before_next_action": True,
        })
        yaw = next_yaw
    return rows


def make_arm(root: Path, scope: str, *, bounded=False):
    forced = scope == "forced_reject_native"
    result = root / scope / "result"
    result.mkdir(parents=True)
    write_json(result / "summary.json", {
        "server_backend": "cec_portability",
        "runtime_role_visibility": "none",
        "queries": 2,
        "role_counts": {"novel": 1, "revisit": 1},
    })
    rows = []
    for role in ("novel", "revisit"):
        query_id = f"pair_00_{role}"
        success = int(role == "revisit" and not forced)
        rows.append({
            "scene": "scene0", "episode": "episode_0000",
            "pair_id": "pair_00", "query_id": query_id,
            "analysis_role": role, "seed": "7",
            "shared_A_frames": "80", "shared_A_decision_frames": "10",
            "shared_A_hashes_ok": "1", "shared_A_diffusion_samples": "0",
            "metric_depth_sensor_consumed_any": "0",
            "runtime_failure_plans": "0", "reached": str(success),
            "geodesic_m": "4.0", "final_goal_dist_m": (
                "0.8" if success else "2.2"),
            "path_len_m": "3.5", "steps": "16",
        })
        if bounded:
            aligned = role == "revisit" and not forced
            rows[-1].update({
                "cec_initial_bearing_alignment_mode": (
                    "first_certified_bounded" if not forced else "off"),
                "cec_initial_bearing_alignment_count": "1" if aligned else "0",
                "cec_initial_bearing_alignment_action_count": (
                    "3" if aligned else "0"),
            })
        takeover = role == "revisit"
        rollout = ([{"x": 1.0}] if takeover and not forced
                   else [{"x": 0.0}])
        query_result = {
            "reached": bool(success), "path_len_m": 3.5,
            "steps": 16, "final_goal_dist_m": 0.8 if success else 2.2,
        }
        plan = make_plan(
            takeover=takeover,
            forced=forced,
            direction=((0.0, 1.0) if bounded else (1.0, 0.0)),
        )
        trace = (
            bounded_trace(plan["cec_handoff_packet_sha256"])
            if bounded and takeover and not forced else []
        )
        write_json(result / f"episode_0000_{query_id}_plans.json", {
            "query_runtime_fields": ["query_id", "goal_rgb"],
            "analysis_role_not_forwarded": True,
            "replay": {"online_frames": 80},
            "legA": [{"step": 0}],
            "query_leg": [plan],
            "cec_initial_bearing_alignment_trace": trace,
            "rollout_traces": {"query": rollout},
            "query_result": query_result,
        })
    with (result / "metric.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / scope / "compute_identity.json", {
        "host": "node", "gpu_uuid": "GPU-0",
        "memnav": {"pid": 10, "process_start_ticks": 100},
        "navdp": {"pid": 11, "process_start_ticks": 101},
        "accepted_controller": {"pid": 12, "process_start_ticks": 102},
        "controller_proxy": {"pid": 13, "process_start_ticks": 103},
    })
    write_json(root / scope / "hub_health.json", {
        "reject_policy": "controller_native_exact",
        "reject_controller": "vint",
    })


def make_pair(root: Path, *, bounded=False):
    write_json(root / "authority_pair_contract.json", {
        "schema_version": (
            "cec_authority_pair_contract_v3_20260829" if bounded
            else "cec_authority_pair_contract_v2_20260828"),
        "controller": "vint", "scene": "scene0",
        "episode": "episode_0000",
        "reject_policy": "controller_native_exact",
        "runtime_role_visibility": "none",
        "authority_order": ["grant", "forced_reject_native"],
        **({
            "grant_bearing_alignment": "first_certified_bounded",
            "forced_reject_bearing_alignment": "off",
        } if bounded else {}),
    })
    make_arm(root, "grant", bounded=bounded)
    make_arm(root, "forced_reject_native", bounded=bounded)
    return root


def test_mixed_role_pair_audit(tmp_path):
    output = audit_pair(make_pair(tmp_path))
    assert output["verified"] is True
    assert output["query_count"] == 2
    by_role = {row["analysis_role"]: row for row in output["query_results"]}
    assert by_role["novel"]["exact_fallback_trace_match"] is True
    assert by_role["revisit"]["paired_gain"] == 1


def test_pair_fails_if_reject_switches_to_navdp(tmp_path):
    root = make_pair(tmp_path)
    path = root / "grant/result/episode_0000_pair_00_novel_plans.json"
    payload = json.loads(path.read_text())
    payload["query_leg"][0]["cec_reject_controller"] = "navdp"
    write_json(path, payload)
    with pytest.raises(RuntimeError, match="both branches"):
        audit_pair(root)


def test_bounded_pair_consumes_certified_bearing_with_fresh_receipts(tmp_path):
    output = audit_pair(make_pair(tmp_path, bounded=True))
    revisit = next(
        row for row in output["query_results"]
        if row["analysis_role"] == "revisit"
    )
    receipt = revisit["grant_bearing_alignment"]
    assert output["grant_bearing_alignment"] == "first_certified_bounded"
    assert receipt["required"] is True
    assert receipt["validated"] is True
    assert receipt["action_count"] == 3
    assert receipt["fresh_observation_receipts"] == 3


def test_bounded_pair_rejects_nonsequential_observation_receipts(tmp_path):
    root = make_pair(tmp_path, bounded=True)
    path = root / "grant/result/episode_0000_pair_00_revisit_plans.json"
    payload = json.loads(path.read_text())
    payload["cec_initial_bearing_alignment_trace"][1]["memory_frame_idx"] = 80
    write_json(path, payload)
    with pytest.raises(RuntimeError, match="sequential"):
        audit_pair(root)
