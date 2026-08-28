import csv
import hashlib
import json

from MemNavData.audit_vint_cec_bearing_alignment_cell import audit
from MemNavData.cec_handoff_contract import build_handoff_packet


ARMS = (
    "anchor_unaligned",
    "native_bearing_aligned",
    "anchor_bearing_aligned",
)


def handoff() -> dict:
    anchor = b"anchor"
    proof = {
        "certified_relocalization_schema_version": 3,
        "frame_idx": 4,
        "ok": True,
        "accepted": True,
        "reason": "certificate_accepted",
        "selected_anchor": 2,
        "selected_anchor_image_sha256": hashlib.sha256(anchor).hexdigest(),
        "direction_vector": [-1.0, 0.0],
        "pointgoal_units": "lingbot_raw_direction_only",
        "certificate": {"accepted": True},
    }
    return build_handoff_packet(
        proof,
        current_rgb=b"current",
        goal_rgb=b"goal",
        anchor_jpeg=anchor,
        causal_history_sha256="0" * 64,
    )


def test_three_arm_cell_audit(tmp_path) -> None:
    (tmp_path / "direction_triple_contract.json").write_text(json.dumps({
        "schema_version": "vint_cec_direction_triple_contract_v1_20260828",
        "scope": "consumed",
        "scene": "scene",
        "episode": "episode_0000",
        "arm_order": list(ARMS),
    }))
    process = {"pid": 1, "process_start_ticks": 2}
    for arm in ARMS:
        result = tmp_path / arm / "result"
        result.mkdir(parents=True)
        aligned = arm != "anchor_unaligned"
        forced = arm == "native_bearing_aligned"
        first = {
            "step": 0,
            "cec_handoff_packet": handoff(),
            "cec_takeover": not forced,
            "cec_shadow_takeover": True,
            "cec_forced_reject_native": forced,
            "cec_initial_bearing_alignment_executed": aligned,
            "evaluation_gt_goal_distance_m": 2.0,
        }
        second = {"step": 8, "evaluation_gt_goal_distance_m": 1.7}
        (result / "episode_pair_revisit_plans.json").write_text(json.dumps({
            "query_leg": [first, second],
            "rollout_traces": {"query": [
                {"step": index, "x": 0.0,
                 "z": (0.1 * index if aligned else -0.1 * index),
                 "yaw": 0.0}
                for index in range(9)
            ]},
        }))
        (result / "summary.json").write_text(json.dumps({
            "queries": 1,
            "role_counts": {"novel": 0, "revisit": 1},
        }))
        with (result / "metric.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "reached", "steps", "final_goal_dist_m",
                "cec_initial_bearing_alignment_count",
            ])
            writer.writeheader()
            writer.writerow({
                "reached": 1 if aligned else 0,
                "steps": 20,
                "final_goal_dist_m": 0.9 if aligned else 2.5,
                "cec_initial_bearing_alignment_count": 1 if aligned else 0,
            })
        (tmp_path / arm / "compute_identity.json").write_text(json.dumps({
            "host": "node",
            "gpu_uuid": "GPU-1",
            "memnav": process,
            "navdp": process,
            "accepted_controller": process,
            "controller_proxy": process,
        }))

    value = audit(tmp_path)
    assert value["verified"] is True
    assert value["arms"]["anchor_unaligned"]["alignment_count"] == 0
    assert value["arms"]["native_bearing_aligned"]["moved_closer"] is True
    assert value["arms"]["anchor_bearing_aligned"][
        "bearing_execution_error_deg"] == 0.0
