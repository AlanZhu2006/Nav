import hashlib
import json
import sys

from MemNavData import analyze_hm3d_longrange_rear_alignment as analyze
from MemNavData import independent_verify_hm3d_longrange_rear_alignment as verify
from MemNavData.hm3d_longrange_rear_alignment import (
    ARMS,
    SELECTED_HISTORY_INDICES,
)


def _write(path, value):
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n")


def test_analyzer_and_independent_verifier_recount(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "schema_version": analyze.PROTOCOL_SCHEMA,
        "selected_history_indices": list(SELECTED_HISTORY_INDICES),
        "arms": list(ARMS),
    }) + "\n")
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
    run = tmp_path / "run"
    for ordinal, index in enumerate(SELECTED_HISTORY_INDICES):
        root = run / "evaluation" / f"{index:03d}_scene{ordinal}_episode"
        root.mkdir(parents=True)
        gain = index in (3, 4)
        outcomes = {ARMS[0]: 0 if gain else 1,
                    ARMS[1]: 1}
        packet_sha = hashlib.sha256(str(index).encode()).hexdigest()
        actions = 6
        row = {
            "schema_version": analyze.CELL_SCHEMA,
            "history_index": index,
            "scene": f"scene{ordinal}",
            "protocol_sha256": protocol_sha,
            "prefix_equality": True,
            "initial_cec_proof_equal": True,
            "initial_route_packet_equal": True,
            "initial_route_packet_sha256": packet_sha,
            "formal_parent_partition": (
                "stuck" if index in (3, 4, 8, 10, 21) else "success"),
            "formal_parent_outcome": 0 if index in (3, 4, 8, 10, 21) else 1,
            "arms": list(ARMS),
            "outcomes": outcomes,
            "termination_reason": {arm: "success" if outcomes[arm]
                                   else "stuck" for arm in ARMS},
            "geometry_stream_stop_plans": {arm: 0 for arm in ARMS},
            "audits": {
                ARMS[0]: {"rear_alignment": {
                    "alignment_actions": 0,
                    "packet_sha256": packet_sha,
                }},
                ARMS[1]: {"rear_alignment": {
                    "alignment_actions": actions,
                    "fresh_observation_receipts": actions,
                    "control_edges_consumed": actions,
                    "visual_pnp_edges_during_alignment": 0,
                    "zero_translation": True,
                    "simulator_pose_receipt_used": False,
                    "packet_sha256": packet_sha,
                    "initial_heading_deg": 180.0,
                    "max_abs_action_deg": 30.0,
                }},
            },
        }
        _write(root / "completion.json", row)

    monkeypatch.setattr(sys, "argv", [
        "analyze", "--run-root", str(run), "--protocol", str(protocol),
        "--expected-protocol-sha256", protocol_sha])
    analyze.main()
    summary = run / "result" / "summary.json"
    result = json.loads(summary.read_text())
    assert result["successes"] == {ARMS[0]: 7, ARMS[1]: 9}
    assert result["aligned_vs_unaligned"]["gain"] == 2
    assert result["decision"].startswith("advance_")

    out = run / "result" / "independent_verification.json"
    monkeypatch.setattr(sys, "argv", [
        "verify", "--run-root", str(run),
        "--expected-protocol-sha256", protocol_sha,
        "--summary", str(summary), "--out", str(out)])
    verify.main()
    assert json.loads(out.read_text())["verified"] is True
