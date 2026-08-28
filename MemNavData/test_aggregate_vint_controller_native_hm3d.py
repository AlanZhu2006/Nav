import json
from pathlib import Path

from MemNavData.aggregate_vint_controller_native_hm3d import aggregate
from MemNavData.independent_verify_vint_controller_native_hm3d import verify


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def query(scene, episode, role, native, cec, takeover, *, bounded=False):
    row = {
        "scene": scene, "episode": episode, "pair_id": "pair_00",
        "query_id": f"pair_00_{role}", "analysis_role": role,
        "first_proof_sha256": "a" * 64,
        "first_shadow_takeover": takeover, "first_anchor": 4 if takeover else None,
        "first_packet_verified": takeover, "first_packet_sha256": None,
        "grant_takeover_plans": int(takeover), "forced_takeover_plans": 0,
        "grant_success": cec, "native_success": native,
        "paired_gain": int(cec == 1 and native == 0),
        "paired_loss": int(cec == 0 and native == 1),
        "initial_geodesic_m": 4.0,
        "grant_final_distance_m": 0.8 if cec else 2.0,
        "native_final_distance_m": 0.8 if native else 2.0,
        "grant_path_len_m": 4.5, "native_path_len_m": 5.0,
        "grant_steps": 24, "native_steps": 32,
        "exact_fallback_trace_match": True if not takeover else None,
        "post_divergence_proof_equality_required": False,
        "files": {},
    }
    if bounded:
        row["grant_bearing_alignment"] = {
            "mode": "first_certified_bounded",
            "required": takeover,
            "validated": True,
            "action_count": 3 if takeover else 0,
            "fresh_observation_receipts": 3 if takeover else 0,
        }
    return row


def make_run(tmp_path, *, bounded=False):
    episodes = []
    for index in range(4):
        scene = f"scene{index}"
        episode = "episode_0000"
        episodes.append({"scene": scene, "episode": episode})
        cell = tmp_path / "run/evaluation" / f"{index:03d}_{scene}" / "vint"
        rows = [
            query(scene, episode, "novel", 0, 0, False, bounded=bounded),
            query(scene, episode, "revisit", 0, 1, True, bounded=bounded),
        ]
        write_json(cell / "controller_native_pair_audit.json", {
            "schema_version": "vint_controller_native_pair_audit_v1_20260828",
            "verified": True, "controller": "vint",
            "reject_policy": "controller_native_exact",
            **({"grant_bearing_alignment": "first_certified_bounded"}
               if bounded else {}),
            "scene": scene, "episode": episode,
            "authority_order": (
                ["grant", "forced_reject_native"] if index % 2 == 0
                else ["forced_reject_native", "grant"]),
            "query_count": 2, "query_results": rows,
        })
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"episodes": episodes})
    return tmp_path / "run", manifest


def test_aggregate_and_independent_verifier(tmp_path):
    run, manifest = make_run(tmp_path)
    summary = aggregate(
        run, manifest, expected_histories=4, expected_scenes=4,
        history_indices=(0, 1, 2, 3), claim_scope="pilot")
    assert summary["results"]["all"]["native_success"] == 0
    assert summary["results"]["all"]["cec_success"] == 4
    assert summary["results"]["revisit"]["paired_gain"] == 4
    summary_path = run / "summary.json"
    write_json(summary_path, summary)
    checked = verify(run, summary_path, manifest)
    assert checked["verified"] is True
    assert checked["raw_queries"] == 8


def test_bounded_alignment_aggregate_and_independent_verifier(tmp_path):
    run, manifest = make_run(tmp_path, bounded=True)
    summary = aggregate(
        run, manifest, expected_histories=4, expected_scenes=4,
        history_indices=(0, 1, 2, 3), claim_scope="paper_heldout",
        expected_grant_alignment="first_certified_bounded")
    assert summary["grant_bearing_alignment"] == "first_certified_bounded"
    assert summary["treatment"] == (
        "proof_bound_bearing_turn_then_history_anchor_imagegoal")
    summary_path = run / "summary.json"
    write_json(summary_path, summary)
    checked = verify(run, summary_path, manifest)
    assert checked["verified"] is True
    assert checked["grant_bearing_alignment"] == "first_certified_bounded"
