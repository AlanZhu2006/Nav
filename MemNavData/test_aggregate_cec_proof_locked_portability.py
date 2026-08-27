import json
from pathlib import Path

from MemNavData.aggregate_cec_proof_locked_portability import aggregate
from MemNavData.independent_verify_cec_proof_locked_portability import verify


CONTROLLERS = ("navdp", "vint", "iplanner")
HISTORIES = (0, 7, 14, 21)


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def make_pilot(tmp_path: Path):
    run = tmp_path / "run"
    entries = []
    for slot, history in enumerate(HISTORIES):
        scene = f"scene{slot}"
        episode = f"episode_{history:04d}"
        query = "pair_00_revisit"
        entries.append({
            "history_index": history, "scene": scene, "episode": episode,
            "query_id": query,
        })
        label = f"{history:03d}_{scene}_{episode}_{query}"
        proof = f"{slot + 1:064x}"
        packet = f"{slot + 101:064x}"
        for controller_index, controller in enumerate(CONTROLLERS):
            cell = run / "evaluation" / label / controller
            order = (["grant", "forced_reject_native"]
                     if (history + controller_index) % 2 == 0
                     else ["forced_reject_native", "grant"])
            write_json(cell / "authority_pair_contract.json", {
                "controller": controller, "authority_order": order,
                "query_manifest_sha256": "pending",
            })
            write_json(cell / "authority_pair_audit.json", {
                "verified": True, "same_process_pair": True,
                "handoff_packet_verified": True,
                "source_accepted_manifest_match": True,
                "controller": controller, "scene": scene,
                "episode": episode, "query_id": query,
                "first_handoff_proof_sha256": proof,
                "first_handoff_anchor": 10 + slot,
                "handoff_packet_sha256": packet,
                "grant_success": int(controller != "iplanner"),
                "forced_reject_success": 0,
                "paired_gain": int(controller != "iplanner"),
                "paired_loss": 0,
                "grant_progress_m": 1.0 + controller_index,
                "forced_reject_progress_m": 0.5,
            })
    manifest = tmp_path / "accepted.json"
    write_json(manifest, {
        "schema_version": "cec_first_decision_accepted_population_v1_20260827",
        "queries": entries,
    })
    import hashlib
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for path in run.glob("evaluation/*/*/authority_pair_contract.json"):
        payload = json.loads(path.read_text())
        payload["query_manifest_sha256"] = manifest_sha
        write_json(path, payload)
    return run, manifest


def test_pilot_aggregate_and_independent_verifier(tmp_path):
    run, manifest = make_pilot(tmp_path)
    summary = aggregate(run, manifest)
    assert summary["verified"] is True
    assert summary["same_packet_across_controller_triads"] is True
    assert summary["controller_results"]["vint"]["paired_gain"] == 4
    summary_path = run / "summary.json"
    write_json(summary_path, summary)
    result = verify(run, summary_path, manifest)
    assert result["verified"] is True
    assert result["raw_cells"] == 12
