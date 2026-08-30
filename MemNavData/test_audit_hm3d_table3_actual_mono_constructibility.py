import hashlib
import json
from pathlib import Path

import pytest

from MemNavData.audit_hm3d_table3_actual_mono_constructibility import audit


def _write_receipt(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )
    return digest


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    protocol = {
        "schema_version": "hm3d_table3_actual_mono_protocol_v1_20260830",
        "history": {"minimum_frames": 40},
        "length_definition": {"bins_m": [
            {"name": "0_to_20_m"},
            {"name": "20_to_30_m"},
            {"name": "30_to_50_m"},
        ]},
        "population_gate": {
            "minimum_histories_per_bin": 1,
            "minimum_scene_clusters_per_bin": 1,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol) + "\n")
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    episodes = []
    run_root = tmp_path / "run"
    for index in range(125):
        bin_name = ("0_to_20_m", "20_to_30_m", "30_to_50_m")[index % 3]
        scene = f"scene_{index:03d}"
        episode = f"{index:04d}"
        identity = hashlib.sha256(f"candidate-{index}".encode()).hexdigest()
        episodes.append({
            "history_index": index, "scene": scene, "episode": episode,
            "bin_name": bin_name, "candidate_identity_sha256": identity,
        })
        trace_path = run_root / "traces" / f"{index:03d}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = {
            "reached": False, "steps": 40,
            "poses": [{"step": step} for step in range(40)],
        }
        trace_path.write_text(json.dumps(trace) + "\n")
        trace_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        factual = {
            "history_index": index, "scene": scene, "episode": f"episode_{episode}",
            "bin_name": bin_name, "candidate_identity_sha256": identity,
            "query_policy_outcomes_read": False, "trace_path": str(trace_path),
            "trace_sha256": trace_sha, "reached_A": False,
            "history_eligible": False, "steps_A": 40,
        }
        factual_path = (
            run_root / "factual_a" / f"{index:03d}_{scene}_episode_{episode}"
            / "completion.json"
        )
        factual_sha = _write_receipt(factual_path, factual)
        fragment = {
            "history_index": index, "scene": scene, "bin_name": bin_name,
            "candidate_identity_sha256": identity,
            "factual_A_completion_sha256": factual_sha,
            "query_policy_outcomes_read": False,
            "status": "factual_A_ineligible", "constructed": False,
        }
        _write_receipt(
            run_root / "construction_fragments" / f"{index:03d}"
            / "completion.json", fragment,
        )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "protocol_sha256": protocol_sha, "episodes": episodes,
    }) + "\n")
    return run_root, plan_path, protocol_path


def test_audit_reports_pre_query_constructibility_failure(tmp_path: Path) -> None:
    run_root, plan, protocol = _fixture(tmp_path)
    result = audit(run_root, plan, protocol)
    assert result["verified"] is True
    assert result["population_gate_passed"] is False
    assert result["formal_policy_evaluation_authorized"] is False
    assert result["query_metric_files"] == 0
    assert result["bins"]["0_to_20_m"]["factual_A_reached"] == 0


def test_audit_rejects_changed_receipt_sidecar(tmp_path: Path) -> None:
    run_root, plan, protocol = _fixture(tmp_path)
    sidecar = (
        run_root / "construction_fragments/000/completion.json.sha256"
    )
    sidecar.write_text("0" * 64 + "  completion.json\n")
    with pytest.raises(RuntimeError, match="invalid receipt sidecar"):
        audit(run_root, plan, protocol)
