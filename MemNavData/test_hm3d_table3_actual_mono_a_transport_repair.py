from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "MemNavData/freeze_hm3d_table3_actual_mono_a_transport_repair.py"
VERIFY = (
    ROOT
    / "MemNavData/independent_verify_hm3d_table3_actual_mono_a_transport_repair.py"
)
LAUNCH = ROOT / "MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair_launch.sbatch"
REPAIR = ROOT / "MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair.sbatch"
FINISH = ROOT / "MemNavData/slurm_hm3d_table3_actual_mono_a_transport_repair_finish.sbatch"
CONTRACT = (
    ROOT / "MemNavData/hm3d_table3_directed_geodesic_repair_contract_20260830.json"
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_receipt(path: Path, payload: bytes = b"not-json-outcome-payload\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.with_name(path.name + ".sha256").write_text(
        f"{sha(path)}  {path.name}\n"
    )


def test_exact_repair_membership_is_byte_only_and_archives_partials(tmp_path: Path):
    episodes = [
        {
            "history_index": index,
            "scene": f"scene{index % 7}",
            "episode": f"{index:04d}",
            "bin_name": ("0_to_20_m", "20_to_30_m", "30_to_50_m")[index % 3],
            "candidate_identity_sha256": f"identity-{index:03d}",
        }
        for index in range(125)
    ]
    candidate = tmp_path / "candidate_plan.json"
    candidate.write_text(json.dumps({"episodes": episodes}) + "\n")
    run = tmp_path / "run"
    missing = {2, 74}
    for index, row in enumerate(episodes):
        label = f"{index:03d}_{row['scene']}_episode_{row['episode']}"
        root = run / "factual_a" / label
        if index not in missing:
            # Deliberately invalid JSON proves outcome receipts are not parsed.
            write_receipt(root / "completion.json")
    partial = run / "factual_a/002_scene2_episode_0002"
    partial.mkdir(parents=True)
    (partial / "partial.log").write_text("transport failed\n")
    carrier = run / "carriers/scene2/episode_0002"
    carrier.mkdir(parents=True)
    (carrier / "frame.jpg").write_bytes(b"frame")
    runtime = run / "runtime/table3_a_2_formal002"
    runtime.mkdir(parents=True)
    (runtime / "run.log").write_text("port collision\n")
    repair_root = run / "repair"
    archive = run / "archive"
    plan = repair_root / "repair_plan.json"
    subprocess.run([
        sys.executable, str(FREEZE), "--run-root", str(run),
        "--candidate-plan", str(candidate),
        "--expected-candidate-plan-sha256", sha(candidate),
        "--archive-root", str(archive), "--out", str(plan),
    ], check=True)
    verify = repair_root / "independent_verification.json"
    subprocess.run([
        sys.executable, str(VERIFY), "--run-root", str(run),
        "--candidate-plan", str(candidate),
        "--expected-candidate-plan-sha256", sha(candidate),
        "--repair-plan", str(plan), "--archive-root", str(archive),
        "--out", str(verify),
    ], check=True)
    payload = json.loads(plan.read_text())
    audit = json.loads(verify.read_text())
    assert payload["missing_history_indices"] == [2, 74]
    assert payload["completed_history_count"] == 123
    assert payload["completion_payloads_deserialized"] is False
    assert payload["fallback_completion_allowed"] is False
    assert audit["verified"] is True
    assert audit["navigation_outcomes_read"] is False
    assert not partial.exists() and not carrier.exists() and not runtime.exists()


def test_repair_dag_is_exact_fail_closed_and_a100_pinned():
    launch = LAUNCH.read_text()
    repair = REPAIR.read_text()
    finish = FINISH.read_text()
    assert "afterok:${r_job}" in launch
    assert "--partition=a100_tandon" in launch
    assert "missing_history_indices" in launch
    assert "--array=0-124" not in launch
    assert "i in p['missing_history_indices']" in repair
    assert "RUNTIME_ATTEMPT=\"exactRepair" in repair
    assert "REPAIR_CONTRACT" in launch
    assert "EXPECTED_REPAIR_CONTRACT_SHA" in launch
    assert "REPAIR_NAMESPACE" in launch
    assert "completion_payloads_deserialized':False" in finish
    assert "json.load(completion" not in finish


def test_directed_geodesic_repair_changes_no_scientific_identity_or_threshold():
    contract = json.loads(CONTRACT.read_text())
    assert contract["diagnosis"]["navigation_outcomes_read"] is False
    assert contract["diagnosis"]["capacity_direction"] == (
        "query_start_to_first_goal"
    )
    assert contract["diagnosis"]["factual_goal_A_direction"] == (
        "first_goal_to_query_start"
    )
    repair = contract["repair"]
    for field in (
        "candidate_identities_changed",
        "goals_changed",
        "distance_bins_changed",
        "model_or_controller_changed",
        "seed_changed",
        "step_budget_changed",
        "success_definition_changed",
        "scientific_thresholds_changed",
        "fallback_completion_allowed",
    ):
        assert repair[field] is False
