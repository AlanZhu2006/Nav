import hashlib
import json

import pytest

from MemNavData.summarize_goat_sequential_revisit import (
    exact_mcnemar_two_sided,
    summarize,
)
from MemNavData.verify_goat_sequential_revisit import verify


def _entry(index, scene):
    return {
        "index": index,
        "arm_order": ["native", "cec"] if index % 2 == 0 else ["cec", "native"],
        "scene_id": scene,
        "episode_id": str(index),
        "target_subtask_index": 1,
        "target_instance_id": "rug_{}".format(index),
        "prior_instance_subtasks": [{
            "subtask_index": 0,
            "modality": "description",
            "instance_id": "rug_{}".format(index),
        }],
    }


def _arm(name, scene, episode, success, accept=0):
    return {
        "arm": name,
        "scene_id": scene,
        "episode_id": episode,
        "target_entered": True,
        "complete_through_target": True,
        "target_success": float(success),
        "steps": 1,
        "termination_reason": "transitioned_past_target",
        "certificate_accept_count": accept,
        "navdp_plan_count": accept,
        "first_override_step": 0 if accept else None,
        "metrics": {"success": {"subtask_success": [1.0, float(success)]}},
        "records": [{
            "subtask_before": 1,
            "official_action_id": 6,
            "executed_action_id": 6,
            "position_before": [0.0, 0.0, 0.0],
            "certificate": ({
                "ok": True,
                "accepted": True,
                "probe_candidates": [{"anchor": 0, "score": 0.9}],
            } if name == "cec" and accept else None),
        }],
    }


def _write_result(root, manifest, manifest_sha, index, native_success,
                  cec_success, accept=0):
    entry = manifest["episodes"][index]
    native = _arm(
        "native", entry["scene_id"], entry["episode_id"], native_success)
    cec = _arm(
        "cec", entry["scene_id"], entry["episode_id"], cec_success, accept)
    payload = {
        "complete": True,
        "manifest_sha256": manifest_sha,
        "manifest_entry": entry,
        "paper_claim_authorized": True,
        "role_label_read_by_controller": False,
        "official_goat_stop_authority_retained": True,
        "official_goat_subtask_stop_preserved_exactly": True,
        "official_policy_uses_released_stochastic_eval_semantics": True,
        "policy_device": "cuda:0",
        "runtime_provenance": {
            "slurm_job_id": str(1000 + index),
            "slurm_array_job_id": "1000",
            "slurm_array_task_id": str(index),
            "cuda_visible_devices": "0",
        },
        "max_steps": 5000,
        "checkpoint_sha256": "c" * 64,
        "goat_commit": "g" * 40,
        "pairs": [{
            "scene_id": entry["scene_id"],
            "episode_id": entry["episode_id"],
            "executed_arm_order": entry["arm_order"],
            "native": native,
            "cec": cec,
            "prefix_audit": {
                "prefix_paired_before_first_override": True,
            },
        }],
    }
    destination = root / "episodes" / "{:03d}".format(index)
    destination.mkdir(parents=True)
    (destination / "goat_sequential_revisit_pilot.json").write_text(
        json.dumps(payload))


def test_exact_mcnemar_matches_six_gains_no_losses():
    assert exact_mcnemar_two_sided(6, 0) == pytest.approx(0.03125)
    assert exact_mcnemar_two_sided(0, 0) == 1.0


def test_strict_summary_reports_itt_and_exact_fallback(tmp_path):
    manifest = {
        "schema_version": "manifest-test",
        "evaluation_stage": "formal_targeted_external_evaluation",
        "paper_claim_authorized": True,
        "analysis_contract": {
            "maximum_steps_per_arm": 5000,
            "minimum_mechanistic_coverage_for_interpretation": {
                "paired_episodes_entering_target": 2,
                "distinct_scenes": 2,
            },
        },
        "episodes": [_entry(0, "sceneA"), _entry(1, "sceneB")],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _write_result(tmp_path, manifest, digest, 0, 0, 1, accept=1)
    _write_result(tmp_path, manifest, digest, 1, 1, 1, accept=0)
    result = summarize(manifest_path, tmp_path)
    assert result["primary_intention_to_treat"]["native_successes"] == 1
    assert result["primary_intention_to_treat"]["cec_successes"] == 2
    assert result["primary_intention_to_treat"]["paired_gains"] == 1
    assert result["audit"]["all_no_accepts_are_exact_fallback"] is True
    assert result["constructibility"]["mechanistic_coverage_gate_passed"] is True
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(result, sort_keys=True))
    receipt = verify(manifest_path, tmp_path, summary_path)
    assert receipt["verified"] is True
    assert receipt["independent_primary_recompute"]["paired_gains"] == 1


def test_summary_rejects_non_cuda_result(tmp_path):
    manifest = {
        "paper_claim_authorized": True,
        "analysis_contract": {
            "maximum_steps_per_arm": 5000,
            "minimum_mechanistic_coverage_for_interpretation": {
                "paired_episodes_entering_target": 1,
                "distinct_scenes": 1,
            },
        },
        "episodes": [_entry(0, "sceneA")],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    _write_result(tmp_path, manifest, digest, 0, 0, 0)
    result_path = next((tmp_path / "episodes").glob(
        "*/goat_sequential_revisit_pilot.json"))
    payload = json.loads(result_path.read_text())
    payload["policy_device"] = "cpu"
    result_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="non-CUDA"):
        summarize(manifest_path, tmp_path)
