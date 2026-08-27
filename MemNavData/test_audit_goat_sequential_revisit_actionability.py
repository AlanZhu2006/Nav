import hashlib
import json

from MemNavData.audit_goat_sequential_revisit_actionability import audit


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_classifies_stop_only_accept_as_degenerate_noop(tmp_path):
    entry = {
        "index": 0,
        "scene_id": "scene",
        "episode_id": "episode",
        "target_subtask_index": 1,
    }
    manifest = tmp_path / "manifest.json"
    manifest_sha = _write_json(manifest, {"episodes": [entry]})
    summary = tmp_path / "summary.json"
    _write_json(summary, {"manifest_sha256": manifest_sha})
    record = {
        "step": 7,
        "subtask_before": 1,
        "position_before": [1.0, 0.0, 2.0],
        "official_action_id": 6,
        "official_action": "subtask_stop",
        "executed_action_id": 6,
        "executed_action": "subtask_stop",
        "action_source": "official_goat_subtask_stop",
        "navdp_plan": None,
        "certificate": {
            "ok": True,
            "accepted": True,
            "pointgoal_units": "lingbot_raw_direction_only",
        },
    }
    result = (tmp_path / "run" / "episodes" / "000"
              / "goat_sequential_revisit_pilot.json")
    result_sha = _write_json(result, {
        "complete": True,
        "manifest_sha256": manifest_sha,
        "manifest_entry": entry,
        "pairs": [{
            "scene_id": "scene",
            "episode_id": "episode",
            "native": {"records": [record]},
            "cec": {
                "records": [record],
                "navdp_plan_count": 0,
                "first_override_step": None,
            },
        }],
    })
    result.with_suffix(result.suffix + ".sha256").write_text(
        result_sha + "  " + str(result) + "\n")

    payload = audit(manifest, tmp_path / "run", summary)
    assert payload["certificate_accept_events"] == 1
    assert payload["actionable_non_stop_accept_events"] == 0
    assert payload["navdp_plan_count"] == 0
    assert payload["executed_override_events"] == 0
    assert payload["cross_arm_action_pose_exact_episodes"] == 1
    assert payload["formal_effect_is_degenerate_noop"] is True
    assert payload["formal_null_classification"] == (
        "degenerate_noop_no_executed_intervention")
