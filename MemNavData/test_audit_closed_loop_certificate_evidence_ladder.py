import json

import pytest

from MemNavData.audit_closed_loop_certificate_evidence_ladder import (
    query_record,
    summarize,
)


def plan(*, cached=False, precheck=True, pose=True, accepted=False,
         reason="minimum_inliers"):
    pnp = {
        "status": "ok" if pose else "precheck_fundamental_inliers",
        "pose9": [0.0] * 9 if pose else None,
    }
    return {
        "certified_relocalization_cached": cached,
        "certified_relocalization_proposal_attempts": [{
            "precheck_passed": precheck,
        }],
        "certified_relocalization_pnp": pnp,
        "certified_relocalization_accepted": accepted,
        "certified_relocalization_certificate": {"accepted": accepted},
        "certified_relocalization_reason": reason,
        "router_selected_anchor": 20,
    }


def write_receipt(tmp_path, role, plans):
    directory = tmp_path / "000_scene_episode_0000" / "mono_cec"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_0000_pair_00_{role}_plans.json"
    path.write_text(json.dumps({"query_leg": plans}))
    return path


def test_receipt_audit_separates_precheck_pose_and_certificate(tmp_path):
    novel = query_record(write_receipt(
        tmp_path, "novel", [plan(), plan(cached=True)]))
    revisit_plan = plan(
        accepted=True, reason="certificate_accepted")
    revisit = query_record(write_receipt(
        tmp_path, "revisit", [revisit_plan, {
            **revisit_plan, "certified_relocalization_cached": True,
        }]))
    result = summarize([novel, revisit])
    assert result["by_role"]["novel"]["precheck_plus_pnp_pose"] == 1
    assert result["by_role"]["novel"]["full_certificate"] == 0
    assert result["by_role"]["revisit"]["full_certificate"] == 1
    assert result["interpretation_contract"]["navigation_outcomes_read"] is False


def test_cached_decision_drift_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="cached decision drifted"):
        query_record(write_receipt(
            tmp_path, "novel", [plan(), plan(cached=True, precheck=False,
                                             pose=False,
                                             reason="precheck_fundamental_inliers")]))
