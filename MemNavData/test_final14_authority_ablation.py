import csv
import hashlib
import json
from pathlib import Path
import sys

from MemNavData.final14_authority_ablation import (
    ARMS,
    AUTHORITY_POLICY,
    DEPTH_SOURCE,
    EVALUATOR_ARM,
    HYBRID_ROUTE,
    REVISIT_ADAPTER,
    rotated_arm_order,
)
from MemNavData import independent_verify_final14_authority_ablation as verifier
from MemNavData import summarize_final14_authority_ablation as summarizer
from MemNavData.run_final14_authority_ablation_episode import (
    audit_initial_proposal_pair,
)


def test_authority_ablation_is_one_variable_and_balanced():
    assert ARMS == ("mono_cec", "mono_unthresholded_witness")
    assert {DEPTH_SOURCE} == {"monocular_sidecar"}
    assert {REVISIT_ADAPTER} == {"verified_bearing_v1"}
    assert set(HYBRID_ROUTE) == set(ARMS)
    assert set(EVALUATOR_ARM) == set(ARMS)
    assert AUTHORITY_POLICY == {
        "mono_cec": "strict_certificate",
        "mono_unthresholded_witness": "pnp_pose_available",
    }
    assert rotated_arm_order(0) == ARMS
    assert rotated_arm_order(1) == tuple(reversed(ARMS))
    assert rotated_arm_order(2) == ARMS


def test_route_names_make_geometry_explicit():
    assert HYBRID_ROUTE["mono_cec"] == "certified_relocalization"
    assert HYBRID_ROUTE["mono_unthresholded_witness"] == (
        "certified_unthresholded_witness")
    assert "dino" not in HYBRID_ROUTE["mono_unthresholded_witness"]


def _payload(policy, accepted, *, anchor=8, pnp=None):
    return {"query_leg": [{
        "certified_relocalization_authority_policy": policy,
        "certified_relocalization_proposal_order": "geometry_first",
        "certified_relocalization_accepted": accepted,
        "certified_relocalization_reason": (
            "certificate_accepted" if policy == "strict_certificate"
            else "pnp_pose_available"),
        "certified_relocalization_certificate": {"accepted": accepted},
        "certified_relocalization_pnp": pnp,
        "router_candidate_order_dino": [9, 8],
        "router_candidate_order_used": [8, 9],
        "router_selected_anchor": anchor,
        "router_selected_candidate_dino_rank": 2,
        "aux_pose": [1.0, 0.0] if accepted else None,
    }]}


def test_initial_pair_audit_exposes_authority_discordance():
    strict = _payload("strict_certificate", False, pnp={"status": "precheck"})
    witness = _payload(
        "pnp_pose_available", True,
        pnp={"status": "insufficient_inliers", "pose9": [0.0] * 9})
    result = audit_initial_proposal_pair(strict, witness, label="unit/novel")
    assert result["proposal_fields_equal"] is True
    assert result["authority_discordant"] is True


def test_initial_pair_audit_requires_identical_accepted_witness():
    pnp = {"status": "ok", "pose9": [0.0] * 9}
    strict = _payload("strict_certificate", True, pnp=pnp)
    witness = _payload("pnp_pose_available", True, pnp=dict(pnp))
    result = audit_initial_proposal_pair(strict, witness, label="unit/revisit")
    assert result["strict_accepted"] is True
    assert result["unthresholded_witness_accepted"] is True


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def test_summary_and_independent_verifier_recompute_raw_receipts(
        tmp_path, monkeypatch):
    bench = tmp_path / "bench"
    run = tmp_path / "run"
    manifest = {"episodes": [
        {"scene": f"scene{i:02d}", "episode": "episode_0000"}
        for i in range(21)
    ]}
    manifest_path = bench / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    fields = [
        "analysis_role", "query_id", "reached", "final_goal_dist_m",
        "navdp_depth_source", "metric_depth_sensor_consumed_any",
        "monocular_receipt_plans", "monocular_active_receipt_plans",
        "monocular_scale_receipt_hashes", "runtime_failure_plans",
        "certificate_accept_plans",
    ]
    for index, item in enumerate(manifest["episodes"]):
        scene, episode = item["scene"], item["episode"]
        label = f"{index:03d}_{scene}_{episode}"
        root = run / "evaluation/natural_direction" / label
        audits = {}
        outcomes = {arm: {} for arm in ARMS}
        distances = {arm: {} for arm in ARMS}
        accepts = {arm: {} for arm in ARMS}
        for arm in ARMS:
            arm_root = root / arm
            arm_root.mkdir(parents=True)
            rows = []
            for role in ("novel", "revisit"):
                strict = arm == "mono_cec"
                accepted = role == "revisit" or not strict
                reached = role == "revisit" or strict
                query_id = f"pair_00_{role}"
                rows.append({
                    "analysis_role": role,
                    "query_id": query_id,
                    "reached": int(reached),
                    "final_goal_dist_m": 0.5 if reached else 2.0,
                    "navdp_depth_source": "monocular_sidecar",
                    "metric_depth_sensor_consumed_any": 0,
                    "monocular_receipt_plans": 1,
                    "monocular_active_receipt_plans": 1,
                    "monocular_scale_receipt_hashes": 1,
                    "runtime_failure_plans": 0,
                    "certificate_accept_plans": int(accepted),
                })
                first = {
                    "certified_relocalization_authority_policy":
                        AUTHORITY_POLICY[arm],
                    "certified_relocalization_proposal_order":
                        "geometry_first",
                    "certified_relocalization_accepted": accepted,
                    "router_candidate_order_dino": [9, 8],
                    "router_candidate_order_used": [8, 9],
                    "router_selected_anchor": 8,
                    "router_selected_candidate_dino_rank": 2,
                }
                _write_json(
                    arm_root / f"{episode}_{query_id}_plans.json",
                    {"analysis_role_not_forwarded": True,
                     "query_leg": [first]},
                )
                outcomes[arm][role] = int(reached)
                distances[arm][role] = 0.5 if reached else 2.0
                accepts[arm][role] = int(accepted)
            with (arm_root / "metric.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        for role in ("novel", "revisit"):
            strict_accept = role == "revisit"
            witness_accept = True
            audits[role] = {
                "proposal_fields_equal": True,
                "strict_accepted": strict_accept,
                "unthresholded_witness_accepted": witness_accept,
                "authority_discordant": witness_accept and not strict_accept,
            }
        completion = {
            "benchmark_manifest_sha256": manifest_sha,
            "arm_order": list(rotated_arm_order(index)),
            "prefix_equality": True,
            "initial_proposal_equality": True,
            "runtime_role_visibility": "none",
            "outcomes": outcomes,
            "final_distance_m": distances,
            "operational_accept_plans": accepts,
            "initial_proposal_audit": audits,
        }
        completion_path = root / "completion.json"
        _write_json(completion_path, completion)
        digest = hashlib.sha256(completion_path.read_bytes()).hexdigest()
        (root / "completion.json.sha256").write_text(
            f"{digest}  completion.json\n")

    summary_path = run / "POSTHOC/summary.json"
    monkeypatch.setattr(summarizer, "BOOTSTRAP_RESAMPLES", 100)
    monkeypatch.setattr(sys, "argv", [
        "summarize", "--run-root", str(run), "--bench-root", str(bench),
        "--expected-manifest-sha256", manifest_sha,
        "--out", str(summary_path),
    ])
    summarizer.main()
    summary = json.loads(summary_path.read_text())
    assert summary["results"]["novel"]["arms"]["mono_cec"][
        "successes"] == 21
    assert summary["results"]["novel"]["arms"][
        "mono_unthresholded_witness"]["successes"] == 0
    assert summary["authority_discordant_queries_by_role"] == {
        "novel": 21, "revisit": 0}

    verify_path = run / "POSTHOC/verify.json"
    monkeypatch.setattr(sys, "argv", [
        "verify", "--run-root", str(run), "--bench-root", str(bench),
        "--expected-manifest-sha256", manifest_sha,
        "--summary", str(summary_path), "--out", str(verify_path),
    ])
    verifier.main()
    receipt = json.loads(verify_path.read_text())
    assert receipt["verified"] is True
    assert receipt["proposal_pairs_verified"] == 42
