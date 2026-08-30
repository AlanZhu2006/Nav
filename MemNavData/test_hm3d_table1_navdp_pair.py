import csv
import hashlib
import json
from pathlib import Path

import pytest

from MemNavData.aggregate_hm3d_table1_navdp_pair import (
    _direction_stratum,
    aggregate,
    digest,
)
from MemNavData.independent_verify_hm3d_table1_navdp_pair import verify


ARMS = ("mono_native", "mono_cec")
ROLES = ("novel", "revisit")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_metric(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plan(*, accepted: bool, trace: list[list[float]], reached: int) -> dict:
    return {
        "analysis_role_not_forwarded": True,
        "query_leg": [{
            "navdp_depth_source": "monocular_sidecar",
            "metric_depth_sensor_consumed": False,
            "monocular_depth_receipt": {
                "frame_index": 40,
                "scale_active": True,
                "scale_receipt_sha256": "a" * 64,
            },
            "certified_relocalization_accepted": accepted,
            "certified_relocalization_reason": (
                "accepted" if accepted else "insufficient_support"
            ),
            "cec_takeover": accepted,
        }],
        "rollout_traces": {"query": trace},
        "query_result": {"reached": bool(reached), "trace": trace},
    }


def make_fixture(tmp_path: Path, *, authorized: bool = True):
    benchmark = tmp_path / "population/natural_direction"
    episodes = []
    for index in range(2):
        scene = f"scene{index}"
        episode = "episode_0000"
        episodes.append({
            "scene": scene,
            "episode": episode,
            "pairs": [{
                "pair_id": "pair_00",
                "queries": [
                    {
                        "analysis_role": "novel",
                        "query_id": "pair_00_novel",
                        "assigned_direction_stratum": (
                            "front" if index == 0 else "side"
                        ),
                    },
                    {"analysis_role": "revisit", "query_id": "pair_00_revisit"},
                ],
            }],
        })
    manifest = {"episodes": episodes}
    write_json(benchmark / "manifest.json", manifest)
    verification_path = tmp_path / "construction_verification.json"
    write_json(verification_path, {
        "verified": True,
        "construction_only": True,
        "formal_policy_evaluation_authorized": authorized,
        "benchmark_manifest_sha256": digest(benchmark / "manifest.json"),
        "histories": 2,
        "scene_clusters": 2,
    })

    run = tmp_path / "run"
    for index, item in enumerate(episodes):
        scene, episode = item["scene"], item["episode"]
        root = (run / "evaluation/natural_direction"
                / f"{index:03d}_{scene}_{episode}")
        order = list(ARMS if index % 2 == 0 else reversed(ARMS))
        completion = {
            "benchmark_manifest_sha256": digest(benchmark / "manifest.json"),
            "arms": list(ARMS),
            "arm_order": order,
            "prefix_equality": True,
            "runtime_role_visibility": "none",
            "online_a_depth_source": "monocular_sidecar",
            "fully_rejected_exact_native": {"novel": True, "revisit": False},
        }
        write_json(root / "completion.json", completion)
        completion_hash = hashlib.sha256(
            (root / "completion.json").read_bytes()).hexdigest()
        (root / "completion.json.sha256").write_text(
            completion_hash + "  completion.json\n")
        for arm in ARMS:
            metric_rows = []
            for role in ROLES:
                accepted = arm == "mono_cec" and role == "revisit"
                reached = int(accepted)
                if arm == "mono_native":
                    reached = 0
                trace = [[0.0, 0.0], [1.0, 0.0]]
                payload = plan(accepted=accepted, trace=trace, reached=reached)
                write_json(
                    root / arm / f"{episode}_pair_00_{role}_plans.json",
                    payload,
                )
                metric_rows.append({
                    "analysis_role": role,
                    "reached": reached,
                    "final_goal_dist_m": 0.8 if reached else 2.0,
                    "geodesic_m": 4.0,
                    "path_len_m": 4.5,
                    "steps": 20,
                    "navdp_depth_source": "monocular_sidecar",
                    "metric_depth_sensor_consumed_any": 0,
                    "monocular_receipt_plans": 1,
                    "monocular_active_receipt_plans": 1,
                    "monocular_scale_receipt_hashes": 1,
                    "certificate_accept_plans": int(accepted),
                    "runtime_failure_plans": 0,
                })
            write_metric(root / arm / "metric.csv", metric_rows)
    return run, benchmark, verification_path


def test_aggregate_and_independent_verifier(tmp_path):
    run, benchmark, construction = make_fixture(tmp_path)
    summary = aggregate(
        run, benchmark, construction,
        claim_scope="conference_table_hm3d_fresh_query_scene_overlap",
        bootstrap_samples=1000,
    )
    assert summary["results"]["all"]["native_success"] == 0
    assert summary["results"]["all"]["cec_success"] == 2
    assert summary["results"]["revisit"]["paired_gain"] == 2
    assert summary["safety"]["novel_takeover_queries"] == 0
    summary_path = run / "summary.json"
    write_json(summary_path, summary)
    checked = verify(run, benchmark, construction, summary_path)
    assert checked["verified"] is True
    assert checked["raw_metric_rows"] == 8
    assert checked["fully_rejected_exact_native_queries"] == 2


def test_direction_stratum_uses_canonical_novel_query(tmp_path):
    _run, benchmark, _construction = make_fixture(tmp_path)
    manifest = json.loads((benchmark / "manifest.json").read_text())
    assert _direction_stratum(manifest["episodes"][0]) == "front"
    assert _direction_stratum(manifest["episodes"][1]) == "side"


def test_power_gate_blocks_aggregation(tmp_path):
    run, benchmark, construction = make_fixture(tmp_path, authorized=False)
    with pytest.raises(RuntimeError, match="did not authorize"):
        aggregate(
            run, benchmark, construction,
            claim_scope="must_not_run", bootstrap_samples=10,
        )


def test_runtime_failure_is_infrastructure_not_a_policy_outcome(tmp_path):
    run, benchmark, construction = make_fixture(tmp_path)
    root = (run / "evaluation/natural_direction"
            / "000_scene0_episode_0000" / "mono_cec")
    metric_path = root / "metric.csv"
    with metric_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["runtime_failure_plans"] = 1
    write_metric(metric_path, rows)
    with pytest.raises(RuntimeError, match="runtime failure"):
        aggregate(
            run, benchmark, construction,
            claim_scope="must_not_seal", bootstrap_samples=10,
        )

    # The independent verifier must reject the same defect even if a stale
    # clean aggregate were presented to it.
    rows[0]["runtime_failure_plans"] = 0
    write_metric(metric_path, rows)
    summary = aggregate(
        run, benchmark, construction,
        claim_scope="must_not_seal", bootstrap_samples=10,
    )
    summary_path = run / "summary.json"
    write_json(summary_path, summary)
    rows[0]["runtime_failure_plans"] = 1
    write_metric(metric_path, rows)
    payload_path = root / "episode_0000_pair_00_novel_plans.json"
    payload = json.loads(payload_path.read_text())
    payload["query_leg"][0][
        "certified_relocalization_reason"
    ] = "certificate_endpoint_failure"
    write_json(payload_path, payload)
    with pytest.raises(RuntimeError, match="runtime failure"):
        verify(run, benchmark, construction, summary_path)
