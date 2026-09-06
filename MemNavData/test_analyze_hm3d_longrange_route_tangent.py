import hashlib
import json

from MemNavData.analyze_hm3d_longrange_route_tangent import analyze
from MemNavData.independent_verify_hm3d_longrange_route_tangent_result import (
    verify,
)


ARMS = (
    "mono_native",
    "mono_cec_endpoint",
    "mono_cec_route_tangent",
)


def _write_json_with_receipt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n")
    return digest


def _accepted_route_plan():
    return {
        "certified_relocalization_accepted": True,
        "local_tangent_evaluator_pose_consumed": False,
        "local_tangent_habitat_path_consumed": False,
        "local_tangent_executor_odometry_consumed": False,
        "local_tangent_metric_depth_sensor_consumed": False,
        "local_tangent_distance_regime_present": False,
        "local_tangent_stuck_trigger_present": False,
        "local_tangent_native_fallback_available": False,
        "local_tangent_endpoint_fallback_available": False,
    }


def test_complete_fresh_result_is_analyzed_and_independently_reproduced(
        tmp_path):
    population = tmp_path / "population"
    run_root = tmp_path / "run"
    manifest_path = population / "role_pairs" / "manifest.json"
    episodes = []
    # Seven paired gains and no loss imply exact McNemar p=0.015625.  The
    # order mirrors the result-blind scene-round-robin population: the first
    # 15 histories are exactly the max-two-per-scene sensitivity subset.
    scene_order = [
        *[f"scene_{index:02d}" for index in range(8)],
        *[f"scene_{index:02d}" for index in range(1, 8)],
        "scene_01", "scene_06", "scene_07",
        "scene_06", "scene_07",
        "scene_06", "scene_07",
        "scene_06",
    ]
    assert len(scene_order) == 23
    for index, scene in enumerate(scene_order):
        episodes.append({
            "scene": scene,
            "episode": f"episode_{index:03d}",
            "candidate_identity_sha256": f"{index + 1:064x}",
        })
    manifest_digest = _write_json_with_receipt(
        manifest_path, {"episodes": episodes})
    _write_json_with_receipt(
        population / "independent_verification.json",
        {
            "verified": True,
            "formal_policy_evaluation_authorized": True,
            "benchmark_manifest_sha256": manifest_digest,
        },
    )

    for index, item in enumerate(episodes):
        label = f"{index:03d}_{item['scene']}_{item['episode']}"
        root = run_root / "evaluation" / label
        endpoint = int(index < 10)
        tangent = int(index < 17)
        native = int(index < 2)
        outcomes = {
            "mono_native": native,
            "mono_cec_endpoint": endpoint,
            "mono_cec_route_tangent": tangent,
        }
        final = {
            arm: 0.5 if success else 2.0
            for arm, success in outcomes.items()
        }
        completion = {
            "schema_version": (
                "hm3d_longrange_route_tangent_history_result_v1_20260903"
            ),
            "history_index": index,
            "scene": item["scene"],
            "episode": item["episode"],
            "candidate_identity_sha256": item[
                "candidate_identity_sha256"],
            "benchmark_manifest_sha256": manifest_digest,
            "arms": list(ARMS),
            "arm_order": list(ARMS[index % 3:] + ARMS[:index % 3]),
            "prefix_equality": True,
            "initial_cec_proof_equal_between_methods": True,
            "success_geometry": "three_dimensional_euclidean",
            "success_distance_m": 1.0,
            "revisit_vertical_error_m": 0.1,
            "initial_geodesic_m": {arm: 25.0 for arm in ARMS},
            "outcomes": outcomes,
            "final_goal_3d_dist_m": final,
            "final_goal_planar_dist_m": final,
            "final_goal_vertical_error_m": {arm: 0.1 for arm in ARMS},
            "path_len_m": {arm: 20.0 for arm in ARMS},
            "steps": {arm: 100 for arm in ARMS},
            "termination_reason": {arm: "success" for arm in ARMS},
            "geometry_stream_stop_plans": 0,
            "fully_rejected_exact_native": {
                "mono_cec_endpoint": False,
                "mono_cec_route_tangent": False,
            },
        }
        _write_json_with_receipt(root / "completion.json", completion)
        (root / "mono_cec_route_tangent").mkdir(parents=True)
        (root / "mono_cec_route_tangent" / "episode_plans.json").write_text(
            json.dumps({
                "analysis_role_not_forwarded": True,
                "query_leg": [_accepted_route_plan()],
            }) + "\n")

    summary = analyze(population=population, run_root=run_root)
    assert summary["histories"] == 23
    assert summary["scene_clusters"] == 8
    assert summary["successes"] == {
        "mono_native": 2,
        "mono_cec_endpoint": 10,
        "mono_cec_route_tangent": 17,
    }
    assert summary["primary_contrast"]["gains"] == 7
    assert summary["primary_contrast"]["losses"] == 0
    assert summary["primary_contrast"][
        "exact_mcnemar_two_sided_p"] == 0.015625
    assert summary["scene_balanced_max2_sensitivity"]["n"] == 15
    assert summary["scene_balanced_max2_sensitivity"][
        "risk_difference"] > 0.0
    assert summary["decision"] == (
        "confirm_route_tangent_for_longrange_extension")

    summary_path = run_root / "result" / "summary.json"
    _write_json_with_receipt(summary_path, summary)
    checked = verify(
        population=population,
        run_root=run_root,
        summary_path=summary_path,
    )
    assert checked["verified"] is True
    assert checked["primary_gains"] == 7
    assert checked["primary_losses"] == 0
    assert checked["successes"] == summary["successes"]
