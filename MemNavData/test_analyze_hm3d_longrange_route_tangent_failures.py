import hashlib
import json

from MemNavData.analyze_hm3d_longrange_route_tangent_failures import analyze


ARMS = (
    "mono_native", "mono_cec_endpoint", "mono_cec_route_tangent",
)


def write_with_receipt(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n")
    return digest


def test_fixed_failure_audit_closes_against_formal_totals(tmp_path):
    population = tmp_path / "population"
    run_root = tmp_path / "run"
    manifest_path = population / "role_pairs/manifest.json"
    episodes = [{
        "scene": f"scene_{index % 8}",
        "episode": f"episode_{index:03d}",
        "pairs": [{"queries": [
            {"analysis_role": "novel", "floor_position": [9.0, 0.0, 9.0]},
            {"analysis_role": "revisit", "floor_position": [1.0, 0.0, 0.0]},
        ]}],
    } for index in range(23)]
    manifest_sha = write_with_receipt(manifest_path, {"episodes": episodes})

    outcomes = {
        "mono_native": [int(index < 2) for index in range(23)],
        "mono_cec_endpoint": [int(index < 10) for index in range(23)],
        "mono_cec_route_tangent": [int(index < 17) for index in range(23)],
    }
    for index, item in enumerate(episodes):
        label = f"{index:03d}_{item['scene']}_{item['episode']}"
        episode_root = run_root / "evaluation" / label
        completion = {
            "prefix_equality": True,
            "initial_cec_proof_equal_between_methods": True,
            "outcomes": {arm: outcomes[arm][index] for arm in ARMS},
            "final_goal_3d_dist_m": {
                arm: 0.5 if outcomes[arm][index] else 2.0 for arm in ARMS
            },
            "initial_geodesic_m": {arm: 1.0 for arm in ARMS},
            "path_len_m": {
                arm: 0.5 if outcomes[arm][index] else 1.0
                for arm in ARMS
            },
            "steps": {arm: 10 for arm in ARMS},
            "termination_reason": {
                arm: "success" if outcomes[arm][index] else "max_steps"
                for arm in ARMS
            },
            "fully_rejected_exact_native": {
                "mono_cec_endpoint": False,
                "mono_cec_route_tangent": False,
            },
            "wall_time_seconds": {arm: 2.0 for arm in ARMS},
        }
        write_with_receipt(episode_root / "completion.json", completion)
        for arm in ARMS:
            reached = bool(outcomes[arm][index])
            end_position = [0.5, 0.0, 0.0] if reached else [-1.0, 0.0, 0.0]
            path_len_m = 0.5 if reached else 1.0
            plan = {
                "navdp_critic_max": -0.25,
                "evaluation_gt_goal_distance_m": 2.0,
                "cec_total_decision_ms": 25.0,
                "certified_relocalization_ms": 5.0,
                "cec_controller_ms": 15.0,
                "cec_depth_sidecar_ms": 3.0,
                "certified_relocalization_accepted": arm != "mono_native",
                "router_selected_anchor": (
                    None if arm == "mono_native" else 8),
            }
            if arm == "mono_cec_route_tangent":
                plan.update({
                    "local_tangent_status": "active",
                    "local_tangent_unit_bearing": [1.0, 0.0],
                    "local_tangent_controller_pointgoal": [2.5, 0.0],
                    "local_tangent_progress_fraction": 0.25,
                    "local_tangent_cross_track_error_m": 0.1,
                    "local_tangent_history_edge_count": 12,
                    "local_tangent_route_extent_m": 22.0,
                    "local_tangent_update_edge_receipts": [{
                        "pnp_inliers": 24,
                        "pnp_reprojection_rmse_px": 0.8,
                    }],
                })
            payload = {
                "query_leg": [plan],
                "rollout_traces": {
                    "query": [{
                        "x": 0.0, "y": 0.0, "z": 0.0,
                        "executed_yaw_rad_since_previous_frame": 0.25,
                    }],
                },
                "query_result": {
                    "end_position": end_position,
                    "path_len_m": path_len_m,
                    "blocked_step_count": 0,
                },
            }
            write_with_receipt(
                episode_root / arm / f"{item['episode']}_plans.json", payload)

    successes = {arm: sum(outcomes[arm]) for arm in ARMS}
    formal_summary_path = run_root / "result/summary.json"
    formal_summary_sha = write_with_receipt(formal_summary_path, {
        "schema_version": "hm3d_longrange_route_tangent_result_v1_20260903",
        "successes": successes,
    })
    formal_verification_path = run_root / "result/independent_verification.json"
    write_with_receipt(formal_verification_path, {
        "schema_version": (
            "hm3d_longrange_route_tangent_result_verification_v1_20260903"),
        "verified": True,
        "summary_sha256": formal_summary_sha,
        "histories": 23,
        "benchmark_manifest_sha256": manifest_sha,
    })
    freeze_path = tmp_path / "failure_freeze.json"
    write_with_receipt(freeze_path, {
        "schema_version": (
            "hm3d_longrange_route_tangent_failure_audit_freeze_v1_20260903"),
        "formal_run_root": str(run_root.resolve()),
    })

    result = analyze(
        population=population,
        run_root=run_root,
        formal_summary_path=formal_summary_path,
        formal_verification_path=formal_verification_path,
        freeze_path=freeze_path,
    )
    assert {arm: result["panels"][arm]["successes"] for arm in ARMS} == (
        successes)
    assert len(result["primary_discordant_pairs"]) == 7
    route = result["panels"]["mono_cec_route_tangent"]
    native = result["panels"]["mono_native"]
    assert native["initial_certificate_reject_episode_count"] == 23
    assert route["initial_certificate_reject_episode_count"] == 0
    assert route["selected_anchor_counts"] == {"8": 23}
    assert route["initial_proof_equality_count"] == 23
    assert route["final_route_progress_fraction"]["median"] == 0.25
    assert route["max_route_progress_fraction"]["median"] == 0.25
    assert route["requested_heading_above_90deg_fraction"] == 0.0
    assert route["live_update_pnp_inliers"]["median"] == 24.0
    assert route["reported_minus_recomputed_path_m"]["max"] == 0.0
    assert route["minimum_actionwise_3d_goal_distance_m"]["median"] == 0.5
    assert route["minimum_actionwise_planar_goal_distance_m"]["median"] == 0.5
    assert route["reported_final_minus_actionwise_terminal_3d_m"]["max"] == 0.0
    assert route["runtime"]["initial_relocalization_ms"]["median"] == 5.0
    assert route["runtime"]["total_decision_ms"]["median"] == 25.0
    assert route["runtime"]["arm_wall_time_seconds"]["median"] == 2.0
    assert route["spl"] == 17 / 23
    assert route["recomputed_3d_path_m"]["median"] == 0.5
    assert route["total_absolute_yaw_deg"]["median"] > 14.0
