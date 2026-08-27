import csv
import hashlib
import json

import pytest

from summarize_certified_relocalization_closed_loop import (
    ARMS,
    WILLIAMS_ORDERS,
    decision_branch,
    summarize,
    validate_certified_episode,
)


def _metric(**updates):
    payload = {
        "episode": "episode_0000",
        "certified_relocalization_request_count": "1",
        "certified_relocalization_uncached_count": "1",
        "certified_relocalization_accept_count": "1",
        "certified_relocalization_runtime_failure_count": "0",
        "revisit_adapter_takeover_plan_count": "1",
        "revisit_adapter_abstain_plan_count": "0",
    }
    payload.update(updates)
    return payload


def _accepted_plan():
    return {
        "certified_relocalization_metric_scale": None,
        "certified_relocalization_ok": True,
        "certified_relocalization_accepted": True,
        "certified_relocalization_cached": False,
        "certified_relocalization_uncached_ms": 2100.0,
        "certified_relocalization_pointgoal_units": (
            "lingbot_raw_direction_only"),
        "certified_relocalization_pnp": {"inliers": 42},
        "revisit_adapter_mode": "verified_bearing_v1",
        "revisit_adapter_source": "lightglue_lingbot_pnp_v2_scale_free",
        "revisit_adapter_takeover": True,
        "memory_unbounded_pointgoal_units": "lingbot_raw_direction_only",
        "memory_unbounded_pointgoal_distance_m": None,
        "memory_controller_pointgoal_distance_m": 2.5,
        "pose_controller": "navdp_image_point_mix",
        "router_selected_candidate_dino_rank": 3,
    }


def test_williams_orders_balance_positions_and_use_every_arm_once():
    for order in WILLIAMS_ORDERS:
        assert set(order) == set(ARMS)
    for position in range(len(ARMS)):
        assert {order[position] for order in WILLIAMS_ORDERS} == set(ARMS)


def test_certified_episode_acceptance_contract(tmp_path):
    (tmp_path / "episode_0000_plans.json").write_text(json.dumps({
        "legB": [_accepted_plan()],
    }))
    audit = validate_certified_episode(
        tmp_path, _metric(), reached_a=True)
    assert audit == {
        "requests": 1,
        "accepted_plans": 1,
        "takeover_episode": True,
        "fallback_episode": False,
        "selected_dino_ranks": [3],
        "uncached_ms": [2100.0],
        "pnp_inliers": [42],
    }


def test_certified_episode_rejection_must_use_native_fallback(tmp_path):
    rejected = {
        "certified_relocalization_metric_scale": None,
        "certified_relocalization_ok": True,
        "certified_relocalization_accepted": False,
        "certified_relocalization_cached": False,
        "certified_relocalization_uncached_ms": 900.0,
        "certified_relocalization_pointgoal_units": None,
        "certified_relocalization_pnp": {"status": "insufficient_inliers"},
        "revisit_adapter_mode": "verified_bearing_v1",
        "revisit_adapter_source": "lightglue_lingbot_pnp_v2_scale_free",
        "revisit_adapter_takeover": False,
        "revisit_adapter_controller_contract": "native_imagegoal",
        "pose_controller": "navdp_image_router",
        "router_selected_candidate_dino_rank": 1,
    }
    (tmp_path / "episode_0000_plans.json").write_text(json.dumps({
        "legB": [rejected],
    }))
    audit = validate_certified_episode(
        tmp_path,
        _metric(
            certified_relocalization_accept_count="0",
            revisit_adapter_takeover_plan_count="0",
            revisit_adapter_abstain_plan_count="1"),
        reached_a=True,
    )
    assert audit["fallback_episode"] is True
    assert audit["accepted_plans"] == 0


def test_certified_episode_rejects_metric_scale_leak(tmp_path):
    plan = _accepted_plan()
    plan["certified_relocalization_metric_scale"] = 2.0
    (tmp_path / "episode_0000_plans.json").write_text(json.dumps({
        "legB": [plan],
    }))
    with pytest.raises(RuntimeError, match="metric scale leaked"):
        validate_certified_episode(tmp_path, _metric(), reached_a=True)


@pytest.mark.parametrize(
    ("delta", "pvalue", "interval", "expected"),
    [
        (0.1, 0.01, [0.02, 0.2],
         "certified_router_has_closed_loop_value_"
         "seek_fresh_scene_open_set_confirmation"),
        (-0.1, 0.01, [-0.2, -0.02],
         "reject_certified_router_retain_known_role_system"),
        (0.1, 0.2, [-0.01, 0.2],
         "inconclusive_do_not_retune_on_consumed_pool"),
    ],
)
def test_decision_branch(delta, pvalue, interval, expected):
    assert decision_branch({
        "joint_sr_delta_right_minus_left": delta,
        "mcnemar_exact_two_sided_p": pvalue,
        "scene_cluster_bootstrap_risk_difference_95": interval,
    }) == expected


def test_complete_synthetic_four_arm_run_passes_every_receipt(tmp_path):
    scenes = [f"scene_{index:02d}" for index in range(20)]
    episodes = {
        scene: [{"episode": f"episode_{index:04d}"} for index in range(8)]
        for scene in scenes
    }
    manifest = {
        "audit": {"status": "ok", "training_scene_overlap": []},
        "data_role_guards": {"blind_allowed": False},
        "scenes": scenes,
        "episodes": episodes,
        "analysis": {
            "cluster_bootstrap_seed": 17,
            "cluster_bootstrap_resamples": 200,
        },
    }
    manifest_path = tmp_path / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (tmp_path / "source_bundle.sha256").write_text(
        "synthetic receipt\n", encoding="utf-8")

    common_summary = {
        "episodes": 8,
        "max_steps": 500,
        "exec_horizon": 8,
        "leg1_mode": "shared_trace",
        "write_leg1_trace": False,
        "deterministic_plan_seeds": True,
        "trajectory_selector": "server",
        "trajectory_selector_scope": "all",
    }
    certified_contract = {
        "schema_version": 3,
        "geometry_certificate_version": 2,
        "candidate_top_k": 8,
        "candidate_min_gap": 4,
        "minimum_anchor": 8,
        "candidate_lifecycle": "frozen_at_first_goal_query",
        "empty_candidate_semantics": "cached_native_abstention",
        "output": "scale_free_relative_bearing",
        "pointgoal_units": "lingbot_raw_direction_only",
        "metric_distance_certified": False,
        "controller_adapter": "verified_bearing_v1_fixed_2.5m",
        "fallback": "native_imagegoal",
    }

    for scene_index, scene in enumerate(scenes):
        scene_root = tmp_path / "scenes" / f"{scene_index:02d}_{scene}"
        scene_root.mkdir(parents=True)
        (scene_root / "scene_contract.json").write_text(json.dumps({
            "schema_version": "certified_relocalization_closed_loop_v1",
            "scene": scene,
            "scene_index": scene_index,
            "manifest_sha256": manifest_sha,
            "arm_order": list(WILLIAMS_ORDERS[scene_index % 4]),
        }), encoding="utf-8")
        trace_root = scene_root / "trace_source"
        trace_root.mkdir()
        (trace_root / "summary.json").write_text(json.dumps({
            "episodes": 8,
            "server_backend": "hybrid_pose",
            "hybrid_route": "phase",
            "leg1_mode": "policy",
            "stop_after_leg1": True,
            "write_leg1_trace": True,
            "deterministic_plan_seeds": True,
        }), encoding="utf-8")

        trace_hashes = {}
        for episode_row in episodes[scene]:
            episode = episode_row["episode"]
            trace_path = trace_root / f"{episode}_leg1_trace.json"
            trace_path.write_text(json.dumps({
                "scene": scene, "episode": episode,
            }), encoding="utf-8")
            trace_hashes[episode] = hashlib.sha256(
                trace_path.read_bytes()).hexdigest()

        for arm in ARMS:
            arm_root = scene_root / arm
            arm_root.mkdir()
            summary = dict(common_summary)
            if arm == "certified_relocalization":
                summary.update({
                    "server_backend": "hybrid_pose",
                    "hybrid_route": "certified_relocalization",
                    "revisit_controller": "navdp_mixed",
                    "revisit_adapter": "verified_bearing_v1",
                    "revisit_adapter_fixed_radius_m": 2.5,
                    "graph_subgoal_spacing_m": 0.0,
                    "graph_subgoal_arrival_m": 0.6,
                    "certified_relocalization_server": {
                        "enabled": True,
                        "runtime_contract": certified_contract,
                    },
                })
            elif arm == "known_revisit_direct":
                summary.update({
                    "server_backend": "hybrid_pose",
                    "hybrid_route": "phase",
                    "revisit_controller": "navdp_mixed",
                    "revisit_adapter": "legacy_metric",
                    "graph_subgoal_spacing_m": 0.0,
                    "graph_subgoal_arrival_m": 0.6,
                })
            elif arm == "geometry_router":
                summary.update({
                    "server_backend": "hybrid_pose",
                    "hybrid_route": "memory_geometry",
                    "revisit_controller": "navdp_mixed",
                    "revisit_adapter": "legacy_metric",
                    "graph_subgoal_spacing_m": 0.0,
                    "graph_subgoal_arrival_m": 0.6,
                    "router_visual_floor": 0.88,
                    "router_min_matches": 20,
                    "router_min_inliers": 12,
                    "router_min_inlier_ratio": 0.5,
                    "router_confirm_plans": 2,
                    "router_verify_top_k": 8,
                })
            else:
                summary.update({
                    "server_backend": "navdp",
                    "hybrid_route": "phase",
                    "revisit_adapter": "legacy_metric",
                    "graph_subgoal_spacing_m": None,
                    "graph_subgoal_arrival_m": None,
                })
            (arm_root / "summary.json").write_text(
                json.dumps(summary), encoding="utf-8")

            metrics = []
            for episode_index, episode_row in enumerate(episodes[scene]):
                episode = episode_row["episode"]
                reached_b = arm in (
                    "certified_relocalization", "known_revisit_direct")
                plan = _accepted_plan()
                plan.update({
                    "requested_diffusion_seed": episode_index,
                    "diffusion_seed": episode_index,
                })
                plans = {
                    "legA": [],
                    "legB": ([plan]
                             if arm == "certified_relocalization" else []),
                    "leg1_trace_sha256": trace_hashes[episode],
                }
                (arm_root / f"{episode}_plans.json").write_text(
                    json.dumps(plans), encoding="utf-8")
                metric = {
                    "episode": episode,
                    "seed": 1000 + episode_index,
                    "recall_gap": 32,
                    "reached_A": 1,
                    "reached_B": int(reached_b),
                    "spl_A": 0.8,
                    "spl_B": 0.7 if reached_b else 0.0,
                    "geo_A": 5.0,
                    "geo_B": 6.0,
                    "len_A": 6.0,
                    "len_B": 7.0,
                    "final_dist_A": 0.5,
                    "terminal_final_goal_dist_m": (
                        0.5 if reached_b else 5.0),
                    "steps_A": 16,
                    "steps_B": 24,
                    "deterministic_plan_seeds": True,
                    "leg1_trace_sha256": trace_hashes[episode],
                }
                if arm == "certified_relocalization":
                    metric.update({
                        "certified_relocalization_request_count": 1,
                        "certified_relocalization_uncached_count": 1,
                        "certified_relocalization_accept_count": 1,
                        "certified_relocalization_runtime_failure_count": 0,
                        "revisit_adapter_takeover_plan_count": 1,
                        "revisit_adapter_abstain_plan_count": 0,
                    })
                metrics.append(metric)
            with (arm_root / "metric.csv").open(
                    "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=metrics[0].keys())
                writer.writeheader()
                writer.writerows(metrics)

    report = summarize(manifest_path, tmp_path)
    assert report["audit"]["status"] == "ok"
    assert report["arms"]["certified_relocalization"]["joint"][
        "successes"] == 160
    assert report["contrasts"]["certified_minus_native"]["joint"][
        "outcomes"]["right_only_joint_success"] == 160
    assert report["decision"]["branch"] == (
        "certified_router_has_closed_loop_value_"
        "seek_fresh_scene_open_set_confirmation")
