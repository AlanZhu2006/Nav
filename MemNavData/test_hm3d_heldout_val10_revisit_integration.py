from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.summarize_hm3d_heldout_val10_revisit import (
    RUNTIME_REPAIR_SCENE_SCHEMA,
    WILLIAMS_ORDERS,
    summarize,
)
from MemNavData.verify_hm3d_heldout_val10_revisit import verify


ARMS = (
    "native",
    "raw_fixed_oracle_role",
    "geometry_router",
    "certified_relocalization",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n",
                    encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _common_summary() -> dict:
    return {
        "episodes": 4,
        "max_steps": 500,
        "exec_horizon": 8,
        "trajectory_selector": "server",
        "trajectory_selector_scope": "all",
        "deterministic_plan_seeds": True,
        "base_seed": 2026081602,
        "leg1_goal_source": "own",
        "certified_cdec_rescue": "off",
        "certified_stagnation_graph": "off",
        "retrieval_override": "off",
        "revisit_controller": "navdp_mixed",
    }


def _arm_summary(arm: str) -> dict:
    payload = _common_summary()
    payload.update({
        "leg1_mode": "shared_trace",
        "stop_after_leg1": False,
        "write_leg1_trace": False,
    })
    if arm == "native":
        payload.update({
            "server_backend": "navdp", "hybrid_route": "phase",
            "revisit_adapter": "legacy_metric",
            "revisit_adapter_fixed_radius_m": None,
        })
    elif arm == "raw_fixed_oracle_role":
        payload.update({
            "server_backend": "hybrid_pose", "hybrid_route": "phase",
            "revisit_adapter": "raw_fixed_bearing_v1",
            "revisit_adapter_fixed_radius_m": None,
        })
    elif arm == "geometry_router":
        payload.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "memory_geometry",
            "revisit_adapter": "legacy_metric",
            "revisit_adapter_fixed_radius_m": None,
            "router_visual_floor": 0.88, "router_min_matches": 20,
            "router_min_inliers": 12, "router_min_inlier_ratio": 0.5,
            "router_confirm_plans": 2, "router_verify_top_k": 8,
        })
    else:
        payload.update({
            "server_backend": "hybrid_pose",
            "hybrid_route": "certified_relocalization",
            "revisit_adapter": "verified_bearing_v1",
            "revisit_adapter_fixed_radius_m": 2.5,
            "certified_relocalization_server": {
                "enabled": True,
                "runtime_contract": {
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
                },
            },
        })
    return payload


class Hm3dHeldoutVal10IntegrationTest(unittest.TestCase):
    def test_constructible_nine_of_ten_summary_and_independent_recount(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenes = [f"scene{i:02d}" for i in range(10)]
            episodes = [f"episode_{index:04d}" for index in range(4)]
            manifest = {
                "schema_version":
                    "hm3d_heldout_val10_causal_revisit_manifest_v2_20260816",
                "audit": {"status": "ok", "no_mp3d_evaluation": True},
                "frozen_guards": {
                    "no_scene_or_episode_filtering_after_outcomes": True,
                    "failed_scene_retained_as_explicit_attrition": True,
                },
                "scenes": scenes, "scene_count": 10,
                "selected_scene_count": 10,
                "constructible_scene_count": 9,
                "evaluation_scene_indices": list(range(8)) + [9],
                "episode_count": 36, "episodes_per_scene": 4,
                "episodes": {
                    scene: ([] if scene_index == 8 else
                            [{"episode": episode} for episode in episodes])
                    for scene_index, scene in enumerate(scenes)
                },
                "construction_attrition": {
                    "target_met": False,
                    "underpowered": True,
                    "navigation_outcomes_read": False,
                    "receipts": [{"scene": scenes[8], "scene_index": 8}],
                },
                "evaluation": {"base_seed": 2026081602},
                "analysis": {"cluster_bootstrap_seed": 17,
                             "cluster_bootstrap_resamples": 1000},
            }
            manifest_path = root / "data_manifest.json"
            _write_json(manifest_path, manifest)
            manifest_sha = _sha(manifest_path)

            for scene_index, scene in enumerate(scenes):
                if scene_index == 8:
                    continue
                scene_root = root / "scenes" / f"{scene_index:02d}_{scene}"
                _write_json(scene_root / "scene_contract.json", {
                    "schema_version": RUNTIME_REPAIR_SCENE_SCHEMA,
                    "scene": scene, "scene_index": scene_index,
                    "manifest_sha256": manifest_sha,
                    "arm_order": list(WILLIAMS_ORDERS[scene_index % 4]),
                    "actual_online_goal_a_trace": True,
                    "certified_runtime_role_label_visible": False,
                    "raw_fixed_role_oracle": True,
                    "runtime_repair_method_change": False,
                })
                trace_summary = _common_summary()
                trace_summary.update({
                    "server_backend": "hybrid_pose",
                    "hybrid_route": "phase", "leg1_mode": "policy",
                    "stop_after_leg1": True, "write_leg1_trace": True,
                    "revisit_adapter": "legacy_metric",
                })
                _write_json(scene_root / "trace_source/summary.json",
                            trace_summary)
                trace_rows = []
                trace_shas = {}
                for episode_index, episode in enumerate(episodes):
                    trace = scene_root / "trace_source" / (
                        f"{episode}_leg1_trace.json")
                    _write_json(trace, {"scene": scene, "episode": episode})
                    trace_shas[episode] = _sha(trace)
                    trace_rows.append({
                        "episode": episode,
                        "reached_A": float(episode_index < 3),
                        "leg1_trace_sha256": trace_shas[episode],
                    })
                _write_csv(scene_root / "trace_source/metric.csv", trace_rows)

                for arm in ARMS:
                    arm_root = scene_root / arm
                    _write_json(arm_root / "summary.json", _arm_summary(arm))
                    rows = []
                    for episode_index, episode in enumerate(episodes):
                        reached_a = episode_index < 3
                        if arm == "native":
                            reached_b = reached_a and episode_index == 0
                        elif arm == "raw_fixed_oracle_role":
                            reached_b = reached_a and episode_index in {0, 1}
                        elif arm == "geometry_router":
                            reached_b = reached_a and episode_index in {0, 2}
                        else:
                            reached_b = reached_a and episode_index in {0, 1}
                        steps_b = 0 if not reached_a else (50 + episode_index)
                        row: dict[str, object] = {
                            "episode": episode,
                            "reached_A": float(reached_a),
                            "reached_B": float(reached_b),
                            "steps_B": steps_b,
                            "leg1_trace_sha256": trace_shas[episode],
                            "termination_reason_B": (
                                "not_run" if not reached_a else
                                "success" if reached_b else "timeout"),
                            "len_B": 0.0 if not reached_a else 3.0,
                            "final_dist_B": 0.5 if reached_b else 2.0,
                            "blocked_steps_B": 0,
                        }
                        if arm == "certified_relocalization":
                            accepted = reached_a and episode_index in {0, 1}
                            fallback = reached_a and episode_index == 2
                            plans = []
                            if reached_a:
                                plan: dict[str, object] = {
                                    "certified_relocalization_metric_scale": None,
                                    "revisit_adapter_mode":
                                        "verified_bearing_v1",
                                    "certified_relocalization_accepted": accepted,
                                    "revisit_adapter_takeover": accepted,
                                    "certified_relocalization_cached": False,
                                    "certified_relocalization_uncached_ms": 10.0,
                                    "certified_relocalization_pnp": (
                                        {"inliers": 24} if accepted else
                                        {"status": "insufficient_inliers"}),
                                    "router_selected_candidate_dino_rank": (
                                        1 if accepted else None),
                                }
                                if accepted:
                                    plan.update({
                                        "certified_relocalization_ok": True,
                                        "certified_relocalization_pointgoal_units":
                                            "lingbot_raw_direction_only",
                                        "memory_unbounded_pointgoal_distance_m": None,
                                        "memory_controller_pointgoal_distance_m":
                                            2.5,
                                        "pose_controller":
                                            "navdp_image_point_mix",
                                    })
                                else:
                                    plan.update({
                                        "revisit_adapter_controller_contract":
                                            "native_imagegoal",
                                        "pose_controller": "navdp_image_router",
                                    })
                                plans.append(plan)
                            _write_json(arm_root / f"{episode}_plans.json",
                                        {"legB": plans})
                            row.update({
                                "certified_relocalization_request_count":
                                    int(reached_a),
                                "certified_relocalization_uncached_count":
                                    int(reached_a),
                                "certified_relocalization_accept_count":
                                    int(accepted),
                                "certified_relocalization_runtime_failure_count":
                                    0,
                                "revisit_adapter_takeover_plan_count":
                                    int(accepted),
                                "revisit_adapter_abstain_plan_count":
                                    int(fallback),
                            })
                            if fallback:
                                row.update({
                                    "reached_B": 0.0,
                                    "steps_B": 52,
                                    "termination_reason_B": "timeout",
                                    "len_B": 3.0,
                                    "final_dist_B": 2.0,
                                    "blocked_steps_B": 0,
                                })
                        rows.append(row)
                    _write_csv(arm_root / "metric.csv", rows)

            report = summarize(manifest_path, root)
            report_path = root / "hm3d_heldout_val10_revisit_summary.json"
            _write_json(report_path, report)
            independent = verify(manifest_path, root, report_path)
            self.assertTrue(independent["verified"])
            self.assertEqual(independent["selected_scene_count"], 10)
            self.assertEqual(independent["scene_count"], 9)
            self.assertEqual(independent["episode_count"], 36)
            self.assertEqual(
                report["certificate_audit"]["takeover_episodes"], 18)
            self.assertEqual(
                report["certificate_audit"]["fallback_episodes"], 9)
            self.assertEqual(
                report["certificate_audit"]
                ["fallback_behavior_mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
