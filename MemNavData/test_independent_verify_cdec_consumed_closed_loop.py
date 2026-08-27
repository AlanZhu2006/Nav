import csv
import json
import tempfile
import unittest
from pathlib import Path

from MemNavData.independent_verify_cdec_consumed_closed_loop import (
    EXPECTED_ARTIFACT_SHA256,
    _neutral,
    exact_mcnemar,
    verify,
)


class IndependentVerifyCDECConsumedClosedLoopTest(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(6, 0), 0.03125)
        self.assertEqual(exact_mcnemar(5, 1), 0.21875)

    def test_neutralization_keeps_geometry_and_drops_learned_probe(self):
        geometry = {
            "router_ranking_mode": "geometry",
            "certified_relocalization_learned_rescue_requested": False,
            "certified_relocalization_learned_proposal": {"status": "off"},
            "certified_relocalization_proposal_attempts": [
                {"source": "geometry", "selected_anchor": 3, "accepted": False}
            ],
            "trajectory": [[0.1, 0.2]],
        }
        cdec = {
            **geometry,
            "router_ranking_mode": "cascade",
            "certified_relocalization_learned_rescue_requested": True,
            "certified_relocalization_learned_proposal": {
                "status": "certificate_rejected"},
            "certified_relocalization_proposal_attempts": [
                geometry["certified_relocalization_proposal_attempts"][0],
                {
                    "source": "learned_on_geometry_reject",
                    "selected_anchor": 5,
                    "accepted": False,
                },
            ],
        }
        self.assertEqual(_neutral(geometry), _neutral(cdec))
        cdec["trajectory"] = [[0.2, 0.1]]
        self.assertNotEqual(_neutral(geometry), _neutral(cdec))

    def test_full_raw_reconstruction_on_synthetic_20_scene_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            scenes = [f"s{index:02d}" for index in range(20)]
            episode_ids = [f"episode_{index:04d}" for index in range(8)]
            manifest = {
                "audit": {"status": "ok"},
                "data_role_guards": {"blind_allowed": False},
                "scenes": scenes,
                "episodes": {
                    scene: [{"episode": episode} for episode in episode_ids]
                    for scene in scenes
                },
                "analysis": {
                    "cluster_bootstrap_seed": 17,
                    "cluster_bootstrap_resamples": 100,
                },
            }
            manifest_path = run / "data_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            import hashlib
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            trace_sha = "a" * 64
            receipt = {
                "manifest_sha256": manifest_sha,
                "trace_payload_decoded": False,
                "episode_target_or_outcome_fields_accessed": False,
                "development_read": False,
                "blind_read": False,
                "scenes": {
                    scene: {"episodes": {
                        episode: trace_sha for episode in episode_ids}}
                    for scene in scenes
                },
            }
            receipt_path = run / "trace_receipt.json"
            receipt_path.write_text(json.dumps(receipt))

            metric_fields = [
                "episode", "seed", "leg1_trace_sha256", "reached_A",
                "reached_B", "spl_A", "spl_B", "spl_B_with_terminal",
                "geo_A", "geo_B", "len_A", "len_B", "len_B_at_reach",
                "final_dist_A", "final_dist_B", "steps_A", "steps_B",
                "steps_B_diagnostic", "steps_B_at_reach",
                "termination_reason_A", "termination_reason_B",
                "blocked_steps_A", "blocked_steps_B", "blocked_step_rate_B",
                "terminal_final_goal_dist_m",
            ]
            geometry_attempt = {
                "source": "geometry", "selected_anchor": 7,
                "accepted": False, "reason": "rejected",
            }
            common_plan = {
                "step": 0, "requested_diffusion_seed": 101,
                "diffusion_seed": 101, "frame_idx": 20,
                "goal_start_frame": 20, "candidate_ceiling": 20,
                "router_candidate_order_dino": [7, 9],
                "router_candidate_order_used": [7, 9],
                "certified_relocalization_selected_proposal_source": "geometry",
                "certified_relocalization_accepted": False,
                "revisit_adapter_takeover": False,
            }
            summaries = {}
            for arm in ("geometry_certificate", "cdec_cascade"):
                summaries[arm] = {
                    "episodes": 8, "server_backend": "hybrid_pose",
                    "hybrid_route": "certified_relocalization",
                    "revisit_controller": "navdp_mixed",
                    "revisit_adapter": "verified_bearing_v1",
                    "revisit_adapter_fixed_radius_m": 2.5,
                    "leg1_mode": "shared_trace",
                    "deterministic_plan_seeds": True,
                    "trajectory_selector": "server",
                    "trajectory_selector_scope": "all", "max_steps": 500,
                    "exec_horizon": 8, "graph_subgoal_spacing_m": 0.0,
                    "certified_cdec_rescue": (
                        "on" if arm == "cdec_cascade" else "off"),
                    "cdec_server_status": {
                        "enabled": True,
                        "artifact_sha256": EXPECTED_ARTIFACT_SHA256,
                        "deployment_approved": False,
                        "authority": "rank_frozen_causal_shortlist_only",
                        "activation_authority": (
                            "independent_atomic_pnp_certificate"),
                    },
                    "certified_cdec_uncached_runtime_failure_count": 0,
                    "certified_cdec_requested_plan_count": (
                        8 if arm == "cdec_cascade" else 0),
                    "certified_cdec_learned_selected_plan_count": 0,
                    "certified_cdec_uncached_invocation_count": (
                        8 if arm == "cdec_cascade" else 0),
                }
            for scene_index, scene in enumerate(scenes):
                scene_root = run / "scenes" / f"{scene_index:02d}_{scene}"
                scene_root.mkdir(parents=True)
                (scene_root / "scene_contract.json").write_text(json.dumps({
                    "schema_version": "cdec_consumed_closed_loop_scene_v1_20260813",
                    "scene": scene, "scene_index": scene_index,
                    "arm_order": list((
                        ("geometry_certificate", "cdec_cascade"),
                        ("cdec_cascade", "geometry_certificate"),
                    )[scene_index % 2]),
                    "cdec_artifact_sha256": EXPECTED_ARTIFACT_SHA256,
                    "stagnation_graph": "off",
                }))
                for arm in ("geometry_certificate", "cdec_cascade"):
                    arm_root = scene_root / arm
                    arm_root.mkdir()
                    (arm_root / "summary.json").write_text(
                        json.dumps(summaries[arm]))
                    with (arm_root / "metric.csv").open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=metric_fields)
                        writer.writeheader()
                        for episode_index, episode in enumerate(episode_ids):
                            writer.writerow({
                                "episode": episode, "seed": 100 + episode_index,
                                "leg1_trace_sha256": trace_sha,
                                "reached_A": 1, "reached_B": 0,
                                "spl_A": 0.5, "spl_B": 0,
                                "spl_B_with_terminal": 0, "geo_A": 2,
                                "geo_B": 3, "len_A": 2.5, "len_B": 5,
                                "len_B_at_reach": "", "final_dist_A": 0.5,
                                "final_dist_B": 2, "steps_A": 10,
                                "steps_B": 20, "steps_B_diagnostic": 20,
                                "steps_B_at_reach": "",
                                "termination_reason_A": "success",
                                "termination_reason_B": "budget",
                                "blocked_steps_A": 0, "blocked_steps_B": 0,
                                "blocked_step_rate_B": 0,
                                "terminal_final_goal_dist_m": 2,
                            })
                            plan = {
                                **common_plan,
                                "router_ranking_mode": (
                                    "cascade" if arm == "cdec_cascade"
                                    else "geometry"),
                                "certified_relocalization_learned_rescue_requested": (
                                    arm == "cdec_cascade"),
                                "certified_relocalization_proposal_attempts": [
                                    geometry_attempt],
                                "certified_relocalization_learned_proposal": {
                                    "status": "not_requested",
                                    "activation_authorized": False,
                                },
                            }
                            if arm == "cdec_cascade":
                                plan["certified_relocalization_proposal_attempts"] = [
                                    geometry_attempt,
                                    {
                                        "source": "learned_on_geometry_reject",
                                        "selected_anchor": 9,
                                        "accepted": False,
                                        "reason": "rejected",
                                    },
                                ]
                                plan["certified_relocalization_learned_proposal"] = {
                                    "status": "certificate_rejected",
                                    "activation_authorized": False,
                                }
                            (arm_root / f"{episode}_plans.json").write_text(
                                json.dumps({
                                    "leg1_trace_sha256": trace_sha,
                                    "legA": [{"step": 0}],
                                    "legA_memory_trace": [{"frame_idx": 0}],
                                    "legB": [plan],
                                }))

            checks = {
                "gain_in_at_least_two_scene_clusters": False,
                "zero_paired_losses": True,
                "exact_mcnemar_below_0_05": False,
                "cluster_interval_lower_above_zero": False,
                "all_causal_and_safety_audits_pass": True,
                "every_gain_has_learned_certified_takeover": True,
            }
            official = {
                "audit": {
                    "development_read": False, "blind_read": False,
                    "stagnation_graph_disabled": True,
                },
                "arms": {
                    arm: {
                        "episodes": 160, "novel": {"successes": 160},
                        "joint": {"successes": 0},
                        "revisit_given_novel_success": {
                            "eligible": 160, "successes": 0},
                    } for arm in ("geometry_certificate", "cdec_cascade")
                },
                "contrasts": {
                    "joint": {
                        "outcomes": {
                            "both_joint_success": 0,
                            "left_only_joint_success": 0,
                            "right_only_joint_success": 0,
                            "neither_joint_success": 160,
                        },
                        "joint_sr_delta_right_minus_left": 0.0,
                        "mcnemar_exact_two_sided_p": 1.0,
                        "scene_cluster_bootstrap_risk_difference_95": [0.0, 0.0],
                    },
                    "conditional_b": {
                        "eligible_shared_novel_success": 160,
                        "outcomes": {
                            "both_revisit_success": 0,
                            "left_only_revisit_success": 0,
                            "right_only_revisit_success": 0,
                            "neither_revisit_success": 160,
                        },
                        "risk_difference_right_minus_left": 0.0,
                        "mcnemar_exact_two_sided_p": 1.0,
                        "gains": [], "losses": [],
                        "scene_cluster_bootstrap_risk_difference_95": [0.0, 0.0],
                    },
                },
                "learned_runtime": {
                    "a_success_episodes": 160, "invoked_episodes": 160,
                    "takeover_episodes": 0, "geometry_accepted_episodes": 0,
                    "no_treatment_exact_episodes": 160,
                    "gain_episodes": [], "loss_episodes": [],
                },
                "decision": {
                    "pass": False, "checks": checks,
                    "branch": "do_not_promote_cdec",
                    "authorize_retuning_on_consumed_pool": False,
                    "authorize_blind_opening_without_explicit_user_approval": False,
                },
            }
            official_path = run / "report.json"
            official_path.write_text(json.dumps(official))
            result = verify(
                manifest_path=manifest_path, trace_receipt_path=receipt_path,
                run_root=run, official_report_path=official_path)
            self.assertTrue(result["verified"])
            self.assertFalse(result["reconstructed"]["decision"]["pass"])


if __name__ == "__main__":
    unittest.main()
