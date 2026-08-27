#!/usr/bin/env python3

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import hm3d_fullmono_lifelong as contract
import aggregate_hm3d_fullmono_lifelong as aggregate_module
import independent_verify_hm3d_fullmono_lifelong as verifier_module
from aggregate_hm3d_fullmono_lifelong import (
    arm_endpoint_counts,
    paired_prefix_comparison,
)


class FullMonoLifelongContractTest(unittest.TestCase):
    def test_deferred_arrays_use_the_exact_sealed_population(self):
        path = Path(__file__).with_name(
            "slurm_hm3d_fullmono_shared_c_deferred.sbatch"
        )
        source = path.read_text()
        self.assertIn("sealed_population_count", source)
        self.assertIn('--array="${array_spec}"', source)
        self.assertNotIn('--array="0-259', source)
        self.assertIn('"submitted_array":f"0-{int(population_count) - 1}"', source)

    def test_frozen_protocol_loads(self):
        path = Path(__file__).with_name(
            "hm3d_fullmono_lifelong_protocol_20260824.json"
        )
        payload = contract.load_protocol(path)
        self.assertEqual(
            tuple(row["name"] for row in payload["query_runtime"]["arms"]),
            contract.ARMS,
        )

    def test_powered_expansion_protocol_is_prospectively_gated(self):
        path = Path(__file__).with_name(
            "hm3d_fullmono_lifelong_power_expansion_protocol_20260826.json"
        )
        payload = contract.load_protocol(path)
        self.assertEqual(
            payload["schema_version"],
            contract.POWERED_EXPANSION_PROTOCOL_SCHEMA,
        )
        self.assertTrue(payload["construction_power_gate"][
            "halt_before_factual_B_if_not_met"
        ])
        self.assertFalse(payload["prior_result_use"][
            "used_to_select_v3_candidate_identity"
        ])

    def test_direct_natural_v4_protocol_freezes_five_leg_gate(self):
        path = Path(__file__).with_name(
            "hm3d_fullmono_lifelong_direct_natural_protocol_20260827.json"
        )
        payload = contract.load_protocol(path)
        self.assertEqual(
            payload["schema_version"],
            contract.DIRECT_NATURAL_PROTOCOL_SCHEMA,
        )
        self.assertEqual(
            payload["query_runtime"]["sequence"],
            ["A", "B", "C", "B2", "C2"],
        )
        self.assertTrue(payload["construction_power_gate"][
            "halt_before_factual_B_if_not_met"
        ])
        self.assertTrue(payload["population"][
            "halt_before_factual_C_if_target_not_met"
        ])
        self.assertFalse(payload["post_prefix_query_outcomes_read_before_freeze"])

    def test_arm_rotation_is_balanced_and_deterministic(self):
        self.assertEqual(contract.rotated_arm_order(0), contract.ARMS)
        self.assertEqual(
            contract.rotated_arm_order(1),
            ("initial_leg_only", "forced_reject_native", "all_prior"),
        )
        self.assertEqual(contract.rotated_arm_order(3), contract.ARMS)

    def test_donor_selection_is_result_blind_and_strict(self):
        rows = [
            {
                "donor_episode": "episode_0000",
                "donor_episode_rank": 0,
                "a_to_b_geodesic_m": 4.0,
                "b_to_c_geodesic_m": 3.0,
                "max_recipient_a_covis": 0.0,
                "navigation_success": 0,
            },
            {
                "donor_episode": "episode_0001",
                "donor_episode_rank": 1,
                "a_to_b_geodesic_m": 4.1,
                "b_to_c_geodesic_m": 3.0,
                "max_recipient_a_covis": 0.01,
                "navigation_success": 1,
            },
            {
                "donor_episode": "episode_0002",
                "donor_episode_rank": 2,
                "a_to_b_geodesic_m": 4.0,
                "b_to_c_geodesic_m": 3.0,
                "max_recipient_a_covis": 0.10,
                "navigation_success": 1,
            },
        ]
        selected = contract.select_donor(
            rows, recipient_episode="episode_9999"
        )
        self.assertEqual(selected["donor_episode"], "episode_0000")
        flipped = [dict(row, navigation_success=1-int(row["navigation_success"]))
                   for row in rows]
        self.assertEqual(
            contract.select_donor(flipped, recipient_episode="episode_9999")[
                "donor_episode"
            ],
            "episode_0000",
        )

    def test_recipient_cannot_donate_to_itself(self):
        row = {
            "donor_episode": "episode_0000",
            "donor_episode_rank": 0,
            "a_to_b_geodesic_m": 4.0,
            "b_to_c_geodesic_m": 3.0,
            "max_recipient_a_covis": 0.0,
        }
        self.assertIsNone(contract.select_donor(
            [row], recipient_episode="episode_0000"
        ))

    def test_power_expansion_is_multi_candidate_and_result_blind(self):
        rows = [
            {
                "donor_episode": "episode_0001",
                "donor_episode_rank": 1,
                "donor_frame_index": 10,
                "a_to_b_geodesic_m": 4.0,
                "b_to_c_geodesic_m": 3.0,
                "max_recipient_a_covis": 0.01,
                "assigned_direction_stratum": "front",
                "navigation_success": 0,
            },
            {
                "donor_episode": "episode_0001",
                "donor_episode_rank": 1,
                "donor_frame_index": 20,
                "a_to_b_geodesic_m": 4.1,
                "b_to_c_geodesic_m": 3.0,
                "max_recipient_a_covis": 0.01,
                "assigned_direction_stratum": "side",
                "navigation_success": 1,
            },
            {
                "donor_episode": "episode_0002",
                "donor_episode_rank": 2,
                "donor_frame_index": 30,
                "a_to_b_geodesic_m": 4.2,
                "b_to_c_geodesic_m": 3.0,
                "max_recipient_a_covis": 0.01,
                "assigned_direction_stratum": "rear",
                "navigation_success": 0,
            },
        ]
        selected = contract.select_donors(
            rows, recipient_episode="episode_9999", maximum_candidates=2,
            maximum_per_donor=1, prefer_distinct_direction_strata=True,
        )
        self.assertEqual(
            [row["donor_episode"] for row in selected],
            ["episode_0001", "episode_0002"],
        )
        flipped = [
            dict(row, navigation_success=1-int(row["navigation_success"]))
            for row in rows
        ]
        self.assertEqual(
            [(row["donor_episode"], row["donor_frame_index"])
             for row in selected],
            [(row["donor_episode"], row["donor_frame_index"])
             for row in contract.select_donors(
                 flipped, recipient_episode="episode_9999",
                 maximum_candidates=2, maximum_per_donor=1,
                 prefer_distinct_direction_strata=True,
             )],
        )

    def test_powered_expansion_enforces_spatially_distinct_candidates(self):
        rows = []
        for frame, x, stratum in (
            (10, 0.0, "front"),
            (20, 0.5, "side"),
            (30, 2.1, "rear"),
            (40, 4.2, "front"),
        ):
            rows.append({
                "donor_episode": "episode_0001",
                "donor_episode_rank": 1,
                "donor_frame_index": frame,
                "goal_floor_position": [x, 0.0, 0.0],
                "a_to_b_geodesic_m": 4.0 + frame / 1000.0,
                "b_to_c_geodesic_m": 3.0,
                "max_recipient_a_covis": 0.01,
                "assigned_direction_stratum": stratum,
            })
        selected = contract.select_donors(
            rows,
            recipient_episode="episode_9999",
            maximum_candidates=4,
            maximum_per_donor=4,
            prefer_distinct_direction_strata=True,
            minimum_planar_separation_m=2.0,
        )
        self.assertEqual(
            [row["donor_frame_index"] for row in selected],
            [10, 30, 40],
        )

    def test_exact_mcnemar(self):
        self.assertEqual(contract.exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(contract.exact_mcnemar(12, 0), 0.00048828125)
        self.assertEqual(contract.exact_mcnemar(1, 1), 1.0)

    def test_parent_binding_checks_both_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sealed_inputs").mkdir()
            (root / "benchmarks").mkdir()
            manifest = root / "sealed_inputs/parent_manifest.json"
            population = root / "benchmarks/population_receipt.json"
            manifest.write_text("{}\n")
            population.write_text("{}\n")
            protocol = {
                "parent": {
                    "run_root": str(root),
                    "parent_manifest": "sealed_inputs/parent_manifest.json",
                    "parent_manifest_sha256": contract.sha256_file(manifest),
                    "fullmono_population_receipt": "benchmarks/population_receipt.json",
                    "fullmono_population_receipt_sha256": contract.sha256_file(population),
                }
            }
            self.assertEqual(
                set(contract.bind_parent(protocol, root)),
                {"manifest", "population"},
            )

    def test_endpoint_counts_keep_causal_denominators(self):
        rows = [
            {
                "reached_C": "1", "reached_B2": "1", "reached_C2": "0",
                "evaluated_B2": "1", "evaluated_C2": "1",
                "queries_completed_before_first_failure": "2",
                "query_joint_success": "0", "B2_used_factual_B_anchor": "1",
            },
            {
                "reached_C": "0", "reached_B2": "0", "reached_C2": "0",
                "evaluated_B2": "0", "evaluated_C2": "0",
                "queries_completed_before_first_failure": "0",
                "query_joint_success": "0", "B2_used_factual_B_anchor": "0",
            },
        ]
        counts = arm_endpoint_counts(rows)
        self.assertEqual(counts["C"], {"success": 1, "evaluated": 2})
        self.assertEqual(
            counts["B2_given_C"], {"success": 1, "evaluated": 1}
        )
        self.assertEqual(
            counts["C2_given_C_B2"], {"success": 0, "evaluated": 1}
        )
        self.assertEqual(counts["prefix_survival"], {"1": 1, "2": 1, "3": 0})

    def test_prefix_comparison_is_unconditioned_and_paired(self):
        first = {
            ("s0", "e0"): {"queries_completed_before_first_failure": "3"},
            ("s1", "e1"): {"queries_completed_before_first_failure": "1"},
        }
        second = {
            ("s0", "e0"): {"queries_completed_before_first_failure": "2"},
            ("s1", "e1"): {"queries_completed_before_first_failure": "0"},
        }
        result = paired_prefix_comparison(
            first, second, first_name="first", second_name="second"
        )
        self.assertEqual(result["1"]["paired_gains"], 1)
        self.assertEqual(result["2"]["paired_gains"], 0)
        self.assertEqual(result["3"]["paired_gains"], 1)
        self.assertEqual(result["3"]["n"], 2)

    def test_aggregate_and_independent_verifier_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            population_root = root / "population"
            population_root.mkdir()
            mono_receipt = {
                "depth_contract": "raw_lingbot_depth_first40_v1",
                "metric_depth_sensor_consumed": False,
                "image_sha256": "image",
                "depth_png_sha256": "depth",
                "frame_index": 40,
                "depth_nonzero_fraction": 0.5,
                "scale_active": True,
                "scale_receipt_sha256": "scale-hash",
                "scale_receipt": {
                    "scale_evidence_contract": "causal_first_prefix_rgb_only_v1",
                    "whole_episode_ground_cache_consumed": False,
                    "scale_valid": True,
                },
            }
            mono_plan = {
                "navdp_depth_source": "monocular_sidecar",
                "metric_depth_sensor_consumed": False,
                "monocular_depth_receipt": mono_receipt,
            }
            source_a = root / "source_a"
            source_a.mkdir()
            trace_a = {
                "reached": True,
                "source_hybrid_route": "native_sidecar",
                "plans": [mono_plan],
            }
            trace_a_path = source_a / "online_a_trace.json"
            trace_a_path.write_text(json.dumps(trace_a) + "\n")
            receipt_a = {
                "scene": "s0", "episode": "e0",
                "online_a_trace_sha256": contract.sha256_file(trace_a_path),
            }
            receipt_a_path = source_a / "receipt.json"
            receipt_a_path.write_text(json.dumps(receipt_a) + "\n")

            benchmark_root = population_root / "benchmark/s0/e0"
            benchmark_root.mkdir(parents=True)
            trace_b = {
                "reached": True,
                "source_hybrid_route": "native_sidecar",
                "plans": [mono_plan],
            }
            trace_b_path = benchmark_root / "e0_legB_trace.json"
            trace_b_path.write_text(json.dumps(trace_b) + "\n")
            b_depth_audit = {
                "metric_sensor_plan_count": 0,
                "monocular_receipt_plan_count": 1,
                "monocular_scale_hash_count": 1,
            }
            completion_b = {
                "controller": "frozen_navdp_native_sidecar",
                "navdp_depth_source": "monocular_sidecar",
                "metric_depth_sensor_reads": 0,
                "depth_audit": b_depth_audit,
            }
            completion_b_path = benchmark_root / "factual_B_completion.json"
            completion_b_path.write_text(json.dumps(completion_b) + "\n")
            benchmark = {
                "scene": "s0", "episode": "e0",
                "source_online_A_episode": str(source_a),
                "source_online_A_receipt_sha256": contract.sha256_file(
                    receipt_a_path
                ),
                "source_online_A_trace_sha256": contract.sha256_file(
                    trace_a_path
                ),
                "online_B_trace": trace_b_path.name,
                "online_B_trace_sha256": contract.sha256_file(trace_b_path),
                "factual_B_completion": completion_b_path.name,
                "factual_B_completion_sha256": contract.sha256_file(
                    completion_b_path
                ),
            }
            benchmark_path = benchmark_root / "benchmark.json"
            benchmark_path.write_text(json.dumps(benchmark) + "\n")
            benchmark_hash = contract.sha256_file(benchmark_path)
            population = {
                "selection_reads_C_B2_C2_navigation_outcomes": False,
                "accepted": [{
                    "population_index": 0,
                    "scene": "s0",
                    "episode": "e0",
                    "benchmark": "benchmark/s0/e0/benchmark.json",
                    "benchmark_sha256": benchmark_hash,
                }],
            }
            population_path = population_root / "population.json"
            population_path.write_text(json.dumps(population) + "\n")
            (population_root / "population.json.sha256").write_text(
                contract.sha256_file(population_path) + "  population.json\n"
            )
            (population_root / "SEALED").write_text("sealed\n")

            outcomes = {
                "all_prior": (1, 1, 1),
                "initial_leg_only": (1, 0, 0),
                "forced_reject_native": (0, 0, 0),
            }
            memory_a = [{"frame_idx": value} for value in range(3)]
            memory_b = [{"frame_idx": value} for value in range(3, 6)]
            c_plan = {
                "step": 0,
                "cec_candidate_ceiling": 2,
                "cec_takeover": False,
                "metric_depth_sensor_consumed": False,
            }
            b2_start = [{"x": 1.0, "y": 0.0, "z": 2.0, "yaw": 0.5}]
            for arm, reached in outcomes.items():
                reached_c, reached_b2, reached_c2 = reached
                result = root / "evaluation/000_s0_e0" / arm / "result"
                result.mkdir(parents=True)
                completed = 0
                for value in reached:
                    if not value:
                        break
                    completed += 1
                row = {
                    "result_schema": contract.RESULT_SCHEMA,
                    "scene": "s0", "episode": "e0",
                    "benchmark_sha256": benchmark_hash,
                    "history_scope": arm, "runtime_role_visible": 0,
                    "metric_depth_reads_queries": 0,
                    "online_A_candidate_ceiling": 2,
                    "online_B_candidate_ceiling": 5,
                    "reached_C": reached_c, "reached_B2": reached_b2,
                    "reached_C2": reached_c2,
                    "evaluated_B2": reached_c,
                    "evaluated_C2": int(reached_c and reached_b2),
                    "queries_completed_before_first_failure": completed,
                    "query_joint_success": int(all(reached)),
                    "B2_used_factual_B_anchor": int(
                        arm == "all_prior" and reached_b2
                    ),
                    "steps_B2": 4 if reached_c else 0,
                    "len_B2": 1.25 if reached_c else 0.0,
                }
                with (result / "metric.csv").open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader(); writer.writerow(row)
                queries = {"C": [dict(c_plan)], "B2": [], "C2": []}
                receipts = [{"candidate_ceiling": 2}]
                if reached_c:
                    b2_plan = dict(
                        c_plan,
                        cec_candidate_ceiling=(
                            2 if arm == "initial_leg_only" else 5
                        ),
                    )
                    queries["B2"] = [b2_plan]
                    receipts.append({
                        "candidate_ceiling": (
                            2 if arm == "initial_leg_only" else 5
                        )
                    })
                if reached_c and reached_b2:
                    queries["C2"] = [dict(c_plan, cec_candidate_ceiling=6)]
                    receipts.append({"candidate_ceiling": 6})
                if arm == "forced_reject_native":
                    queries["C"][0].update({
                        "cec_forced_reject_native": True,
                        "cec_shadow_takeover": True,
                        "cec_action_state": "forced_reject",
                    })
                payload = {
                    "history_scope": arm,
                    "runtime_role_visible": False,
                    "benchmark_sha256": benchmark_hash,
                    "goal_session_receipts": receipts,
                    "queries": queries,
                    "memory_traces": {
                        "A": memory_a, "B": memory_b,
                        "C": [], "B2": [], "C2": [],
                    },
                    "rollout_traces": {
                        "C": [{"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}],
                        "B2": b2_start if reached_c else [],
                        "C2": [],
                    },
                }
                (result / "e0_plans.json").write_text(json.dumps(payload) + "\n")
                (result / "summary.json").write_text("{}\n")
                process_base = 20 if arm == "forced_reject_native" else 10
                compute = {
                    "schema_version": "cec_compute_identity_v1_20260824",
                    "host": "node0", "gpu_uuid": "gpu0",
                    "cuda_visible_devices": "0",
                    "runtime_scope": arm,
                    "memnav": {"pid": process_base,
                               "process_start_ticks": process_base * 100},
                    "navdp": {"pid": process_base + 1,
                              "process_start_ticks": process_base * 100 + 1},
                    "cec_hub": {"pid": process_base + 2,
                                "process_start_ticks": process_base * 100 + 2},
                    "paired_scope_order": (
                        [] if arm == "forced_reject_native"
                        else ["all_prior", "initial_leg_only"]
                    ),
                }
                (result.parent / "compute_identity.json").write_text(
                    json.dumps(compute) + "\n"
                )

            aggregate_root = root / "aggregate"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "aggregate", "--population", str(population_path),
                    "--evaluation-root", str(root / "evaluation"),
                    "--out", str(aggregate_root),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    aggregate_module.main()
                verification_path = root / "verification.json"
                sys.argv = [
                    "verify", "--population", str(population_path),
                    "--aggregate", str(aggregate_root),
                    "--evaluation-root", str(root / "evaluation"),
                    "--out", str(verification_path),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    verifier_module.main()
            finally:
                sys.argv = old_argv
            verification = json.loads(verification_path.read_text())
            self.assertTrue(verification["verified"])
            self.assertEqual(
                verification["primary_B2_after_shared_C"]["paired_gains"], 1
            )
            self.assertEqual(
                verification["arms"]["all_prior"]["query_joint"]["success"], 1
            )
            self.assertEqual(
                verification["primary_pair_same_GPU_process_receipts"], 1
            )
            self.assertEqual(
                verification["forced_baseline_same_GPU_node_receipts"], 1
            )


if __name__ == "__main__":
    unittest.main()
