#!/usr/bin/env python3

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import aggregate_lifelong_shared_c_b2 as aggregate_module
import independent_verify_lifelong_shared_c_b2 as verifier_module
from finalize_lifelong_shared_c_population import finalize
from lifelong_shared_c_contract import (
    ARMS,
    RESULT_SCHEMA,
    TRACE_SCHEMA,
    load_trace,
    sha256_file,
    validate_trace,
    write_trace,
)


def trace_payload(*, reached=True):
    return {
        "schema_version": TRACE_SCHEMA,
        "scene": "scene0",
        "episode": "episode_0000",
        "controller": "navdp",
        "benchmark_sha256": "a" * 64,
        "online_A_trace_sha256": "b" * 64,
        "online_B_trace_sha256": "c" * 64,
        "episode_seed": 7,
        "goal_C_sha256": "d" * 64,
        "online_A_candidate_ceiling": 2,
        "online_B_candidate_ceiling": 5,
        "C_goal_start_frame": 6,
        "C_candidate_ceiling": 2,
        "runtime_role_visible": False,
        "reached_C": bool(reached),
        "geodesic_C_m": 2.0,
        "path_len_C_m": 1.5,
        "steps_C": 1,
        "final_goal_dist_C_m": 0.8,
        "termination_reason": "success" if reached else "max_steps",
        "start_position": [0.0, 0.0, 0.0],
        "start_yaw": 0.0,
        "end_position": [1.0, 0.0, 0.0],
        "end_yaw": 0.1,
        "poses": [{
            "step": 0, "x": 1.0, "y": 0.0, "z": 0.0,
            "yaw": 0.1, "jpg_sha256": "e" * 64,
        }],
        "plans": [{"step": 0, "cec_takeover": False}],
        "memory_trace": [{
            "frame_idx": 6, "step": 0, "x": 1.0, "z": 0.0,
            "yaw": 0.1,
        }],
        "navdp_short_fifo_reset_receipt": {"ok": True},
        "B2_navigation_outcomes_read": False,
    }


def write_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


class SharedCContractTest(unittest.TestCase):
    def test_trace_roundtrip_and_contiguous_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.json"
            digest = write_trace(path, trace_payload())
            self.assertEqual(digest, sha256_file(path))
            self.assertTrue(load_trace(path, expected_sha256=digest)["reached_C"])
            broken = trace_payload()
            broken["C_goal_start_frame"] = 7
            with self.assertRaisesRegex(RuntimeError, "goal boundary"):
                validate_trace(broken)

    def test_freeze_then_aggregate_and_verify_b2_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_population = root / "source_population.json"
            source_population.write_text(json.dumps({
                "accepted": [{
                    "scene": "scene0", "episode": "episode_0000",
                    "benchmark_sha256": "a" * 64,
                }]
            }) + "\n")
            collection = root / "collection/000_scene0_episode_0000"
            result = collection / "result"
            result.mkdir(parents=True)
            trace_path = result / "episode_0000_shared_C_trace.json"
            trace_sha = write_trace(trace_path, trace_payload())
            write_csv(result / "metric.csv", {
                "scene": "scene0", "episode": "episode_0000",
                "controller": "navdp", "shared_C_trace_sha256": trace_sha,
                "B2_outcomes_read": 0,
            })
            (result / "episode_0000_plans.json").write_text("{}\n")
            (result / "summary.json").write_text("{}\n")
            (collection / "compute_identity.json").write_text("{}\n")
            frozen = root / "frozen"
            population = finalize(
                source_population=source_population,
                collection_root=root / "collection",
                controller="navdp",
                run_root=root / "run",
                out=frozen,
            )
            self.assertEqual(population["accepted_histories"], 1)
            self.assertFalse(population["selection_reads_B2_navigation_outcomes"])

            evaluation = root / "evaluation/000_scene0_episode_0000"
            shared = {
                "frozen_legA": [{"step": 0}],
                "frozen_legB": [{"step": 0}],
                "frozen_legC": [{"step": 0}],
                "rollout_traces": {
                    "A": [{"step": 0}], "B": [{"step": 0}],
                    "C": [{"step": 0}], "B2": [],
                },
                "memory_traces": {
                    "A": [{"frame_idx": 0}],
                    "B": [{"frame_idx": 3}],
                    "C": [{"frame_idx": 6}], "B2": [],
                },
            }
            for arm in ARMS:
                arm_root = evaluation / arm
                arm_result = arm_root / "result"
                arm_result.mkdir(parents=True)
                reached = int(arm == "all_prior")
                ceiling = 2 if arm == "initial_leg_only" else 5
                write_csv(arm_result / "metric.csv", {
                    "result_schema": RESULT_SCHEMA,
                    "scene": "scene0", "episode": "episode_0000",
                    "history_scope": arm,
                    "shared_C_prefix_replayed": 1,
                    "shared_C_trace_sha256": population["accepted"][0][
                        "shared_C_trace_sha256"],
                    "shared_C_start_x": 0.0, "shared_C_start_y": 0.0,
                    "shared_C_start_z": 0.0, "shared_C_start_yaw": 0.0,
                    "B2_start_x": 1.0, "B2_start_y": 0.0,
                    "B2_start_z": 0.0, "B2_start_yaw": 0.1,
                    "online_A_candidate_ceiling": 2,
                    "online_B_candidate_ceiling": 5,
                    "B2_candidate_ceiling": ceiling,
                    "reached_B2": reached,
                })
                b2 = [{"step": 0, "cec_takeover": False}]
                if arm == "forced_reject_native":
                    b2[0]["cec_forced_reject_native"] = True
                payload = dict(shared)
                payload["B2"] = b2
                (arm_result / "episode_0000_plans.json").write_text(
                    json.dumps(payload) + "\n")
                (arm_result / "summary.json").write_text("{}\n")
                process = 20 if arm == "forced_reject_native" else 10
                (arm_root / "compute_identity.json").write_text(json.dumps({
                    "host": "node", "gpu_uuid": "gpu",
                    "memnav": {"pid": process, "process_start_ticks": 100},
                    "navdp": {"pid": process + 1,
                              "process_start_ticks": 101},
                    "cec_hub": {"pid": process + 2,
                                "process_start_ticks": 102},
                }) + "\n")

            aggregate = root / "aggregate"
            verification = root / "verification.json"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "aggregate", "--population", str(frozen / "population.json"),
                    "--evaluation-root", str(root / "evaluation"),
                    "--out", str(aggregate),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    aggregate_module.main()
                sys.argv = [
                    "verify", "--population", str(frozen / "population.json"),
                    "--aggregate", str(aggregate), "--out", str(verification),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    verifier_module.main()
            finally:
                sys.argv = old_argv
            checked = json.loads(verification.read_text())
            self.assertTrue(checked["verified"])
            self.assertEqual(
                checked["B2"]["all_prior_vs_initial_leg_only"]
                ["paired_gains"], 1)
            self.assertTrue(checked["shared_C_prefix_exact_across_primary_arms"])


if __name__ == "__main__":
    unittest.main()
