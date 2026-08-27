import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MemNavData.novel_a_bearing_gate import (
    ARMS,
    critic_shadow_diagnostics,
    normalize_selected_trajectory,
    rotated_arm_order,
    token_request_deg,
    wrap_deg,
)
from MemNavData.summarize_novel_a_bearing_gate import (
    exact_mcnemar_two_sided,
    gate_decision,
    load_and_audit,
    scene_cluster_bootstrap,
    summarize,
)
from MemNavData.deterministic_eval_protocol import diffusion_plan_seed


class BearingProtocolHelperTest(unittest.TestCase):
    def test_arm_order_is_balanced_rotation(self):
        self.assertEqual(rotated_arm_order(0, 0), ARMS)
        self.assertEqual(rotated_arm_order(0, 1), ARMS[1:] + ARMS[:1])
        self.assertEqual(rotated_arm_order(2, 1), ARMS)
        with self.assertRaises(ValueError):
            rotated_arm_order(-1, 0)

    def test_wrap_and_token_clip(self):
        self.assertAlmostEqual(wrap_deg(181), -179)
        self.assertAlmostEqual(wrap_deg(-181), 179)
        self.assertAlmostEqual(token_request_deg(35), 35)
        self.assertAlmostEqual(token_request_deg(179), 100)
        self.assertAlmostEqual(token_request_deg(-179), -100)

    def test_selected_trajectory_normalization(self):
        trajectory = np.zeros((1, 24, 3), dtype=float)
        self.assertEqual(normalize_selected_trajectory(trajectory).shape,
                         (24, 3))
        with self.assertRaises(ValueError):
            normalize_selected_trajectory(np.zeros((24, 2)))

    def test_critic_shadow_is_diagnostic_only_and_well_formed(self):
        candidates = np.zeros((1, 3, 4, 3), dtype=float)
        candidates[0, 0, :, 0] = np.linspace(0.2, 1.0, 4)
        candidates[0, 1, :, :2] = np.stack([
            np.linspace(0.2, 0.5, 4), np.linspace(0.2, 1.0, 4)], axis=1)
        candidates[0, 2, :, :2] = np.stack([
            np.linspace(0.2, 0.5, 4), -np.linspace(0.2, 1.0, 4)], axis=1)
        response = {
            "trajectory": candidates[:, 1],
            "all_trajectory": candidates,
            "all_values": [[0.9, 0.1, 0.5]],
        }
        diag = critic_shadow_diagnostics(
            response, requested_heading_deg=60.0)
        self.assertEqual(diag["candidate_count"], 3)
        self.assertEqual(diag["selected_candidate_index"], 1)
        self.assertEqual(diag["critic_unique_4dp"], 3)
        self.assertIsNotNone(diag["best_direction_critic_rank"])
        self.assertGreaterEqual(diag["heading_resultant_r"], 0.0)
        self.assertLessEqual(diag["heading_resultant_r"], 1.0)


class BearingStatisticsTest(unittest.TestCase):
    def test_exact_mcnemar(self):
        self.assertEqual(exact_mcnemar_two_sided(0, 0), 1.0)
        self.assertAlmostEqual(exact_mcnemar_two_sided(6, 0), 0.03125)
        self.assertAlmostEqual(exact_mcnemar_two_sided(8, 1), 0.0390625)

    def test_gate_decision_is_frozen(self):
        self.assertEqual(gate_decision({
            "gain_count": 4, "loss_count": 1, "net_gain_count": 3}),
            "go_to_unseen_526_pool_not_paper_confirmation")
        self.assertEqual(gate_decision({
            "gain_count": 2, "loss_count": 0, "net_gain_count": 2}),
            "no_go")
        self.assertEqual(gate_decision({
            "gain_count": 3, "loss_count": 0, "net_gain_count": 3}),
            "ambiguous_retest_disjoint_before_building_frontier_ranker")

    def test_scene_bootstrap_clusters_two_episodes(self):
        native = {}
        right = {}
        for scene, outcomes in {
                "scene_a": [(False, True), (False, True)],
                "scene_b": [(True, True), (True, True)]}.items():
            for index, (left_value, right_value) in enumerate(outcomes):
                key = (scene, f"episode_{index:04d}")
                native[key] = {"reached": left_value}
                right[key] = {"reached": right_value}
        interval = scene_cluster_bootstrap(
            {"native": native, "ideal_periodic_yaw": right},
            "ideal_periodic_yaw", resamples=1000, seed=7)
        self.assertEqual(interval, [0.0, 1.0])


class FrozenArtifactTest(unittest.TestCase):
    def test_machine_protocol_matches_documented_arm_contract(self):
        root = Path(__file__).resolve().parent
        protocol = json.loads((
            root / "novel_a_bearing_gate_protocol_20260808.json").read_text())
        self.assertEqual(tuple(protocol["arms"]), ARMS)
        self.assertEqual(protocol["evaluation"]["episodes"], 40)
        self.assertEqual(protocol["evaluation"]["scenes"], 20)
        self.assertEqual(protocol["token"]["resample_seed"],
                         "same_as_native_plan_seed")
        self.assertEqual(len(protocol["input_overlay"]["sha256"]), 64)
        overlay_path = root / "novel_a_bearing_inputs_20260808.json"
        self.assertEqual(hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
                         protocol["input_overlay"]["sha256"])
        overlay = json.loads(overlay_path.read_text())
        self.assertEqual(overlay["parent_manifest_sha256"],
                         protocol["manifest"]["sha256"])

    def test_fail_closed_smoke_audit_checks_seed_and_fifo_hash(self):
        scene = "scene_a"
        episode = "episode_0000"
        protocol_sha = "a" * 64
        manifest_sha = "b" * 64
        protocol = {
            "_sha256": protocol_sha,
            "manifest": {"sha256": manifest_sha},
            "input_overlay": {"sha256": "d" * 64},
            "evaluation": {
                "base_seed": 17,
                "execution_horizon": 8,
                "episodes_per_scene": 1,
                "episodes": 1,
                "scenes": 1,
            },
        }
        manifest = {
            "selection": {"selected_scenes": [scene]},
            "training_scenes": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            scene_root = run_root / "scenes" / f"00_{scene}"
            scene_root.mkdir(parents=True)
            order = rotated_arm_order(0, 0)
            rows = []
            for position, arm in enumerate(order):
                seed = diffusion_plan_seed(17, 0, 0)
                token = arm == "oracle_token_periodic"
                plan = {
                    "plan_index": 0,
                    "step": 0,
                    "requested_diffusion_seed": seed,
                    "native_diffusion_seed": seed,
                    "trajectory_source": "oracle_token" if token else "native",
                    "token_request_deg": 60.0 if token else None,
                    "token_diffusion_seed": seed if token else None,
                    "token_queue_hashes_before": ["hash"] if token else None,
                    "token_queue_hashes_after": ["hash"] if token else None,
                    "token_shadow": {} if token else None,
                    "ideal_turn_deg": (
                        30.0 if arm == "ideal_periodic_yaw" else 0.0),
                    "executed_steps": 8,
                    "path_m": 0.3,
                }
                plans_name = f"{episode}_{arm}_plans.json"
                (scene_root / plans_name).write_text(json.dumps({
                    "scene_index": 0,
                    "scene": scene,
                    "episode": episode,
                    "episode_index": 0,
                    "episode_seed": 17,
                    "arm": arm,
                    "arm_position": position,
                    "arm_order": list(order),
                    "plans": [plan],
                }))
                rows.append({
                    "formal": False,
                    "scene_index": 0,
                    "scene": scene,
                    "episode": episode,
                    "episode_index": 0,
                    "seed": 17,
                    "arm": arm,
                    "arm_position": position,
                    "arm_order": json.dumps(list(order)),
                    "protocol_sha256": protocol_sha,
                    "manifest_sha256": manifest_sha,
                    "input_overlay_sha256": "d" * 64,
                    "plans_file": plans_name,
                    "reached": arm != "native",
                    "geo_A": 4.0,
                    "path_len_m": 5.0,
                    "final_dist_m": 0.9 if arm != "native" else 2.0,
                    "steps": 8,
                    "plan_count": 1,
                    "ideal_turn_count": int(arm == "ideal_periodic_yaw"),
                    "ideal_turn_abs_deg": (
                        30.0 if arm == "ideal_periodic_yaw" else 0.0),
                    "token_plan_count": int(token),
                    "token_path_m": 0.3 if token else 0.0,
                    "token_disabled_reason": "",
                    "goal_jpg_sha256": "c" * 64,
                })
            with (scene_root / "bearing_arms.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (scene_root / "run_meta.json").write_text(json.dumps({
                "status": "complete",
                "formal": False,
                "scene_index": 0,
                "scene": scene,
                "arms": list(ARMS),
                "protocol_sha256": protocol_sha,
                "manifest_sha256": manifest_sha,
                "input_overlay_sha256": "d" * 64,
            }))
            audited, audit = load_and_audit(
                run_root, manifest, protocol, allow_smoke=True)
            self.assertEqual(audit["episodes"], 1)
            self.assertTrue(audited["ideal_periodic_yaw"][(scene, episode)][
                "reached"])
            summary_protocol = {
                **protocol,
                "protocol_version": 1,
                "benchmark_role": "smoke",
                "bootstrap": {"resamples": 100, "seed": 1},
            }
            self.assertEqual(
                summarize(audited, summary_protocol, audit)["decision"],
                "not_evaluated_transport_smoke")

            token_path = scene_root / f"{episode}_oracle_token_periodic_plans.json"
            payload = json.loads(token_path.read_text())
            payload["plans"][0]["token_queue_hashes_after"] = ["changed"]
            token_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(RuntimeError, "FIFO content changed"):
                load_and_audit(run_root, manifest, protocol, allow_smoke=True)


if __name__ == "__main__":
    unittest.main()
