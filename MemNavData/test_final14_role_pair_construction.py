#!/usr/bin/env python3

import math
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_final14_role_pair_scene as builder
import construct_hm3d_fullmono_lifelong_ab as lifelong_builder
import finalize_final14_role_pairs as finalizer
from audit_final14_role_pairs import audit as audit_final14


class Final14RolePairConstructionTest(unittest.TestCase):
    def test_parent_certified_missing_online_root_is_zero_history_attrition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = "upstream_empty_scene"
            scene_index = 4
            parent_manifest = root / "parent_manifest.json"
            parent_population = root / "population_receipt.json"
            protocol = root / "protocol.json"
            parent_manifest.write_text("{}\n")
            parent_population.write_text("{}\n")
            protocol.write_text("{}\n")
            parent_scene = (
                root / "construction/scenes"
                / f"{scene_index:02d}_{scene}"
            )
            parent_scene.mkdir(parents=True)
            upstream = {
                "status": "complete",
                "scene": scene,
                "scene_index": scene_index,
                "query_policy_outcomes_read": False,
                "construction_attrition": [{
                    "scene": scene,
                    "stage": "source_generation",
                    "reason": "fixed_attempt_source_generation_incomplete",
                }],
                "materialization": {
                    "materialized": 0,
                    "manifest_sha256": None,
                },
            }
            completion_path = parent_scene / "completion.json"
            completion_path.write_text(json.dumps(upstream, sort_keys=True) + "\n")
            upstream_sha = lifelong_builder.sha256_file(completion_path)
            (parent_scene / "completion.json.sha256").write_text(
                upstream_sha + "  completion.json\n"
            )
            fragment = {
                "scene": scene,
                "scene_index": scene_index,
                "materialized_histories": 0,
                "goal_a_successes": 0,
                "retained_histories": 0,
                "construction_completion_sha256": upstream_sha,
            }
            out = root / "repair" / f"{scene_index:02d}_{scene}"
            receipt = lifelong_builder.write_upstream_empty_scene(
                parent_root=root,
                protocol_path=protocol,
                parent_manifest_path=parent_manifest,
                parent_population_path=parent_population,
                population_fragment=fragment,
                scene=scene,
                scene_index=scene_index,
                online_root=parent_scene / "online_a",
                out=out,
            )
            self.assertEqual(receipt["materialized_A_histories"], 0)
            self.assertEqual(receipt["constructible_AB_C_histories"], 0)
            self.assertFalse(receipt["query_policy_outcomes_read"])
            self.assertIsNone(receipt["online_A_manifest_sha256"])
            self.assertEqual(
                receipt["upstream_parent_completion_sha256"], upstream_sha
            )
            manifest = json.loads(
                (out / "role_pairs/manifest.json").read_text()
            )
            self.assertEqual(manifest["episodes"], [])

    def test_missing_online_root_cannot_hide_materialized_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("parent_manifest.json", "population_receipt.json",
                         "protocol.json"):
                (root / name).write_text("{}\n")
            with self.assertRaisesRegex(
                RuntimeError, "despite materialized histories"
            ):
                lifelong_builder.write_upstream_empty_scene(
                    parent_root=root,
                    protocol_path=root / "protocol.json",
                    parent_manifest_path=root / "parent_manifest.json",
                    parent_population_path=root / "population_receipt.json",
                    population_fragment={
                        "scene": "scene",
                        "scene_index": 0,
                        "materialized_histories": 1,
                        "goal_a_successes": 1,
                        "retained_histories": 1,
                        "construction_completion_sha256": "0" * 64,
                    },
                    scene="scene",
                    scene_index=0,
                    online_root=root / "construction/scenes/00_scene/online_a",
                    out=root / "out",
                )

    def test_empty_scene_is_retained_as_fail_closed_attrition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            online = root / "online_a"
            online.mkdir()
            (online / "manifest.json").write_text(json.dumps({
                "schema_version": "shared_online_a_materialized_v1",
                "episodes": [],
                "attrition": [],
                "selection": {
                    "all_eligible_traces_attempted": True,
                    "eligible_count": 0,
                    "requested_count": None,
                },
                "source_trace_count": 0,
            }))
            output = root / "role_pairs"
            receipt = builder.build(
                online,
                output,
                scene_rank=3,
                source_episode_order=[],
                only_scene="frozen_scene",
            )
            self.assertEqual(receipt["source_scene"], "frozen_scene")
            self.assertEqual(receipt["source_materialized_histories"], 0)
            self.assertEqual(receipt["retained_standard_natural_histories"], 0)
            self.assertEqual(receipt["retained_hard_support_histories"], 0)
            self.assertFalse(receipt["policy_outcomes_read"])
            for protocol in ("natural_direction", "hard_support"):
                manifest = json.loads(
                    (output / protocol / "manifest.json").read_text()
                )
                self.assertEqual(manifest["episodes"], [])

    def test_global_lexicographic_stratum_cycle(self):
        self.assertEqual(
            [builder.assigned_direction_stratum(0, rank) for rank in range(8)],
            ["front", "side", "rear", "front", "side", "rear", "front", "side"],
        )
        self.assertEqual(builder.assigned_direction_stratum(1, 0), "rear")
        self.assertEqual(builder.assigned_direction_stratum(1, 1), "front")

    def test_stratum_boundaries(self):
        self.assertTrue(builder.direction_in_stratum(-60.0, "front"))
        self.assertTrue(builder.direction_in_stratum(60.0, "front"))
        self.assertFalse(builder.direction_in_stratum(60.0001, "front"))
        self.assertTrue(builder.direction_in_stratum(60.0001, "side"))
        self.assertTrue(builder.direction_in_stratum(-120.0, "side"))
        self.assertFalse(builder.direction_in_stratum(120.0001, "side"))
        self.assertTrue(builder.direction_in_stratum(120.0001, "rear"))
        self.assertTrue(builder.direction_in_stratum(180.0, "rear"))

    def test_support_bands_are_disjoint(self):
        self.assertEqual(builder.support_band(0.55, 24), "standard")
        self.assertEqual(builder.support_band(0.90, 24), "standard")
        self.assertEqual(builder.support_band(0.25, 32), "hard")
        self.assertEqual(builder.support_band(0.549999, 32), "hard")
        self.assertIsNone(builder.support_band(0.55, 25))
        self.assertIsNone(builder.support_band(0.24, 0))

    def test_goal_yaw_is_identity_bound(self):
        first = builder.goal_yaw_bin("scene", "episode_0001")
        second = builder.goal_yaw_bin("scene", "episode_0001")
        self.assertEqual(first, second)
        self.assertIn(first, range(8))
        yaw = builder.goal_yaw_radians("scene", "episode_0001")
        expected = (first * math.pi / 4.0 + math.pi) % (2.0 * math.pi) - math.pi
        self.assertAlmostEqual(yaw, expected)

    def test_local_grid_covers_only_requested_intended_stratum(self):
        endpoint = np.asarray([2.0, 0.0, -3.0])
        endpoint_yaw = 0.3
        for stratum in builder.STRATA:
            rows = builder.deterministic_novel_position_grid(
                endpoint,
                endpoint_yaw,
                scene="scene",
                episode=f"episode_{stratum}",
                stratum=stratum,
            )
            self.assertGreater(len(rows), 50)
            for position in rows:
                delta = position[[0, 2]] - endpoint[[0, 2]]
                intended_yaw = math.atan2(-delta[0], -delta[1])
                relative = builder.relative_direction_degrees(
                    intended_yaw, endpoint_yaw
                )
                self.assertTrue(
                    builder.direction_in_stratum(relative, stratum),
                    (stratum, relative),
                )

    def test_population_targets_are_joint_history_and_scene_requirements(self):
        result = finalizer._population_summary(
            {("a", "e0"), ("a", "e1"), ("b", "e0")},
            target_histories=3,
            target_scenes=2,
        )
        self.assertTrue(result["target_met"])
        under_scene = finalizer._population_summary(
            {("a", "e0"), ("a", "e1"), ("a", "e2")},
            target_histories=3,
            target_scenes=2,
        )
        self.assertFalse(under_scene["target_met"])

    def test_consumed_two_scene_finalizer_integration_when_assets_exist(self):
        repository = Path(__file__).resolve().parent.parent
        diagnostic = (
            repository
            / ".diagnostics/final14_population_v3_consumed_20260817"
        )
        fragments = (
            ("00_gxdoqLR6rwA", diagnostic / "gxdoqLR6rwA_attempt2"),
            ("01_pLe4wQe7qrG", diagnostic / "pLe4wQe7qrG_attempt4"),
        )
        online_manifest = (
            repository
            / ".diagnostics/shared_online_a_v0v1_pilot_20260812/manifest.json"
        )
        if not online_manifest.is_file() or not all(
            (fragment / "construction_receipt.json").is_file()
            for _name, fragment in fragments
        ):
            self.skipTest("consumed final14 fragments are not installed")
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            for name, fragment in fragments:
                scene_root = run_root / "traces" / name
                shutil.copytree(fragment, scene_root / "role_pairs")
                (scene_root / "online_a").mkdir(parents=True)
                shutil.copy2(online_manifest, scene_root / "online_a/manifest.json")
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "online_a_inventory.json").write_text(json.dumps({
                "source_scenes": 2,
                "manifest_sha256": "0" * 64,
                "source_episodes": 2,
                "goal_a_successes": 2,
                "materialized_histories": 2,
            }))
            output = run_root / "benchmarks"
            receipt = finalizer.finalize(
                run_root,
                output,
                expected_scene_count=2,
                natural_target_histories=2,
                natural_target_scenes=2,
                hard_target_histories=2,
                hard_target_scenes=2,
            )
            self.assertTrue(receipt["populations"]["natural_standard"]["target_met"])
            self.assertTrue(receipt["populations"]["hard_support"]["target_met"])
            audited = audit_final14(output)
            self.assertTrue(audited["ok"])
            self.assertEqual(audited["natural_standard_histories"], 2)
            self.assertEqual(audited["hard_support_histories"], 2)


if __name__ == "__main__":
    unittest.main()
