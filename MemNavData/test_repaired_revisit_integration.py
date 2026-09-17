"""CPU checks of the standalone orchestration, not policy performance."""
import inspect
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import repaired_revisit_integration as r


class IntegrationContractTests(unittest.TestCase):
    def test_fixed_population_and_order(self):
        self.assertEqual([s[0] for s in r.SOURCES], ["rJhMRvNn4DS", "jgPBycuV1Jq"])
        self.assertEqual(r.arm_order(0), ("native", "raw_fixed", "cec"))
        self.assertEqual(r.arm_order(1), ("cec", "raw_fixed", "native"))
        with self.assertRaises(ValueError):
            r.arm_order(2)

    def test_no_research_or_novel_construction(self):
        source = inspect.getsource(r.build)
        self.assertNotIn("search_revisit_candidates(", source)
        self.assertNotIn("sample_natural_novel(", source)
        self.assertNotIn("write_protocol_episode(", source)
        self.assertIn('"new_a_rollouts": 0', source)

    def test_saved_identity_and_measurement_must_match(self):
        selected = dict(source_frame=55, render_attempt=1, max_online_a_covis_frame=39,
                        support_band="standard", query_geodesic_m=3., initial_path_bearing_rad=.4,
                        translation_m=.46, yaw_delta_deg=45., source_anchor_covis=.6,
                        max_online_a_covis=.719, pixel_mae=20.)
        r.check_selected(selected, dict(selected))
        for key, value in (("render_attempt", 2), ("max_online_a_covis", .718),
                           ("query_geodesic_m", 3.1), ("support_band", "hard")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.check_selected(selected, dict(selected, **{key: value}))

    def test_pose_reconstruction_uses_one_saved_grid_index(self):
        import build_final14_role_pair_scene as b
        class Pathfinder:
            def snap_point(self, point):
                return point
            def is_navigable(self, point):
                return True
        h = {"floor_positions": [np.array([1., .2, 2.])], "poses": [{"yaw": .3}]}
        selected = {"source_frame": 0, "render_attempt": 2}
        with patch.object(b, "deterministic_pose_grid", return_value=[(.2, 0., 15.), (.46, 0., 30.)]) as grid:
            point, yaw = r.reconstruct_pose(Pathfinder(), h, selected, "scene", "episode_0000")
        grid.assert_called_once_with("scene/episode_0000/0")
        np.testing.assert_allclose(point, [1.46, .2, 2.])
        self.assertAlmostEqual(yaw, .3 + np.pi / 6)

    def test_command_changes_entry_only(self):
        from MemNavData.run_repaired_fullmono_local import evaluator_command as original
        source = {"scene": "rJhMRvNn4DS", "episode": "episode_0000", "asset": "/unused.glb",
                  "source_episode": "/unused/episode_0000", "seed": 2026082200}
        for arm in r.ARMS:
            old = original(source, Path("/unused/out"), 21750, 21751, arm=arm,
                           role="revisit", benchmark=Path("/unused/bench"))
            new = r.evaluator_command(source, Path("/unused/out"), 21750, 21751,
                                      arm=arm, benchmark=Path("/unused/bench"))
            self.assertEqual(new[:2], old[:2])
            self.assertEqual(new[4:], old[4:])
            self.assertEqual(new[3], "eval")

    def test_single_query_does_not_accept_pair_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "query.json"
            r.dump(p, {"schema_version": r.SCHEMA, "scene": "x", "pairs": [], "mixed_role_pairs": 0})
            with self.assertRaisesRegex(ValueError, "not a role-pair"):
                r.load_query(p, "x")


if __name__ == "__main__":
    unittest.main()
