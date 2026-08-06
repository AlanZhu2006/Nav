import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from MemNavData.deterministic_eval_protocol import (
    bytes_sha256,
    diffusion_plan_seed,
    diffusion_resample_seed,
    load_leg1_trace,
    validate_leg1_trace,
    write_leg1_trace,
)


class DeterministicEvalProtocolTest(unittest.TestCase):
    @staticmethod
    def payload():
        return {
            "schema_version": 1,
            "episode": "episode_0003",
            "episode_seed": 17,
            "goal_sha256": bytes_sha256(b"goal"),
            "goal_source_episode": "episode_0003",
            "source_scene": "scene",
            "source_backend": "hybrid_pose",
            "source_hybrid_route": "memory_geometry",
            "source_retrieval_candidate_min_gap": 16,
            "source_graph_subgoal_spacing_m": 0.0,
            "source_graph_subgoal_arrival_m": 0.60,
            "reached": True,
            "path_len": 1.25,
            "path_len_at_reach": 1.25,
            "step_at_reach": 2,
            "steps": 2,
            "final_goal_dist_m": 0.4,
            "end_position": [1.0, 0.0, 2.0],
            "end_yaw": 0.2,
            "poses": [
                {"step": 0, "x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0,
                 "jpg_sha256": bytes_sha256(b"frame-0")},
                {"step": 1, "x": 0.1, "y": 0.0, "z": 0.0, "yaw": 0.1,
                 "jpg_sha256": bytes_sha256(b"frame-1")},
            ],
            "plans": [],
        }

    def test_plan_seeds_are_stable_and_disjoint(self):
        self.assertEqual(diffusion_plan_seed(17, 0, 0), 1_700_000)
        self.assertEqual(diffusion_plan_seed(17, 1, 0), 1_710_000)
        self.assertNotEqual(
            diffusion_plan_seed(17, 1, 7),
            diffusion_plan_seed(18, 0, 7),
        )
        self.assertEqual(
            diffusion_resample_seed(diffusion_plan_seed(17, 1, 7), 1),
            171_000_701,
        )
        with self.assertRaisesRegex(ValueError, "resample_index"):
            diffusion_resample_seed(17, 0)

    def test_trace_round_trip_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            expected_hash = write_leg1_trace(path, self.payload())
            loaded, actual_hash = load_leg1_trace(
                path,
                expected_episode="episode_0003",
                expected_seed=17,
                expected_goal_sha256=bytes_sha256(b"goal"),
                expected_source_scene="scene",
            )
            self.assertEqual(loaded["steps"], 2)
            self.assertEqual(actual_hash, expected_hash)

    def test_trace_rejects_wrong_goal_and_sparse_steps(self):
        payload = self.payload()
        with self.assertRaisesRegex(ValueError, "Goal-A image mismatch"):
            validate_leg1_trace(
                payload, expected_goal_sha256=bytes_sha256(b"other"))
        broken = json.loads(json.dumps(payload))
        broken["poses"][1]["step"] = 3
        with self.assertRaisesRegex(ValueError, "dense and ordered"):
            validate_leg1_trace(broken)
        with self.assertRaisesRegex(ValueError, "source scene mismatch"):
            validate_leg1_trace(payload, expected_source_scene="other")

    def test_navdp_http_seed_parser_is_strict(self):
        path = (Path(__file__).resolve().parents[1]
                / "NavDP/baselines/navdp/deterministic_seed.py")
        spec = importlib.util.spec_from_file_location(
            "navdp_deterministic_seed", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        self.assertEqual(module.normalize_seed("17"), 17)
        self.assertIsNone(module.normalize_seed(""))
        self.assertEqual(module.apply_seed(2_026_080_300_000),
                         2_026_080_300_000)
        with self.assertRaisesRegex(ValueError, "canonical"):
            module.normalize_seed("017")
        with self.assertRaisesRegex(ValueError, "not bool"):
            module.normalize_seed(True)


if __name__ == "__main__":
    unittest.main()
