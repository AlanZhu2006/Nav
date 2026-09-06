import hashlib
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np
import torch

from NavDP.baselines.memnav.policy_agent import MemNavAgent


def pose9(x: float, z: float) -> torch.Tensor:
    value = np.zeros(9, dtype=np.float32)
    value[0] = x
    value[2] = z
    value[6] = 1.0
    return torch.from_numpy(value)


class PolicyAgentPathFieldTest(unittest.TestCase):
    @staticmethod
    def agent(scale=1.0):
        agent = MemNavAgent.__new__(MemNavAgent)
        # Frames 0..2 are the immutable history: goal, corner, causal tail.
        # Frame 3 is the first query observation at that same tail.
        agent.cam_pose = [
            pose9(4.0, 4.0),
            pose9(4.0, 0.0),
            pose9(0.0, 0.0),
            pose9(0.0, 0.0),
        ]
        agent._certified_path_field_routes = {}
        agent._first40_local_pose_metric_scale = lambda: {
            "available": True,
            "metric_scale_m_per_raw": scale,
            "scale_receipt_sha256": "a" * 64,
        }
        return agent

    def test_runtime_readout_uses_local_ordered_history(self):
        agent = self.agent()
        direction, receipt = agent._certified_path_field_direction(
            goal_key="goal",
            target_anchor=0,
            goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy(),
        )
        np.testing.assert_allclose(direction, [0.0, -1.0], atol=1e-7)
        self.assertEqual(receipt["episodic_path_field_status"], "active")
        self.assertFalse(receipt["episodic_path_field_evaluator_pose_consumed"])
        self.assertFalse(receipt["episodic_path_field_habitat_path_consumed"])
        self.assertAlmostEqual(receipt["path_progress_m"], 0.0)
        self.assertAlmostEqual(receipt["path_remaining_m"], 8.0)

    def test_causal_pose_trace_receipt_contains_only_model_state(self):
        agent = self.agent()
        agent.n = len(agent.cam_pose)
        agent._first40_local_pose_metric_scale = lambda: {
            "available": True,
            "metric_scale_m_per_raw": 1.25,
        }
        receipt = agent.causal_pose_trace_receipt()
        self.assertEqual(receipt["pose_count"], 4)
        self.assertEqual(receipt["stream_observation_count"], 4)
        self.assertEqual(len(receipt["pose9"]), 4)
        self.assertFalse(receipt["runtime_evaluator_pose_visible"])
        self.assertFalse(receipt["metric_depth_sensor_consumed"])
        self.assertEqual(
            receipt["longrange_metric_scale"]["metric_scale_m_per_raw"],
            1.25,
        )

    def test_causal_visual_similarity_splits_one_stream(self):
        agent = self.agent()
        agent.device = torch.device("cpu")
        agent.dino_cls = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 0.0]),
        ]
        receipt = agent.causal_visual_similarity_receipt(2)
        self.assertEqual(receipt["history_count"], 2)
        self.assertEqual(receipt["query_count"], 1)
        np.testing.assert_allclose(receipt["similarity"], [[1.0, 0.0]])
        self.assertFalse(receipt["runtime_evaluator_pose_visible"])

    def test_runtime_progress_is_cached_and_monotone(self):
        agent = self.agent()
        _, first = agent._certified_path_field_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy())
        agent.cam_pose[-1] = pose9(2.0, 0.0)
        _, second = agent._certified_path_field_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy())
        agent.cam_pose[-1] = pose9(1.0, 0.0)
        _, regressed_observation = agent._certified_path_field_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy())
        self.assertGreater(second["path_progress_m"], first["path_progress_m"])
        self.assertEqual(
            regressed_observation["path_progress_m"],
            second["path_progress_m"],
        )

    def test_missing_scale_is_explicit_geometry_failure(self):
        agent = self.agent()
        agent._first40_local_pose_metric_scale = lambda: {
            "available": False,
            "reason": "test_scale_missing",
        }
        direction, receipt = agent._certified_path_field_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy())
        self.assertIsNone(direction)
        self.assertEqual(
            receipt["episodic_path_field_status"], "geometry_failure")
        self.assertEqual(
            receipt["episodic_path_field_error"], "test_scale_missing")

    def test_reanchored_pose_replaces_drifted_global_pose(self):
        agent = self.agent()
        # Create the route while the causal tail is still correct.
        agent._certified_path_field_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy())
        # The live global pose drifts away, while a local PnP witness places
        # the camera two metres along the demonstrated first segment.
        agent.cam_pose[-1] = pose9(20.0, 20.0)
        direction, receipt = agent._certified_path_field_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy(),
            current_pose9_override=pose9(2.0, 0.0).numpy(),
        )
        self.assertAlmostEqual(receipt["path_progress_m"], 2.0)
        expected = np.asarray([0.5, -2.0], dtype=np.float64)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(direction, expected, atol=1e-7)

    def test_visual_route_address_can_advance_beyond_control_horizon(self):
        agent = self.agent()
        agent._certified_path_field_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy())
        direction, receipt = agent._certified_path_field_direction(
            goal_key="goal", target_anchor=0, goal_start_frame=3,
            goal_pose9=pose9(4.0, 4.0).numpy(),
            current_pose9_override=pose9(4.0, 0.0).numpy(),
            route_progress_hint_m=4.0,
        )
        self.assertGreater(receipt["path_progress_m"], 2.5)
        self.assertAlmostEqual(receipt["path_progress_m"], 4.0)
        np.testing.assert_allclose(direction, [1.0, 0.0], atol=1e-7)

    def test_route_coordinate_uses_monotone_route_without_fallback(self):
        goal = b"target-goal"
        target_key = hashlib.md5(goal).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            agent = MemNavAgent.__new__(MemNavAgent)
            agent.device = torch.device("cpu")
            agent.S = 16
            agent.n = 24
            agent.rgb_dir = temporary
            Path(temporary, "23.jpg").write_bytes(b"current-rgb")
            agent.cam_pose = [
                pose9(0.0, float(index)) for index in range(agent.n)
            ]
            agent.dino_cls = [
                torch.tensor([float(index), 1.0])
                for index in range(agent.n)
            ]
            # Make frame 21 the best appearance inside the 22->20 local
            # control window; a globally similar remote frame cannot enter.
            agent.dino_cls[-1] = agent.dino_cls[21].clone()
            agent._certified_route_depth_cached_anchors = {20, 21, 22}
            agent._first40_local_pose_metric_scale = lambda: {
                "available": True,
                "metric_scale_m_per_raw": 1.0,
                "scale_receipt_sha256": "a" * 64,
            }
            agent._certified_path_field_routes = {}
            agent._certified_relocalization_cache = {}
            agent._goal_start_frame = {target_key: 23}
            agent._goal_cache = {}
            agent._anchor_state = {}
            agent._graph_routes = {}
            agent._retrieval_verification_cache = {}
            agent._phase_b_rank_cache = {}
            agent._phase_b_scale_cache = {}
            agent._phase_b_geometry_cache = {}
            agent._pi3x_relocalization_cache = {}
            agent._certified_graph_routes = {}
            agent._certified_candidate_cache = {}
            agent._certified_route_live_depth_cache = {}

            agent._certified_path_field_direction(
                goal_key=target_key,
                target_anchor=16,
                goal_start_frame=23,
                goal_pose9=pose9(0.0, 16.0).numpy(),
            )
            agent._certified_relocalization_cache[target_key] = {
                "result": {"accepted": True},
                "goal_pose9": pose9(0.0, 16.0).numpy().tolist(),
            }
            observed = {}

            def fake_relocalize(this, _image, candidates, **kwargs):
                observed["anchors"] = [
                    int(candidate["anchor"]) for candidate in candidates]
                observed["kwargs"] = kwargs
                return {
                    "accepted": True,
                    "selected_anchor": 21,
                    "certificate": {"accepted": False},
                    "pnp": {
                        "status": "ok",
                        "pose9": pose9(0.0, 21.0).numpy().tolist(),
                    },
                }

            agent.certified_relocalize = types.MethodType(
                fake_relocalize, agent)
            result = agent.certified_path_field_reanchor(goal)

            self.assertTrue(result["state_updated"])
            self.assertEqual(result["status"], "active")
            self.assertFalse(result["endpoint_fallback_available"])
            self.assertFalse(result["native_fallback_available"])
            self.assertFalse(result["distance_gate_present"])
            self.assertEqual(observed["anchors"], [21])
            self.assertEqual(
                observed["kwargs"]["authority_policy"],
                "pnp_pose_available",
            )
            self.assertEqual(
                observed["kwargs"]["reference_depth_source"],
                "route_sparse",
            )
            self.assertEqual(result["path_progress_m"], 1.0)


if __name__ == "__main__":
    unittest.main()
