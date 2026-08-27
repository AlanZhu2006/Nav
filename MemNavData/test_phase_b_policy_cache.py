import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from NavDP.baselines.memnav.policy_agent import MemNavAgent


class _Ranker:
    checkpoint_sha256 = "a" * 64

    def __init__(self):
        self.rows = []

    def rank(self, rows):
        self.rows.append(rows)
        count = len(rows)
        return {
            "order": list(range(count)),
            "rank_probability": [1.0 / count] * count,
            "candidate_validity": [0.5] * count,
            "no_match_probability_diagnostic": 0.25,
        }


class PhaseBPolicyCacheTest(unittest.TestCase):
    def test_geometry_cache_ignores_dino_float_jitter_but_rank_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for anchor in (40, 56):
                (root / f"{anchor}.jpg").write_bytes(b"frame")

            agent = MemNavAgent.__new__(MemNavAgent)
            agent.phase_b_ranker = _Ranker()
            agent.amargin = 0
            agent.n = 100
            agent.rgb_dir = str(root)
            agent.device = "cpu"
            agent.camera_height = 1.25
            agent._goal_start_frame = {"goal-key": 100}
            agent._phase_b_rank_cache = {}
            agent._phase_b_scale_cache = {}
            agent._phase_b_geometry_cache = {}
            agent._snapshot = lambda: {}
            agent._restore = lambda _snapshot: None
            agent._live_cache = lambda: {"cam_pose_enc": torch.zeros(100, 9)}
            agent.lb = mock.Mock()
            agent.lb.load_images.return_value = torch.zeros(1, 3, 4, 4)

            scale_calls = []
            geometry_calls = []

            def external_scale(*_args, **_kwargs):
                scale_calls.append(True)
                return 2.0, {
                    "external_scale_valid_frame_ratio": 1.0,
                    "external_scale_relative_h_iqr": 0.1,
                    "external_scale_clamped": 0.0,
                }

            def append_geometry(_lb, _cache, _root, _goal, anchor):
                geometry_calls.append(anchor)
                return {"anchor": anchor, "cloud_overlap_f1": 0.75}

            def feature_row(measurement, *, dino_cosine, **_kwargs):
                value = float(dino_cosine)
                return {
                    "dino_cosine": value,
                    "predicted_relative_xy_m": np.array([value, 0.0]),
                    "anchor_goal_distance_norm_center": 1.0,
                    "goal_refine_translation_norm_median": 0.1,
                    "goal_refine_rotation_deg_median": 2.0,
                }

            candidates = [
                {"anchor": 40, "score": 0.9},
                {"anchor": 56, "score": 0.8},
            ]
            jittered = [
                {"anchor": 40, "score": 0.90000001},
                {"anchor": 56, "score": 0.80000001},
            ]
            with (
                mock.patch("hashlib.md5") as md5,
                mock.patch(
                    "MemNavData.phase_b_runtime.external_causal_metric_scale",
                    side_effect=external_scale,
                ),
                mock.patch(
                    "MemNavData.phase_b_runtime.append_goal_geometry",
                    side_effect=append_geometry,
                ),
                mock.patch(
                    "MemNavData.phase_b_runtime.measurement_feature_row",
                    side_effect=feature_row,
                ),
            ):
                md5.return_value.hexdigest.return_value = "goal-key"
                first = agent.rank_retrieval_candidates(b"goal", candidates)
                second = agent.rank_retrieval_candidates(b"goal", jittered)
                third = agent.rank_retrieval_candidates(b"goal", jittered)

            self.assertTrue(first["ok"], first)
            self.assertEqual(first["geometry_cache_miss_count"], 2)
            self.assertEqual(first["geometry_cache_hit_count"], 0)
            self.assertFalse(first["scale_cached"])
            self.assertTrue(second["ok"], second)
            self.assertFalse(second["cached"])
            self.assertEqual(second["geometry_cache_miss_count"], 0)
            self.assertEqual(second["geometry_cache_hit_count"], 2)
            self.assertTrue(second["scale_cached"])
            self.assertTrue(third["cached"])
            self.assertEqual(len(scale_calls), 1)
            self.assertEqual(geometry_calls, [40, 56])
            self.assertEqual(len(agent.phase_b_ranker.rows), 2)
            self.assertEqual(
                agent.phase_b_ranker.rows[1][0]["dino_cosine"],
                jittered[0]["score"],
            )


if __name__ == "__main__":
    unittest.main()
