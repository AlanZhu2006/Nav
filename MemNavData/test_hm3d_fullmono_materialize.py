from pathlib import Path
import tempfile
import unittest

from MemNavData.materialize_hm3d_fullmono_online_a import _candidate


class FullMonoMaterializationTest(unittest.TestCase):
    def test_single_anchor_candidate_enforces_causal_margin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode_0000_leg1_trace.json"
            poses = [
                {"step": index, "x": float(index) / 10.0,
                 "y": 0.0, "z": 0.0, "yaw": 0.0}
                for index in range(60)
            ]
            payload = {"source_scene": "scene", "episode": "episode_0000",
                       "poses": poses}
            candidate = _candidate(path, payload)
            self.assertGreaterEqual(candidate.anchor, 39)
            self.assertLess(candidate.anchor, len(poses) - 16)
            with self.assertRaisesRegex(ValueError, "insufficient_history"):
                _candidate(path, {**payload, "poses": poses[:55]})


if __name__ == "__main__":
    unittest.main()
