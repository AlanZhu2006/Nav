import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MemNavData.build_unknown_goal_natural_stream_teacher import (
    camera_to_world_habitat,
    index_natural_frames,
    shortlist_support_label,
)


class NaturalStreamTeacherTests(unittest.TestCase):
    def payload(self, digest):
        return {
            "rollout_traces": {
                "legA": [{"step": 0, "x": 1.0, "y": 2.0, "z": 3.0,
                          "yaw": 0.1, "jpg_sha256": digest}],
                "legB": [{"step": 0, "x": 2.0, "y": 2.0, "z": 3.0,
                          "yaw": 0.2, "jpg_sha256": digest}],
                "legC": [{"step": 0, "x": 3.0, "y": 2.0, "z": 3.0,
                          "yaw": 0.3, "jpg_sha256": digest}],
            },
            "memory_traces": {
                "legA": [{"frame_idx": 0, "step": 0, "x": 1.0,
                          "z": 3.0, "yaw": 0.1}],
                "legB": [{"frame_idx": 1, "step": 0, "x": 2.0,
                          "z": 3.0, "yaw": 0.2}],
                "legC": [{"frame_idx": 2, "step": 0, "x": 3.0,
                          "z": 3.0, "yaw": 0.3}],
            },
        }

    def test_index_binds_frame_pose_and_exact_rgb(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            content = b"same-jpeg-bytes"
            digest = hashlib.sha256(content).hexdigest()
            for frame in range(3):
                (root / f"{frame}.jpg").write_bytes(content)
            frames = index_natural_frames(self.payload(digest), root)
            self.assertEqual(sorted(frames), [0, 1, 2])
            np.testing.assert_allclose(
                frames[1].camera_position(0.5), [2.0, 2.5, 3.0])

    def test_index_rejects_pose_mismatch(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            content = b"same-jpeg-bytes"
            digest = hashlib.sha256(content).hexdigest()
            for frame in range(3):
                (root / f"{frame}.jpg").write_bytes(content)
            payload = self.payload(digest)
            payload["memory_traces"]["legB"][0]["x"] = 99.0
            with self.assertRaisesRegex(ValueError, "pose mismatch"):
                index_natural_frames(payload, root)

    def test_censored_downstream_legs_are_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            content = b"same-jpeg-bytes"
            digest = hashlib.sha256(content).hexdigest()
            (root / "0.jpg").write_bytes(content)
            payload = self.payload(digest)
            for leg in ("legB", "legC"):
                payload["rollout_traces"][leg] = []
                payload["memory_traces"][leg] = []
            with self.assertRaisesRegex(ValueError, "legB rollout trace"):
                index_natural_frames(payload, root)
            frames = index_natural_frames(
                payload, root, allow_censored_legs=True)
            self.assertEqual(sorted(frames), [0])

    def test_camera_rotation_matches_habitat_yaw_convention(self):
        transform = camera_to_world_habitat(
            np.asarray([1.0, 2.0, 3.0]), np.pi / 2)
        np.testing.assert_allclose(transform[:3, 3], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            transform[:3, :3],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
             [-1.0, 0.0, 0.0]],
            atol=1e-12,
        )

    def test_shortlist_labels_are_fail_closed(self):
        self.assertEqual(shortlist_support_label([])[0], 0)
        self.assertEqual(shortlist_support_label([0, 0])[0], 0)
        self.assertEqual(shortlist_support_label([0, -1])[0], -1)
        self.assertEqual(shortlist_support_label([0, 1, -1])[0], 1)


if __name__ == "__main__":
    unittest.main()
