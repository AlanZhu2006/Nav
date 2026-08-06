import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MemNavData.navdp_native_first_audit_server import (
    NativeFirstAuditError,
    ndarray_sha256,
    padded_fifo_tensor,
    resample_point_image_goal_read_only,
    snapshot_memory,
    verify_source_file,
)


class FakePolicy:
    def __init__(self, owner, mutate=False):
        self.owner = owner
        self.mutate = mutate

    def predict_ip_action(
        self,
        point_goal,
        image_goal,
        input_images,
        input_depth,
    ):
        if self.mutate:
            self.owner.memory_queue[0][-1][0, 0, 0] += 0.5
        batch = len(point_goal)
        all_trajectory = np.ones((batch, 4, 24, 2), dtype=np.float32)
        all_values = np.ones((batch, 4), dtype=np.float32)
        good = all_trajectory[:, :1].copy()
        bad = all_trajectory[:, 1:2].copy()
        return all_trajectory, all_values, good, bad


class FakeAgent:
    def __init__(self, *, mutate=False):
        self.memory_size = 3
        self.batch_size = 1
        self.stop_threshold = -0.5
        current = np.full((2, 2, 3), 7.0 / 255.0, dtype=np.float32)
        self.memory_queue = [[
            np.zeros_like(current),
            current.copy(),
        ]]
        self.navi_former = FakePolicy(self, mutate=mutate)

    @staticmethod
    def process_image(images):
        return np.asarray(images, dtype=np.float32) / 255.0

    @staticmethod
    def process_depth(depths):
        return np.asarray(depths, dtype=np.float32)

    @staticmethod
    def process_pointgoal(goals):
        goals = np.asarray(goals, dtype=np.float32).copy()
        goals[:, 0] = np.clip(goals[:, 0], 0.0, 10.0)
        return goals


def inputs(current_value=7):
    point = np.asarray([[2.0, 0.5, 0.0]], dtype=np.float32)
    goal = np.full((1, 2, 2, 3), 9, dtype=np.uint8)
    current = np.full((1, 2, 2, 3), current_value, dtype=np.uint8)
    depth = np.ones((1, 2, 2, 1), dtype=np.float32)
    return point, goal, current, depth


class NavDPNativeFirstAuditServerTest(unittest.TestCase):
    def test_ndarray_hash_includes_dtype_shape_and_bytes(self):
        value = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertEqual(ndarray_sha256(value), ndarray_sha256(value.copy()))
        self.assertNotEqual(ndarray_sha256(value), ndarray_sha256(value + 1))
        self.assertNotEqual(
            ndarray_sha256(value), ndarray_sha256(value.astype(np.float64)))
        self.assertNotEqual(
            ndarray_sha256(value), ndarray_sha256(value.reshape(3, 2)))

    def test_padding_matches_navdp_left_zero_semantics(self):
        one = np.ones((2, 2, 1), dtype=np.float32)
        two = np.full((2, 2, 1), 2.0, dtype=np.float32)
        padded = padded_fifo_tensor([[one, two]], 3)
        self.assertEqual(padded.shape, (1, 3, 2, 2, 1))
        self.assertTrue(np.array_equal(padded[0, 0], np.zeros_like(one)))
        self.assertTrue(np.array_equal(padded[0, 1], one))
        self.assertTrue(np.array_equal(padded[0, 2], two))

    def test_snapshot_hashes_real_items_and_model_tensor(self):
        agent = FakeAgent()
        first = snapshot_memory(agent)
        second = snapshot_memory(agent)
        self.assertEqual(first, second)
        self.assertEqual(first["queue_lengths"], [2])
        self.assertEqual(len(first["queue_item_sha256"][0]), 2)
        self.assertEqual(len(first["padded_model_tensor_sha256"]), 64)
        agent.memory_queue[0][-1][0, 0, 0] += 0.1
        self.assertNotEqual(first["fifo_sha256"], snapshot_memory(agent)["fifo_sha256"])

    def test_read_only_mixed_resample_preserves_fifo(self):
        agent = FakeAgent()
        before = snapshot_memory(agent)
        execute, trajectories, values, audit = (
            resample_point_image_goal_read_only(agent, *inputs()))
        self.assertEqual(execute.shape, (1, 24, 2))
        self.assertEqual(trajectories.shape, (1, 4, 24, 2))
        self.assertEqual(values.shape, (1, 4))
        self.assertEqual(before, snapshot_memory(agent))
        self.assertEqual(audit["fifo_before"], audit["fifo_after"])
        self.assertFalse(audit["memory_mutated"])

    def test_current_image_must_equal_fifo_tail(self):
        with self.assertRaisesRegex(NativeFirstAuditError, "FIFO tail"):
            resample_point_image_goal_read_only(FakeAgent(), *inputs(8))

    def test_predictor_fifo_mutation_fails_closed(self):
        with self.assertRaisesRegex(NativeFirstAuditError, "mutated"):
            resample_point_image_goal_read_only(
                FakeAgent(mutate=True), *inputs())

    def test_ragged_or_mixed_empty_batch_fails(self):
        one = np.zeros((2, 2, 1), dtype=np.float32)
        with self.assertRaisesRegex(NativeFirstAuditError, "ragged"):
            padded_fifo_tensor([[one, np.zeros((3, 2, 1), dtype=np.float32)]], 3)
        agent = FakeAgent()
        agent.memory_queue = [[], [one]]
        agent.batch_size = 2
        with self.assertRaisesRegex(NativeFirstAuditError, "mixed"):
            snapshot_memory(agent)

    def test_nonfinite_array_fails_closed(self):
        with self.assertRaisesRegex(NativeFirstAuditError, "non-finite"):
            ndarray_sha256(np.asarray([np.nan], dtype=np.float32))
        with self.assertRaisesRegex(NativeFirstAuditError, "numeric"):
            ndarray_sha256(np.asarray(["not-an-image"]))

    def test_source_verification_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.py"
            path.write_bytes(b"print('ok')\n")
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                verify_source_file(path, expected, "test source"), expected)
            with self.assertRaisesRegex(NativeFirstAuditError, "mismatch"):
                verify_source_file(path, "0" * 64, "test source")


if __name__ == "__main__":
    unittest.main()
