import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MemNavData.navdp_native_first_audit_server import (
    NativeFirstAuditError,
    atomic_native_first_plan,
    ndarray_sha256,
    padded_fifo_tensor,
    resample_point_image_goal_read_only,
    snapshot_memory,
    verify_same_prefix_plan_receipts,
    verify_source_file,
)


class FakePolicy:
    def __init__(self, owner, mutate=False, extra_append=False):
        self.owner = owner
        self.mutate = mutate
        self.extra_append = extra_append
        self.calls = []

    def _predict(self, mode):
        self.calls.append((mode, self.owner.applied_seed))
        if self.extra_append:
            self.owner.memory_queue[0].append(
                np.full_like(self.owner.memory_queue[0][-1], 0.25))
        if self.mutate:
            self.owner.memory_queue[0][-1][0, 0, 0] += 0.5
        batch = self.owner.batch_size
        all_trajectory = np.ones((batch, 4, 24, 2), dtype=np.float32)
        all_values = np.ones((batch, 4), dtype=np.float32)
        good = all_trajectory[:, :1].copy()
        bad = all_trajectory[:, 1:2].copy()
        return all_trajectory, all_values, good, bad

    def predict_imagegoal_action(self, image_goal, input_images, input_depth):
        del image_goal, input_images, input_depth
        return self._predict("native")

    def predict_ip_action(
        self,
        point_goal,
        image_goal,
        input_images,
        input_depth,
    ):
        del point_goal, image_goal, input_images, input_depth
        return self._predict("image_point")


class FakeAgent:
    def __init__(self, *, mutate=False, extra_append=False, first_value=0.0):
        self.memory_size = 3
        self.batch_size = 1
        self.stop_threshold = -0.5
        self.applied_seed = None
        current = np.full((2, 2, 3), 7.0 / 255.0, dtype=np.float32)
        self.memory_queue = [[
            np.full_like(current, first_value),
            current.copy(),
        ]]
        self.navi_former = FakePolicy(
            self, mutate=mutate, extra_append=extra_append)

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

    def apply_seed(self, seed):
        self.applied_seed = seed
        return seed


def inputs(current_value=7):
    point = np.asarray([[2.0, 0.5, 0.0]], dtype=np.float32)
    goal = np.full((1, 2, 2, 3), 9, dtype=np.uint8)
    current = np.full((1, 2, 2, 3), current_value, dtype=np.uint8)
    depth = np.ones((1, 2, 2, 1), dtype=np.float32)
    return point, goal, current, depth


def atomic_inputs(current_value=8, goal_value=9):
    point, _goal, _current, depth = inputs(current_value)
    goal = np.full((1, 2, 2, 3), goal_value, dtype=np.uint8)
    current = np.full((1, 2, 2, 3), current_value, dtype=np.uint8)
    return point, goal, current, depth


def run_atomic(agent, mode, *, seed=123, goal_value=9, current_value=8):
    point, goal, current, depth = atomic_inputs(
        current_value=current_value, goal_value=goal_value)
    return atomic_native_first_plan(
        agent,
        mode=mode,
        image_goal=goal,
        current_images=current,
        current_depths=depth,
        diffusion_seed=seed,
        apply_seed=agent.apply_seed,
        point_goal=point if mode == "image_point" else None,
    )


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

    def test_atomic_native_and_residual_share_exact_prefix(self):
        native_agent = FakeAgent()
        residual_agent = FakeAgent()
        native_before = snapshot_memory(native_agent)
        _execute, _trajectories, _values, native = run_atomic(
            native_agent, "native")
        _execute, _trajectories, _values, residual = run_atomic(
            residual_agent, "image_point")

        pair_sha = verify_same_prefix_plan_receipts(native, residual)
        self.assertEqual(len(pair_sha), 64)
        self.assertEqual(native["diffusion_call_count"], 1)
        self.assertEqual(residual["diffusion_call_count"], 1)
        self.assertEqual(
            native["fifo_item_sha256_before"],
            residual["fifo_item_sha256_before"],
        )
        self.assertEqual(native["fifo_item_sha256"], residual["fifo_item_sha256"])
        self.assertEqual(native["current_sha256"], residual["current_sha256"])
        self.assertEqual(native["goal_sha256"], residual["goal_sha256"])
        self.assertEqual(native_agent.navi_former.calls, [("native", 123)])
        self.assertEqual(
            residual_agent.navi_former.calls, [("image_point", 123)])
        self.assertNotEqual(native_before, snapshot_memory(native_agent))
        self.assertEqual(
            snapshot_memory(native_agent), snapshot_memory(residual_agent))

    def test_atomic_prefix_comparison_is_order_independent(self):
        residual = run_atomic(FakeAgent(), "image_point")[3]
        native = run_atomic(FakeAgent(), "native")[3]
        self.assertEqual(
            verify_same_prefix_plan_receipts(residual, native),
            verify_same_prefix_plan_receipts(native, residual),
        )

    def test_atomic_pair_rejects_wrong_goal_seed_or_fifo(self):
        native = run_atomic(FakeAgent(), "native")[3]
        wrong_goal = run_atomic(
            FakeAgent(), "image_point", goal_value=10)[3]
        with self.assertRaisesRegex(NativeFirstAuditError, "goal_sha256"):
            verify_same_prefix_plan_receipts(native, wrong_goal)

        wrong_seed = run_atomic(FakeAgent(), "image_point", seed=124)[3]
        with self.assertRaisesRegex(NativeFirstAuditError, "diffusion_seed"):
            verify_same_prefix_plan_receipts(native, wrong_seed)

        wrong_fifo = run_atomic(
            FakeAgent(first_value=0.125), "image_point")[3]
        with self.assertRaisesRegex(NativeFirstAuditError, "fifo_before"):
            verify_same_prefix_plan_receipts(native, wrong_fifo)

    def test_atomic_rejects_non_tail_or_extra_append_and_rolls_back(self):
        mutating = FakeAgent(mutate=True)
        before = snapshot_memory(mutating)
        with self.assertRaisesRegex(NativeFirstAuditError, "FIFO tail"):
            run_atomic(mutating, "native")
        self.assertEqual(before, snapshot_memory(mutating))

        extra = FakeAgent(extra_append=True)
        before = snapshot_memory(extra)
        with self.assertRaisesRegex(NativeFirstAuditError, "FIFO tail"):
            run_atomic(extra, "image_point")
        self.assertEqual(before, snapshot_memory(extra))

    def test_atomic_rejects_invalid_mode_goal_and_seed_without_append(self):
        point, goal, current, depth = atomic_inputs()
        agent = FakeAgent()
        before = snapshot_memory(agent)
        with self.assertRaisesRegex(NativeFirstAuditError, "unsupported"):
            atomic_native_first_plan(
                agent,
                mode="unknown",
                image_goal=goal,
                current_images=current,
                current_depths=depth,
                diffusion_seed=123,
                apply_seed=agent.apply_seed,
            )
        self.assertEqual(before, snapshot_memory(agent))

        with self.assertRaisesRegex(NativeFirstAuditError, "forbids"):
            atomic_native_first_plan(
                agent,
                mode="native",
                image_goal=goal,
                current_images=current,
                current_depths=depth,
                diffusion_seed=123,
                apply_seed=agent.apply_seed,
                point_goal=point,
            )
        self.assertEqual(before, snapshot_memory(agent))

        with self.assertRaisesRegex(NativeFirstAuditError, "integer"):
            atomic_native_first_plan(
                agent,
                mode="image_point",
                image_goal=goal,
                current_images=current,
                current_depths=depth,
                diffusion_seed=True,
                apply_seed=agent.apply_seed,
                point_goal=point,
            )
        self.assertEqual(before, snapshot_memory(agent))

    def test_tampered_receipt_or_call_count_fails_closed(self):
        native = run_atomic(FakeAgent(), "native")[3]
        residual = run_atomic(FakeAgent(), "image_point")[3]
        residual["diffusion_call_count"] = 2
        with self.assertRaisesRegex(NativeFirstAuditError, "call count"):
            verify_same_prefix_plan_receipts(native, residual)

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
