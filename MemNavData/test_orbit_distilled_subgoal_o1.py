import unittest

import numpy as np
import torch

from MemNavData.train_orbit_distilled_subgoal_o1 import (
    SingleViewCompass,
    camera_relative_rings,
    primary_row_mask,
    relative_logits_to_world,
    single_view_gate,
    single_view_metric_report,
    view_dose_report,
)


class OrbitDistilledSubgoalO1Test(unittest.TestCase):
    def test_relative_ring_transport_is_exact(self):
        world = np.arange(16, dtype=np.float64).reshape(2, 8)
        relative = camera_relative_rings(world)
        self.assertEqual(relative.shape, (2, 8, 8))
        for view in range(8):
            np.testing.assert_array_equal(
                relative[:, view], np.roll(world, -view, axis=1))
        reconstructed = relative_logits_to_world(relative)
        np.testing.assert_array_equal(
            reconstructed,
            np.broadcast_to(world[:, None], reconstructed.shape))

    def test_single_view_readout_has_one_complete_ring(self):
        torch.manual_seed(4)
        model = SingleViewCompass(16)
        features = torch.randn(5, 16)
        logits = model(features)
        self.assertEqual(tuple(logits.shape), (5, 8))
        loss = logits.square().mean()
        loss.backward()
        gradient = sum(
            float(parameter.grad.square().sum())
            for parameter in model.parameters()
            if parameter.grad is not None)
        self.assertGreater(gradient, 0.0)

    @staticmethod
    def _perfect_fixture():
        advantages = np.asarray([
            [-0.6, 1.0, 0.4, 0.1, -0.2, -0.3, -0.4, -0.5],
            [-0.2, 0.0, 0.3, 1.1, 0.5, 0.2, -0.1, -0.3],
        ], dtype=np.float64)
        teacher = np.zeros_like(advantages)
        teacher[0, 1] = 1.0
        teacher[1, 3] = 1.0
        relative_teacher = camera_relative_rings(teacher)
        correct = relative_teacher * 12.0
        swapped = np.zeros_like(correct)
        arrays = {
            "advantages_m": advantages,
            "teacher_distribution": teacher,
            "scene": np.asarray(["scene_a", "scene_b"]),
            "state_name": np.asarray(["goal_b_t0", "goal_b_t0"]),
            "goal_variant": np.asarray(["factual", "factual"]),
        }
        return arrays, correct, swapped

    def test_perfect_single_view_predictions_pass_causal_gate(self):
        arrays, correct, swapped = self._perfect_fixture()
        primary = primary_row_mask(arrays)
        report = single_view_metric_report(
            correct, swapped, arrays, row_mask=primary,
            off_axis_only=True, seed=7)
        self.assertEqual(report["top1_exact_bin"]["scene_macro_mean"], 1.0)
        self.assertEqual(
            report["paired_progress_counts"]["losses_lt_minus_0p25m"], 0)
        self.assertGreater(
            report["selected_minus_camera_forward_progress_m"]
            ["scene_cluster_bootstrap_95"]["lower_95"], 0.0)
        self.assertGreater(
            report["goal_swap_nll_increase"]
            ["scene_cluster_bootstrap_95"]["lower_95"], 0.0)
        passed, conditions = single_view_gate(
            {"primary_factual_t0_off_axis": report},
            {"best_bin_different_rate": 0.5})
        self.assertTrue(passed)
        self.assertTrue(all(conditions.values()))

    def test_view_dose_aligns_relative_predictions_before_averaging(self):
        arrays, correct, swapped = self._perfect_fixture()
        report = view_dose_report(
            correct, swapped, arrays,
            row_mask=np.asarray([True, True]), seed=11)
        for dose in ("1", "2", "4", "8"):
            self.assertEqual(
                report[dose]["all_starts_top1"]["scene_macro_mean"], 1.0)
            self.assertEqual(
                report[dose]["off_axis_losses_lt_minus_0p25m"], 0)


if __name__ == "__main__":
    unittest.main()
