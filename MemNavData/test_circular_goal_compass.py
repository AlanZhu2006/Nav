import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MemNavData.circular_goal_compass import (
    NUM_DIRECTIONS,
    circular_bin_error,
    deterministic_gauge_bin,
    deterministic_scene_folds,
    native_scan_index,
    scan_yaws,
    scene_cluster_bootstrap,
    teacher_distribution,
    world_forward_xz,
)
from MemNavData.build_cgc_multiyaw_dataset import (
    DatasetBuildError,
    apply_group_eligibility,
    resolve_relative_input,
)


class CircularGoalCompassPureTest(unittest.TestCase):
    @staticmethod
    def _eligibility(**overrides):
        value = {
            "schema_version": "orbit_distilled_subgoal_eligibility_v1",
            "scope": "unit-test physical-state eligibility",
            "input_physical_group_count": 3,
            "input_scene_count": 2,
            "excluded_scenes": ["scene_b"],
            "excluded_group_ids": ["scene_a/group_1"],
            "expected_selected_physical_group_count": 1,
            "expected_selected_scene_count": 1,
        }
        value.update(overrides)
        return value

    @staticmethod
    def _groups():
        def pair(scene):
            return [
                {"scene": scene, "goal_variant": "counterfactual"},
                {"scene": scene, "goal_variant": "factual"},
            ]
        return [
            ("scene_a/group_0", pair("scene_a")),
            ("scene_a/group_1", pair("scene_a")),
            ("scene_b/group_0", pair("scene_b")),
        ]

    def test_pre_model_eligibility_selects_only_complete_groups(self):
        selected, audit = apply_group_eligibility(
            self._groups(), self._eligibility())
        self.assertEqual([group_id for group_id, _ in selected],
                         ["scene_a/group_0"])
        self.assertEqual(audit["selected_physical_group_count"], 1)
        self.assertEqual(audit["excluded_scene_count"], 1)
        self.assertEqual(audit["excluded_individual_group_count"], 1)

    def test_pre_model_eligibility_fails_closed_on_scope_drift(self):
        with self.assertRaisesRegex(
                DatasetBuildError, "redundant with a scene exclusion"):
            apply_group_eligibility(
                self._groups(), self._eligibility(
                    excluded_group_ids=["scene_b/group_0"],
                    expected_selected_physical_group_count=2))
        with self.assertRaisesRegex(
                DatasetBuildError, "selected physical-group count changed"):
            apply_group_eligibility(
                self._groups(), self._eligibility(
                    expected_selected_physical_group_count=2))

    def test_gauge_and_native_index_are_content_stable(self):
        first = deterministic_gauge_bin("scene/state", salt="unit")
        second = deterministic_gauge_bin("scene/state", salt="unit")
        self.assertEqual(first, second)
        self.assertIn(first, range(NUM_DIRECTIONS))
        yaws = scan_yaws(0.37, first)
        native = native_scan_index(first)
        delta = (float(yaws[native]) - 0.37 + math.pi) % (2 * math.pi) - math.pi
        self.assertAlmostEqual(delta, 0.0, places=12)

    def test_scan_is_one_complete_c8_orbit(self):
        yaws = scan_yaws(-1.2, 3)
        self.assertEqual(yaws.shape, (NUM_DIRECTIONS,))
        steps = np.diff(np.unwrap(np.r_[yaws, yaws[0] + 2 * math.pi]))
        np.testing.assert_allclose(steps, math.pi / 4, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.linalg.norm([
            world_forward_xz(yaw) for yaw in yaws], axis=1), 1.0,
            atol=1e-12, rtol=0)

    def test_teacher_distribution_masks_invalid_directions(self):
        advantages = [0.0, 1.0, 100.0, -1.0, 0.5, 0.2, -0.2, 0.1]
        valid = [True, True, False, True, True, True, True, True]
        distribution = teacher_distribution(advantages, valid)
        self.assertAlmostEqual(float(distribution.sum()), 1.0, places=12)
        self.assertEqual(float(distribution[2]), 0.0)
        self.assertEqual(int(np.argmax(distribution)), 1)

    def test_circular_error_and_scene_folds(self):
        self.assertEqual(circular_bin_error(0, 7), 1)
        self.assertEqual(circular_bin_error(1, 5), 4)
        folds = deterministic_scene_folds(
            [f"scene_{index}" for index in range(40)], folds=5, salt="unit")
        self.assertEqual([len(fold) for fold in folds], [8] * 5)
        self.assertEqual(len({scene for fold in folds for scene in fold}), 40)

    def test_cluster_bootstrap_uses_scenes(self):
        interval = scene_cluster_bootstrap(
            [1.0, 1.0, 3.0, 3.0], ["a", "a", "b", "b"],
            seed=7, resamples=1000)
        self.assertEqual(interval["scene_clusters"], 2)
        self.assertLessEqual(interval["lower_95"], 2.0)
        self.assertGreaterEqual(interval["upper_95"], 2.0)

    def test_episode_fallback_is_unique_and_cannot_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            primary = base / "primary"
            fallback = base / "fallback"
            outside = base / "outside"
            for root in (primary, fallback, outside):
                root.mkdir()
            relative = Path("scene/episode/goal.jpg")
            target = fallback / relative
            target.parent.mkdir(parents=True)
            target.write_bytes(b"goal")
            self.assertEqual(
                resolve_relative_input(
                    str(relative), (primary, fallback), "goal"),
                target,
            )

            duplicate = primary / relative
            duplicate.parent.mkdir(parents=True)
            duplicate.write_bytes(b"goal")
            with self.assertRaises(DatasetBuildError):
                resolve_relative_input(
                    str(relative), (primary, fallback), "goal")

            duplicate.unlink()
            escaped = outside / "escaped.jpg"
            escaped.write_bytes(b"outside")
            (primary / "escape.jpg").symlink_to(escaped)
            with self.assertRaises(DatasetBuildError):
                resolve_relative_input(
                    "escape.jpg", (primary, fallback), "goal")


try:
    import torch
    from MemNavData.circular_goal_compass import (
        CyclicGoalCompass,
        CyclicLinearCompass,
        masked_listwise_loss,
    )
    from MemNavData.train_cgc_scene_oof import risk_coverage
except ImportError:  # pragma: no cover - Habitat-only interpreter
    torch = None


@unittest.skipIf(torch is None, "torch is unavailable")
class CircularGoalCompassTorchTest(unittest.TestCase):
    def _assert_equivariant(self, model):
        torch.manual_seed(3)
        features = torch.randn(4, NUM_DIRECTIONS, 16)
        reference = model(features)
        for shift in range(NUM_DIRECTIONS):
            actual = model(torch.roll(features, shift, dims=1))
            expected = torch.roll(reference, shift, dims=1)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)

    def test_both_models_are_exactly_cyclic(self):
        self._assert_equivariant(CyclicLinearCompass(16).eval())
        self._assert_equivariant(CyclicGoalCompass(16, hidden_dim=8).eval())

    def test_no_absolute_direction_prior(self):
        features = torch.zeros(3, NUM_DIRECTIONS, 16)
        for model in (
                CyclicLinearCompass(16).eval(),
                CyclicGoalCompass(16, hidden_dim=8).eval()):
            scores = model(features)
            torch.testing.assert_close(
                scores, scores[:, :1].expand_as(scores), atol=1e-7, rtol=0)

    def test_masked_listwise_loss_is_finite_and_trains(self):
        model = CyclicGoalCompass(16, hidden_dim=8)
        features = torch.randn(5, NUM_DIRECTIONS, 16)
        advantages = torch.randn(5, NUM_DIRECTIONS)
        valid = torch.ones(5, NUM_DIRECTIONS, dtype=torch.bool)
        valid[:, 3] = False
        logits = model(features)
        loss = masked_listwise_loss(logits, advantages, valid)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        norm = sum(float(parameter.grad.square().sum())
                   for parameter in model.parameters()
                   if parameter.grad is not None)
        self.assertGreater(norm, 0.0)

    def test_risk_budget_never_forces_a_negative_margin_takeover(self):
        rows = 8
        logits = np.zeros((rows, NUM_DIRECTIONS), dtype=np.float64)
        advantages = np.zeros_like(logits)
        valid = np.ones_like(logits, dtype=bool)
        native = np.zeros(rows, dtype=np.int64)
        for index in range(4):
            logits[index, 1] = 2.0 - 0.1 * index
            advantages[index, 1] = 0.5 if index < 2 else -0.5
        logits[4:, 0] = 2.0
        scenes = np.asarray([f"scene_{index // 2}" for index in range(rows)])
        result = risk_coverage(
            logits, advantages, valid, native, scenes,
            np.arange(rows), seed=1)
        self.assertEqual(result["50"]["budget_rows"], 4)
        self.assertEqual(result["50"]["intervened_rows"], 4)
        self.assertEqual(result["100"]["budget_rows"], 8)
        self.assertEqual(result["100"]["intervened_rows"], 4)
        self.assertEqual(result["100"]["actual_coverage"], 0.5)


if __name__ == "__main__":
    unittest.main()
