"""Property tests for the GLP mechanism core (MemNavData/goal_posterior.py)."""

from __future__ import annotations

import math
import unittest

from MemNavData.goal_posterior import UNMODELED_ID, GoalPosterior


def build_basic() -> GoalPosterior:
    posterior = GoalPosterior(unmodeled_log_weight=0.0)
    posterior.add_node("n0", frame_index=10, log_ratio=1.0)
    posterior.add_node("n1", frame_index=20, log_ratio=0.5)
    posterior.add_frontier("f0", log_area_prior=0.0, log_ratio=0.2)
    return posterior


class TestRegistration(unittest.TestCase):
    def test_empty_posterior_is_all_unmodeled(self) -> None:
        posterior = GoalPosterior(unmodeled_log_weight=0.0)
        self.assertEqual(posterior.posterior(), {UNMODELED_ID: 1.0})
        self.assertEqual(posterior.match_probability(), 0.0)

    def test_duplicate_and_reserved_ids_fail(self) -> None:
        posterior = build_basic()
        with self.assertRaises(ValueError):
            posterior.add_node("n0", frame_index=1, log_ratio=0.0)
        with self.assertRaises(ValueError):
            posterior.add_frontier("n0", log_area_prior=0.0, log_ratio=0.0)
        with self.assertRaises(ValueError):
            posterior.add_node(UNMODELED_ID, frame_index=1, log_ratio=0.0)

    def test_non_finite_evidence_fails(self) -> None:
        posterior = GoalPosterior(unmodeled_log_weight=0.0)
        with self.assertRaises(ValueError):
            posterior.add_node("n0", frame_index=0, log_ratio=float("nan"))
        with self.assertRaises(ValueError):
            posterior.add_frontier(
                "f0", log_area_prior=float("inf"), log_ratio=0.0)

    def test_posterior_normalizes(self) -> None:
        posterior = build_basic()
        values = posterior.posterior()
        self.assertAlmostEqual(sum(values.values()), 1.0, places=12)
        self.assertIn(UNMODELED_ID, values)


class TestCarving(unittest.TestCase):
    def test_carving_is_monotone_and_floored(self) -> None:
        posterior = build_basic()
        before = posterior.posterior()["n0"]
        posterior.carve("n0", detection_power=0.5)
        middle = posterior.posterior()["n0"]
        self.assertLess(middle, before)
        for _ in range(200):
            posterior.carve("n0", detection_power=0.9)
        floored = posterior.posterior()["n0"]
        self.assertGreater(floored, 0.0)
        # survival never drops below the floor relative to its own weight
        posterior.carve("n0", detection_power=0.9)
        self.assertAlmostEqual(posterior.posterior()["n0"], floored, places=12)

    def test_detection_power_bounds(self) -> None:
        posterior = build_basic()
        with self.assertRaises(ValueError):
            posterior.carve("n0", detection_power=1.0)
        with self.assertRaises(ValueError):
            posterior.carve("n0", detection_power=-0.1)
        with self.assertRaises(ValueError):
            posterior.carve("missing", detection_power=0.5)

    def test_carving_shifts_mass_toward_frontiers(self) -> None:
        posterior = build_basic()
        frontier_before = posterior.posterior()["f0"]
        posterior.carve("n0", detection_power=0.9)
        posterior.carve("n1", detection_power=0.9)
        frontier_after = posterior.posterior()["f0"]
        self.assertGreater(frontier_after, frontier_before)


class TestFrontierLineage(unittest.TestCase):
    def test_full_inheritance_conserves_mass(self) -> None:
        posterior = build_basic()
        total_before = posterior.total_unnormalized_weight()
        posterior.retire_frontier("f0", heirs={"f1": 0.7, "f2": 0.3})
        self.assertAlmostEqual(
            posterior.total_unnormalized_weight(), total_before, places=9)

    def test_partial_inheritance_destroys_remainder(self) -> None:
        posterior = build_basic()
        weight_f0 = posterior.total_unnormalized_weight()
        posterior.retire_frontier("f0", heirs={"f1": 0.4})
        destroyed = weight_f0 - posterior.total_unnormalized_weight()
        self.assertGreater(destroyed, 0.0)

    def test_invalid_shares_fail(self) -> None:
        posterior = build_basic()
        with self.assertRaises(ValueError):
            posterior.retire_frontier("f0", heirs={"f1": 0.8, "f2": 0.3})
        with self.assertRaises(ValueError):
            posterior.retire_frontier("f0", heirs={"n0": 0.5})
        with self.assertRaises(ValueError):
            posterior.retire_frontier("n0", heirs={"f1": 1.0})


class TestApproachEvidence(unittest.TestCase):
    def test_accumulates_and_saturates(self) -> None:
        posterior = build_basic()
        before = posterior.posterior()["n0"]
        posterior.add_approach_evidence("n0", log_ratio=1.0)
        self.assertGreater(posterior.posterior()["n0"], before)
        for _ in range(100):
            posterior.add_approach_evidence("n0", log_ratio=1.0)
        capped = posterior.posterior()["n0"]
        posterior.add_approach_evidence("n0", log_ratio=1.0)
        self.assertAlmostEqual(posterior.posterior()["n0"], capped, places=12)

    def test_disconfirming_evidence_reverses(self) -> None:
        posterior = build_basic()
        posterior.add_approach_evidence("n0", log_ratio=2.0)
        high = posterior.posterior()["n0"]
        posterior.add_approach_evidence("n0", log_ratio=-4.0)
        self.assertLess(posterior.posterior()["n0"], high)

    def test_frontier_rejects_approach_evidence(self) -> None:
        posterior = build_basic()
        with self.assertRaises(ValueError):
            posterior.add_approach_evidence("f0", log_ratio=1.0)


class TestGoalSwitch(unittest.TestCase):
    def test_stale_posterior_fails_closed(self) -> None:
        posterior = build_basic()
        posterior.reset_goal()
        with self.assertRaises(RuntimeError):
            posterior.posterior()

    def test_resupply_restores_and_structure_survives(self) -> None:
        posterior = build_basic()
        posterior.add_approach_evidence("n0", log_ratio=2.0)
        posterior.reset_goal()
        for hypothesis_id, ratio in (("n0", 0.0), ("n1", 0.0), ("f0", 0.0)):
            posterior.resupply_evidence(hypothesis_id, ratio)
        values = posterior.posterior()
        # approach evidence must not leak across goals
        self.assertAlmostEqual(values["n0"], values["n1"], places=12)

    def test_double_resupply_fails(self) -> None:
        posterior = build_basic()
        posterior.reset_goal()
        posterior.resupply_evidence("n0", 0.0)
        with self.assertRaises(ValueError):
            posterior.resupply_evidence("n0", 0.0)


class TestRegionsAndSummary(unittest.TestCase):
    def test_cluster_aggregation_beats_isolated_peak(self) -> None:
        """A coherent cluster of moderate evidence must outweigh a single
        isolated higher-scoring frame (the argmax-vs-region property)."""
        posterior = GoalPosterior(unmodeled_log_weight=0.0, cluster_gap=8)
        for frame in (50, 55, 60, 65):
            posterior.add_node(f"c{frame}", frame_index=frame, log_ratio=1.2)
        posterior.add_node("alias", frame_index=200, log_ratio=1.8)
        regions = posterior.node_regions()
        top_ids, top_mass = regions[0]
        self.assertNotIn("alias", top_ids)
        self.assertGreater(top_mass, posterior.posterior()["alias"])
        summary = posterior.summary()
        self.assertIn(summary.best_region_anchor, top_ids)

    def test_summary_fields_consistent(self) -> None:
        posterior = build_basic()
        summary = posterior.summary()
        self.assertAlmostEqual(
            summary.p_match, posterior.match_probability(), places=12)
        self.assertGreaterEqual(summary.entropy, 0.0)
        self.assertEqual(summary.best_frontier, "f0")


if __name__ == "__main__":
    unittest.main()
