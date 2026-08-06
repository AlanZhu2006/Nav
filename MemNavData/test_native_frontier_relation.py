"""Tests for goal-conditioned native/frontier deployment features."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from MemNavData.native_frontier_relation import (
    NATIVE_RELATION_FEATURE_NAMES,
    NativeFrontierRelationError,
    native_conditioned_union_shortlist,
    native_frontier_relation,
)


def trajectories() -> np.ndarray:
    return np.asarray([
        [[0.5, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.5], [0.0, 1.0], [0.0, 2.0]],
        [[0.4, 0.4], [0.8, 0.8], [1.5, 1.5]],
    ], dtype=np.float64)


def candidate(candidate_id: str, forward: float, left: float,
              topology: float, patch: float | None = None) -> dict:
    return {
        "candidate_id": candidate_id,
        "subgoal_forward_m": forward,
        "subgoal_left_m": left,
        "topology_score": topology,
        "goal_patch_relation_present": patch is not None,
        "goal_patch_relation_score": float(patch or 0.0),
    }


class NativeFrontierRelationTest(unittest.TestCase):
    def test_exact_forward_axis_and_selected_plan(self):
        feature = native_frontier_relation(
            trajectories(), [2.0, 0.0], selected_index=0,
            native_values=[3.0, 1.0, 2.0])
        self.assertEqual(feature.shape, (len(NATIVE_RELATION_FEATURE_NAMES),))
        self.assertTrue(np.isfinite(feature).all())
        by_name = dict(zip(NATIVE_RELATION_FEATURE_NAMES, feature))
        self.assertAlmostEqual(by_name["selected_endpoint_distance_m"], 0.0)
        self.assertAlmostEqual(by_name["selected_path_distance_m"], 0.0)
        self.assertAlmostEqual(by_name["selected_direction_cosine"], 1.0)
        self.assertAlmostEqual(by_name["best_endpoint_distance_m"], 0.0)
        self.assertEqual(by_name["native_value_present"], 1.0)
        self.assertGreater(by_name["selected_value_zscore"], 0.0)

    def test_left_axis_and_missing_values_have_explicit_mask(self):
        feature = native_frontier_relation(
            trajectories(), [0.0, 2.0], selected_index=1)
        by_name = dict(zip(NATIVE_RELATION_FEATURE_NAMES, feature))
        self.assertAlmostEqual(by_name["selected_direction_cosine"], 1.0)
        self.assertEqual(by_name["native_value_present"], 0.0)
        for name in (
            "value_weighted_endpoint_distance_m",
            "value_weighted_direction_cosine",
            "native_value_std",
            "selected_value_zscore",
        ):
            self.assertEqual(by_name[name], 0.0)

    def test_goal_swap_changes_native_relation_without_gt(self):
        first = trajectories()
        swapped = first.copy()
        swapped[..., 0], swapped[..., 1] = (
            first[..., 1].copy(), -first[..., 0].copy())
        candidate_local = [2.0, 0.0]
        before = native_frontier_relation(
            first, candidate_local, selected_index=0)
        after = native_frontier_relation(
            swapped, candidate_local, selected_index=0)
        self.assertFalse(np.array_equal(before, after))
        self.assertGreater(before[2], after[2])

    def test_zero_motion_is_finite_and_marked_not_moving(self):
        stopped = np.zeros((2, 3, 2), dtype=np.float64)
        feature = native_frontier_relation(
            stopped, [1.0, 0.0], selected_index=0)
        by_name = dict(zip(NATIVE_RELATION_FEATURE_NAMES, feature))
        self.assertEqual(by_name["native_moving_fraction"], 0.0)
        self.assertEqual(by_name["native_heading_resultant"], 0.0)
        self.assertEqual(by_name["best_direction_cosine"], 0.0)

    def test_bad_shapes_nonfinite_or_indices_fail_closed(self):
        bad = (
            (np.zeros((3, 2)), [1.0, 0.0], 0, None),
            (np.full((1, 2, 2), np.nan), [1.0, 0.0], 0, None),
            (trajectories(), [0.0, 0.0], 0, None),
            (trajectories(), [1.0, 0.0], 3, None),
            (trajectories(), [1.0, 0.0], 0, [1.0]),
        )
        for native, local, selected, values in bad:
            with self.subTest(selected=selected):
                with self.assertRaises(NativeFrontierRelationError):
                    native_frontier_relation(
                        native, local, selected_index=selected,
                        native_values=values)

    def test_union_shortlist_is_deterministic_and_source_balanced(self):
        rows = [
            candidate("forward", 2.0, 0.0, 1.0),
            candidate("left", 0.0, 2.0, 2.0),
            candidate("diagonal", 1.5, 1.5, 3.0, patch=0.8),
            candidate("back", -2.0, 0.0, 10.0),
            candidate("right", 0.0, -2.0, 4.0, patch=0.9),
        ]
        original = copy.deepcopy(rows)
        first = native_conditioned_union_shortlist(
            rows, trajectories(), selected_index=0,
            max_candidates=4, native_slots=1, patch_slots=1,
            topology_slots=1)
        second = native_conditioned_union_shortlist(
            list(reversed(rows)), trajectories(), selected_index=0,
            max_candidates=4, native_slots=1, patch_slots=1,
            topology_slots=1)
        self.assertEqual(
            [row["candidate_id"] for row in first],
            [row["candidate_id"] for row in second],
        )
        self.assertEqual(rows, original)
        self.assertEqual(first[0]["candidate_id"], "forward")
        self.assertIn("native_aligned", first[0][
            "native_relation_selection_sources"])
        self.assertIn("right", [row["candidate_id"] for row in first])
        self.assertIn("back", [row["candidate_id"] for row in first])
        self.assertTrue(all(len(row["native_proposal_relation"])
                            == len(NATIVE_RELATION_FEATURE_NAMES)
                            for row in first))

    def test_duplicate_ids_or_malformed_candidate_fail(self):
        duplicate = [
            candidate("same", 1.0, 0.0, 1.0),
            candidate("same", 0.0, 1.0, 2.0),
        ]
        with self.assertRaisesRegex(NativeFrontierRelationError, "unique"):
            native_conditioned_union_shortlist(
                duplicate, trajectories(), selected_index=0)
        malformed = [candidate("x", 1.0, 0.0, 1.0)]
        malformed[0]["topology_score"] = float("nan")
        with self.assertRaises(NativeFrontierRelationError):
            native_conditioned_union_shortlist(
                malformed, trajectories(), selected_index=0,
                topology_slots=1)


if __name__ == "__main__":
    unittest.main()
