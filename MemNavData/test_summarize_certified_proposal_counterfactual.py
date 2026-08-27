#!/usr/bin/env python3

import unittest

from summarize_certified_proposal_counterfactual import summarize


def row(index, *, geometry, top1, ordered, same_anchor=False, rank=1):
    geometry_anchor = 8
    dino_anchor = geometry_anchor if same_anchor else 9
    return {
        "schema_version": (
            "certified_proposal_counterfactual_episode_v1_20260815"),
        "scope": "consumed_posthoc_mechanism_diagnostic",
        "is_closed_loop_evaluation": False,
        "is_method_selection_evidence": True,
        "is_confirmation_evidence": False,
        "query_role_selected_for_analysis": "revisit",
        "scene": f"scene_{index}",
        "episode": "episode_0000",
        "population_index": index,
        "benchmark_root": "/benchmark",
        "dino_top1_anchor": dino_anchor,
        "geometry_selected_anchor": geometry_anchor,
        "geometry_accepted": geometry,
        "counterfactual_dino_top1": {
            "accepted": top1,
            "action_authority": False,
        },
        "counterfactual_dino_first_certified": {
            "accepted": ordered,
            "selected_anchor": dino_anchor if ordered else None,
            "selected_dino_rank": rank if ordered else None,
            "attempt_count": rank,
            "action_authority": False,
        },
        "method_action_unchanged": True,
    }


class CounterfactualSummaryTest(unittest.TestCase):
    def test_paired_acceptance_and_selection(self):
        rows = [
            row(0, geometry=True, top1=True, ordered=True, same_anchor=True),
            row(1, geometry=False, top1=True, ordered=True),
            row(2, geometry=True, top1=False, ordered=True, rank=2),
            row(3, geometry=False, top1=False, ordered=False, rank=2),
        ]
        result = summarize(rows, expected_count=4)
        self.assertEqual(
            result["proposal_acceptance"],
            {
                "deployed_geometry": 2,
                "dino_top1_same_certificate": 2,
                "dino_order_first_certificate": 3,
                "denominator": 4,
            },
        )
        self.assertEqual(
            result["paired_acceptance"]["dino_top1_minus_geometry"][
                "gain_loss"],
            [1, 1],
        )
        self.assertEqual(
            result["paired_acceptance"]["dino_order_minus_geometry"][
                "gain_loss"],
            [1, 0],
        )
        self.assertEqual(
            result["selection"]["accepted_dino_rank_histogram"],
            {"1": 2, "2": 1},
        )
        self.assertEqual(result["selection"]["same_anchor"], 1)


if __name__ == "__main__":
    unittest.main()
