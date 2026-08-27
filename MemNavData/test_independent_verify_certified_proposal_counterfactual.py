import unittest

from independent_verify_certified_proposal_counterfactual import verify_records


def row(index, geometry, top1, ordered, geometry_anchor, dino_anchor,
        ordered_rank=1, attempts=1):
    return {
        "schema_version":
            "certified_proposal_counterfactual_episode_v1_20260815",
        "benchmark_root": "/frozen",
        "population_index": index,
        "is_closed_loop_evaluation": False,
        "method_action_unchanged": True,
        "query_role_selected_for_analysis": "revisit",
        "geometry_accepted": geometry,
        "geometry_selected_anchor": geometry_anchor,
        "dino_top1_anchor": dino_anchor,
        "counterfactual_dino_top1": {
            "accepted": top1, "action_authority": False,
        },
        "counterfactual_dino_first_certified": {
            "accepted": ordered,
            "selected_dino_rank": ordered_rank if ordered else None,
            "attempt_count": attempts,
            "action_authority": False,
        },
    }


class IndependentCounterfactualVerifierTest(unittest.TestCase):
    def test_recomputes_without_importing_formal_summarizer(self):
        rows = [
            row(0, True, True, True, 8, 8),
            row(1, False, True, True, 8, 9),
            row(2, True, False, True, 8, 9, ordered_rank=2, attempts=2),
        ]
        summary = {
            "schema_version":
                "certified_proposal_counterfactual_summary_v1_20260815",
            "proposal_acceptance": {
                "deployed_geometry": 2,
                "dino_top1_same_certificate": 2,
                "dino_order_first_certificate": 3,
                "denominator": 3,
            },
            "paired_acceptance": {
                "dino_top1_minus_geometry": {
                    "gain_loss": [1, 1], "exact_mcnemar_p": 1.0,
                },
                "dino_order_minus_geometry": {
                    "gain_loss": [1, 0], "exact_mcnemar_p": 1.0,
                },
            },
            "selection": {
                "geometry_changed_dino_top1": 2,
                "same_anchor": 1,
                "accepted_dino_rank_histogram": {"1": 2, "2": 1},
                "ordered_attempts_mean": 4 / 3,
                "ordered_attempts_max": 2,
            },
        }
        recomputed = verify_records(rows, summary, 3)
        self.assertEqual(
            recomputed["proposal_acceptance"]["dino_order_first_certificate"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
