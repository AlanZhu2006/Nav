import unittest

from MemNavData.summarize_semantic_proposal_gate_b import summarize
from MemNavData.verify_semantic_proposal_gate_b import verify


def row(index, geometry, semantic, *, geometry_runtime_failures=0,
        semantic_runtime_failures=0):
    order = (["geometry_first", "semantic_first"] if index % 2 == 0
             else ["semantic_first", "geometry_first"])
    return {
        "schema_version": "semantic_proposal_gate_b_completion_v2_20260815",
        "scope": "consumed_closed_loop_development_never_confirmation",
        "population_index": index,
        "cohort": "attempt7" if index < 2 else "phase2",
        "scene": f"scene{index}",
        "episode": f"episode_{index:04d}",
        "query_role": "revisit",
        "runtime_role_visibility": "none",
        "prefix_equality": True,
        "arm_order": order,
        "raw_outcomes": {"geometry_first": geometry,
                         "semantic_first": semantic},
        "outcomes": {
            "geometry_first": int(
                bool(geometry) and geometry_runtime_failures == 0),
            "semantic_first": int(
                bool(semantic) and semantic_runtime_failures == 0),
        },
        "runtime_failure_plans": {
            "geometry_first": geometry_runtime_failures,
            "semantic_first": semantic_runtime_failures,
        },
        "proposal_orders": {
            "geometry_first": ["geometry_first"],
            "semantic_first": ["dino_first_certified"],
        },
        "selected_anchors": {
            "geometry_first": [8],
            "semantic_first": [9 if index == 0 else 8],
        },
    }


class SemanticProposalGateBTest(unittest.TestCase):
    def test_summary_and_independent_verifier_agree(self):
        rows = [row(0, 0, 1), row(1, 1, 1), row(2, 1, 0), row(3, 0, 1)]
        summary = summarize(rows, 4)
        self.assertEqual(summary["successes"]["geometry_first"], 2)
        self.assertEqual(summary["successes"]["semantic_first"], 3)
        self.assertEqual(
            summary["paired_semantic_minus_geometry"]["gains"], 2)
        self.assertEqual(
            summary["paired_semantic_minus_geometry"]["losses"], 1)
        self.assertTrue(summary["gate_b_passed"])
        checked = verify(rows, summary, 4)
        self.assertTrue(checked["verified"])
        self.assertTrue(checked["gate_b_passed"])

    def test_runtime_failure_is_retained_and_counted_as_arm_failure(self):
        rows = [
            row(0, 1, 1, semantic_runtime_failures=1),
            row(1, 0, 1),
            row(2, 1, 1),
            row(3, 0, 0),
        ]
        summary = summarize(rows, 4)
        self.assertEqual(
            summary["raw_physical_successes_before_runtime_failure_penalty"]
            ["semantic_first"],
            3,
        )
        self.assertEqual(summary["successes"]["semantic_first"], 2)
        self.assertEqual(summary["execution_audit"]["runtime_failure_plans"]
                         ["semantic_first"], 1)
        checked = verify(rows, summary, 4)
        self.assertTrue(checked["verified"])


if __name__ == "__main__":
    unittest.main()
