from __future__ import annotations

import math
import unittest

import pandas as pd

from MemNavData.analyze_revisit_geometry_expert import (
    exact_mcnemar_p,
    hard_gate_metrics,
    paired_top1_comparison,
    session_metrics,
    within_session_concordance,
)
from MemNavData.build_revisit_geometry_expert_table import (
    classify_geometry_state,
    parse_fallback_maps,
    parse_explicit_true,
)


class GeometryStateTest(unittest.TestCase):
    def test_states_keep_missingness_distinct_from_rejection(self) -> None:
        self.assertEqual(
            classify_geometry_state(
                descriptors_available=False,
                matches=0,
                essential_rate=0.0,
                pass_rate=0.0,
            ),
            "insufficient_features",
        )
        self.assertEqual(
            classify_geometry_state(
                descriptors_available=True,
                matches=7,
                essential_rate=0.0,
                pass_rate=0.0,
            ),
            "insufficient_matches",
        )
        self.assertEqual(
            classify_geometry_state(
                descriptors_available=True,
                matches=21,
                essential_rate=1.0,
                pass_rate=0.4,
            ),
            "unstable",
        )
        self.assertEqual(
            classify_geometry_state(
                descriptors_available=True,
                matches=21,
                essential_rate=1.0,
                pass_rate=1.0,
            ),
            "stable_support",
        )
        self.assertEqual(
            classify_geometry_state(
                descriptors_available=True,
                matches=21,
                essential_rate=1.0,
                pass_rate=0.0,
            ),
            "estimable_reject",
        )

    def test_fallback_maps_are_absolute(self) -> None:
        self.assertEqual(
            parse_fallback_maps(["/old/root=/new/root/"]),
            (("/old/root", "/new/root"),),
        )
        with self.assertRaises(ValueError):
            parse_fallback_maps(["relative=/new"])

    def test_causal_boolean_parser_is_fail_closed(self) -> None:
        self.assertTrue(parse_explicit_true("1"))
        self.assertTrue(parse_explicit_true("True"))
        self.assertFalse(parse_explicit_true("yes"))
        self.assertFalse(parse_explicit_true(""))


class PairedMetricsTest(unittest.TestCase):
    @staticmethod
    def frame() -> pd.DataFrame:
        rows = []
        # Two positive sessions and one strict no-match session.  Fusion fixes
        # s1, preserves s2, and never turns an ignored label into a top-1 hit.
        specs = {
            "s1": [(1, 0.7, 0.9, 1), (0, 0.9, 0.8, 1)],
            "s2": [(1, 0.8, 0.9, 1), (0, 0.6, 0.1, 0)],
            "s3": [(0, 0.8, 0.6, 1), (0, 0.7, 0.5, 0)],
        }
        for scene_index, (session, candidates) in enumerate(specs.items()):
            for rank, (label, dino, fusion, hard_pass) in enumerate(candidates, 1):
                rows.append(
                    {
                        "session_id": session,
                        "scene": f"scene{scene_index}",
                        "candidate_rank": rank,
                        "candidate_frame": rank * 10,
                        "label": label,
                        "covisibility": 0.7 if label == 1 else 0.0,
                        "score_dino": dino,
                        "score_fusion": fusion,
                        "geometry_hard_pass": hard_pass,
                    }
                )
        return pd.DataFrame(rows)

    def test_exact_mcnemar_matches_project_reference(self) -> None:
        self.assertTrue(math.isclose(exact_mcnemar_p(12, 0), 0.00048828125))
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)

    def test_within_session_concordance(self) -> None:
        metrics, sessions = within_session_concordance(self.frame(), "score_fusion")
        self.assertEqual(metrics["sessions"], 2)
        self.assertEqual(metrics["pairs"], 2)
        self.assertEqual(metrics["session_macro_auc"], 1.0)
        self.assertEqual(set(sessions), {"s1", "s2"})

    def test_session_pairing_and_hard_gate(self) -> None:
        frame = self.frame()
        _, dino = session_metrics(frame, "score_dino")
        _, fusion = session_metrics(frame, "score_fusion")
        comparison = paired_top1_comparison(
            dino, fusion, bootstrap_samples=200, seed=7
        )
        self.assertEqual(comparison["wins"], 1)
        self.assertEqual(comparison["losses"], 0)
        hard = hard_gate_metrics(frame)
        self.assertEqual(hard["positive_correct_selected"], 2)
        self.assertEqual(hard["strict_no_match_false_activations"], 1)


if __name__ == "__main__":
    unittest.main()
