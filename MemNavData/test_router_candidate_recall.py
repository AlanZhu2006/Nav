import unittest

import pandas as pd

from MemNavData.audit_router_candidate_recall import (
    parse_int_list,
    select_raw_topk,
    select_temporal_nms,
    selector_summary,
)
from MemNavData.relabel_router_covisibility import selected_indices


def frame(rows):
    return pd.DataFrame(rows, columns=(
        "session_id", "scene", "kind", "candidate_path",
        "candidate_frame", "dino_cosine", "teacher_covis"))


class CandidateRecallTest(unittest.TestCase):
    def test_deterministic_raw_tie_break(self):
        data = frame([
            ("s", "x", "k", "b", 8, 0.9, 0.0),
            ("s", "x", "k", "a", 4, 0.9, 1.0),
            ("s", "x", "k", "c", 2, 0.8, 0.0),
        ])
        selected = select_raw_topk(data, 2)
        self.assertEqual(selected["candidate_frame"].tolist(), [4, 8])

    def test_temporal_nms_recovers_nonredundant_positive(self):
        data = frame([
            ("s", "x", "k", "f0", 0, 0.99, 0.0),
            ("s", "x", "k", "f1", 1, 0.98, 0.0),
            ("s", "x", "k", "f2", 2, 0.97, 0.0),
            ("s", "x", "k", "f20", 20, 0.90, 0.8),
        ])
        raw = select_raw_topk(data, 2)
        diverse = select_temporal_nms(data, 2, 4)
        self.assertFalse(raw["teacher_covis"].ge(0.5).any())
        self.assertTrue(diverse["teacher_covis"].ge(0.5).any())
        self.assertEqual(diverse["candidate_frame"].tolist(), [0, 20])

    def test_summary_excludes_novel_from_recall_denominator(self):
        data = frame([
            ("positive", "x", "k", "a", 0, 0.9, 0.8),
            ("positive", "x", "k", "b", 8, 0.8, 0.0),
            ("novel", "x", "k", "c", 0, 0.9, 0.0),
            ("novel", "x", "k", "d", 8, 0.8, 0.0),
        ])
        result = selector_summary(
            data, lambda group: select_raw_topk(group, 1), 0.5)
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["oracle_positive_sessions"], 1)
        self.assertEqual(result["no_positive_in_candidate_pool"], 1)
        self.assertEqual(result["conditional_candidate_recall"], 1.0)

    def test_integer_list_validation(self):
        self.assertEqual(parse_int_list(["8", "4", "8"], minimum=1), (4, 8))
        with self.assertRaises(ValueError):
            parse_int_list(["0"], minimum=1)

    def test_relabel_top_zero_selects_complete_pool(self):
        data = frame([
            ("a", "x", "k", "a0", 0, 0.8, 0.0),
            ("a", "x", "k", "a1", 1, 0.9, 0.0),
            ("b", "x", "k", "b0", 0, 0.7, 0.0),
        ])
        self.assertEqual(len(selected_indices(data, 0)), 3)
        self.assertEqual(len(selected_indices(data, 1)), 2)


if __name__ == "__main__":
    unittest.main()
