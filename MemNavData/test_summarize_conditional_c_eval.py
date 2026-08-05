import unittest

from MemNavData.summarize_conditional_c_eval import compare, summarize_rows


class ConditionalCSummaryTest(unittest.TestCase):
    @staticmethod
    def row(success):
        return {
            "seed": 11,
            "success": success,
            "spl": 0.5 if success else 0.0,
            "geodesic": 3.0,
            "path": 4.0,
            "steps": 80,
            "final_distance": 0.8 if success else 4.0,
            "prefix_last_frame": 416,
            "prefix_source_frames": 417,
            "memory_prefix_frames": 417,
            "recall_gap": 353,
            "gt_anchor": 63,
            "router_active": success,
        }

    def test_summary_and_pairing(self):
        rows = [self.row(True), self.row(False)]
        summary = summarize_rows(rows)
        self.assertEqual(summary["successes"], 1)
        self.assertEqual(summary["conditional_C_SR"], 0.5)

        key_a = ("a", "episode_0000")
        key_b = ("b", "episode_0000")
        left = {key_a: self.row(False), key_b: self.row(True)}
        right = {key_a: self.row(True), key_b: self.row(True)}
        paired = compare("top1", "topk", left, right, {key_a, key_b})
        self.assertEqual(paired["outcomes"]["right_only"], 1)
        self.assertEqual(paired["conditional_C_SR_delta_right_minus_left"], 0.5)


if __name__ == "__main__":
    unittest.main()
