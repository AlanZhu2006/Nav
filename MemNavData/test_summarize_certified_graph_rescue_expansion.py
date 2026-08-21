import unittest

from MemNavData.summarize_certified_graph_rescue_expansion import (
    FULL_INDICES,
    arm_contrast,
    expansion_gate,
)


class Fresh20ExpansionGateTest(unittest.TestCase):
    @staticmethod
    def records():
        rows = []
        for index in FULL_INDICES:
            direct = index not in (2, 7, 14)
            graph = 1 if not direct else 0
            no_attempt = {"attempted": False}
            rows.append({
                "selection_index": index,
                "scene": f"scene_{index // 2}",
                "direct": {"B": direct, "joint": direct,
                           "graph_active_plans_B": 0,
                           "graph_active_plans_C": 0},
                "budget_control": {"B": direct, "joint": direct,
                                   "graph_active_plans_B": 0,
                                   "graph_active_plans_C": 0},
                "rescue": {"B": True, "joint": True,
                           "graph_active_plans_B": graph,
                           "graph_active_plans_C": 0},
                "prefix_audits": {
                    "budget_control": {"B": no_attempt, "C": no_attempt},
                    "rescue": {"B": no_attempt, "C": no_attempt},
                },
            })
        return rows

    def test_freezes_clean_three_gain_total_effect(self):
        gate = expansion_gate(self.records())
        self.assertEqual(gate["rescue_gain_indices"], [2, 7, 14])
        self.assertEqual(gate["rescue_loss_indices"], [])
        self.assertEqual(gate["decision"], "freeze_internal_fresh20_result")

    def test_stops_on_a_new_loss(self):
        records = self.records()
        records[5]["rescue"]["joint"] = False
        self.assertEqual(
            expansion_gate(records)["decision"],
            "stop_and_audit_expansion",
        )

    def test_stops_on_an_unselected_success_intervention(self):
        records = self.records()
        records[5]["prefix_audits"]["rescue"]["B"] = {
            "attempted": True,
        }
        self.assertEqual(
            expansion_gate(records)["decision"],
            "stop_and_audit_expansion",
        )

    def test_contrast_reports_exact_small_sample_uncertainty(self):
        result = arm_contrast(
            self.records(), "rescue", "direct", "joint",
            seed=7, resamples=1_000)
        self.assertEqual(result["gain"], 3)
        self.assertEqual(result["loss"], 0)
        self.assertEqual(result["risk_difference_pp"], 15.0)
        self.assertEqual(result["exact_mcnemar_p"], 0.25)


if __name__ == "__main__":
    unittest.main()
