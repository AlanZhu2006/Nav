import tempfile
import unittest
from pathlib import Path

import pandas as pd

from MemNavData.summarize_lingbot_pnp_arrival import summarize


class ArrivalSummaryTest(unittest.TestCase):
    @staticmethod
    def rows() -> pd.DataFrame:
        records = []
        for index in range(24):
            positive = index < 20
            records.append({
                "state_id": f"s{index}",
                "scene": f"scene{index % 10}",
                "episode": "episode_0000",
                "euclidean_distance_m": 0.1 if positive else 0.4,
                "arrival_025_strict": positive,
                "native_selected_zero_sample0": True,
                "precheck_passed": True,
                "certificate_accepted": True,
                "predicted_distance_m": 0.10 if positive else 0.30,
            })
        return pd.DataFrame(records)

    def test_frozen_gate_can_pass_without_false_positives(self):
        rows = self.rows()
        report = summarize(rows, rows.copy())
        self.assertTrue(report["primary_gate_passed"])
        winner = report["selected_train_operating_point"]
        self.assertEqual(winner["native_zero_plus_pnp"]["tp"], 20)
        self.assertEqual(winner["native_zero_plus_pnp"]["fp"], 0)
        self.assertEqual(winner["predicted_distance_max_m"], 0.1)

    def test_native_trigger_is_part_of_primary_rule(self):
        rows = self.rows()
        rows.loc[20:, "native_selected_zero_sample0"] = False
        rows.loc[20:, "predicted_distance_m"] = 0.01
        report = summarize(rows, rows.copy())
        point = next(item for item in report["operating_points"]
                     if item["predicted_distance_max_m"] == 0.1)
        self.assertEqual(point["pnp_only"]["fp"], 4)
        self.assertEqual(point["native_zero_plus_pnp"]["fp"], 0)

    def test_exact_state_cover_is_required(self):
        rows = self.rows()
        with self.assertRaisesRegex(RuntimeError, "exactly cover"):
            summarize(rows.iloc[:-1].copy(), rows.copy())


if __name__ == "__main__":
    unittest.main()
