import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from MemNavData.summarize_lingbot_pnp_arrival import atomic_json, summarize
from MemNavData.verify_lingbot_pnp_arrival import sha256_file, verify


class IndependentArrivalVerifierTest(unittest.TestCase):
    def test_verifies_a_sealed_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
                    "predicted_distance_m": 0.1 if positive else 0.3,
                })
            rows = pd.DataFrame(records)
            states = rows[[
                "state_id", "scene", "episode", "euclidean_distance_m"]]
            rows.to_csv(root / "rows.csv", index=False)
            states.to_csv(root / "states.csv", index=False)
            report = summarize(rows, states)
            report["states_sha256"] = sha256_file(root / "states.csv")
            atomic_json(root / "report.json", report)
            atomic_json(root / "SHA256SUMS.json", {
                "rows.csv": sha256_file(root / "rows.csv"),
                "report.json": sha256_file(root / "report.json"),
            })
            (root / "SEALED").touch()
            result = verify(root, root / "states.csv")
            self.assertTrue(result["verified"])
            self.assertTrue(result["primary_gate_passed"])


if __name__ == "__main__":
    unittest.main()
