from __future__ import annotations

import unittest

from MemNavData.hm3d_table1_fresh_query_contract import (
    identity_set,
    power,
    stratum_order,
)
from MemNavData.final14_role_pair_contract import (
    STRATA,
    assigned_direction_stratum,
)


def row(scene: str, episode: str, stratum: str) -> dict:
    return {
        "scene": scene,
        "episode": episode,
        "pairs": [{
            "queries": [{
                "analysis_role": "novel",
                "assigned_direction_stratum": stratum,
            }, {
                "analysis_role": "revisit",
            }],
        }],
    }


class Table1FreshQueryReserveTest(unittest.TestCase):
    def test_consumed_identity_ledger_rejects_duplicates(self):
        manifest = {"episodes": [
            {"scene": "s0", "episode": "e0"},
            {"scene": "s0", "episode": "e0"},
        ]}
        with self.assertRaisesRegex(RuntimeError, "duplicates"):
            identity_set(manifest)

    def test_stratum_order_is_preferred_first_and_complete(self):
        order = stratum_order(3, 2, "scene", "episode")
        self.assertEqual(order[0], assigned_direction_stratum(3, 2))
        self.assertEqual(set(order), set(STRATA))
        self.assertEqual(len(order), len(set(order)))
        self.assertEqual(order, stratum_order(3, 2, "scene", "episode"))

    def test_power_gate_requires_histories_scenes_and_direction_coverage(self):
        rows = []
        for index in range(24):
            rows.append(row(
                f"scene_{index % 15}", f"episode_{index}",
                STRATA[index % len(STRATA)],
            ))
        result = power(
            rows, target_histories=24, target_scenes=15,
            minimum_per_stratum=4,
        )
        self.assertTrue(result["target_met"])
        self.assertEqual(result["direction_strata"], {
            "front": 8, "side": 8, "rear": 8,
        })
        result = power(
            [item for item in rows if item["pairs"][0]["queries"][0][
                "assigned_direction_stratum"] != "rear"],
            target_histories=16, target_scenes=10,
            minimum_per_stratum=4,
        )
        self.assertFalse(result["target_met"])


if __name__ == "__main__":
    unittest.main()
