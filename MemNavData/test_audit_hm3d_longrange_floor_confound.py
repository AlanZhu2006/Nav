import unittest

from MemNavData.audit_hm3d_longrange_floor_confound import audit_manifest


class LongRangeFloorConfoundAuditTest(unittest.TestCase):
    def test_three_balanced_bins(self):
        episodes = []
        for bin_index, name in enumerate(
                ("0_to_20_m", "20_to_30_m", "30_to_50_m")):
            for local_index in range(16):
                index = 16 * bin_index + local_index
                episodes.append({
                    "population_index": index,
                    "scene": f"scene_{local_index}",
                    "episode": f"episode_{index}",
                    "bin_name": name,
                    "online_a_endpoint": {
                        "floor_position": [0.0, float(bin_index), 0.0],
                    },
                    "pairs": [{"queries": [
                        {
                            "analysis_role": "novel",
                            "floor_position": [1.0, 0.0, 0.0],
                            "geodesic_from_a_end_m": 1.0,
                        },
                        {
                            "analysis_role": "revisit",
                            "floor_position": [1.0, 0.0, 0.0],
                            "geodesic_from_a_end_m": 8.0 + 10 * bin_index,
                        },
                    ]}],
                })
        result = audit_manifest({"episodes": episodes})
        self.assertEqual(result["history_count"], 48)
        self.assertEqual(
            result["by_bin"]["0_to_20_m"]
            ["vertical_at_least_1m_count"], 0)
        self.assertEqual(
            result["by_bin"]["20_to_30_m"]
            ["vertical_at_least_1m_count"], 16)
        self.assertEqual(
            result["by_bin"]["30_to_50_m"]
            ["vertical_at_least_1m_count"], 16)


if __name__ == "__main__":
    unittest.main()
