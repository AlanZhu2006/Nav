import unittest

from build_paper_role_pair_scene import revisit_contract, role_contract
from materialize_paper_online_a_scene import (
    PAPER_ANCHOR_END_MARGIN,
    PAPER_MINIMUM_ELIGIBLE_FRAME,
)


class PaperSingleRevisitContractTest(unittest.TestCase):
    def test_runtime_anchor_boundary_is_consistent(self) -> None:
        contract = revisit_contract()
        self.assertEqual(PAPER_MINIMUM_ELIGIBLE_FRAME, 39)
        self.assertEqual(contract["minimum_eligible_online_frame"], 39)
        self.assertEqual(PAPER_ANCHOR_END_MARGIN, 16)
        self.assertEqual(contract["source_anchor_end_margin_frames"], 16)
        self.assertEqual(contract["source_anchor_stride_frames"], 8)

    def test_single_query_candidate_contract_is_frozen(self) -> None:
        contract = revisit_contract()
        self.assertEqual(contract["minimum_query_geodesic_m"], 2.0)
        self.assertEqual(contract["maximum_query_geodesic_m"], 9.0)
        self.assertEqual(contract["target_query_geodesic_m"], 3.0)
        self.assertEqual(contract["maximum_revisit_candidates"], 4)
        self.assertEqual(contract["v1_min_max_online_a_covis"], 0.50)
        self.assertEqual(contract["v1_max_max_online_a_covis"], 0.98)

    def test_protocols_differ_only_in_bearing_tolerance(self) -> None:
        controlled = role_contract(30.0)
        natural = role_contract(180.0)
        self.assertEqual(controlled["pairs_per_online_history"], 1)
        self.assertEqual(natural["pairs_per_online_history"], 1)
        differing = {
            key for key in controlled if controlled[key] != natural[key]
        }
        self.assertEqual(
            differing, {"maximum_role_initial_path_bearing_error_deg"}
        )


if __name__ == "__main__":
    unittest.main()
