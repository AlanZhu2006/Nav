import unittest

import numpy as np

from MemNavData.run_local_reverse_sequence_address_audit import (
    causal_monotone_readout,
)


class ReverseSequenceAddressAuditTest(unittest.TestCase):
    def test_causal_readout_tracks_ordered_evidence(self):
        similarity = np.full((4, 12), -1.0, dtype=np.float64)
        for time, state in enumerate((2, 4, 7, 9)):
            similarity[time, state] = 2.0
        self.assertEqual(
            causal_monotone_readout(similarity, maximum_advance=3),
            [2, 4, 7, 9],
        )

    def test_transition_cannot_regress_or_jump_past_bound(self):
        similarity = np.zeros((2, 10), dtype=np.float64)
        similarity[0, 3] = 10.0
        similarity[1, 1] = 20.0
        similarity[1, 9] = 30.0
        self.assertEqual(
            causal_monotone_readout(similarity, maximum_advance=3),
            [3, 3],
        )


if __name__ == "__main__":
    unittest.main()
