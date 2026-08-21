import unittest
from unittest.mock import patch

import numpy as np

import build_shared_online_role_pairs as builder


class DisconnectedPathfinder:
    pass


class OptionalGeodesicTest(unittest.TestCase):
    def test_rejects_disconnected_candidate(self):
        with patch.object(
            builder,
            "geodesic",
            return_value=(False, float("inf"), []),
        ):
            self.assertIsNone(
                builder.optional_geodesic(
                    DisconnectedPathfinder(), np.zeros(3), np.ones(3)
                )
            )

    def test_accepts_finite_candidate(self):
        with patch.object(
            builder,
            "geodesic",
            return_value=(True, 3.25, []),
        ):
            self.assertEqual(
                builder.optional_geodesic(
                    DisconnectedPathfinder(), np.zeros(3), np.ones(3)
                ),
                3.25,
            )


if __name__ == "__main__":
    unittest.main()
