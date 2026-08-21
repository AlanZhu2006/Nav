import unittest

from MemNavData.goat_certified_arrival_contract import (
    ARRIVAL_DISTANCE_THRESHOLD_M,
    ArrivalEvidence,
    decide_subtask_stop,
)


def evidence(**overrides):
    values = {
        "native_zero_proposal": True,
        "stream_frame_count": 64,
        "certificate_accepted": True,
        "predicted_distance_m": ARRIVAL_DISTANCE_THRESHOLD_M,
        "metric_scale_available": True,
    }
    values.update(overrides)
    return ArrivalEvidence(**values)


class GoatCertifiedArrivalContractTest(unittest.TestCase):
    def test_frozen_boundary_is_inclusive(self):
        result = decide_subtask_stop(evidence())
        self.assertTrue(result["authorized_subtask_stop"])
        self.assertEqual(result["reason"], "certified_arrival")

    def test_distance_above_boundary_abstains(self):
        result = decide_subtask_stop(evidence(predicted_distance_m=0.075001))
        self.assertFalse(result["authorized_subtask_stop"])
        self.assertEqual(
            result["reason"], "predicted_distance_above_frozen_threshold")

    def test_incomplete_scale_prefix_abstains(self):
        result = decide_subtask_stop(evidence(stream_frame_count=63))
        self.assertFalse(result["authorized_subtask_stop"])
        self.assertEqual(result["reason"], "causal_scale_prefix_incomplete")

    def test_geometry_rejection_abstains(self):
        result = decide_subtask_stop(evidence(
            certificate_accepted=False,
            predicted_distance_m=None,
            metric_scale_available=False,
        ))
        self.assertFalse(result["authorized_subtask_stop"])
        self.assertEqual(result["reason"], "geometry_certificate_rejected")

    def test_no_native_trigger_cannot_stop(self):
        result = decide_subtask_stop(evidence(native_zero_proposal=False))
        self.assertFalse(result["authorized_subtask_stop"])
        self.assertEqual(result["reason"], "native_zero_trigger_absent")

    def test_nonfinite_distance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            decide_subtask_stop(evidence(predicted_distance_m=float("nan")))


if __name__ == "__main__":
    unittest.main()
