import unittest

from MemNavData.final14_zero_depth import audit_zero_depth_plans


class Final14ZeroDepthTest(unittest.TestCase):
    def test_explicit_zero_contract(self) -> None:
        result = audit_zero_depth_plans([{
            "navdp_depth_source": "zero",
            "metric_depth_sensor_consumed": False,
            "monocular_depth_receipt": None,
        }])
        self.assertEqual(result["explicit_zero_depth_plan_count"], 1)

    def test_metric_or_mono_payload_is_rejected(self) -> None:
        for row in (
            {"navdp_depth_source": "metric_request",
             "metric_depth_sensor_consumed": True,
             "monocular_depth_receipt": None},
            {"navdp_depth_source": "zero",
             "metric_depth_sensor_consumed": False,
             "monocular_depth_receipt": {"unexpected": True}},
        ):
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                audit_zero_depth_plans([row])


if __name__ == "__main__":
    unittest.main()
