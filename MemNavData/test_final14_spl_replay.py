import copy
import types
import unittest

from MemNavData.final14_spl_replay import (
    ARMS, exact_p, independent_measurement, install_profile, measurement, rotated_arm_order,
)


class ReplayTests(unittest.TestCase):
    def leg(self):
        return {"rollout_trace": [{"step": 0, "x": 0.0, "z": 0.0},
                                  {"step": 1, "x": 1.0, "z": 0.0}],
                "steps": 2, "end_pos": [1.5, 0.0, 0.0], "end_psi": 0.3,
                "reached": True, "final_goal_dist_m": 0.5, "path_len": 9.0}

    def fixture(self):
        leg = self.leg()
        receipt = measurement(leg, [2.0, 0.0], 1.0)
        row = {"steps": 2, "reached": 1, "final_goal_dist_m": 0.5,
               "geodesic_m": 1.0, "path_len_m": 1.5, "spl": 2 / 3,
               "end_x_m": 1.5, "end_y_m": 0.0, "end_z_m": 0.0}
        payload = {"query_result": receipt, "rollout_traces": {"query": leg["rollout_trace"]}}
        return row, payload

    def test_measurement_does_not_mutate_control_result(self):
        leg = self.leg()
        before = copy.deepcopy(leg)
        result = measurement(leg, [2, 0], 1)
        self.assertEqual(leg, before)
        self.assertEqual(result["actual_path_len_m"], 1.5)
        self.assertEqual(result["commanded_path_len_m"], 9.0)
        self.assertAlmostEqual(result["spl"], 2 / 3)

    def test_independent_recount_includes_last_action(self):
        row, payload = self.fixture()
        self.assertAlmostEqual(independent_measurement(row, payload), 2 / 3)

    def test_missing_terminal_rejected(self):
        row, payload = self.fixture()
        del payload["query_result"]["end_position"]
        with self.assertRaises(KeyError):
            independent_measurement(row, payload)

    def test_commanded_distance_cannot_masquerade_as_actual(self):
        row, payload = self.fixture()
        row["path_len_m"] = 9.0
        with self.assertRaisesRegex(RuntimeError, "executed path"):
            independent_measurement(row, payload)

    def test_changed_success_rejected(self):
        row, payload = self.fixture()
        row["reached"] = 0
        with self.assertRaisesRegex(RuntimeError, "success mismatch"):
            independent_measurement(row, payload)

    def test_incomplete_trace_rejected(self):
        row, payload = self.fixture()
        payload["rollout_traces"]["query"] = payload["rollout_traces"]["query"][1:]
        with self.assertRaises(RuntimeError):
            independent_measurement(row, payload)

    def test_five_arm_rotation_and_zero_contract(self):
        from MemNavData import final14_mono_factorial as old
        runner = types.SimpleNamespace(**{name: getattr(old, name) for name in (
            "DEPTH_SOURCE", "HYBRID_ROUTE", "REVISIT_ADAPTER", "EVALUATOR_ARM", "audit_depth_plans")})
        install_profile(runner)
        self.assertEqual(set(runner.DEPTH_SOURCE), set(ARMS))
        self.assertEqual(runner.DEPTH_SOURCE["zero_native"], "zero")
        for i in range(21):
            self.assertEqual(set(rotated_arm_order(i)), set(ARMS))
        self.assertEqual(rotated_arm_order(17)[0], "mono_cec")
        runner.audit_depth_plans("zero_native", [{"navdp_depth_source": "zero",
                                                  "metric_depth_sensor_consumed": False,
                                                  "monocular_depth_receipt": None}])

    def test_known_exact_mcnemar(self):
        self.assertEqual(exact_p(0, 0), 1)
        self.assertEqual(exact_p(12, 0), 0.00048828125)


if __name__ == "__main__":
    unittest.main()
