import unittest

from MemNavData.summarize_certified_graph_rescue_pilot import (
    as_bool,
    causal_payload,
    pilot_gate,
    verify_leg_prefix,
)


class BoolParsingTest(unittest.TestCase):
    def test_csv_and_native_bools(self):
        for value in (True, "True", "true", "1", "1.0"):
            self.assertTrue(as_bool(value))
        for value in (False, "False", "false", "0", "0.0", ""):
            self.assertFalse(as_bool(value))

    def test_rejects_ambiguous_value(self):
        with self.assertRaises(ValueError):
            as_bool("yes")


class CausalPayloadTest(unittest.TestCase):
    def test_drops_only_wall_clock_latency_recursively(self):
        value = {
            "decision": 7,
            "relocalization_ms": 13.2,
            "nested": [{"ranking_ms": 2.0, "score": 0.9}],
            "candidate_path_pairwise_rms_mean": 0.3,
        }
        self.assertEqual(causal_payload(value), {
            "decision": 7,
            "nested": [{"score": 0.9}],
            "candidate_path_pairwise_rms_mean": 0.3,
        })


class PrefixAuditTest(unittest.TestCase):
    @staticmethod
    def payload(extra_plan, graph=False):
        plans = [
            {"step": 0, "decision": "a", "relocalization_ms": 1.0},
            {"step": 8, "decision": "b", "relocalization_ms": 2.0},
        ]
        rollout = [{"step": 0}, {"step": 1}]
        memory = [{"step": 0}, {"step": 1}]
        if extra_plan:
            plans.append({
                "step": 16,
                "decision": "treatment",
                "certified_graph_rescue_requested": graph,
            })
            rollout.append({"step": 2})
            memory.append({"step": 2})
        return {
            "legB": plans,
            "rollout_traces": {"legB": rollout},
            "memory_traces": {"legB": memory},
        }

    def test_graph_and_budget_share_the_exact_direct_prefix(self):
        direct = self.payload(False)
        graph = self.payload(True, graph=True)
        budget = self.payload(True, graph=False)
        self.assertTrue(verify_leg_prefix(
            direct, graph, "B", attempted=True, intervention_step=16,
            expects_graph=True)["plan_prefix_exact"])
        self.assertTrue(verify_leg_prefix(
            direct, budget, "B", attempted=True, intervention_step=16,
            expects_graph=False)["plan_prefix_exact"])

    def test_graph_is_forbidden_in_budget_control(self):
        with self.assertRaisesRegex(RuntimeError, "budget control"):
            verify_leg_prefix(
                self.payload(False), self.payload(True, graph=True), "B",
                attempted=True, intervention_step=16, expects_graph=False)


class FrozenGateTest(unittest.TestCase):
    @staticmethod
    def record(index, *, budget=False, rescue=None, graph_active=1):
        control = index in (0, 1, 3)
        if rescue is None:
            rescue = control
        outcomes = {
            "direct": {"B": control, "joint": control},
            "budget_control": {"B": control or budget,
                               "joint": control},
            "rescue": {"B": bool(rescue), "joint": control,
                       "graph_active_plans_B": graph_active},
        }
        no_attempt = {"attempted": False}
        return {
            "selection_index": index,
            **outcomes,
            "prefix_audits": {
                "budget_control": {"B": no_attempt, "C": no_attempt},
                "rescue": {"B": no_attempt, "C": no_attempt},
            },
        }

    def test_passes_only_when_graph_beats_extra_budget(self):
        records = [self.record(index) for index in (0, 1, 3)]
        records += [self.record(index, rescue=True) for index in (2, 7)]
        records += [self.record(14, rescue=False)]
        self.assertEqual(
            pilot_gate(records)["gate"]["decision"],
            "expand_to_unselected_fresh20",
        )

    def test_stops_when_extra_budget_explains_the_rescues(self):
        records = [self.record(index) for index in (0, 1, 3)]
        records += [
            self.record(index, budget=True, rescue=True)
            for index in (2, 7, 14)
        ]
        self.assertEqual(
            pilot_gate(records)["gate"]["decision"],
            "stop_or_repair_before_expansion",
        )

    def test_stops_when_success_did_not_execute_graph(self):
        records = [self.record(index) for index in (0, 1, 3)]
        records += [
            self.record(index, rescue=True, graph_active=0)
            for index in (2, 7)
        ]
        records += [self.record(14, rescue=False)]
        self.assertEqual(
            pilot_gate(records)["gate"]["decision"],
            "stop_or_repair_before_expansion",
        )


if __name__ == "__main__":
    unittest.main()
