import unittest

from MemNavData.summarize_goat_certified_arrival import summarize
from MemNavData.verify_goat_certified_arrival import verify


class GoatCertifiedArrivalSummaryTest(unittest.TestCase):
    def test_summary_and_independent_verifier_agree(self):
        contract = {"frozen": True}
        manifest = {
            "episodes": [
                {"scene_id": "a", "episode_id": "1"},
                {"scene_id": "b", "episode_id": "2"},
            ],
            "arrival_contract": contract,
            "primary_confirmation_gate": {
                "maximum_false_certified_stops": 0,
                "minimum_true_certified_stops": 1,
                "minimum_true_stop_scenes": 1,
            },
        }
        tasks = []
        for index, (scene, success, legacy) in enumerate([
                ("a", True, False), ("b", False, False)]):
            decision = ({
                "authorized_subtask_stop": True,
                "post_decision_official_distance_m": 0.1,
                "reason": "certified_arrival",
            } if success else None)
            tasks.append({
                "complete": True,
                "episode_index": index,
                "ground_truth_used_by_decision": False,
                "arrival_contract": contract,
                "record": {
                    "status": "complete",
                    "scene_id": scene,
                    "episode_id": str(index + 1),
                    "first_task": ["x", "image"],
                    "certified_success": success,
                    "certified_stop": success,
                    "legacy_first_zero_success_counterfactual": legacy,
                    "safe_stall": not success,
                    "forced_guard_stop": not success,
                    "same_batch_fallback_count": 0,
                    "extra_resample_count": 0,
                    "plans": [{"plan_index": 0, "stop_decision": decision}],
                },
            })
        report = summarize(manifest, tasks)
        self.assertTrue(report["primary_gate_passed"])
        self.assertEqual(report["certified"]["false_stops"], 0)
        checked = verify(manifest, tasks, report)
        self.assertTrue(checked["verified"])


if __name__ == "__main__":
    unittest.main()
