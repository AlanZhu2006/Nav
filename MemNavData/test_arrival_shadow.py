import inspect
import unittest

from MemNavData.arrival_shadow import (
    ArrivalShadowConfig,
    ArrivalShadowDetector,
)


def signal(distance=0.5, *, anchor=7, route_complete=True,
           router_active=True, critic_stop=False):
    return {
        "goal_aux_pose": [distance, 0.0],
        "anchor": anchor,
        "router_selected_anchor": anchor,
        "router_active": router_active,
        "graph_subgoal_enabled": True,
        "graph_subgoal_complete": route_complete,
        "navdp_critic_max": -0.7,
        "navdp_stop_evidence": critic_stop,
    }


class ArrivalShadowTest(unittest.TestCase):
    def test_api_cannot_receive_privileged_goal_coordinates(self):
        parameters = inspect.signature(ArrivalShadowDetector.update).parameters
        self.assertEqual(list(parameters), ["self", "signal", "step"])

    def test_requires_consecutive_stable_completed_plans(self):
        detector = ArrivalShadowDetector(ArrivalShadowConfig(window_plans=3))
        self.assertFalse(detector.update(signal(), step=0)[
            "arrival_shadow_pose_ready"])
        self.assertFalse(detector.update(signal(), step=8)[
            "arrival_shadow_pose_ready"])
        ready = detector.update(signal(), step=16)
        self.assertTrue(ready["arrival_shadow_pose_ready"])
        self.assertFalse(ready["arrival_shadow_strict_ready"])
        self.assertIn("critic_not_stopped", ready["arrival_shadow_reason"])
        self.assertEqual(detector.summary()["arrival_shadow_first_pose_step"], 16)

    def test_critic_is_a_separate_strict_consensus(self):
        detector = ArrivalShadowDetector(ArrivalShadowConfig(window_plans=2))
        detector.update(signal(critic_stop=False), step=0)
        ready = detector.update(signal(critic_stop=True), step=8)
        self.assertTrue(ready["arrival_shadow_pose_ready"])
        self.assertTrue(ready["arrival_shadow_strict_ready"])
        self.assertEqual(detector.summary()["arrival_shadow_first_strict_step"], 8)

    def test_route_may_complete_on_current_plan(self):
        detector = ArrivalShadowDetector(ArrivalShadowConfig(window_plans=3))
        detector.update(signal(route_complete=False), step=0)
        detector.update(signal(route_complete=False), step=8)
        result = detector.update(signal(route_complete=True), step=16)
        self.assertTrue(result["arrival_shadow_pose_ready"])

    def test_anchor_change_and_incomplete_route_fail_closed(self):
        detector = ArrivalShadowDetector(ArrivalShadowConfig(window_plans=3))
        detector.update(signal(anchor=7), step=0)
        detector.update(signal(anchor=8), step=8)
        changed = detector.update(signal(anchor=8), step=16)
        self.assertFalse(changed["arrival_shadow_pose_ready"])
        self.assertIn("anchor_unstable", changed["arrival_shadow_reason"])

        incomplete = ArrivalShadowDetector(ArrivalShadowConfig(window_plans=2))
        incomplete.update(signal(route_complete=False), step=0)
        result = incomplete.update(signal(route_complete=False), step=8)
        self.assertFalse(result["arrival_shadow_pose_ready"])
        self.assertIn("route_incomplete", result["arrival_shadow_reason"])

    def test_noisy_or_missing_goal_pose_fails_closed(self):
        detector = ArrivalShadowDetector(ArrivalShadowConfig(
            window_plans=3, max_distance_mad_m=0.05))
        detector.update(signal(distance=0.2), step=0)
        detector.update(signal(distance=0.7), step=8)
        noisy = detector.update(signal(distance=0.45), step=16)
        self.assertFalse(noisy["arrival_shadow_pose_ready"])
        self.assertIn("goal_distance_unstable_or_far", noisy[
            "arrival_shadow_reason"])

        missing = signal()
        missing["goal_aux_pose"] = None
        detector = ArrivalShadowDetector(ArrivalShadowConfig(window_plans=2))
        detector.update(missing, step=0)
        result = detector.update(missing, step=8)
        self.assertFalse(result["arrival_shadow_pose_ready"])
        self.assertIn("goal_pose_missing", result["arrival_shadow_reason"])

    def test_invalid_thresholds_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 2"):
            ArrivalShadowConfig(window_plans=1)
        with self.assertRaisesRegex(ValueError, "distance_m"):
            ArrivalShadowConfig(distance_m=-1.0)


if __name__ == "__main__":
    unittest.main()
