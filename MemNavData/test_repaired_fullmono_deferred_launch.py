import copy
import unittest

from launch_repaired_fullmono_a_after_probe import EXPECTED_SCENES, validate_gate


class DeferredLaunchTests(unittest.TestCase):
    def fixture(self):
        summary = dict(completed=True, navigation_rollouts=0, query_outcomes_read=False,
                       scene_count=2, pair_count=2)
        payloads = []
        for scene in EXPECTED_SCENES:
            payloads.append(dict(scene=scene, episode="episode_0000", pair_constructible=True,
                navigation_rollouts=0, query_outcomes_read=False, all_historical_rgb_rerender_hashes_match=True,
                selected_novel_direction="rear", novel_by_direction={"rear": dict(constructible=True, query_id="novel_rear")},
                queries=[dict(query_id="revisit", analysis_role="revisit", max_online_a_covis=.7, geodesic_from_a_end_m=3.),
                         dict(query_id="novel_rear", analysis_role="novel", max_online_a_covis=.08, geodesic_from_a_end_m=4.)]))
        return summary, payloads

    def test_complete_construction_only_gate(self):
        validate_gate(*self.fixture())

    def test_partial_or_failed_gate_cannot_launch(self):
        for changes in ({"completed": False}, {"pair_count": 1}, {"query_outcomes_read": True},
                        {"navigation_rollouts": 1}):
            s, p = self.fixture()
            s.update(changes)
            with self.assertRaises(ValueError):
                validate_gate(s, p)

    def test_cannot_relax_support_or_distance(self):
        _, original = self.fixture()
        for role, field, value in ((1, "max_online_a_covis", .10), (0, "max_online_a_covis", .5),
                                   (0, "geodesic_from_a_end_m", 1.9)):
            s, _ = self.fixture()
            p = copy.deepcopy(original)
            p[0]["queries"][role][field] = value
            with self.assertRaises(ValueError):
                validate_gate(s, p)

    def test_wrong_source_or_missing_rgb_binding_rejected(self):
        for field, value in (("episode", "episode_0001"), ("all_historical_rgb_rerender_hashes_match", False)):
            s, p = self.fixture()
            p[0][field] = value
            with self.assertRaises(ValueError):
                validate_gate(s, p)


if __name__ == "__main__":
    unittest.main()
