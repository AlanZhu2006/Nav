import unittest

from MemNavData.multigoal_policy_contract import three_leg_policy_backends


AUTOMATIC_ROUTES = {
    "memory_advantage",
    "memory_geometry",
    "learned_rank_geometry",
    "certified_relocalization",
}


class MultigoalPolicyContractTest(unittest.TestCase):
    def test_native_runtime_keeps_all_three_legs_native(self):
        self.assertEqual(
            three_leg_policy_backends(
                server_backend="navdp",
                hybrid_route="phase",
                automatic_routes=AUTOMATIC_ROUTES,
            ),
            (None, None, None),
        )

    def test_known_roles_switch_only_revisit_c_to_memory_residual(self):
        self.assertEqual(
            three_leg_policy_backends(
                server_backend="hybrid_pose",
                hybrid_route="phase",
                automatic_routes=AUTOMATIC_ROUTES,
            ),
            ("navdp", "navdp", "navdp_mix"),
        )

    def test_known_roles_switch_b_and_c_for_double_revisit(self):
        self.assertEqual(
            three_leg_policy_backends(
                server_backend="hybrid_pose",
                hybrid_route="phase",
                automatic_routes=AUTOMATIC_ROUTES,
                role_sequence=("initial_imagegoal", "revisit", "revisit"),
            ),
            ("navdp", "navdp_mix", "navdp_mix"),
        )

    def test_double_revisit_can_disable_only_c_for_causal_ablation(self):
        self.assertEqual(
            three_leg_policy_backends(
                server_backend="hybrid_pose",
                hybrid_route="phase",
                automatic_routes=AUTOMATIC_ROUTES,
                role_sequence=("initial_imagegoal", "revisit", "revisit"),
                known_revisit_leg_indices={1},
            ),
            ("navdp", "navdp_mix", "navdp"),
        )

    def test_known_revisit_selection_rejects_non_revisit_leg(self):
        with self.assertRaisesRegex(ValueError, "non-Revisit"):
            three_leg_policy_backends(
                server_backend="hybrid_pose",
                hybrid_route="phase",
                automatic_routes=AUTOMATIC_ROUTES,
                role_sequence=("initial_imagegoal", "revisit", "revisit"),
                known_revisit_leg_indices={0, 1},
            )

    def test_known_revisit_selection_cannot_modify_automatic_route(self):
        with self.assertRaisesRegex(ValueError, "automatic route"):
            three_leg_policy_backends(
                server_backend="hybrid_pose",
                hybrid_route="certified_relocalization",
                automatic_routes=AUTOMATIC_ROUTES,
                role_sequence=("initial_imagegoal", "revisit", "revisit"),
                known_revisit_leg_indices={1},
            )

    def test_role_free_certificate_must_decide_on_every_leg(self):
        self.assertEqual(
            three_leg_policy_backends(
                server_backend="hybrid_pose",
                hybrid_route="certified_relocalization",
                automatic_routes=AUTOMATIC_ROUTES,
            ),
            ("navdp_auto", "navdp_auto", "navdp_auto"),
        )

    def test_native_runtime_cannot_masquerade_as_automatic_method(self):
        with self.assertRaisesRegex(ValueError, "automatic route"):
            three_leg_policy_backends(
                server_backend="navdp",
                hybrid_route="certified_relocalization",
                automatic_routes=AUTOMATIC_ROUTES,
            )

    def test_unknown_hybrid_route_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            three_leg_policy_backends(
                server_backend="hybrid_pose",
                hybrid_route="not_a_route",
                automatic_routes=AUTOMATIC_ROUTES,
            )

    def test_unknown_role_sequence_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "role sequence"):
            three_leg_policy_backends(
                server_backend="hybrid_pose",
                hybrid_route="phase",
                automatic_routes=AUTOMATIC_ROUTES,
                role_sequence=("initial_imagegoal", "revisit", "mystery"),
            )


if __name__ == "__main__":
    unittest.main()
