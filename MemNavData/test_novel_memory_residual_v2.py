import math
import unittest

from MemNavData.novel_memory_residual_v2 import (
    ResidualCandidate,
    ResidualContext,
    ResidualFeedback,
    ResidualSource,
    ResidualThresholds,
    SelectiveResidualController,
)


def context(
    *,
    session="session-a",
    epoch="goal-a",
    plan=0,
    revision=None,
    match_lower=0.01,
    match_upper=0.05,
    coverage=0.95,
    stagnation=3,
    fresh=True,
    calibrated=True,
    feedback=None,
):
    if revision is None:
        revision = plan
    return ResidualContext(
        session_id=session,
        goal_epoch=epoch,
        plan_index=plan,
        graph_revision=revision,
        match_probability_lower=match_lower,
        match_probability_upper=match_upper,
        candidate_coverage_probability_lower=coverage,
        native_stagnation_plans=stagnation,
        graph_fresh=fresh,
        calibration_supported=calibrated,
        previous_residual_feedback=feedback,
    )


def candidate(
    candidate_id="f1",
    source=ResidualSource.FRONTIER,
    mean=0.60,
    std=0.10,
    harm=0.02,
    pose_t=0.20,
    pose_yaw=10.0,
    clearance=0.40,
    feasible=True,
):
    return ResidualCandidate(
        candidate_id=candidate_id,
        source=source,
        advantage_mean_m=mean,
        advantage_std_m=std,
        harm_probability_upper=harm,
        pose_translation_p90_m=pose_t,
        pose_yaw_p90_deg=pose_yaw,
        clearance_lower_m=clearance,
        route_feasible=feasible,
    )


def feedback(
    candidate_id="f1",
    source=ResidualSource.FRONTIER,
    *,
    executed_plan=0,
    executed_revision=None,
    improved=True,
):
    if executed_revision is None:
        executed_revision = executed_plan
    return ResidualFeedback(
        source=source,
        candidate_id=candidate_id,
        executed_plan_index=executed_plan,
        executed_graph_revision=executed_revision,
        graph_displacement_improved=improved,
        native_novelty_improved=False,
        goal_evidence_improved=False,
    )


class SelectiveResidualControllerTest(unittest.TestCase):
    def test_thresholds_require_a_match_defer_gap(self):
        with self.assertRaisesRegex(ValueError, "defer gap"):
            ResidualThresholds(
                match_ucb_for_frontier=0.50,
                match_lcb_for_memory=0.50)

    def test_plan_count_thresholds_require_strict_integers(self):
        for invalid in (True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integers"):
                    ResidualThresholds(confirmation_plans=invalid)

    def test_empty_or_ambiguous_evidence_uses_native(self):
        controller = SelectiveResidualController()
        self.assertTrue(controller.decide(context(), []).uses_native)
        decision = controller.decide(
            context(plan=1, match_lower=0.20, match_upper=0.80),
            [candidate()])
        self.assertTrue(decision.uses_native)
        self.assertEqual(decision.reason, "no_eligible_residual")

    def test_frontier_requires_stagnation_and_temporal_confirmation(self):
        controller = SelectiveResidualController()
        first = controller.decide(context(stagnation=2), [candidate()])
        self.assertTrue(first.uses_native)
        second = controller.decide(
            context(plan=1, stagnation=3), [candidate()])
        self.assertTrue(second.uses_native)
        self.assertEqual(second.reason, "confirming_residual")
        third = controller.decide(
            context(plan=2, stagnation=3), [candidate()])
        self.assertEqual(third.source, "frontier")
        self.assertEqual(third.candidate_id, "f1")

    def test_memory_graph_requires_match_but_not_frontier_stagnation(self):
        controller = SelectiveResidualController()
        memory = candidate(
            candidate_id="node-17", source=ResidualSource.MEMORY_GRAPH)
        low_match = context(
            match_lower=0.80, match_upper=0.95, stagnation=0)
        self.assertTrue(controller.decide(low_match, [memory]).uses_native)
        high_match = context(
            plan=1, match_lower=0.92, match_upper=0.97, stagnation=0)
        self.assertTrue(controller.decide(high_match, [memory]).uses_native)
        activated = controller.decide(context(
            plan=2, match_lower=0.92, match_upper=0.97, stagnation=0),
            [memory])
        self.assertEqual(activated.source, "memory_graph")

    def test_selector_uses_lower_confidence_bound_not_raw_mean(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        uncertain = candidate(
            candidate_id="high-mean", mean=1.20, std=0.50)
        reliable = candidate(
            candidate_id="high-lcb", mean=0.65, std=0.05)
        decision = controller.decide(context(), [uncertain, reliable])
        self.assertEqual(decision.candidate_id, "high-lcb")

    def test_missing_nonfinite_or_unsafe_fields_fail_closed(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        bad = [
            candidate(candidate_id="nan", std=math.nan),
            candidate(candidate_id="harm", harm=0.20),
            candidate(candidate_id="pose", pose_t=0.31),
            candidate(candidate_id="yaw", pose_yaw=15.1),
            candidate(candidate_id="clear", clearance=0.29),
            candidate(candidate_id="blocked", feasible=False),
        ]
        decision = controller.decide(context(), bad)
        self.assertTrue(decision.uses_native)
        self.assertEqual(decision.reason, "malformed_candidate_set")

    def test_goal_switch_discards_confirmation(self):
        controller = SelectiveResidualController()
        self.assertTrue(controller.decide(context(epoch="a"), [candidate()]).uses_native)
        switched = controller.decide(context(epoch="b"), [candidate()])
        self.assertTrue(switched.uses_native)
        self.assertEqual(switched.reason, "confirming_residual")
        activated = controller.decide(
            context(epoch="b", plan=1), [candidate()])
        self.assertEqual(activated.source, "frontier")

    def test_changed_winner_restarts_confirmation(self):
        controller = SelectiveResidualController()
        self.assertTrue(controller.decide(
            context(), [candidate(candidate_id="a")]).uses_native)
        changed = controller.decide(
            context(plan=1), [candidate(candidate_id="b")])
        self.assertTrue(changed.uses_native)
        self.assertEqual(changed.reason, "confirming_residual")

    def test_residual_commitment_is_bounded(self):
        controller = SelectiveResidualController(
            ResidualThresholds(
                confirmation_plans=1, max_residual_burst_plans=2))
        self.assertEqual(
            controller.decide(context(), [candidate()]).source, "frontier")
        self.assertEqual(
            controller.decide(
                context(plan=1, feedback=feedback()),
                [candidate()]).source,
            "frontier")
        bounded = controller.decide(
            context(plan=2, feedback=feedback(executed_plan=1)),
            [candidate()])
        self.assertTrue(bounded.uses_native)
        self.assertEqual(bounded.reason, "residual_burst_limit")

    def test_invalid_context_fails_closed(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        invalid = context(match_lower=0.8, match_upper=0.2)
        decision = controller.decide(invalid, [candidate()])
        self.assertTrue(decision.uses_native)
        self.assertEqual(decision.reason, "invalid_or_uncertain_context")

    def test_candidate_coverage_or_ood_gap_fails_closed(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        low_coverage = controller.decide(
            context(coverage=0.89), [candidate()])
        self.assertTrue(low_coverage.uses_native)
        unsupported = controller.decide(
            context(calibrated=False), [candidate()])
        self.assertTrue(unsupported.uses_native)

    def test_duplicate_candidate_identity_fails_closed(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        duplicate = candidate(candidate_id="duplicate")
        decision = controller.decide(
            context(), [duplicate, duplicate])
        self.assertTrue(decision.uses_native)
        self.assertEqual(decision.reason, "duplicate_candidate_identity")

    def test_repeated_identical_decision_is_idempotent(self):
        controller = SelectiveResidualController()
        first = controller.decide(context(), [candidate()])
        repeated = controller.decide(context(), [candidate()])
        self.assertEqual(first, repeated)
        self.assertEqual(repeated.reason, "confirming_residual")
        activated = controller.decide(
            context(plan=1), [candidate()])
        self.assertEqual(activated.reason, "activated_residual")

    def test_repeated_decision_with_changed_input_fails_closed(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        activated = controller.decide(context(), [candidate()])
        self.assertFalse(activated.uses_native)
        changed = controller.decide(
            context(), [candidate(candidate_id="changed")])
        self.assertTrue(changed.uses_native)
        self.assertEqual(changed.reason, "inconsistent_repeated_decision")
        repeated_original = controller.decide(context(), [candidate()])
        self.assertTrue(repeated_original.uses_native)
        # The inconsistent retry cannot erase the feedback obligation for the
        # residual that the first call may already have executed.
        next_plan = controller.decide(context(plan=1), [candidate()])
        self.assertTrue(next_plan.uses_native)
        self.assertEqual(
            next_plan.reason, "missing_or_mismatched_residual_feedback")

    def test_plan_and_graph_revision_must_advance(self):
        controller = SelectiveResidualController()
        controller.decide(context(), [candidate()])
        stale_revision = controller.decide(
            context(plan=1, revision=0), [candidate()])
        self.assertTrue(stale_revision.uses_native)
        self.assertEqual(
            stale_revision.reason, "graph_revision_not_advanced")

    def test_stale_or_invalid_rpc_cannot_erase_feedback_obligation(self):
        for intervening in ("stale", "invalid"):
            with self.subTest(intervening=intervening):
                controller = SelectiveResidualController(
                    ResidualThresholds(confirmation_plans=1))
                activated = controller.decide(context(plan=1), [candidate()])
                self.assertFalse(activated.uses_native)
                if intervening == "stale":
                    rejected = controller.decide(
                        context(plan=0, revision=0), [candidate()])
                    self.assertEqual(
                        rejected.reason, "non_monotonic_plan_index")
                else:
                    rejected = controller.decide(
                        context(plan=2, match_lower=0.8, match_upper=0.2),
                        [candidate()],
                    )
                    self.assertEqual(
                        rejected.reason, "invalid_or_uncertain_context")
                missing = controller.decide(
                    context(plan=2, revision=2), [candidate()])
                self.assertTrue(missing.uses_native)
                self.assertEqual(
                    missing.reason,
                    "missing_or_mismatched_residual_feedback",
                )

    def test_stale_malformed_rpc_cannot_erase_feedback_obligation(self):
        def raising_candidates():
            raise RuntimeError("stale payload must not be consumed")
            yield candidate()  # pragma: no cover

        malformed_sets = (
            [candidate(source="unknown")],
            [candidate(candidate_id=f"overflow-{index}")
             for index in range(33)],
            raising_candidates(),
        )
        for malformed in malformed_sets:
            with self.subTest(kind=type(malformed).__name__):
                controller = SelectiveResidualController(
                    ResidualThresholds(confirmation_plans=1))
                activated = controller.decide(
                    context(plan=1, revision=1), [candidate()])
                self.assertFalse(activated.uses_native)

                stale = controller.decide(
                    context(plan=0, revision=0), malformed)
                self.assertTrue(stale.uses_native)
                self.assertEqual(stale.reason, "non_monotonic_plan_index")

                missing = controller.decide(
                    context(plan=2, revision=2), [candidate()])
                self.assertTrue(missing.uses_native)
                self.assertEqual(
                    missing.reason,
                    "missing_or_mismatched_residual_feedback",
                )

    def test_session_switch_discards_confirmation(self):
        controller = SelectiveResidualController()
        controller.decide(context(session="a"), [candidate()])
        switched = controller.decide(
            context(session="b"), [candidate()])
        self.assertTrue(switched.uses_native)
        self.assertEqual(switched.reason, "confirming_residual")
        activated = controller.decide(
            context(session="b", plan=1), [candidate()])
        self.assertFalse(activated.uses_native)

    def test_residual_requires_feedback_and_no_improvement_cools_down(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        self.assertFalse(
            controller.decide(context(), [candidate()]).uses_native)
        missing = controller.decide(
            context(plan=1), [candidate()])
        self.assertTrue(missing.uses_native)
        self.assertEqual(
            missing.reason, "missing_or_mismatched_residual_feedback")
        reactivated = controller.decide(
            context(plan=2), [candidate()])
        self.assertFalse(reactivated.uses_native)
        cooled = controller.decide(
            context(plan=3, feedback=feedback(
                executed_plan=2, improved=False)),
            [candidate()])
        self.assertTrue(cooled.uses_native)
        self.assertEqual(
            cooled.reason, "residual_no_improvement_cooldown")
        self.assertFalse(controller.decide(
            context(plan=4), [candidate()]).uses_native)

        stale_controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        stale_controller.decide(context(), [candidate()])
        stale = stale_controller.decide(
            context(plan=1, feedback=feedback(executed_plan=99)),
            [candidate()])
        self.assertTrue(stale.uses_native)
        self.assertEqual(
            stale.reason, "missing_or_mismatched_residual_feedback")

    def test_malformed_flags_and_unknown_source_fail_entire_set_closed(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        malformed_context = controller.decide(
            context(fresh="false", calibrated="false"), [candidate()])
        self.assertTrue(malformed_context.uses_native)
        self.assertEqual(
            malformed_context.reason, "invalid_or_uncertain_context")

        unknown = controller.decide(
            context(plan=1),
            [candidate(candidate_id="good"),
             candidate(candidate_id="bad", source="unknown")],
        )
        self.assertTrue(unknown.uses_native)
        self.assertEqual(unknown.reason, "malformed_candidate_set")

    def test_nonboolean_feasibility_and_negative_uncertainty_fail_closed(self):
        for index, malformed in enumerate((
                candidate(feasible="false"),
                candidate(pose_t=-0.1),
                candidate(pose_yaw=-1.0))):
            with self.subTest(index=index):
                controller = SelectiveResidualController(
                    ResidualThresholds(confirmation_plans=1))
                decision = controller.decide(context(), [malformed])
                self.assertTrue(decision.uses_native)
                self.assertEqual(
                    decision.reason, "malformed_candidate_set")

    def test_candidate_k_and_ties_are_deterministic(self):
        controller = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        too_many = [candidate(candidate_id=f"f-{index}")
                    for index in range(33)]
        bounded = controller.decide(context(), too_many)
        self.assertTrue(bounded.uses_native)
        self.assertEqual(bounded.reason, "malformed_candidate_set")

        left = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        right = SelectiveResidualController(
            ResidualThresholds(confirmation_plans=1))
        a = candidate(candidate_id="a")
        b = candidate(candidate_id="b")
        self.assertEqual(
            left.decide(context(), [b, a]).candidate_id, "a")
        self.assertEqual(
            right.decide(context(), [a, b]).candidate_id, "a")


if __name__ == "__main__":
    unittest.main()
