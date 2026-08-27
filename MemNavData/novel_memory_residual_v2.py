"""Fail-closed policy logic for a frozen-NavDP memory residual.

The native ImageGoal proposal is implicit and always available.  This module
only decides whether a separately generated memory-graph or frontier proposal
has enough calibrated evidence to replace it for one short commitment.  It is
deliberately dependency-free: feature extraction, LingBot pose/depth, NavDP
sampling, and calibration live outside this safety boundary.

The logic returns a native-default decision when it abstains.  Exact NavDP FIFO,
RNG, and trajectory equivalence additionally require a native-first executor
that this dependency-free selector deliberately does not implement.  Nor does
it turn empirical uncertainty estimates into a mathematical navigation-SR
guarantee; scene-disjoint risk calibration and paired evaluation remain needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
from typing import Iterable


class ResidualSource(str, Enum):
    """Kinds of non-native proposal understood by the controller."""

    MEMORY_GRAPH = "memory_graph"
    FRONTIER = "frontier"


@dataclass(frozen=True)
class ResidualThresholds:
    """Frozen deployment thresholds for selective intervention.

    Defaults are conservative engineering starting points.  A deployment must
    replace or confirm them using scene-disjoint calibration, then freeze the
    resulting artifact before blind evaluation.
    """

    match_lcb_for_memory: float = 0.90
    match_ucb_for_frontier: float = 0.10
    advantage_z: float = 1.645
    min_advantage_lcb_m: float = 0.25
    max_harm_probability_upper: float = 0.05
    max_pose_translation_p90_m: float = 0.30
    max_pose_yaw_p90_deg: float = 15.0
    min_clearance_lower_m: float = 0.30
    min_candidate_coverage_probability_lower: float = 0.90
    frontier_stagnation_plans: int = 3
    confirmation_plans: int = 2
    max_residual_burst_plans: int = 3

    def __post_init__(self) -> None:
        probabilities = (
            self.match_lcb_for_memory,
            self.match_ucb_for_frontier,
            self.max_harm_probability_upper,
            self.min_candidate_coverage_probability_lower,
        )
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(float(value))
               or not 0.0 <= float(value) <= 1.0
               for value in probabilities):
            raise ValueError("probability thresholds must lie in [0, 1]")
        if self.match_ucb_for_frontier >= self.match_lcb_for_memory:
            raise ValueError(
                "frontier and memory match regions must leave a defer gap")
        positive = (
            self.advantage_z,
            self.min_advantage_lcb_m,
            self.max_pose_translation_p90_m,
            self.max_pose_yaw_p90_deg,
            self.min_clearance_lower_m,
        )
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(float(value)) or float(value) < 0.0
               for value in positive):
            raise ValueError("metric thresholds must be finite and non-negative")
        counts = (
            self.frontier_stagnation_plans,
            self.confirmation_plans,
            self.max_residual_burst_plans,
        )
        if any(isinstance(value, bool) or not isinstance(value, Integral)
               or int(value) < 1 for value in counts):
            raise ValueError("plan-count thresholds must be positive integers")


@dataclass(frozen=True)
class ResidualFeedback:
    """Observed improvement after executing one residual plan."""

    source: ResidualSource | str
    candidate_id: str
    executed_plan_index: int
    executed_graph_revision: int
    graph_displacement_improved: bool
    native_novelty_improved: bool
    goal_evidence_improved: bool


@dataclass(frozen=True)
class ResidualContext:
    """Set-level evidence shared by all proposals at one policy decision."""

    session_id: str
    goal_epoch: str
    plan_index: int
    graph_revision: int
    match_probability_lower: float | None
    match_probability_upper: float | None
    candidate_coverage_probability_lower: float | None
    native_stagnation_plans: int
    graph_fresh: bool = True
    calibration_supported: bool = True
    previous_residual_feedback: ResidualFeedback | None = None


@dataclass(frozen=True)
class ResidualCandidate:
    """Calibrated evidence for one graph/frontier point-goal proposal.

    Missing and non-finite uncertainty is intentionally representable.  Any
    malformed candidate makes the complete decision set abstain rather than
    being skipped or accidentally treated as zero uncertainty.
    """

    candidate_id: str
    source: ResidualSource | str
    advantage_mean_m: float | None
    advantage_std_m: float | None
    harm_probability_upper: float | None
    pose_translation_p90_m: float | None
    pose_yaw_p90_deg: float | None
    clearance_lower_m: float | None
    route_feasible: bool


@dataclass(frozen=True)
class ResidualDecision:
    """One bounded selection; ``source='native'`` means selector abstention."""

    source: str
    candidate_id: str | None
    reason: str
    advantage_lcb_m: float | None = None

    @property
    def uses_native(self) -> bool:
        return self.source == "native"


MAX_RESIDUAL_CANDIDATES = 32


def _finite_real(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


class SelectiveResidualController:
    """Stateful, fail-closed selector with idempotent plan decisions."""

    def __init__(self, thresholds: ResidualThresholds | None = None) -> None:
        self.thresholds = thresholds or ResidualThresholds()
        self._session_id: str | None = None
        self._goal_epoch: str | None = None
        self._last_plan_index: int | None = None
        self._last_graph_revision: int | None = None
        self._pending_key: tuple[str, str] | None = None
        self._pending_count = 0
        self._pending_revision: int | None = None
        self._active_key: tuple[str, str] | None = None
        self._active_plans = 0
        self._last_residual_key: tuple[str, str, int, int] | None = None
        self._cached_decision_key: tuple[str, str, int, int] | None = None
        self._cached_signature: tuple | None = None
        self._cached_decision: ResidualDecision | None = None

    def reset(
        self,
        session_id: str | None = None,
        goal_epoch: str | None = None,
    ) -> None:
        """Clear temporal evidence after an episode/session or goal switch."""
        self._session_id = session_id
        self._goal_epoch = goal_epoch
        self._last_plan_index = None
        self._last_graph_revision = None
        self._clear_selection()
        self._last_residual_key = None
        self._cached_decision_key = None
        self._cached_signature = None
        self._cached_decision = None

    def _clear_selection(self) -> None:
        self._pending_key = None
        self._pending_count = 0
        self._pending_revision = None
        self._active_key = None
        self._active_plans = 0

    @staticmethod
    def _native(reason: str) -> ResidualDecision:
        return ResidualDecision(
            source="native", candidate_id=None, reason=reason)

    def _context_valid(self, context: ResidualContext) -> bool:
        if not isinstance(context, ResidualContext):
            return False
        lower = context.match_probability_lower
        upper = context.match_probability_upper
        coverage = context.candidate_coverage_probability_lower
        if not (_finite_real(lower) and _finite_real(upper)
                and _finite_real(coverage)):
            return False
        lower = float(lower)
        upper = float(upper)
        coverage = float(coverage)
        feedback = context.previous_residual_feedback
        if feedback is not None and not self._feedback_valid(feedback):
            return False
        return (
            isinstance(context.session_id, str)
            and bool(context.session_id.strip())
            and context.session_id == context.session_id.strip()
            and isinstance(context.goal_epoch, str)
            and bool(context.goal_epoch.strip())
            and context.goal_epoch == context.goal_epoch.strip()
            and isinstance(context.plan_index, Integral)
            and not isinstance(context.plan_index, bool)
            and int(context.plan_index) >= 0
            and isinstance(context.graph_revision, Integral)
            and not isinstance(context.graph_revision, bool)
            and int(context.graph_revision) >= 0
            and 0.0 <= lower <= upper <= 1.0
            and 0.0 <= coverage <= 1.0
            and (coverage
                 >= self.thresholds.min_candidate_coverage_probability_lower)
            and isinstance(context.native_stagnation_plans, Integral)
            and not isinstance(context.native_stagnation_plans, bool)
            and context.native_stagnation_plans >= 0
            and type(context.graph_fresh) is bool
            and context.graph_fresh
            and type(context.calibration_supported) is bool
            and context.calibration_supported
        )

    @staticmethod
    def _feedback_valid(feedback: object) -> bool:
        if not isinstance(feedback, ResidualFeedback):
            return False
        try:
            ResidualSource(feedback.source)
        except (TypeError, ValueError):
            return False
        return (
            isinstance(feedback.candidate_id, str)
            and bool(feedback.candidate_id.strip())
            and feedback.candidate_id == feedback.candidate_id.strip()
            and isinstance(feedback.executed_plan_index, Integral)
            and not isinstance(feedback.executed_plan_index, bool)
            and int(feedback.executed_plan_index) >= 0
            and isinstance(feedback.executed_graph_revision, Integral)
            and not isinstance(feedback.executed_graph_revision, bool)
            and int(feedback.executed_graph_revision) >= 0
            and type(feedback.graph_displacement_improved) is bool
            and type(feedback.native_novelty_improved) is bool
            and type(feedback.goal_evidence_improved) is bool
        )

    @staticmethod
    def _feedback_signature(
        feedback: ResidualFeedback | None,
    ) -> tuple | None:
        if feedback is None:
            return None
        return (
            ResidualSource(feedback.source).value,
            feedback.candidate_id,
            int(feedback.executed_plan_index),
            int(feedback.executed_graph_revision),
            feedback.graph_displacement_improved,
            feedback.native_novelty_improved,
            feedback.goal_evidence_improved,
        )

    @staticmethod
    def _normalized_candidate(
        candidate: object,
    ) -> tuple[ResidualCandidate, ResidualSource] | None:
        if not isinstance(candidate, ResidualCandidate):
            return None
        if (not isinstance(candidate.candidate_id, str)
                or not candidate.candidate_id.strip()
                or candidate.candidate_id != candidate.candidate_id.strip()):
            return None
        try:
            source = ResidualSource(candidate.source)
        except (TypeError, ValueError):
            return None
        values = (
            candidate.advantage_mean_m,
            candidate.advantage_std_m,
            candidate.harm_probability_upper,
            candidate.pose_translation_p90_m,
            candidate.pose_yaw_p90_deg,
            candidate.clearance_lower_m,
        )
        if not all(_finite_real(value) for value in values):
            return None
        if type(candidate.route_feasible) is not bool:
            return None
        if float(candidate.advantage_std_m) < 0.0:
            return None
        if not 0.0 <= float(candidate.harm_probability_upper) <= 1.0:
            return None
        if (float(candidate.pose_translation_p90_m) < 0.0
                or float(candidate.pose_yaw_p90_deg) < 0.0
                or float(candidate.clearance_lower_m) < 0.0):
            return None
        return candidate, source

    @staticmethod
    def _candidate_signature(
        candidate: ResidualCandidate,
        source: ResidualSource,
    ) -> tuple:
        return (
            source.value,
            candidate.candidate_id,
            float(candidate.advantage_mean_m),
            float(candidate.advantage_std_m),
            float(candidate.harm_probability_upper),
            float(candidate.pose_translation_p90_m),
            float(candidate.pose_yaw_p90_deg),
            float(candidate.clearance_lower_m),
            candidate.route_feasible,
        )

    def _decision_signature(
        self,
        context: ResidualContext,
        normalized: list[tuple[ResidualCandidate, ResidualSource]],
    ) -> tuple:
        return (
            float(context.match_probability_lower),
            float(context.match_probability_upper),
            float(context.candidate_coverage_probability_lower),
            int(context.native_stagnation_plans),
            context.graph_fresh,
            context.calibration_supported,
            self._feedback_signature(context.previous_residual_feedback),
            tuple(sorted(
                self._candidate_signature(candidate, source)
                for candidate, source in normalized
            )),
        )

    def _commit(
        self,
        context: ResidualContext,
        signature: tuple | None,
        decision: ResidualDecision,
        *,
        advance: bool = True,
        preserve_last_residual_key: bool = False,
    ) -> ResidualDecision:
        if advance:
            self._last_plan_index = int(context.plan_index)
            self._last_graph_revision = int(context.graph_revision)
        self._cached_decision_key = (
            context.session_id,
            context.goal_epoch,
            int(context.plan_index),
            int(context.graph_revision),
        )
        self._cached_signature = signature
        self._cached_decision = decision
        if not preserve_last_residual_key:
            self._last_residual_key = (
                (
                    decision.source,
                    decision.candidate_id,
                    int(context.plan_index),
                    int(context.graph_revision),
                )
                if not decision.uses_native and decision.candidate_id is not None
                else None
            )
        return decision

    def _commit_native(
        self,
        context: ResidualContext,
        signature: tuple | None,
        reason: str,
        *,
        clear_selection: bool = True,
        advance: bool = True,
        preserve_last_residual_key: bool = False,
    ) -> ResidualDecision:
        if clear_selection:
            self._clear_selection()
        return self._commit(
            context,
            signature,
            self._native(reason),
            advance=advance,
            preserve_last_residual_key=preserve_last_residual_key,
        )

    def _candidate_lcb(
        self,
        context: ResidualContext,
        candidate: ResidualCandidate,
        source: ResidualSource,
    ) -> float | None:
        if not candidate.route_feasible:
            return None

        std = float(candidate.advantage_std_m)
        harm = float(candidate.harm_probability_upper)
        pose_t = float(candidate.pose_translation_p90_m)
        pose_yaw = float(candidate.pose_yaw_p90_deg)
        clearance = float(candidate.clearance_lower_m)
        thresholds = self.thresholds
        if harm > thresholds.max_harm_probability_upper:
            return None
        if pose_t > thresholds.max_pose_translation_p90_m:
            return None
        if pose_yaw > thresholds.max_pose_yaw_p90_deg:
            return None
        if clearance < thresholds.min_clearance_lower_m:
            return None

        if source is ResidualSource.MEMORY_GRAPH:
            if (float(context.match_probability_lower)
                    < thresholds.match_lcb_for_memory):
                return None
        elif source is ResidualSource.FRONTIER:
            if (float(context.match_probability_upper)
                    > thresholds.match_ucb_for_frontier):
                return None
            if (context.native_stagnation_plans
                    < thresholds.frontier_stagnation_plans):
                return None

        advantage_lcb = (
            float(candidate.advantage_mean_m)
            - thresholds.advantage_z * std
        )
        if advantage_lcb < thresholds.min_advantage_lcb_m:
            return None
        return advantage_lcb

    def decide(
        self,
        context: ResidualContext,
        candidates: Iterable[ResidualCandidate],
    ) -> ResidualDecision:
        """Choose a residual only after all selective checks pass.

        The caller must execute the native ImageGoal request first and generate
        residual proposals through a read-only path.  Returning ``native`` can
        then be an exact no-op with respect to NavDP's FIFO and RNG schedule.
        """
        if not self._context_valid(context):
            # An untrusted malformed/stale RPC must not erase the feedback
            # obligation created by a residual that may already have run.
            return self._native("invalid_or_uncertain_context")

        if (context.session_id != self._session_id
                or context.goal_epoch != self._goal_epoch):
            self.reset(context.session_id, context.goal_epoch)

        decision_key = (
            context.session_id,
            context.goal_epoch,
            int(context.plan_index),
            int(context.graph_revision),
        )

        # Reject out-of-order RPCs before touching the candidate iterable.  In
        # particular, a stale request whose payload is malformed (or whose
        # iterator raises) must not enter the malformed-set commit path and
        # erase feedback owed by a residual that may already have executed.
        # The cached key is exempt because an exact retry still needs its
        # candidate signature checked below.
        if (decision_key != self._cached_decision_key
                and self._last_plan_index is not None):
            if int(context.plan_index) <= self._last_plan_index:
                return self._native("non_monotonic_plan_index")
            if (self._last_graph_revision is not None
                    and int(context.graph_revision)
                    <= self._last_graph_revision):
                return self._native("graph_revision_not_advanced")

        try:
            candidate_values = list(candidates)
        except Exception:
            candidate_values = []
            malformed_candidates = True
        else:
            malformed_candidates = len(candidate_values) > MAX_RESIDUAL_CANDIDATES
        normalized: list[tuple[ResidualCandidate, ResidualSource]] = []
        if not malformed_candidates:
            for candidate in candidate_values:
                row = self._normalized_candidate(candidate)
                if row is None:
                    malformed_candidates = True
                    break
                normalized.append(row)
        if malformed_candidates:
            if (decision_key == self._cached_decision_key
                    and self._cached_signature is None
                    and self._cached_decision is not None):
                return self._cached_decision
            same_decision = decision_key == self._cached_decision_key
            advances = (
                self._last_plan_index is None
                or (
                    int(context.plan_index) > self._last_plan_index
                    and (
                        self._last_graph_revision is None
                        or int(context.graph_revision)
                        > self._last_graph_revision
                    )
                )
            )
            return self._commit_native(
                context, None, "malformed_candidate_set",
                advance=advances,
                preserve_last_residual_key=same_decision,
            )

        signature = self._decision_signature(context, normalized)
        if decision_key == self._cached_decision_key:
            if (signature == self._cached_signature
                    and self._cached_decision is not None):
                return self._cached_decision
            return self._commit_native(
                context, ("inconsistent_repeated_decision",),
                "inconsistent_repeated_decision",
                advance=False,
                preserve_last_residual_key=True,
            )

        feedback = context.previous_residual_feedback
        if self._last_residual_key is not None:
            expected_key = self._last_residual_key
            feedback_key = (
                (
                    ResidualSource(feedback.source).value,
                    feedback.candidate_id,
                    int(feedback.executed_plan_index),
                    int(feedback.executed_graph_revision),
                )
                if feedback is not None else None
            )
            improved = bool(feedback is not None and (
                feedback.graph_displacement_improved
                or feedback.native_novelty_improved
                or feedback.goal_evidence_improved
            ))
            if feedback_key != expected_key:
                return self._commit_native(
                    context, signature,
                    "missing_or_mismatched_residual_feedback")
            if not improved:
                return self._commit_native(
                    context, signature,
                    "residual_no_improvement_cooldown")
        elif feedback is not None:
            return self._commit_native(
                context, signature, "unexpected_residual_feedback")

        eligible: list[tuple[float, str, ResidualSource]] = []
        seen_keys: set[tuple[str, str]] = set()
        for candidate, candidate_source in normalized:
            candidate_key = (
                candidate_source.value, candidate.candidate_id)
            if candidate_key in seen_keys:
                return self._commit_native(
                    context, signature, "duplicate_candidate_identity")
            seen_keys.add(candidate_key)
            advantage_lcb = self._candidate_lcb(
                context, candidate, candidate_source)
            if advantage_lcb is None:
                continue
            eligible.append(
                (advantage_lcb, candidate.candidate_id, candidate_source))

        if not eligible:
            return self._commit_native(
                context, signature, "no_eligible_residual")

        # Prefer the largest conservative advantage.  Candidate id provides a
        # deterministic tie-break independent of input enumeration order.
        advantage_lcb, candidate_id, source = sorted(
            eligible, key=lambda item: (-item[0], item[1], item[2].value))[0]
        key = (source.value, candidate_id)

        if self._active_key == key:
            if self._active_plans >= self.thresholds.max_residual_burst_plans:
                return self._commit_native(
                    context, signature, "residual_burst_limit")
            self._active_plans += 1
            return self._commit(context, signature, ResidualDecision(
                source=source.value,
                candidate_id=candidate_id,
                reason="active_residual",
                advantage_lcb_m=advantage_lcb,
            ))

        # A changed winning candidate invalidates the previous confirmation.
        self._active_key = None
        self._active_plans = 0
        if self._pending_key == key:
            if (self._pending_revision is not None
                    and int(context.graph_revision) > self._pending_revision):
                self._pending_count += 1
                self._pending_revision = int(context.graph_revision)
        else:
            self._pending_key = key
            self._pending_count = 1
            self._pending_revision = int(context.graph_revision)

        if self._pending_count < self.thresholds.confirmation_plans:
            return self._commit(
                context, signature, self._native("confirming_residual"))

        self._active_key = key
        self._active_plans = 1
        self._pending_key = None
        self._pending_count = 0
        return self._commit(context, signature, ResidualDecision(
            source=source.value,
            candidate_id=candidate_id,
            reason="activated_residual",
            advantage_lcb_m=advantage_lcb,
        ))
