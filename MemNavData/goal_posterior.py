"""Goal-location posterior over persistent memory nodes and frontiers (GLP).

Design doc: MemNavData/GOAL_POSTERIOR_DECISION_LAYER_20260807.md.

This module is the deterministic mechanism core only.  It never imports
torch, Habitat, or LingBot; evidence values arrive as pre-computed,
pre-calibrated log-likelihood-ratios.  Three properties are load-bearing
and enforced here rather than by convention:

1. Static evidence is attached exactly once per (goal, hypothesis).  The
   posterior recursion is hypothesis-set growth, never repeated
   multiplication of the same evidence (no naive-Bayes drift).
2. Frontier retirement is mass-conserving: heir shares transfer the
   retired weight explicitly; any share not inherited is explicitly
   destroyed mass, never silently renormalized away.
3. Fail closed: duplicate ids, unknown ids, non-finite inputs, and
   querying a posterior that contains stale (goal-switched, not yet
   re-evaluated) hypotheses all raise instead of guessing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

UNMODELED_ID = "__unmodeled__"

_DEFAULT_SURVIVAL_FLOOR = 1e-3
_DEFAULT_APPROACH_CAP = 4.0
_DEFAULT_CLUSTER_GAP = 16


def _require_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


@dataclass
class _Hypothesis:
    kind: str                      # "node" | "frontier"
    log_prior: float
    log_static: float | None       # None => stale / not yet evaluated for this goal
    frame_index: int | None = None
    carve_log: float = 0.0         # log survival mass from visibility carving
    approach_log: float = 0.0      # bounded cumulative approach-time evidence


@dataclass(frozen=True)
class PosteriorSummary:
    p_match: float                 # total mass on node hypotheses
    p_unmodeled: float
    entropy: float                 # over the full normalized posterior
    best_region_mass: float        # mass of the strongest temporal node region
    best_region_anchor: str | None # highest-weight node id inside that region
    best_frontier: str | None
    best_frontier_mass: float


class GoalPosterior:
    """Unnormalized log-weight registry + normalized readouts."""

    def __init__(
        self,
        unmodeled_log_weight: float,
        survival_floor: float = _DEFAULT_SURVIVAL_FLOOR,
        approach_cap: float = _DEFAULT_APPROACH_CAP,
        cluster_gap: int = _DEFAULT_CLUSTER_GAP,
    ) -> None:
        self._unmodeled_log = _require_finite(
            "unmodeled_log_weight", unmodeled_log_weight)
        if not (0.0 < survival_floor < 1.0):
            raise ValueError("survival_floor must be in (0, 1)")
        if approach_cap <= 0.0 or not math.isfinite(approach_cap):
            raise ValueError("approach_cap must be finite and positive")
        if int(cluster_gap) < 1:
            raise ValueError("cluster_gap must be >= 1")
        self._survival_floor_log = math.log(survival_floor)
        self._approach_cap = float(approach_cap)
        self._cluster_gap = int(cluster_gap)
        self._hypotheses: dict[str, _Hypothesis] = {}

    # ------------------------------------------------------------------
    # registration (static evidence: exactly once per goal)
    # ------------------------------------------------------------------
    def add_node(
        self,
        node_id: str,
        frame_index: int,
        log_ratio: float,
        log_prior: float = 0.0,
    ) -> None:
        if node_id in self._hypotheses or node_id == UNMODELED_ID:
            raise ValueError(f"duplicate or reserved hypothesis id: {node_id}")
        if int(frame_index) < 0:
            raise ValueError("frame_index must be non-negative")
        self._hypotheses[node_id] = _Hypothesis(
            kind="node",
            log_prior=_require_finite("log_prior", log_prior),
            log_static=_require_finite("log_ratio", log_ratio),
            frame_index=int(frame_index),
        )

    def add_frontier(
        self, frontier_id: str, log_area_prior: float, log_ratio: float
    ) -> None:
        if frontier_id in self._hypotheses or frontier_id == UNMODELED_ID:
            raise ValueError(
                f"duplicate or reserved hypothesis id: {frontier_id}")
        self._hypotheses[frontier_id] = _Hypothesis(
            kind="frontier",
            log_prior=_require_finite("log_area_prior", log_area_prior),
            log_static=_require_finite("log_ratio", log_ratio),
        )

    # ------------------------------------------------------------------
    # frontier lineage (mass conserving)
    # ------------------------------------------------------------------
    def retire_frontier(self, frontier_id: str, heirs: dict[str, float]) -> None:
        """Retire a frontier, transferring `share` of its current weight to
        each heir (a NEW frontier id).  Shares must be in [0, 1] and sum to
        at most 1; the uninherited remainder is destroyed (explored space
        that contained no goal evidence)."""
        hypothesis = self._hypotheses.get(frontier_id)
        if hypothesis is None or hypothesis.kind != "frontier":
            raise ValueError(f"unknown frontier: {frontier_id}")
        total_share = 0.0
        for heir_id, share in heirs.items():
            share = _require_finite("heir share", share)
            if not (0.0 <= share <= 1.0):
                raise ValueError("heir share must be in [0, 1]")
            if heir_id in self._hypotheses or heir_id == UNMODELED_ID:
                raise ValueError(f"heir id already exists: {heir_id}")
            total_share += share
        if total_share > 1.0 + 1e-9:
            raise ValueError("heir shares must sum to at most 1")
        retired_log = self._log_weight(hypothesis)
        del self._hypotheses[frontier_id]
        for heir_id, share in heirs.items():
            if share <= 0.0:
                continue
            self._hypotheses[heir_id] = _Hypothesis(
                kind="frontier",
                log_prior=retired_log + math.log(share),
                log_static=0.0,
            )

    # ------------------------------------------------------------------
    # accumulating updates
    # ------------------------------------------------------------------
    def carve(self, hypothesis_id: str, detection_power: float) -> None:
        """Observed the hypothesis region with detection power d and saw no
        goal evidence: multiply survival mass by (1 - d), floored so no
        hypothesis is ever fully deleted by carving alone."""
        hypothesis = self._require(hypothesis_id)
        detection_power = _require_finite("detection_power", detection_power)
        if not (0.0 <= detection_power < 1.0):
            raise ValueError("detection_power must be in [0, 1)")
        hypothesis.carve_log = max(
            hypothesis.carve_log + math.log1p(-detection_power),
            self._survival_floor_log,
        )

    def add_approach_evidence(self, node_id: str, log_ratio: float) -> None:
        """Bounded cumulative evidence from NEW keyframes while approaching a
        hypothesized match (the probabilistic form of re-verification)."""
        hypothesis = self._require(node_id)
        if hypothesis.kind != "node":
            raise ValueError("approach evidence applies to node hypotheses")
        updated = hypothesis.approach_log + _require_finite(
            "log_ratio", log_ratio)
        hypothesis.approach_log = max(
            -self._approach_cap, min(self._approach_cap, updated))

    # ------------------------------------------------------------------
    # goal switch
    # ------------------------------------------------------------------
    def reset_goal(self) -> None:
        """A goal change invalidates goal-conditioned quantities (static
        evidence, approach evidence, carving) but keeps the hypothesis
        registry and its structural priors.  Hypotheses become stale and the
        posterior refuses to answer until evidence is re-supplied."""
        for hypothesis in self._hypotheses.values():
            hypothesis.log_static = None
            hypothesis.approach_log = 0.0
            hypothesis.carve_log = 0.0

    def resupply_evidence(self, hypothesis_id: str, log_ratio: float) -> None:
        hypothesis = self._require(hypothesis_id)
        if hypothesis.log_static is not None:
            raise ValueError(
                f"hypothesis already has evidence: {hypothesis_id}")
        hypothesis.log_static = _require_finite("log_ratio", log_ratio)

    # ------------------------------------------------------------------
    # readouts
    # ------------------------------------------------------------------
    def posterior(self) -> dict[str, float]:
        stale = [i for i, h in self._hypotheses.items() if h.log_static is None]
        if stale:
            raise RuntimeError(
                f"posterior queried with stale hypotheses: {sorted(stale)[:4]}")
        logs = {i: self._log_weight(h) for i, h in self._hypotheses.items()}
        logs[UNMODELED_ID] = self._unmodeled_log
        peak = max(logs.values())
        weights = {i: math.exp(v - peak) for i, v in logs.items()}
        total = sum(weights.values())
        return {i: w / total for i, w in weights.items()}

    def match_probability(self) -> float:
        posterior = self.posterior()
        return sum(
            posterior[i]
            for i, h in self._hypotheses.items() if h.kind == "node")

    def node_regions(self) -> list[tuple[list[str], float]]:
        """Temporal clusters of node hypotheses (frame gap <= cluster_gap),
        each with its total posterior mass, sorted by descending mass."""
        posterior = self.posterior()
        nodes = sorted(
            ((h.frame_index, i) for i, h in self._hypotheses.items()
             if h.kind == "node"),
        )
        regions: list[list[str]] = []
        previous_frame: int | None = None
        for frame, node_id in nodes:
            if previous_frame is None or frame - previous_frame > self._cluster_gap:
                regions.append([])
            regions[-1].append(node_id)
            previous_frame = frame
        scored = [
            (ids, sum(posterior[i] for i in ids)) for ids in regions]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def summary(self) -> PosteriorSummary:
        posterior = self.posterior()
        entropy = -sum(p * math.log(p) for p in posterior.values() if p > 0.0)
        regions = self.node_regions()
        best_region_mass, best_anchor = 0.0, None
        if regions:
            ids, best_region_mass = regions[0]
            best_anchor = max(ids, key=lambda i: (posterior[i], i))
        frontier_mass = {
            i: posterior[i] for i, h in self._hypotheses.items()
            if h.kind == "frontier"}
        best_frontier = max(
            frontier_mass, key=lambda i: (frontier_mass[i], i), default=None)
        return PosteriorSummary(
            p_match=sum(
                posterior[i] for i, h in self._hypotheses.items()
                if h.kind == "node"),
            p_unmodeled=posterior[UNMODELED_ID],
            entropy=entropy,
            best_region_mass=best_region_mass,
            best_region_anchor=best_anchor,
            best_frontier=best_frontier,
            best_frontier_mass=(
                frontier_mass[best_frontier] if best_frontier else 0.0),
        )

    # ------------------------------------------------------------------
    def total_unnormalized_weight(self) -> float:
        """Linear-space total hypothesis weight (excluding unmodeled); used
        by mass-conservation tests, not by decisions."""
        return sum(
            math.exp(self._log_weight(h))
            for h in self._hypotheses.values() if h.log_static is not None)

    def _log_weight(self, hypothesis: _Hypothesis) -> float:
        static = 0.0 if hypothesis.log_static is None else hypothesis.log_static
        return (hypothesis.log_prior + static
                + hypothesis.carve_log + hypothesis.approach_log)

    def _require(self, hypothesis_id: str) -> _Hypothesis:
        hypothesis = self._hypotheses.get(hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"unknown hypothesis: {hypothesis_id}")
        return hypothesis
