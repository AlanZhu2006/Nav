"""Fail-closed contract for role-symmetric multi-goal evaluation data."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any


ROLE_SYMMETRIC_PROTOCOL = "multileg_v4_role_paired_20260812"
ROLE_SEQUENCE = ("initial_imagegoal", "novel", "revisit")
DOUBLE_REVISIT_PROTOCOL = "multileg_v5_double_revisit_20260812"
DOUBLE_REVISIT_SEQUENCE = ("initial_imagegoal", "revisit", "revisit")
MAX_ROLE_DISTANCE_MATCH_TOLERANCE_M = 0.50


@dataclass(frozen=True)
class RoleSymmetryObservation:
    geo_a_m: float
    geo_b_m: float
    initial_pose_error_m: float
    a_terminal_pose_error_m: float
    b_terminal_pose_error_m: float
    b_terminal_yaw_error_deg: float
    goal_b_matches_terminal_rgb: bool


@dataclass(frozen=True)
class DoubleRevisitObservation:
    geo_a_m: float
    geo_b_m: float
    geo_c_m: float
    initial_pose_error_m: float
    a_terminal_pose_error_m: float
    goal_b_matches_render: bool
    goal_c_matches_render: bool


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _validate_revisit_entry(
    *,
    goal: dict[str, Any],
    name: str,
    history_end: int,
    anchor_limit: int,
    negative_tail_start: int | None,
    anchor_margin: int,
    covis_band: tuple[float, float],
    negative_limit: float,
    issues: list[str],
) -> int | None:
    if (goal.get("name"), goal.get("kind")) != (name, "revisit"):
        issues.append(f"Goal {name} must be Revisit")
        return None
    curve = goal.get("covis_curve")
    if not isinstance(curve, list) or len(curve) != history_end:
        issues.append(
            f"Revisit {name} covisibility curve must span its full history")
        return None
    try:
        values = [float(value) for value in curve]
        anchor = int(goal.get("covis_argmax"))
        stored_limit = int(goal.get("anchor_frame_limit"))
        stored_covis = float(goal.get("covis"))
        stored_tail = float(goal.get("non_anchor_max_covis"))
    except (TypeError, ValueError):
        issues.append(f"Revisit {name} metadata is invalid")
        return None
    if not all(math.isfinite(value) for value in values):
        issues.append(f"Revisit {name} covisibility curve is non-finite")
        return None
    if stored_limit != anchor_limit:
        issues.append(
            f"Revisit {name} anchor limit must equal the end of leg A")
    if not anchor_margin <= anchor < anchor_limit:
        issues.append(f"Revisit {name} anchor must lie in valid leg-A memory")
        return None
    anchor_covis = values[anchor]
    if (not covis_band[0] <= anchor_covis <= covis_band[1]
            or abs(stored_covis - anchor_covis) > 1e-4):
        issues.append(f"Revisit {name} stored anchor covisibility is invalid")
    tail_start = history_end if negative_tail_start is None else negative_tail_start
    tail_max = max(values[tail_start:], default=0.0)
    if tail_max > negative_limit + 1e-4:
        issues.append(
            f"history tail is not a hard negative for Revisit {name}")
    if abs(stored_tail - tail_max) > 1e-4:
        issues.append(
            f"stored Revisit-{name} non-anchor covisibility disagrees")
    return anchor


def validate_double_revisit_contract(
    metadata: dict[str, Any],
    observation: DoubleRevisitObservation,
    *,
    distance_tolerance_m: float = 0.05,
    pose_tolerance_m: float = 0.02,
) -> dict[str, Any]:
    """Validate ``initial -> Revisit-B -> distinct Revisit-C`` data.

    C must relocalize into leg A while the complete intervening leg B is a
    hard negative.  This prevents the second Revisit from succeeding by using
    only the immediately preceding trajectory.
    """
    issues: list[str] = []
    if metadata.get("gen_protocol") != DOUBLE_REVISIT_PROTOCOL:
        issues.append(
            f"gen_protocol must be {DOUBLE_REVISIT_PROTOCOL!r}")
    if int(metadata.get("n_legs", -1)) != 3:
        issues.append("double-Revisit data must contain exactly three legs")
    if tuple(metadata.get("role_sequence") or ()) != DOUBLE_REVISIT_SEQUENCE:
        issues.append(
            f"role_sequence must be {list(DOUBLE_REVISIT_SEQUENCE)!r}")
    if metadata.get("initial_yaw_mode") != "uniform":
        issues.append("first-leg yaw must be sampled uniformly")
    if metadata.get("initial_goal_pose_source") != "expert_arrival_frame_exact":
        issues.append("Goal A pose must be bound to its terminal expert frame")
    if metadata.get("initial_start_pose_source") != "first_stored_expert_frame_exact":
        issues.append("initial pose must be bound to the first stored frame")
    if metadata.get("double_revisit_goal_image_source") != "metadata_pose_render":
        issues.append("double-Revisit goal images must render their metadata poses")

    initial_band = _band(metadata.get("initial_distance_band_m"),
                         "initial_distance_band_m", issues)
    if initial_band is not None:
        geo_a = _finite_float(observation.geo_a_m)
        if not (initial_band[0] - distance_tolerance_m
                <= geo_a <= initial_band[1] + distance_tolerance_m):
            issues.append("Goal A geodesic is outside its generation band")

    minimums = metadata.get("double_revisit_distance_min_m")
    if not isinstance(minimums, dict):
        issues.append("double-Revisit distance minima are missing")
        minimums = {}
    for name, measured, stored_key in (
        ("B", observation.geo_b_m, "geo_AB"),
        ("C", observation.geo_c_m, "geo_BC"),
    ):
        value = _finite_float(measured)
        minimum = _finite_float(minimums.get(name))
        stored = _finite_float(metadata.get(stored_key))
        if (not math.isfinite(value) or not math.isfinite(minimum)
                or value + distance_tolerance_m < minimum):
            issues.append(f"Revisit {name} geodesic is below its minimum")
        if (not math.isfinite(stored)
                or abs(stored - value) > distance_tolerance_m):
            issues.append(f"stored/measured Revisit {name} geodesic disagrees")

    if (not math.isfinite(float(observation.initial_pose_error_m))
            or observation.initial_pose_error_m > pose_tolerance_m):
        issues.append("initial metadata pose differs from the first stored frame")
    if (not math.isfinite(float(observation.a_terminal_pose_error_m))
            or observation.a_terminal_pose_error_m > pose_tolerance_m):
        issues.append("Goal A metadata differs from its terminal expert frame")
    if not observation.goal_b_matches_render:
        issues.append("goal_1.jpg does not render Revisit B metadata pose")
    if not observation.goal_c_matches_render:
        issues.append("goal_2.jpg does not render Revisit C metadata pose")

    switches = metadata.get("switches")
    goals = metadata.get("goals")
    if (not isinstance(switches, (list, tuple)) or len(switches) != 2
            or not isinstance(goals, list) or len(goals) != 2):
        issues.append("double-Revisit metadata requires two switches and goals")
    else:
        try:
            switch_a, switch_b = (int(switches[0]), int(switches[1]))
            n_frames = int(metadata.get("n_frames"))
            anchor_margin = int(metadata.get("anchor_margin"))
            covis_lo, covis_hi = (
                float(value) for value in metadata.get("covis_band"))
            negative_limit = float(metadata.get("covis_pos_lo"))
        except (TypeError, ValueError):
            issues.append("double-Revisit switch/covisibility metadata is invalid")
        else:
            if not 0 < switch_a < switch_b < n_frames:
                issues.append("switch indices must satisfy 0 < A < B < n_frames")
            anchor_b = _validate_revisit_entry(
                goal=goals[0], name="B", history_end=switch_a,
                anchor_limit=switch_a, negative_tail_start=None,
                anchor_margin=anchor_margin,
                covis_band=(covis_lo, covis_hi),
                negative_limit=negative_limit, issues=issues,
            )
            anchor_c = _validate_revisit_entry(
                goal=goals[1], name="C", history_end=switch_b,
                anchor_limit=switch_a, negative_tail_start=switch_a,
                anchor_margin=anchor_margin,
                covis_band=(covis_lo, covis_hi),
                negative_limit=negative_limit, issues=issues,
            )
            required_gap = _finite_float(
                metadata.get("double_revisit_min_anchor_gap"))
            stored_gap = _finite_float(metadata.get("double_revisit_anchor_gap"))
            if anchor_b is not None and anchor_c is not None:
                measured_gap = abs(anchor_b - anchor_c)
                if (not math.isfinite(required_gap)
                        or measured_gap < required_gap):
                    issues.append("B/C Revisit anchors are not sufficiently distinct")
                if (not math.isfinite(stored_gap)
                        or abs(stored_gap - measured_gap) > 1e-9):
                    issues.append("stored/measured B/C anchor gap disagrees")

    return {
        "ok": not issues,
        "issues": issues,
        "protocol": DOUBLE_REVISIT_PROTOCOL,
        "observation": asdict(observation),
    }


def _band(value: Any, field: str, issues: list[str]) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        issues.append(f"{field} must contain [min,max]")
        return None
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        issues.append(f"{field} contains non-numeric values")
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        issues.append(f"{field} contains non-finite values")
        return None
    if not 0.0 < lo < hi:
        issues.append(f"{field} must satisfy 0 < min < max")
        return None
    return lo, hi


def validate_role_symmetric_contract(
    metadata: dict[str, Any],
    observation: RoleSymmetryObservation,
    *,
    distance_tolerance_m: float = 0.05,
    pose_tolerance_m: float = 0.02,
    yaw_tolerance_deg: float = 0.05,
) -> dict[str, Any]:
    """Validate properties that made the old first/second Novel comparison unfair."""
    issues: list[str] = []
    if metadata.get("gen_protocol") != ROLE_SYMMETRIC_PROTOCOL:
        issues.append(
            f"gen_protocol must be {ROLE_SYMMETRIC_PROTOCOL!r}, got "
            f"{metadata.get('gen_protocol')!r}")
    if int(metadata.get("n_legs", -1)) != 3:
        issues.append("role-symmetric data must contain exactly three legs")
    if tuple(metadata.get("role_sequence") or ()) != ROLE_SEQUENCE:
        issues.append(f"role_sequence must be {list(ROLE_SEQUENCE)!r}")
    if metadata.get("initial_yaw_mode") != "uniform":
        issues.append("first-leg yaw must be sampled uniformly, not path-aligned")
    if metadata.get("initial_goal_pose_source") != "expert_arrival_frame_exact":
        issues.append("Goal A pose must be bound to its exact expert terminal frame")
    if metadata.get("initial_start_pose_source") != "first_stored_expert_frame_exact":
        issues.append("initial pose must be bound to the first stored expert frame")
    if metadata.get("novel_b_goal_yaw") != "expert_arrival_heading":
        issues.append("Goal B yaw must be the expert arrival heading")
    if metadata.get("novel_b_goal_image_source") != "expert_arrival_frame_exact":
        issues.append("Goal B image must be the exact expert terminal RGB")
    if metadata.get("role_pairing") != "same_episode_geodesic":
        issues.append("A/B roles must be paired by within-episode geodesic distance")

    try:
        role_match_tolerance = float(
            metadata.get("role_distance_match_tolerance_m"))
    except (TypeError, ValueError):
        role_match_tolerance = float("nan")
    if (not math.isfinite(role_match_tolerance)
            or not 0.0 < role_match_tolerance
            <= MAX_ROLE_DISTANCE_MATCH_TOLERANCE_M):
        issues.append(
            "role distance-match tolerance must be finite and in (0,0.50] m")

    initial_band = _band(metadata.get("initial_distance_band_m"),
                         "initial_distance_band_m", issues)
    novel_band = _band(metadata.get("novel_distance_band_m"),
                       "novel_distance_band_m", issues)
    if initial_band is not None and novel_band is not None:
        if any(abs(a - b) > 1e-9 for a, b in zip(initial_band, novel_band)):
            issues.append("initial and later-Novel geodesic bands differ")
        for name, distance in (("A", observation.geo_a_m),
                               ("B", observation.geo_b_m)):
            measured = float(distance)
            if (not math.isfinite(measured)
                    or not (initial_band[0] - distance_tolerance_m
                            <= measured
                            <= initial_band[1] + distance_tolerance_m)):
                issues.append(
                    f"Goal {name} geodesic {measured:.3f} m is outside "
                    f"the symmetric band {initial_band}")

    measured_role_error = abs(
        float(observation.geo_a_m) - float(observation.geo_b_m))
    if (not math.isfinite(measured_role_error)
            or (math.isfinite(role_match_tolerance)
                and measured_role_error
                > role_match_tolerance + distance_tolerance_m)):
        issues.append(
            f"A/B geodesics differ by {measured_role_error:.3f} m, exceeding "
            f"the paired-role tolerance {role_match_tolerance:.3f} m")
    try:
        stored_role_error = float(metadata.get("role_distance_error_m"))
    except (TypeError, ValueError):
        stored_role_error = float("nan")
    if (not math.isfinite(stored_role_error)
            or abs(stored_role_error - measured_role_error)
            > distance_tolerance_m):
        issues.append("stored/measured A/B role-distance error disagrees")

    stored_geo_a = metadata.get("geo_startA")
    stored_geo_b = metadata.get("geo_AB")
    for name, stored, measured in (
        ("A", stored_geo_a, observation.geo_a_m),
        ("B", stored_geo_b, observation.geo_b_m),
    ):
        try:
            stored_value = float(stored)
            measured_value = float(measured)
        except (TypeError, ValueError):
            issues.append(f"stored Goal {name} geodesic is missing")
        else:
            if not (math.isfinite(stored_value)
                    and math.isfinite(measured_value)):
                issues.append(
                    f"stored/measured Goal {name} geodesic is non-finite")
                continue
            disagreement = abs(stored_value - measured_value)
            if disagreement > distance_tolerance_m:
                issues.append(
                    f"stored/measured Goal {name} geodesics differ by "
                    f"{disagreement:.3f} m")

    if (not math.isfinite(float(observation.initial_pose_error_m))
            or observation.initial_pose_error_m > pose_tolerance_m):
        issues.append(
            f"initial metadata pose is {observation.initial_pose_error_m:.4f} m "
            "from the first stored expert frame")
    if (not math.isfinite(float(observation.a_terminal_pose_error_m))
            or observation.a_terminal_pose_error_m > pose_tolerance_m):
        issues.append(
            f"Goal A metadata is {observation.a_terminal_pose_error_m:.4f} m "
            "from its terminal expert frame")
    if (not math.isfinite(float(observation.b_terminal_pose_error_m))
            or observation.b_terminal_pose_error_m > pose_tolerance_m):
        issues.append(
            f"Goal B metadata is {observation.b_terminal_pose_error_m:.4f} m "
            "from its terminal expert frame")
    if (not math.isfinite(float(observation.b_terminal_yaw_error_deg))
            or observation.b_terminal_yaw_error_deg > yaw_tolerance_deg):
        issues.append(
            f"Goal B yaw differs from terminal expert yaw by "
            f"{observation.b_terminal_yaw_error_deg:.4f} deg")
    if not observation.goal_b_matches_terminal_rgb:
        issues.append("goal_1.jpg is not byte-identical to Goal B terminal RGB")

    goals = metadata.get("goals")
    if not isinstance(goals, list) or len(goals) != 2:
        issues.append("3-leg metadata must contain exactly Goal B and Goal C")
    else:
        goal_b, goal_c = goals
        if not isinstance(goal_b, dict) or not isinstance(goal_c, dict):
            issues.append("Goal B/C metadata entries must be dictionaries")
            goal_b, goal_c = {}, {}
        if (goal_b.get("name"), goal_b.get("kind")) != ("B", "novel"):
            issues.append("first goal metadata must be Novel B")
        if goal_b.get("covis_argmax") != -1:
            issues.append("Novel B must not contain a retrieval anchor")
        if (goal_c.get("name"), goal_c.get("kind")) != ("C", "revisit"):
            issues.append("second goal metadata must be Revisit C")
        switches = metadata.get("switches")
        if (not isinstance(switches, (list, tuple))
                or len(switches) != 2):
            issues.append("3-leg metadata must contain two switch indices")
        else:
            try:
                switch_a, switch_b = int(switches[0]), int(switches[1])
                n_frames = int(metadata.get("n_frames"))
            except (TypeError, ValueError):
                issues.append("switch indices and n_frames must be integers")
            else:
                if not 0 < switch_a < switch_b < n_frames:
                    issues.append("switch indices must satisfy 0 < A < B < n_frames")
                b_curve = goal_b.get("covis_curve")
                c_curve = goal_c.get("covis_curve")
                if not isinstance(b_curve, list) or len(b_curve) != switch_a:
                    issues.append("Novel B covisibility curve must span leg A")
                else:
                    try:
                        b_max = max(float(value) for value in b_curve)
                        novel_limit = float(metadata.get("novel_covis"))
                    except (TypeError, ValueError):
                        issues.append("Novel B covisibility values are invalid")
                    else:
                        if (not math.isfinite(b_max)
                                or not math.isfinite(novel_limit)
                                or b_max >= novel_limit):
                            issues.append(
                                "Novel B is not strictly non-covisible with leg A")
                if not isinstance(c_curve, list) or len(c_curve) != switch_b:
                    issues.append("Revisit C covisibility curve must span legs A+B")
                else:
                    try:
                        c_values = [float(value) for value in c_curve]
                        c_anchor = int(goal_c.get("covis_argmax"))
                        anchor_margin = int(metadata.get("anchor_margin"))
                        anchor_limit = int(goal_c.get("anchor_frame_limit"))
                        negative_limit = float(metadata.get("covis_pos_lo"))
                        stored_tail_max = float(
                            goal_c.get("non_anchor_max_covis"))
                        stored_c_covis = float(goal_c.get("covis"))
                        covis_lo, covis_hi = (
                            float(value) for value in metadata.get("covis_band"))
                    except (TypeError, ValueError):
                        issues.append("Revisit C anchor/covisibility values are invalid")
                    else:
                        if anchor_limit != switch_a:
                            issues.append("Revisit C anchor limit must equal the end of leg A")
                        if not anchor_margin <= c_anchor < switch_a:
                            issues.append("Revisit C anchor must lie in valid leg-A memory")
                        if not all(math.isfinite(value) for value in c_values):
                            issues.append("Revisit C covisibility curve is non-finite")
                        else:
                            anchor_covis = c_values[c_anchor] if (
                                0 <= c_anchor < len(c_values)) else float("nan")
                            if (not math.isfinite(anchor_covis)
                                    or not covis_lo <= anchor_covis <= covis_hi
                                    or abs(stored_c_covis - anchor_covis) > 1e-4):
                                issues.append(
                                    "Revisit C stored anchor covisibility is invalid")
                            tail_max = max(c_values[switch_a:], default=0.0)
                            if (not math.isfinite(negative_limit)
                                    or tail_max > negative_limit + 1e-4):
                                issues.append(
                                    "leg B is not a hard-negative history segment for Revisit C")
                            if (not math.isfinite(stored_tail_max)
                                    or abs(stored_tail_max - tail_max) > 1e-4):
                                issues.append(
                                    "stored Revisit-C non-anchor covisibility disagrees")

    return {
        "ok": not issues,
        "issues": issues,
        "protocol": ROLE_SYMMETRIC_PROTOCOL,
        "observation": asdict(observation),
        "initial_distance_band_m": (
            list(initial_band) if initial_band is not None else None),
        "novel_distance_band_m": (
            list(novel_band) if novel_band is not None else None),
    }


def require_role_symmetric_contract(
    metadata: dict[str, Any],
    observation: RoleSymmetryObservation,
) -> dict[str, Any]:
    report = validate_role_symmetric_contract(metadata, observation)
    if not report["ok"]:
        raise RuntimeError(
            "role-symmetric 3-leg data contract failed: "
            + "; ".join(report["issues"]))
    return report
