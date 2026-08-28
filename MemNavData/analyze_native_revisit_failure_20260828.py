"""Post-hoc diagnostic: why does mono-native NavDP fail supported Revisits?

Label: POST-HOC DIAGNOSTIC on the consumed fresh HM3D full-mono mixed-role
population (formal_20260820T143609Z_e6dd44c6). No protocol freeze, no new
SR claim; this stratifies and classifies already-sealed outcomes, in the
spirit of ``analyze_certificate_evidence_waterfall.py``.

Questions answered offline, from pulled receipts only (no renderer):

1. Direction strata for REVISIT queries (the sealed summary computes them
   only for Novel): each ``role_pairs.json`` stores the construction-time
   geodesic first-segment bearing ``initial_path_bearing_rad``; relative
   direction = wrap(bearing - online_a_endpoint.yaw_rad), the identical
   quantity behind the Novel ``initial_path_direction_relative_to_a_end_deg``
   field (verified per-episode against the stored Novel value).
2. Per-arm SR within each stratum: does the raw/CEC bearing injection
   rescue exactly the strata where native fails (back/lateral)?
3. Behavior taxonomy of native failures from the executed rollout traces
   (x/z/yaw per step): did the agent ever align its heading with the goal
   direction, how much did it turn in total, did it ever get close?

Run from the repo root:
    python MemNavData/analyze_native_revisit_failure_20260828.py \
        --pulled-root .diagnostics/hm3d_fresh_fullmono_mixed_role_20260820/pulled_20260828 \
        --out-dir .diagnostics/native_revisit_failure_20260828
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARMS = ("mono_native", "mono_raw_fixed", "mono_cec")
ROLES = ("novel", "revisit")
NEAR_MISS_M = 1.5
ALIGN_DEG = 45.0


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def direction_stratum(degrees: float) -> str:
    magnitude = abs(wrap_angle(math.radians(float(degrees))))
    if magnitude <= math.radians(45.0):
        return "front"
    if magnitude < math.radians(135.0):
        return "lateral"
    return "back"


def relative_direction_deg(path_bearing_rad: float, a_end_yaw_rad: float) -> float:
    return math.degrees(wrap_angle(float(path_bearing_rad) - float(a_end_yaw_rad)))


def bearing_to_goal_deg(x: float, z: float, yaw: float,
                        goal_x: float, goal_z: float) -> float:
    """Signed goal bearing in the agent frame (Habitat forward = -Z)."""
    facing = math.atan2(-(goal_x - x), -(goal_z - z))
    return math.degrees(wrap_angle(facing - yaw))


def trace_behavior(trace: list[dict], goal_xz: tuple[float, float]) -> dict:
    """Heading/turning/progress statistics for one executed rollout."""
    xs = [float(r["x"]) for r in trace]
    zs = [float(r["z"]) for r in trace]
    yaws = [float(r["yaw"]) for r in trace]
    dists = [math.hypot(x - goal_xz[0], z - goal_xz[1])
             for x, z in zip(xs, zs)]
    bearings = [bearing_to_goal_deg(x, z, yaw, *goal_xz)
                for x, z, yaw in zip(xs, zs, yaws)]
    yaw_steps = [wrap_angle(b - a) for a, b in zip(yaws, yaws[1:])]
    cumulative = [0.0]
    for step in yaw_steps:
        cumulative.append(cumulative[-1] + step)
    aligned_steps = [i for i, b in enumerate(bearings) if abs(b) <= ALIGN_DEG]
    path_len = sum(math.hypot(b - a, d - c)
                   for a, b, c, d in zip(xs, xs[1:], zs, zs[1:]))
    return {
        "steps": len(trace),
        "initial_goal_distance_m": dists[0],
        "final_goal_distance_m": dists[-1],
        "min_goal_distance_m": min(dists),
        "initial_goal_bearing_deg": bearings[0],
        "path_length_m": path_len,
        "total_unsigned_turn_deg": math.degrees(sum(abs(s) for s in yaw_steps)),
        "max_abs_cumulative_turn_deg": math.degrees(
            max(abs(c) for c in cumulative)),
        "ever_aligned_within_45deg": bool(aligned_steps),
        "first_alignment_step": aligned_steps[0] if aligned_steps else None,
        "aligned_step_fraction": len(aligned_steps) / len(trace),
    }


def classify_failure(behavior: dict) -> str:
    """Taxonomy for one non-successful rollout."""
    if behavior["min_goal_distance_m"] <= NEAR_MISS_M:
        return "near_miss_no_arrival"
    approach = (behavior["initial_goal_distance_m"]
                - behavior["min_goal_distance_m"])
    if approach < 0.5:
        return "never_approached"
    return "partial_approach_stall"


def load_episode(pulled_root: Path, label: str) -> dict:
    """Join receipts + benchmark manifest for one history."""
    _, scene, episode = label.split("_", 2)
    eval_dir = pulled_root / "evaluation_natural_direction" / label
    completion = json.loads((eval_dir / "completion.json").read_text())
    manifest = json.loads(
        (pulled_root / "benchmarks" / "natural_direction" / scene / episode
         / "role_pairs.json").read_text())
    a_end_yaw = float(manifest["online_a_endpoint"]["yaw_rad"])
    queries = {q["analysis_role"]: q for q in manifest["pairs"][0]["queries"]}

    novel_expected = float(
        queries["novel"]["initial_path_direction_relative_to_a_end_deg"])
    novel_recomputed = relative_direction_deg(
        float(queries["novel"]["initial_path_bearing_rad"]), a_end_yaw)
    if abs(wrap_angle(math.radians(novel_recomputed - novel_expected))) > 1e-6:
        raise AssertionError(
            f"{label}: bearing convention drifted "
            f"({novel_recomputed} vs stored {novel_expected})")

    out = {"label": label, "scene": scene, "episode": episode, "roles": {}}
    for role in ROLES:
        query = queries[role]
        goal = query["floor_position"]
        goal_xz = (float(goal[0]), float(goal[2]))
        direction_deg = relative_direction_deg(
            float(query["initial_path_bearing_rad"]), a_end_yaw)
        role_row = {
            "initial_path_direction_deg": direction_deg,
            "stratum": direction_stratum(direction_deg),
            "geodesic_from_a_end_m": float(query["geodesic_from_a_end_m"]),
            "max_online_a_covis": float(query["max_online_a_covis"]),
            "support_band": query.get("construction_support_band"),
            "arms": {},
        }
        for arm in ARMS:
            plans_path = (eval_dir / arm
                          / f"{episode}_pair_00_{role}_plans.json")
            plans = json.loads(plans_path.read_text())
            behavior = trace_behavior(
                plans["rollout_traces"]["query"], goal_xz)
            reached = bool(completion["outcomes"][arm][role])
            behavior["reached"] = reached
            behavior["failure_class"] = (
                None if reached else classify_failure(behavior))
            role_row["arms"][arm] = behavior
        out["roles"][role] = role_row
    return out


def aggregate(episodes: list[dict]) -> dict:
    strata_table: dict = {
        role: {s: {arm: {"n": 0, "successes": 0} for arm in ARMS}
               for s in ("front", "lateral", "back")}
        for role in ROLES
    }
    failure_taxonomy = {arm: {} for arm in ARMS}
    native_revisit_failures = []
    for ep in episodes:
        for role in ROLES:
            row = ep["roles"][role]
            for arm in ARMS:
                cell = strata_table[role][row["stratum"]][arm]
                cell["n"] += 1
                cell["successes"] += int(row["arms"][arm]["reached"])
                if role == "revisit" and not row["arms"][arm]["reached"]:
                    cls = row["arms"][arm]["failure_class"]
                    failure_taxonomy[arm][cls] = (
                        failure_taxonomy[arm].get(cls, 0) + 1)
        native = ep["roles"]["revisit"]["arms"]["mono_native"]
        if not native["reached"]:
            native_revisit_failures.append({
                "label": ep["label"],
                "stratum": ep["roles"]["revisit"]["stratum"],
                "direction_deg": round(
                    ep["roles"]["revisit"]["initial_path_direction_deg"], 1),
                "failure_class": native["failure_class"],
                "min_goal_distance_m": round(
                    native["min_goal_distance_m"], 2),
                "ever_aligned_within_45deg":
                    native["ever_aligned_within_45deg"],
                "max_abs_cumulative_turn_deg": round(
                    native["max_abs_cumulative_turn_deg"], 0),
            })
    for role in ROLES:
        for stratum in strata_table[role].values():
            for cell in stratum.values():
                cell["sr"] = (cell["successes"] / cell["n"]
                              if cell["n"] else None)
    alignment = {}
    for reached_bucket in (False, True):
        rows = [ep["roles"]["revisit"]["arms"]["mono_native"]
                for ep in episodes
                if ep["roles"]["revisit"]["arms"]["mono_native"]["reached"]
                == reached_bucket]
        key = "native_revisit_success" if reached_bucket else \
            "native_revisit_failure"
        alignment[key] = {
            "n": len(rows),
            "ever_aligned_within_45deg": sum(
                r["ever_aligned_within_45deg"] for r in rows),
            "mean_max_abs_cumulative_turn_deg": (
                sum(r["max_abs_cumulative_turn_deg"] for r in rows)
                / len(rows) if rows else None),
            "mean_aligned_step_fraction": (
                sum(r["aligned_step_fraction"] for r in rows)
                / len(rows) if rows else None),
        }
    return {
        "strata_table": strata_table,
        "revisit_failure_taxonomy": failure_taxonomy,
        "native_revisit_failures": sorted(
            native_revisit_failures, key=lambda r: r["label"]),
        "native_revisit_alignment": alignment,
    }


def validate_against_summary(strata_table: dict, summary: dict) -> None:
    """The Novel strata must reproduce the sealed summary exactly."""
    sealed = summary["initial_path_direction_strata"]["novel"]
    for stratum, block in sealed.items():
        for arm, arm_row in block["arms"].items():
            cell = strata_table["novel"][stratum][arm]
            if (cell["n"], cell["successes"]) != (arm_row["n"],
                                                  arm_row["successes"]):
                raise AssertionError(
                    f"novel/{stratum}/{arm}: recomputed "
                    f"{cell['n']}/{cell['successes']} != sealed "
                    f"{arm_row['n']}/{arm_row['successes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pulled-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    labels = sorted(
        p.name for p in
        (args.pulled_root / "evaluation_natural_direction").iterdir()
        if p.is_dir())
    if len(labels) != 28:
        raise AssertionError(f"expected 28 histories, found {len(labels)}")
    episodes = [load_episode(args.pulled_root, label) for label in labels]
    report = aggregate(episodes)

    summary = json.loads(
        (args.pulled_root
         / "hm3d_fullmono_mixed_role_summary.json").read_text())
    validate_against_summary(report["strata_table"], summary)
    report["validation"] = "novel strata reproduce sealed summary exactly"
    report["analysis_label"] = "posthoc_diagnostic"
    report["source_run"] = summary.get("scope")
    report["per_episode"] = episodes

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "native_revisit_failure_analysis.json"
    out_path.write_text(json.dumps(report, indent=1) + "\n")

    print(f"wrote {out_path}\n")
    for role in ROLES:
        print(f"=== {role}: SR by initial-path direction stratum ===")
        for stratum in ("front", "lateral", "back"):
            cells = report["strata_table"][role][stratum]
            n = cells["mono_native"]["n"]
            row = "  ".join(
                f"{arm.removeprefix('mono_')}: "
                f"{cells[arm]['successes']}/{n}" for arm in ARMS)
            print(f"  {stratum:8s} (n={n:2d})  {row}")
    print("\n=== native revisit failure taxonomy ===")
    print(json.dumps(report["revisit_failure_taxonomy"]["mono_native"],
                     indent=1))
    print("\n=== native revisit heading alignment ===")
    print(json.dumps(report["native_revisit_alignment"], indent=1))


if __name__ == "__main__":
    main()
