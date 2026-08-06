#!/usr/bin/env python3
"""Fail-closed paired summary for the three Novel-B upper-bound arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


EXPECTED_MANIFEST_SHA256 = (
    "55096038d0493723d2280fe9abbcb2ea9ea7f2059c50b6eb08d02f560cdb281b"
)
EXPECTED_NAVDP_CHECKPOINT_SHA256 = (
    "3bb3ad4ab241e857bb57a4021cc6aab76d5263e81fbf80298d579053ef011947"
)
EXPECTED_BASE_SIF_BYTES = 10035580928
EXPECTED_BASE_SIF_HEAD_SHA256 = (
    "990c1377491fbf58c1d7d37bcf8043249826bc7a094eacc4c7a443b9905126c7"
)
PROTOCOL = "novel_b_upper_bound_v1"
SCHEMA_VERSION = 1
ARMS = (
    "native_imagegoal",
    "oracle_short_1p25m",
    "oracle_final_point",
)
ARM_SUBGOAL_METRES = {
    "native_imagegoal": None,
    "oracle_short_1p25m": 1.25,
    "oracle_final_point": 100.0,
}
PAIRWISE = (
    ("native_imagegoal", "oracle_short_1p25m"),
    ("native_imagegoal", "oracle_final_point"),
    ("oracle_short_1p25m", "oracle_final_point"),
)
REQUIRED_METRIC_FIELDS = {
    "schema_version",
    "protocol",
    "scene",
    "episode",
    "seed",
    "arm",
    "server_backend",
    "navdp_stop_threshold",
    "goal_A_controller",
    "goal_B_controller",
    "oracle_subgoal_m",
    "deterministic_plan_seeds",
    "navdp_goal_switch_reset",
    "success_dist_m",
    "max_steps",
    "exec_horizon",
    "reached_A",
    "reached_B",
    "B_attempted",
    "spl_A",
    "spl_B",
    "geo_A",
    "geo_B",
    "len_A",
    "len_B",
    "len_B_at_reach",
    "steps_A",
    "steps_B",
    "final_dist_A",
    "final_dist_B",
    "goal_A_plan_count",
    "goal_B_plan_count",
    "goal_a_sha256",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def strict_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("0", "false"):
            return False
        if normalized in ("1", "true"):
            return True
    raise RuntimeError(f"{field} must be an exact boolean/0/1 value")


def finite_float(value: Any, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{field} must be numeric") from exc
    require(math.isfinite(converted), f"{field} must be finite")
    return converted


def exact_int(value: Any, field: str) -> int:
    converted = finite_float(value, field)
    require(converted.is_integer(), f"{field} must be an integer")
    return int(converted)


def optional_float(value: Any, field: str) -> float | None:
    if value is None or value == "":
        return None
    return finite_float(value, field)


def mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def wilson(successes: int, total: int,
           z: float = 1.959963984540054) -> list[float | None]:
    if total == 0:
        return [None, None]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        rate * (1.0 - rate) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def exact_sign_p(right_only: int, discordant: int) -> float:
    if discordant == 0:
        return 1.0
    tail = min(right_only, discordant - right_only)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * probability)


def load_frozen_manifest(path: Path) -> tuple[dict, str]:
    require(path.is_file(), f"missing manifest: {path}")
    actual_sha = file_sha256(path)
    require(
        actual_sha == EXPECTED_MANIFEST_SHA256,
        "frozen three-leg manifest SHA256 mismatch",
    )
    manifest = json.loads(path.read_text())
    require(manifest.get("schema_version") == 1, "manifest schema changed")
    scenes = manifest.get("selection", {}).get("selected_scenes")
    require(isinstance(scenes, list) and len(scenes) == 10,
            "frozen scene selection changed")
    require(len(set(scenes)) == len(scenes), "manifest scenes are duplicated")
    require(set(manifest.get("episodes", {})) == set(scenes),
            "manifest episode scene keys changed")
    evaluation = manifest.get("evaluation", {})
    require(evaluation.get("episodes_per_scene") == 1,
            "episodes-per-scene changed")
    require(evaluation.get("base_seed") == 20260803,
            "base seed changed")
    require(evaluation.get("success_distance_m") == 1.0,
            "success distance changed")
    require(evaluation.get("max_steps_per_leg") == 1200,
            "max-steps protocol changed")
    require(evaluation.get("execution_horizon") == 8,
            "execution horizon changed")
    require(
        evaluation.get("goal_roles")
        == {"A": "novel", "B": "novel", "C": "revisit"},
        "goal roles changed",
    )
    for scene in scenes:
        records = manifest["episodes"][scene]
        require(len(records) == 1, f"{scene} must contain exactly one episode")
        require(
            isinstance(records[0].get("episode"), str),
            f"{scene} episode ID is invalid",
        )
    return manifest, actual_sha


def expected_records(manifest: dict, mode: str) -> list[tuple[int, str, str]]:
    require(mode in ("smoke", "full"), f"unsupported mode: {mode}")
    scenes = manifest["selection"]["selected_scenes"]
    selected = scenes[:1] if mode == "smoke" else scenes
    return [
        (index, scene, record["episode"])
        for index, scene in enumerate(scenes)
        if scene in selected
        for record in manifest["episodes"][scene]
    ]


def verify_plan_seeds(
    plans: Any,
    label: str,
    *,
    episode_seed: int,
    leg_index: int,
) -> None:
    require(isinstance(plans, list), f"{label} plans must be a list")
    for index, plan in enumerate(plans):
        require(isinstance(plan, dict), f"{label} plan {index} is not an object")
        requested = plan.get("requested_diffusion_seed")
        echoed = plan.get("diffusion_seed")
        require(
            requested is not None and echoed is not None,
            f"{label} plan {index} is missing a deterministic seed",
        )
        requested_seed = exact_int(requested, f"{label} requested seed")
        require(
            requested_seed == exact_int(echoed, f"{label} echoed seed"),
            f"{label} plan {index} diffusion seed mismatch",
        )
        expected_seed = episode_seed * 100_000 + leg_index * 10_000 + index
        require(
            requested_seed == expected_seed,
            f"{label} plan {index} does not use the frozen seed mapping",
        )


def load_arm(
    scene_root: Path,
    arm: str,
    scene: str,
    expected_episodes: set[str],
    *,
    expected_seed: int,
) -> dict[tuple[str, str], dict]:
    arm_root = scene_root / arm
    metric_path = arm_root / "metric.csv"
    summary_path = arm_root / "summary.json"
    require(metric_path.is_file(), f"missing metric file: {metric_path}")
    require(summary_path.is_file(), f"missing arm summary: {summary_path}")
    declared_summary = json.loads(summary_path.read_text())
    with metric_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"empty metric header: {metric_path}")
        require(
            set(reader.fieldnames) == REQUIRED_METRIC_FIELDS,
            f"metric schema differs from the frozen schema: {metric_path}",
        )
        metrics = list(reader)
    require(
        {row.get("episode") for row in metrics} == expected_episodes,
        f"{arm} episode rows differ from the manifest for {scene}",
    )

    output: dict[tuple[str, str], dict] = {}
    for metric in metrics:
        episode = metric["episode"]
        label = f"{arm} {scene} {episode}"
        artifact_path = arm_root / f"{episode}_audit.json"
        require(artifact_path.is_file(), f"missing audit artifact: {artifact_path}")
        artifact = json.loads(artifact_path.read_text())
        require(artifact.get("schema_version") == SCHEMA_VERSION,
                f"{label} artifact schema mismatch")
        require(artifact.get("protocol") == PROTOCOL,
                f"{label} artifact protocol mismatch")
        require(artifact.get("scene") == scene, f"{label} artifact scene mismatch")
        require(artifact.get("episode") == episode,
                f"{label} artifact episode mismatch")
        require(artifact.get("arm") == arm, f"{label} artifact arm mismatch")

        require(exact_int(metric["schema_version"], f"{label} schema")
                == SCHEMA_VERSION, f"{label} metric schema mismatch")
        require(metric["protocol"] == PROTOCOL, f"{label} metric protocol mismatch")
        require(metric["scene"] == scene, f"{label} metric scene mismatch")
        require(metric["arm"] == arm, f"{label} metric arm mismatch")
        require(metric["server_backend"] == "navdp",
                f"{label} did not use native NavDP server")
        require(
            finite_float(
                metric["navdp_stop_threshold"], f"{label} NavDP stop threshold"
            ) == -0.5,
            f"{label} NavDP stop threshold changed",
        )
        require(metric["goal_A_controller"] == "native_imagegoal",
                f"{label} Goal A is not native ImageGoal")
        require(metric["goal_B_controller"] == arm,
                f"{label} Goal-B controller label mismatch")
        require(strict_bool(metric["deterministic_plan_seeds"],
                            f"{label} deterministic seeds"),
                f"{label} deterministic plan seeds are disabled")
        require(metric["navdp_goal_switch_reset"] == "carry",
                f"{label} did not carry Goal-A short memory")
        require(finite_float(metric["success_dist_m"], f"{label} success radius")
                == 1.0, f"{label} success radius changed")
        require(exact_int(metric["max_steps"], f"{label} max steps") == 1200,
                f"{label} max steps changed")
        require(exact_int(metric["exec_horizon"], f"{label} horizon") == 8,
                f"{label} execution horizon changed")
        actual_subgoal = optional_float(
            metric["oracle_subgoal_m"], f"{label} oracle subgoal"
        )
        require(actual_subgoal == ARM_SUBGOAL_METRES[arm],
                f"{label} oracle subgoal changed")

        seed = exact_int(metric["seed"], f"{label} seed")
        require(
            seed == expected_seed,
            f"{label} seed differs from the frozen manifest seed",
        )
        require(seed == exact_int(artifact.get("seed"), f"{label} artifact seed"),
                f"{label} seed differs between metric and artifact")
        goal_a = artifact.get("goal_a")
        goal_b = artifact.get("goal_b")
        require(isinstance(goal_a, dict), f"{label} Goal-A record is missing")
        require(isinstance(goal_b, dict), f"{label} Goal-B record is missing")
        actual_goal_a_sha = canonical_sha256(goal_a)
        require(artifact.get("goal_a_sha256") == actual_goal_a_sha,
                f"{label} artifact Goal-A SHA mismatch")
        require(metric["goal_a_sha256"] == actual_goal_a_sha,
                f"{label} metric Goal-A SHA mismatch")

        reached_a = strict_bool(metric["reached_A"], f"{label} reached_A")
        reached_b = strict_bool(metric["reached_B"], f"{label} reached_B")
        attempted_b = strict_bool(metric["B_attempted"], f"{label} B_attempted")
        require(strict_bool(goal_a.get("reached"), f"{label} Goal-A reached")
                == reached_a, f"{label} Goal-A outcome mismatch")
        require(strict_bool(goal_b.get("reached"), f"{label} Goal-B reached")
                == reached_b, f"{label} Goal-B outcome mismatch")
        require(strict_bool(goal_b.get("attempted"), f"{label} Goal-B attempted")
                == attempted_b, f"{label} Goal-B attempted mismatch")
        require(attempted_b == reached_a,
                f"{label} Goal B must be attempted exactly when Goal A succeeds")
        require(not reached_b or attempted_b,
                f"{label} Goal B succeeded without being attempted")

        steps_a = exact_int(metric["steps_A"], f"{label} steps_A")
        steps_b = exact_int(metric["steps_B"], f"{label} steps_B")
        require(steps_a == exact_int(goal_a.get("steps"), f"{label} Goal-A steps"),
                f"{label} Goal-A steps mismatch")
        require(steps_b == exact_int(goal_b.get("steps"), f"{label} Goal-B steps"),
                f"{label} Goal-B steps mismatch")
        plans_a = goal_a.get("plans")
        plans_b = goal_b.get("plans")
        require(isinstance(plans_a, list), f"{label} Goal-A plans are invalid")
        require(isinstance(plans_b, list), f"{label} Goal-B plans are invalid")
        require(len(plans_a) == exact_int(metric["goal_A_plan_count"],
                                          f"{label} Goal-A plan count"),
                f"{label} Goal-A plan count mismatch")
        require(len(plans_b) == exact_int(metric["goal_B_plan_count"],
                                          f"{label} Goal-B plan count"),
                f"{label} Goal-B plan count mismatch")
        verify_plan_seeds(
            plans_a, f"{label} Goal A", episode_seed=seed, leg_index=0
        )
        verify_plan_seeds(
            plans_b, f"{label} Goal B", episode_seed=seed, leg_index=1
        )
        for plan_index, plan in enumerate(plans_b):
            require(plan.get("trajectory_selector") == "server",
                    f"{label} Goal-B plan {plan_index} used an oracle selector")
            if arm == "native_imagegoal":
                require("oracle_subgoal_world" not in plan,
                        f"{label} native Goal-B plan contains an oracle subgoal")
            else:
                require(
                    plan.get("pose_controller")
                    == "oracle_habitat_geodesic_image_point_mix",
                    f"{label} Goal-B oracle controller mismatch",
                )
                require(
                    finite_float(
                        plan.get("oracle_subgoal_distance_cap_m"),
                        f"{label} Goal-B oracle cap",
                    ) == ARM_SUBGOAL_METRES[arm],
                    f"{label} Goal-B per-plan oracle cap mismatch",
                )
                if arm == "oracle_final_point":
                    require(
                        plan.get("oracle_subgoal_is_final_endpoint") is True,
                        f"{label} final-point plan did not target the endpoint",
                    )
        if not attempted_b:
            require(steps_b == 0 and not plans_b,
                    f"{label} unattempted Goal B has rollout activity")

        geo = artifact.get("geodesic_m")
        require(isinstance(geo, dict), f"{label} geodesics are missing")
        geo_a = finite_float(metric["geo_A"], f"{label} geo_A")
        geo_b = finite_float(metric["geo_B"], f"{label} geo_B")
        require(geo_a == finite_float(geo.get("A"), f"{label} artifact geo_A"),
                f"{label} Goal-A geodesic mismatch")
        require(geo_b == finite_float(geo.get("B"), f"{label} artifact geo_B"),
                f"{label} Goal-B geodesic mismatch")
        path_a = finite_float(metric["len_A"], f"{label} len_A")
        path_b = finite_float(metric["len_B"], f"{label} len_B")
        path_b_at_reach = optional_float(
            metric["len_B_at_reach"], f"{label} len_B_at_reach"
        )
        final_a = finite_float(metric["final_dist_A"], f"{label} final_dist_A")
        final_b = finite_float(metric["final_dist_B"], f"{label} final_dist_B")
        require(path_a == finite_float(goal_a.get("path_len"),
                                       f"{label} Goal-A path"),
                f"{label} Goal-A path mismatch")
        require(path_b == finite_float(goal_b.get("path_len"),
                                       f"{label} Goal-B path"),
                f"{label} Goal-B path mismatch")
        artifact_path_b_at_reach = optional_float(
            goal_b.get("path_len_at_reach"),
            f"{label} Goal-B artifact path at reach",
        )
        require(path_b_at_reach == artifact_path_b_at_reach,
                f"{label} Goal-B path-at-reach mismatch")
        require(final_a == finite_float(goal_a.get("final_goal_dist_m"),
                                        f"{label} Goal-A final distance"),
                f"{label} Goal-A final distance mismatch")
        require(final_b == finite_float(goal_b.get("final_goal_dist_m"),
                                        f"{label} Goal-B final distance"),
                f"{label} Goal-B final distance mismatch")
        path_a_for_spl = optional_float(
            goal_a.get("path_len_at_reach"),
            f"{label} Goal-A artifact path at reach",
        )
        if path_a_for_spl is None:
            path_a_for_spl = path_a
        path_b_for_spl = path_b_at_reach if path_b_at_reach is not None else path_b
        expected_spl_a = min(1.0, max(0.0, geo_a / max(path_a_for_spl, 1e-6))) \
            * float(reached_a)
        expected_spl_b = min(1.0, max(0.0, geo_b / max(path_b_for_spl, 1e-6))) \
            * float(reached_b)
        spl_a = finite_float(metric["spl_A"], f"{label} spl_A")
        spl_b = finite_float(metric["spl_B"], f"{label} spl_B")
        require(spl_a == expected_spl_a, f"{label} Goal-A SPL mismatch")
        require(spl_b == expected_spl_b, f"{label} Goal-B SPL mismatch")

        key = (scene, episode)
        require(key not in output, f"duplicate result row: {label}")
        output[key] = {
            "scene": scene,
            "episode": episode,
            "seed": seed,
            "goal_a": goal_a,
            "goal_a_sha256": actual_goal_a_sha,
            "reached_a": reached_a,
            "reached_b": reached_b,
            "attempted_b": attempted_b,
            "spl_a": spl_a,
            "spl_b": spl_b,
            "geo_a": geo_a,
            "geo_b": geo_b,
            "path_a": path_a,
            "path_b": path_b,
            "final_dist_a": final_a,
            "final_dist_b": final_b,
            "steps_a": steps_a,
            "steps_b": steps_b,
        }
    eligible = [row for row in output.values() if row["reached_a"]]
    require(declared_summary.get("schema_version") == SCHEMA_VERSION,
            f"{arm} {scene} summary schema mismatch")
    require(declared_summary.get("protocol") == PROTOCOL,
            f"{arm} {scene} summary protocol mismatch")
    require(declared_summary.get("arm") == arm,
            f"{arm} {scene} summary arm mismatch")
    require(declared_summary.get("episodes") == len(output),
            f"{arm} {scene} summary episode count mismatch")
    require(declared_summary.get("goal_A_successes") == len(eligible),
            f"{arm} {scene} summary Goal-A count mismatch")
    require(declared_summary.get("goal_B_eligible") == len(eligible),
            f"{arm} {scene} summary Goal-B denominator mismatch")
    require(
        declared_summary.get("goal_B_successes")
        == sum(row["reached_b"] for row in eligible),
        f"{arm} {scene} summary Goal-B success count mismatch",
    )
    expected_sr = (
        sum(row["reached_b"] for row in eligible) / len(eligible)
        if eligible
        else None
    )
    require(declared_summary.get("goal_B_sr_given_A") == expected_sr,
            f"{arm} {scene} summary Goal-B SR mismatch")
    return output


def first_difference(left: Any, right: Any, path: str = "goal_a") -> str:
    if type(left) is not type(right):
        return f"{path} type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path} keys {sorted(left)} != {sorted(right)}"
        for key in sorted(left):
            if left[key] != right[key]:
                return first_difference(left[key], right[key], f"{path}.{key}")
        return path
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path} length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            if left_item != right_item:
                return first_difference(
                    left_item, right_item, f"{path}[{index}]"
                )
        return path
    return f"{path}: {left!r} != {right!r}"


def validate_pairing(
    rows: dict[str, dict[tuple[str, str], dict]],
    expected: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    require(set(rows) == set(ARMS), "result arm set changed")
    for arm in ARMS:
        require(set(rows[arm]) == expected,
                f"{arm} result keys differ from selected manifest rows")

    baseline_name = ARMS[0]
    eligible: list[tuple[str, str]] = []
    for key in sorted(expected):
        baseline = rows[baseline_name][key]
        for arm in ARMS[1:]:
            candidate = rows[arm][key]
            require(candidate["seed"] == baseline["seed"],
                    f"paired seed mismatch: {key} {baseline_name} vs {arm}")
            require(candidate["geo_a"] == baseline["geo_a"],
                    f"Goal-A geodesic mismatch: {key} {baseline_name} vs {arm}")
            require(candidate["geo_b"] == baseline["geo_b"],
                    f"Goal-B geodesic mismatch: {key} {baseline_name} vs {arm}")
            if candidate["goal_a_sha256"] != baseline["goal_a_sha256"]:
                difference = first_difference(
                    baseline["goal_a"], candidate["goal_a"]
                )
                raise RuntimeError(
                    f"Goal-A field mismatch: {key} {baseline_name} vs {arm}: "
                    f"{difference}"
                )
            require(candidate["goal_a"] == baseline["goal_a"],
                    f"Goal-A record mismatch despite equal SHA: {key}")
            # These derived fields are intentionally checked independently of
            # the full record so a malformed loader cannot hide a discrepancy.
            for field in (
                "reached_a", "spl_a", "path_a", "final_dist_a", "steps_a"
            ):
                require(candidate[field] == baseline[field],
                        f"Goal-A {field} mismatch: {key} {baseline_name} vs {arm}")
        if all(rows[arm][key]["reached_a"] for arm in ARMS):
            eligible.append(key)
    return eligible


def arm_summary(rows: dict[tuple[str, str], dict],
                eligible: list[tuple[str, str]]) -> dict:
    successes = sum(rows[key]["reached_b"] for key in eligible)
    return {
        "goal_B_given_common_goal_A_success": {
            "eligible": len(eligible),
            "successes": successes,
            "sr": successes / len(eligible) if eligible else None,
            "wilson_95": wilson(successes, len(eligible)),
            "mean_spl": mean([rows[key]["spl_b"] for key in eligible]),
            "mean_final_distance_m": mean([
                rows[key]["final_dist_b"] for key in eligible
            ]),
            "mean_path_length_m": mean([
                rows[key]["path_b"] for key in eligible
            ]),
        }
    }


def paired_b_summary(
    left_name: str,
    right_name: str,
    left: dict[tuple[str, str], dict],
    right: dict[tuple[str, str], dict],
    eligible: list[tuple[str, str]],
) -> dict:
    outcomes = {
        "both_B_success": 0,
        "left_only_B_success": 0,
        "right_only_B_success": 0,
        "neither_B_success": 0,
    }
    episodes = []
    for key in eligible:
        left_success = left[key]["reached_b"]
        right_success = right[key]["reached_b"]
        if left_success and right_success:
            outcome = "both_B_success"
        elif left_success:
            outcome = "left_only_B_success"
        elif right_success:
            outcome = "right_only_B_success"
        else:
            outcome = "neither_B_success"
        outcomes[outcome] += 1
        episodes.append({
            "scene": key[0],
            "episode": key[1],
            "outcome": outcome,
            "left_B_success": left_success,
            "right_B_success": right_success,
        })
    discordant = (
        outcomes["left_only_B_success"] + outcomes["right_only_B_success"]
    )
    return {
        "left": left_name,
        "right": right_name,
        "eligible_common_goal_A_success": len(eligible),
        "outcomes": outcomes,
        "B_sr_delta_right_minus_left": (
            (
                outcomes["right_only_B_success"]
                - outcomes["left_only_B_success"]
            ) / len(eligible)
            if eligible
            else None
        ),
        "mcnemar_exact_two_sided_p": exact_sign_p(
            outcomes["right_only_B_success"], discordant
        ),
        "episodes": episodes,
    }


def summarize_rows(
    rows: dict[str, dict[tuple[str, str], dict]],
    expected: set[tuple[str, str]],
    *,
    mode: str,
    manifest_sha256: str = EXPECTED_MANIFEST_SHA256,
) -> dict:
    eligible = validate_pairing(rows, expected)
    excluded = sorted(expected - set(eligible))
    per_scene = {}
    for scene in sorted({key[0] for key in expected}):
        scene_eligible = [key for key in eligible if key[0] == scene]
        per_scene[scene] = {
            "common_goal_A_success_eligible": len(scene_eligible),
            "arms": {
                arm: {
                    "goal_B_successes": sum(
                        rows[arm][key]["reached_b"] for key in scene_eligible
                    ),
                    "goal_B_sr": (
                        mean([
                            float(rows[arm][key]["reached_b"])
                            for key in scene_eligible
                        ])
                    ),
                }
                for arm in ARMS
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "audit": {
            "status": "ok",
            "mode": mode,
            "manifest_sha256": manifest_sha256,
            "selected_episodes": len(expected),
            "same_live_navdp_server_required_by_runner": True,
            "paired_seed_and_geodesic_match": True,
            "goal_A_full_record_field_match": True,
            "goal_B_denominator": "intersection_of_goal_A_success_across_all_arms",
            "common_goal_A_success_eligible": len(eligible),
            "excluded_goal_A_failures": [
                {"scene": scene, "episode": episode}
                for scene, episode in excluded
            ],
        },
        "arms": {
            arm: arm_summary(rows[arm], eligible) for arm in ARMS
        },
        "pairwise": {
            f"{right}_vs_{left}": paired_b_summary(
                left, right, rows[left], rows[right], eligible
            )
            for left, right in PAIRWISE
        },
        "per_scene": per_scene,
    }


def load_results(
    manifest: dict,
    run_root: Path,
    mode: str,
) -> tuple[dict[str, dict[tuple[str, str], dict]], set[tuple[str, str]]]:
    records = expected_records(manifest, mode)
    expected_seed = exact_int(
        manifest["evaluation"]["base_seed"], "manifest base seed"
    )
    expected = {(scene, episode) for _index, scene, episode in records}
    preflight_path = run_root / "preflight" / "inputs.json"
    require(preflight_path.is_file(), f"missing input preflight: {preflight_path}")
    preflight = json.loads(preflight_path.read_text())
    require(preflight.get("status") == "ok", "input preflight did not pass")
    require(preflight.get("mode") == mode, "input preflight mode mismatch")
    require(preflight.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256,
            "input preflight manifest SHA mismatch")
    require(
        preflight.get("base_manifest_sha256")
        == manifest["base_manifest"]["sha256"],
        "input preflight base-manifest SHA mismatch",
    )
    require(
        preflight.get("navdp_checkpoint_sha256")
        == EXPECTED_NAVDP_CHECKPOINT_SHA256,
        "input preflight NavDP checkpoint SHA mismatch",
    )
    require(preflight.get("policy_training_overlap") == [],
            "policy-training scene leaked into the evaluation")
    environment = preflight.get("environment")
    require(isinstance(environment, dict),
            "input preflight environment provenance is missing")
    container = environment.get("container_image")
    require(isinstance(container, dict),
            "container-image provenance is missing")
    require(container.get("bytes") == EXPECTED_BASE_SIF_BYTES,
            "container-image byte count changed")
    require(
        container.get("head_sha256") == EXPECTED_BASE_SIF_HEAD_SHA256,
        "container-image header SHA256 changed",
    )
    for environment_name in ("navdp", "habitat"):
        record = environment.get(environment_name)
        require(isinstance(record, dict),
                f"{environment_name} environment provenance is missing")
        require(isinstance(record.get("python"), str),
                f"{environment_name} Python version is missing")
        require(isinstance(record.get("packages"), dict)
                and bool(record["packages"]),
                f"{environment_name} package provenance is missing")
    preflight_episodes = preflight.get("episodes")
    require(isinstance(preflight_episodes, list),
            "input preflight episode list is invalid")
    require(
        {
            (record.get("scene"), record.get("episode"))
            for record in preflight_episodes
            if isinstance(record, dict)
        } == expected
        and len(preflight_episodes) == len(expected),
        "input preflight rows differ from selected manifest episodes",
    )
    protocol_path = run_root / "protocol.json"
    require(protocol_path.is_file(), f"missing runner protocol: {protocol_path}")
    runner_protocol = json.loads(protocol_path.read_text())
    require(runner_protocol.get("schema_version") == SCHEMA_VERSION,
            "runner protocol schema changed")
    require(runner_protocol.get("protocol") == PROTOCOL,
            "runner protocol label changed")
    require(runner_protocol.get("mode") == mode, "runner mode mismatch")
    require(
        runner_protocol.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256,
        "runner manifest SHA mismatch",
    )
    require(runner_protocol.get("server_start_count") == 1,
            "runner did not use exactly one NavDP server start")
    require(
        runner_protocol.get("same_live_server_for_all_arms_and_scenes") is True,
        "runner did not attest one live server across all evaluations",
    )
    require(runner_protocol.get("arms_order") == list(ARMS),
            "runner arm order changed")
    require(
        runner_protocol.get("scene_order")
        == [scene for _index, scene, _episode in records],
        "runner scene order changed",
    )
    require(
        isinstance(runner_protocol.get("server_pid"), int)
        and runner_protocol["server_pid"] > 0,
        "runner server PID is invalid",
    )
    require(
        isinstance(runner_protocol.get("server_port"), int)
        and 0 < runner_protocol["server_port"] <= 65535,
        "runner server port is invalid",
    )
    commit = runner_protocol.get("expected_commit")
    require(
        isinstance(commit, str)
        and len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit),
        "runner commit identity is invalid",
    )
    scenes_root = run_root / "scenes"
    require(scenes_root.is_dir(), f"missing scenes output root: {scenes_root}")
    expected_scene_dirs = {
        f"{index:02d}_{scene}" for index, scene, _episode in records
    }
    actual_scene_dirs = {
        path.name for path in scenes_root.iterdir() if path.is_dir()
    }
    require(actual_scene_dirs == expected_scene_dirs,
            "scene output directories differ from selected manifest scenes")

    rows = {arm: {} for arm in ARMS}
    for index, scene, episode in records:
        scene_root = scenes_root / f"{index:02d}_{scene}"
        pairing_path = scene_root / "goal_a_pairing.json"
        require(pairing_path.is_file(),
                f"missing per-scene Goal-A pairing audit: {pairing_path}")
        pairing = json.loads(pairing_path.read_text())
        require(pairing.get("status") == "ok",
                f"per-scene Goal-A pairing failed for {scene}/{episode}")
        require(pairing.get("scene") == scene
                and pairing.get("episode") == episode,
                f"per-scene Goal-A pairing identity mismatch: {scene}")
        require(pairing.get("seed") == expected_seed,
                f"per-scene Goal-A pairing seed mismatch: {scene}")
        require(pairing.get("arms") == list(ARMS),
                f"per-scene Goal-A pairing arm order changed: {scene}")
        for arm in ARMS:
            loaded = load_arm(
                scene_root,
                arm,
                scene,
                {episode},
                expected_seed=expected_seed,
            )
            overlap = set(rows[arm]) & set(loaded)
            require(not overlap, f"duplicate loaded rows for {arm}: {sorted(overlap)}")
            rows[arm].update(loaded)
        key = (scene, episode)
        require(
            pairing.get("goal_a_sha256")
            == rows[ARMS[0]][key]["goal_a_sha256"],
            f"per-scene Goal-A pairing digest mismatch: {scene}/{episode}",
        )
    return rows, expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()

    manifest, manifest_sha = load_frozen_manifest(args.manifest)
    rows, expected = load_results(manifest, args.run_root, args.mode)
    summary = summarize_rows(
        rows,
        expected,
        mode=args.mode,
        manifest_sha256=manifest_sha,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
