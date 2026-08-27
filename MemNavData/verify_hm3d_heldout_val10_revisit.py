#!/usr/bin/env python3
"""Independent raw-file recount for the HM3D held-out val10 Revisit transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ARMS = (
    "native",
    "raw_fixed_oracle_role",
    "geometry_router",
    "certified_relocalization",
)
CONTRASTS = {
    "certified_minus_native": ("native", "certified_relocalization"),
    "certified_minus_raw_fixed_oracle_role": (
        "raw_fixed_oracle_role", "certified_relocalization"),
    "certified_minus_geometry": (
        "geometry_router", "certified_relocalization"),
    "raw_fixed_oracle_role_minus_native": (
        "native", "raw_fixed_oracle_role"),
    "geometry_minus_native": ("native", "geometry_router"),
}
MANIFEST_SCHEMA_V1 = "hm3d_heldout_val10_causal_revisit_manifest_v1_20260816"
MANIFEST_SCHEMA_V2 = "hm3d_heldout_val10_causal_revisit_manifest_v2_20260816"
REPORT_SCHEMA_V1 = "hm3d_heldout_val10_revisit_summary_v1_20260816"
REPORT_SCHEMA_V2 = "hm3d_heldout_val10_revisit_summary_v2_20260816"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: Any) -> bool:
    if value in (None, ""):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes"}:
        return True
    parsed = float(value)
    require(math.isfinite(parsed), f"non-finite truth value: {value!r}")
    return parsed > 0.5


def integer(value: Any) -> int:
    parsed = float(value)
    require(math.isfinite(parsed) and parsed.is_integer(),
            f"invalid integer: {value!r}")
    return int(parsed)


def exact_p(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value)
               for value in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def read_rows(path: Path, expected: list[str]) -> dict[str, dict[str, str]]:
    require(path.is_file(), f"missing metric: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require([row.get("episode") for row in rows] == expected,
            f"episode identity/order differs: {path}")
    return {str(row["episode"]): row for row in rows}


def paired(
    left: dict[tuple[str, str], dict[str, bool]],
    right: dict[tuple[str, str], dict[str, bool]],
    *,
    conditional: bool,
) -> dict[str, Any]:
    keys = sorted(left)
    require(keys == sorted(right), "paired result populations differ")
    target = "b" if conditional else "joint"
    eligible = []
    gains = losses = both = neither = 0
    for key in keys:
        require(left[key]["a"] == right[key]["a"],
                f"Goal-A differs for {key}")
        if conditional and not left[key]["a"]:
            continue
        eligible.append(key)
        lval, rval = left[key][target], right[key][target]
        if lval and rval:
            both += 1
        elif rval:
            gains += 1
        elif lval:
            losses += 1
        else:
            neither += 1
    return {
        "eligible": len(eligible),
        "left_successes": sum(left[key][target] for key in eligible),
        "right_successes": sum(right[key][target] for key in eligible),
        "both_success": both,
        "right_only_gain": gains,
        "left_only_loss": losses,
        "neither_success": neither,
        "mcnemar_exact_two_sided_p": exact_p(gains, losses),
    }


def verify(
    manifest_path: Path,
    run_root: Path,
    report_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_schema = manifest.get("schema_version")
    report_schema = report.get("schema_version")
    require(manifest_schema in {MANIFEST_SCHEMA_V1, MANIFEST_SCHEMA_V2},
            "manifest schema differs")
    expected_report_schema = (
        REPORT_SCHEMA_V1 if manifest_schema == MANIFEST_SCHEMA_V1
        else REPORT_SCHEMA_V2)
    require(report_schema == expected_report_schema,
            "report schema differs")
    require(manifest.get("scene_count") == 10,
            "selected scene population is not ten")
    if manifest_schema == MANIFEST_SCHEMA_V1:
        require(manifest.get("episode_count") == 40,
                "V1 manifest population is not 40 episodes")
    else:
        require(manifest.get("episode_count") == 36 and
                manifest.get("constructible_scene_count") == 9 and
                manifest.get("evaluation_scene_indices") ==
                [0, 1, 2, 3, 4, 5, 6, 7, 9],
                "V2 construction-attrition population differs")
        attrition = manifest.get("construction_attrition", {})
        require(attrition.get("target_met") is False and
                attrition.get("navigation_outcomes_read") is False and
                len(attrition.get("receipts", [])) == 1,
                "V2 construction attrition is invalid")
    require(manifest.get("audit", {}).get("no_mp3d_evaluation") is True and
            report.get("no_mp3d_evaluation") is True,
            "no-MP3D guard absent")
    require(report.get("manifest_sha256") == sha256_file(manifest_path),
            "report references a different manifest")

    scenes = [str(value) for value in manifest["scenes"]]
    results: dict[str, dict[tuple[str, str], dict[str, bool]]] = {
        arm: {} for arm in ARMS}
    requests = accepted = abstained = failures = 0
    takeover_episodes = fallback_episodes = 0
    fallback_mismatches: list[dict[str, str]] = []
    constructible_scenes = 0
    for index, scene in enumerate(scenes):
        expected = [str(row["episode"])
                    for row in manifest["episodes"][scene]]
        require(len(expected) in {0, 4}, f"unbalanced scene: {scene}")
        if not expected:
            require(manifest_schema == MANIFEST_SCHEMA_V2 and index == 8,
                    f"unexpected empty scene: {scene}")
            continue
        constructible_scenes += 1
        scene_root = run_root / "scenes" / f"{index:02d}_{scene}"
        trace_rows = read_rows(scene_root / "trace_source" / "metric.csv",
                               expected)
        arm_rows = {arm: read_rows(scene_root / arm / "metric.csv", expected)
                    for arm in ARMS}
        for episode in expected:
            trace_path = (scene_root / "trace_source" /
                          f"{episode}_leg1_trace.json")
            trace_sha = sha256_file(trace_path)
            trace_a = truth(trace_rows[episode].get("reached_A"))
            require(trace_rows[episode].get("leg1_trace_sha256") == trace_sha,
                    f"trace receipt differs: {scene}/{episode}")
            for arm in ARMS:
                row = arm_rows[arm][episode]
                a_value = truth(row.get("reached_A"))
                b_value = truth(row.get("reached_B"))
                require(a_value == trace_a,
                        f"{arm}: Goal-A differs: {scene}/{episode}")
                require(row.get("leg1_trace_sha256") == trace_sha,
                        f"{arm}: trace SHA differs: {scene}/{episode}")
                steps_b = integer(row.get("steps_B"))
                require(a_value or (not b_value and steps_b == 0),
                        f"{arm}: Goal-B ran after Goal-A failure")
                results[arm][(scene, episode)] = {
                    "a": a_value,
                    "b": b_value,
                    "joint": a_value and b_value,
                }

            certified = arm_rows["certified_relocalization"][episode]
            req = integer(certified["certified_relocalization_request_count"])
            acc = integer(certified["certified_relocalization_accept_count"])
            abst = integer(certified["revisit_adapter_abstain_plan_count"])
            fail = integer(
                certified["certified_relocalization_runtime_failure_count"])
            plans_path = (scene_root / "certified_relocalization" /
                          f"{episode}_plans.json")
            plans = json.loads(plans_path.read_text(encoding="utf-8"))["legB"]
            require(req == len(plans),
                    f"certificate request/plan count differs: {scene}/{episode}")
            counted_accepts = sum(
                plan.get("certified_relocalization_accepted") is True
                for plan in plans)
            require(acc == counted_accepts,
                    f"certificate accept count differs: {scene}/{episode}")
            requests += req
            accepted += acc
            abstained += abst
            failures += fail
            if trace_a:
                require(req > 0 and fail == 0,
                        f"certificate missing/failed: {scene}/{episode}")
                require(acc in {0, len(plans)},
                        f"certificate decision not atomic: {scene}/{episode}")
                if acc:
                    takeover_episodes += 1
                else:
                    fallback_episodes += 1
                    native = arm_rows["native"][episode]
                    changed = [field for field in (
                        "reached_B", "steps_B", "termination_reason_B",
                        "len_B", "final_dist_B", "blocked_steps_B")
                        if certified.get(field) != native.get(field)]
                    if changed:
                        fallback_mismatches.append({
                            "scene": scene, "episode": episode,
                            "fields": ",".join(changed)})
            else:
                require(req == acc == abst == fail == 0,
                        f"certificate ran after Goal-A failure: {scene}/{episode}")

    require(not fallback_mismatches,
            f"fallback behavior differs: {fallback_mismatches}")
    arm_counts: dict[str, Any] = {}
    for arm in ARMS:
        rows = results[arm]
        a_successes = sum(value["a"] for value in rows.values())
        joint = sum(value["joint"] for value in rows.values())
        b_successes = sum(value["b"] for value in rows.values()
                          if value["a"])
        arm_counts[arm] = {
            "episodes": len(rows),
            "goal_a_successes": a_successes,
            "joint_successes": joint,
            "goal_b_successes_given_a": b_successes,
            "goal_b_eligible": a_successes,
        }
        for field, value in arm_counts[arm].items():
            require(report["arms"][arm].get(field) == value,
                    f"report arm count differs: {arm}/{field}")

    contrast_counts: dict[str, Any] = {}
    for name, (left_name, right_name) in CONTRASTS.items():
        contrast_counts[name] = {}
        for label, conditional in (("joint", False), ("conditional_b", True)):
            count = paired(results[left_name], results[right_name],
                           conditional=conditional)
            contrast_counts[name][label] = count
            published = report["contrasts"][name][label]
            for field, value in count.items():
                if isinstance(value, float):
                    require(math.isclose(float(published[field]), value,
                                         rel_tol=0.0, abs_tol=1e-15),
                            f"report contrast differs: {name}/{label}/{field}")
                else:
                    require(published[field] == value,
                            f"report contrast differs: {name}/{label}/{field}")

    certificate_counts = {
        "goal_a_eligible_episodes": takeover_episodes + fallback_episodes,
        "takeover_episodes": takeover_episodes,
        "fallback_episodes": fallback_episodes,
        "exact_native_fallback_episodes": fallback_episodes,
        "fallback_behavior_mismatch_count": 0,
        "requests": requests,
        "accepted_plans": accepted,
        "abstained_plans": abstained,
        "runtime_failures": failures,
    }
    for field, value in certificate_counts.items():
        require(report["certificate_audit"].get(field) == value,
                f"report certificate count differs: {field}")
    require(report.get("scene_count") == constructible_scenes and
            report.get("selected_scene_count") == len(scenes) and
            report.get("constructible_scene_count") == constructible_scenes,
            "report scene accounting differs")
    return {
        "schema_version": (
            "hm3d_heldout_val10_revisit_independent_verification_v1_20260816"
            if manifest_schema == MANIFEST_SCHEMA_V1 else
            "hm3d_heldout_val10_revisit_independent_verification_v2_20260816"),
        "verified": True,
        "scope": "independent raw-file recount; no bootstrap reimplementation",
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "scene_count": constructible_scenes,
        "selected_scene_count": len(scenes),
        "constructible_scene_count": constructible_scenes,
        "episode_count": len(results["native"]),
        "construction_attrition_verified": (
            manifest_schema == MANIFEST_SCHEMA_V2),
        "no_mp3d_evaluation": True,
        "arms": arm_counts,
        "contrasts": contrast_counts,
        "certificate": certificate_counts,
        "fallback_mismatches": fallback_mismatches,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(
        args.manifest.resolve(), args.run_root.resolve(), args.report.resolve())
    out = args.out.resolve()
    write_exclusive(out, payload)
    receipt = out.with_suffix(out.suffix + ".sha256")
    receipt.write_text(f"{sha256_file(out)}  {out.name}\n", encoding="utf-8")
    print(json.dumps({"status": "verified", "verified": True,
                      "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
