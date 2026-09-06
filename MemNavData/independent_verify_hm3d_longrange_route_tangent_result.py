#!/usr/bin/env python3
"""Independent verifier for the fresh long-range route-tangent result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


VERIFY_SCHEMA = "hm3d_longrange_route_tangent_result_verification_v1_20260903"
SUMMARY_SCHEMA = "hm3d_longrange_route_tangent_result_v1_20260903"
COMPLETION_SCHEMA = (
    "hm3d_longrange_route_tangent_history_result_v1_20260903"
)
ARMS = (
    "mono_native", "mono_cec_endpoint", "mono_cec_route_tangent",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index)
               for index in range(min(gains, losses) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def verify(*, population: Path, run_root: Path, summary_path: Path) -> dict:
    manifest_path = population / "role_pairs/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    require(summary.get("schema_version") == SUMMARY_SCHEMA,
            "summary schema changed")
    require(summary.get("benchmark_manifest_sha256")
            == sha256_file(manifest_path),
            "summary references another benchmark")
    receipt = summary_path.with_name(summary_path.name + ".sha256")
    require(receipt.is_file()
            and receipt.read_text().split()
            == [sha256_file(summary_path), summary_path.name],
            "summary receipt is invalid")

    require(len(manifest["episodes"]) == 23,
            "fresh population size changed")
    scene_occurrences = {}
    balanced_indices = set()
    for index, item in enumerate(manifest["episodes"]):
        scene = str(item["scene"])
        count = int(scene_occurrences.get(scene, 0))
        if count < 2:
            balanced_indices.add(index)
        scene_occurrences[scene] = count + 1
    require(len(scene_occurrences) == 8 and len(balanced_indices) == 15,
            "balanced sensitivity population changed")

    successes = {arm: 0 for arm in ARMS}
    gains = losses = 0
    balanced_gains = balanced_losses = 0
    geometry_stops = 0
    completion_hashes = []
    for index, item in enumerate(manifest["episodes"]):
        label = f"{index:03d}_{item['scene']}_{item['episode']}"
        root = run_root / "evaluation" / label
        completion_path = root / "completion.json"
        completion_receipt = root / "completion.json.sha256"
        require(completion_receipt.is_file()
                and completion_receipt.read_text().split()
                == [sha256_file(completion_path), "completion.json"],
                f"{index}: invalid completion receipt")
        result = json.loads(completion_path.read_text())
        require(result.get("schema_version") == COMPLETION_SCHEMA,
                f"{index}: completion schema changed")
        require(result["candidate_identity_sha256"]
                == item["candidate_identity_sha256"]
                and result["prefix_equality"] is True
                and result["initial_cec_proof_equal_between_methods"] is True,
                f"{index}: paired identity/proof contract failed")
        for arm in ARMS:
            outcome = int(result["outcomes"][arm])
            final_3d = float(result["final_goal_3d_dist_m"][arm])
            require(outcome == int(final_3d < 1.0),
                    f"{index}/{arm}: 3-D success mismatch")
            successes[arm] += outcome
        treatment = int(result["outcomes"]["mono_cec_route_tangent"])
        reference = int(result["outcomes"]["mono_cec_endpoint"])
        gains += treatment == 1 and reference == 0
        losses += treatment == 0 and reference == 1
        if index in balanced_indices:
            balanced_gains += treatment == 1 and reference == 0
            balanced_losses += treatment == 0 and reference == 1
        geometry_stops += int(result["geometry_stream_stop_plans"] > 0)

        tangent_payloads = list(
            (root / "mono_cec_route_tangent").glob("*_plans.json"))
        require(len(tangent_payloads) == 1,
                f"{index}: route arm has an unexpected plan-file count")
        payload = json.loads(tangent_payloads[0].read_text())
        require(payload.get("analysis_role_not_forwarded") is True,
                f"{index}: route arm forwarded the role")
        stop_seen = False
        for plan_index, plan in enumerate(payload["query_leg"]):
            if plan.get("geometry_stream_stop") is True:
                require(plan_index == len(payload["query_leg"]) - 1
                        and plan.get("geometry_stream_controller_called")
                        is False
                        and plan.get(
                            "geometry_stream_native_fallback_executed")
                        is False
                        and plan.get(
                            "geometry_stream_endpoint_fallback_executed")
                        is False,
                        f"{index}: geometry stop executed hidden control")
                stop_seen = True
            if plan.get("certified_relocalization_accepted") is True:
                require(plan.get("local_tangent_evaluator_pose_consumed")
                        is False
                        and plan.get("local_tangent_habitat_path_consumed")
                        is False
                        and plan.get(
                            "local_tangent_executor_odometry_consumed")
                        is False
                        and plan.get(
                            "local_tangent_metric_depth_sensor_consumed")
                        is False,
                        f"{index}: route arm consumed privileged geometry")
                require(plan.get("local_tangent_distance_regime_present")
                        is False
                        and plan.get("local_tangent_stuck_trigger_present")
                        is False
                        and plan.get(
                            "local_tangent_native_fallback_available")
                        is False
                        and plan.get(
                            "local_tangent_endpoint_fallback_available")
                        is False,
                        f"{index}: route arm gained a hidden branch")
        require(stop_seen == (result["geometry_stream_stop_plans"] > 0),
                f"{index}: geometry stop accounting changed")
        completion_hashes.append(sha256_file(completion_path))

    require(successes == summary["successes"],
            "independent success totals differ from summary")
    primary = summary["primary_contrast"]
    require(int(primary["gains"]) == gains
            and int(primary["losses"]) == losses
            and math.isclose(
                float(primary["risk_difference"]),
                (gains - losses) / len(manifest["episodes"]),
                abs_tol=1e-12)
            and math.isclose(
                float(primary["exact_mcnemar_two_sided_p"]),
                exact_mcnemar(gains, losses), abs_tol=1e-12),
            "independent primary contrast differs from summary")
    require(int(summary["geometry_stream_stop_episodes"])
            == geometry_stops,
            "geometry stop total differs from summary")
    balanced = summary["scene_balanced_max2_sensitivity"]
    require(int(balanced["n"]) == 15
            and int(balanced["gains"]) == balanced_gains
            and int(balanced["losses"]) == balanced_losses
            and math.isclose(
                float(balanced["risk_difference"]),
                (balanced_gains - balanced_losses) / 15,
                abs_tol=1e-12)
            and math.isclose(
                float(balanced["exact_mcnemar_two_sided_p"]),
                exact_mcnemar(balanced_gains, balanced_losses),
                abs_tol=1e-12),
            "independent balanced sensitivity differs from summary")
    return {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "histories": len(manifest["episodes"]),
        "scene_clusters": len({row["scene"] for row in manifest["episodes"]}),
        "successes": successes,
        "primary_gains": gains,
        "primary_losses": losses,
        "primary_exact_mcnemar_two_sided_p": exact_mcnemar(gains, losses),
        "balanced_histories": 15,
        "balanced_gains": balanced_gains,
        "balanced_losses": balanced_losses,
        "geometry_stream_stop_episodes": geometry_stops,
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "completion_hashes_sha256": hashlib.sha256(
            json.dumps(completion_hashes, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    result = verify(
        population=args.population.resolve(), run_root=args.run_root.resolve(),
        summary_path=args.summary.resolve(),
    )
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
