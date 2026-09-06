#!/usr/bin/env python3
"""Analyze complete paired fresh route-tangent results after one-time unseal."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.hm3d_longrange_route_tangent_experiment import (
    ARMS,
    PRIMARY_CONTRAST,
)
from MemNavData.mdtec_raw_depth_gate_d import (
    paired_contrast,
    scene_cluster_interval,
)


RESULT_SCHEMA = "hm3d_longrange_route_tangent_result_v1_20260903"
COMPLETION_SCHEMA = (
    "hm3d_longrange_route_tangent_history_result_v1_20260903"
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


def _verify_sidecar(path: Path) -> str:
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file()
            and sidecar.read_text().split() == [digest, path.name],
            f"invalid completion receipt: {path}")
    return digest


def analyze(*, population: Path, run_root: Path) -> dict[str, Any]:
    manifest_path = population / "role_pairs/manifest.json"
    verification_path = population / "independent_verification.json"
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads(verification_path.read_text())
    require(verification.get("verified") is True
            and verification.get("formal_policy_evaluation_authorized") is True
            and verification.get("benchmark_manifest_sha256")
            == sha256_file(manifest_path),
            "population is not independently authorized")
    histories = manifest["episodes"]
    require(len(histories) == 23, "fresh population size changed")

    scene_occurrences = Counter()
    balanced_history_indices = set()
    for index, item in enumerate(histories):
        scene = str(item["scene"])
        if scene_occurrences[scene] < 2:
            balanced_history_indices.add(index)
        scene_occurrences[scene] += 1
    require(len(scene_occurrences) == 8
            and len(balanced_history_indices) == 15,
            "scene-balanced sensitivity population changed")

    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    geometry_stops = 0
    rejection_exactness = Counter()
    for index, item in enumerate(histories):
        label = f"{index:03d}_{item['scene']}_{item['episode']}"
        path = run_root / "evaluation" / label / "completion.json"
        digest = _verify_sidecar(path)
        result = json.loads(path.read_text())
        require(result.get("schema_version") == COMPLETION_SCHEMA,
                f"{index}: completion schema changed")
        require(int(result["history_index"]) == index
                and result["scene"] == item["scene"]
                and result["episode"] == item["episode"]
                and result["candidate_identity_sha256"]
                == item["candidate_identity_sha256"],
                f"{index}: completion identity changed")
        require(result["benchmark_manifest_sha256"]
                == sha256_file(manifest_path),
                f"{index}: completion references another population")
        require(result["arms"] == list(ARMS)
                and set(result["arm_order"]) == set(ARMS)
                and result["prefix_equality"] is True
                and result["initial_cec_proof_equal_between_methods"] is True,
                f"{index}: paired arm contract failed")
        require(result["success_geometry"]
                == "three_dimensional_euclidean"
                and math.isclose(float(result["success_distance_m"]), 1.0),
                f"{index}: success contract changed")
        require(float(result["revisit_vertical_error_m"]) <= 0.5 + 1e-12,
                f"{index}: episode escaped same-floor selection")
        initial = [float(result["initial_geodesic_m"][arm]) for arm in ARMS]
        require(max(initial) - min(initial) <= 1e-9
                and 20.0 <= initial[0] < 30.0,
                f"{index}: arm geodesic or distance stratum changed")
        for arm in ARMS:
            success = int(result["outcomes"][arm])
            final_3d = float(result["final_goal_3d_dist_m"][arm])
            require(success in (0, 1)
                    and success == int(final_3d < 1.0),
                    f"{index}/{arm}: success does not match 3-D distance")
            rows.append({
                "scene": result["scene"],
                "episode": result["episode"],
                "history_index": index,
                "arm": arm,
                "reached": success,
                "initial_geodesic_m": initial[0],
                "final_goal_3d_dist_m": final_3d,
                "final_goal_planar_dist_m": float(
                    result["final_goal_planar_dist_m"][arm]),
                "final_goal_vertical_error_m": float(
                    result["final_goal_vertical_error_m"][arm]),
                "path_len_m": float(result["path_len_m"][arm]),
                "steps": int(result["steps"][arm]),
                "termination_reason": result["termination_reason"][arm],
            })
        geometry_stops += int(result["geometry_stream_stop_plans"] > 0)
        for arm, exact in result["fully_rejected_exact_native"].items():
            rejection_exactness[f"{arm}:{bool(exact)}"] += 1
        ledger.append({
            "history_index": index,
            "scene": item["scene"],
            "episode": item["episode"],
            "completion_sha256": digest,
        })

    by_arm = {arm: [row for row in rows if row["arm"] == arm]
              for arm in ARMS}
    sr = {arm: sum(row["reached"] for row in arm_rows)
          for arm, arm_rows in by_arm.items()}
    treatment, reference = PRIMARY_CONTRAST
    primary = paired_contrast(rows, treatment, reference)
    primary["scene_cluster_bootstrap"] = scene_cluster_interval(
        rows, treatment, reference, seed=20260903, resamples=20000,
    )
    balanced_rows = [
        row for row in rows
        if int(row["history_index"]) in balanced_history_indices
    ]
    balanced = paired_contrast(balanced_rows, treatment, reference)
    balanced["scene_cluster_bootstrap"] = scene_cluster_interval(
        balanced_rows, treatment, reference, seed=20260905,
        resamples=20000,
    )
    versus_native = paired_contrast(
        rows, "mono_cec_route_tangent", "mono_native")
    versus_native["scene_cluster_bootstrap"] = scene_cluster_interval(
        rows, "mono_cec_route_tangent", "mono_native",
        seed=20260904, resamples=20000,
    )
    if (primary["risk_difference"] > 0.0
            and primary["exact_mcnemar_two_sided_p"] <= 0.05
            and primary["scene_cluster_bootstrap"][0] > 0.0
            and balanced["risk_difference"] > 0.0):
        decision = "confirm_route_tangent_for_longrange_extension"
    elif primary["risk_difference"] > 0.0:
        decision = "positive_but_underpowered_no_tuning"
    else:
        decision = "reject_route_tangent_keep_canonical_range_claim"

    return {
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "fresh-history reused-scene HM3D same-floor 20--30 m Revisit"
        ),
        "histories": len(histories),
        "scene_clusters": len({row["scene"] for row in histories}),
        "arms": list(ARMS),
        "successes": sr,
        "success_rates": {
            arm: sr[arm] / len(by_arm[arm]) for arm in ARMS
        },
        "primary_contrast": primary,
        "scene_balanced_max2_sensitivity": balanced,
        "scene_balanced_history_indices": sorted(
            balanced_history_indices),
        "route_tangent_vs_native": versus_native,
        "geometry_stream_stop_episodes": geometry_stops,
        "fully_rejected_exact_native_counts": dict(rejection_exactness),
        "mean_final_3d_distance_m": {
            arm: float(np.mean([
                row["final_goal_3d_dist_m"] for row in by_arm[arm]
            ])) for arm in ARMS
        },
        "mean_realized_path_m": {
            arm: float(np.mean([
                row["path_len_m"] for row in by_arm[arm]
            ])) for arm in ARMS
        },
        "decision": decision,
        "decision_rule": (
            "call confirmed only for positive primary paired risk difference, "
            "exact McNemar p<=0.05, positive scene-cluster CI lower bound, "
            "and positive max-two-per-scene sensitivity; perform no "
            "threshold/radius sweep"
        ),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "population_verification_sha256": sha256_file(verification_path),
        "completion_ledger": ledger,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    result = analyze(
        population=args.population.resolve(), run_root=args.run_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n"
    )
    print(json.dumps({
        "histories": result["histories"],
        "scene_clusters": result["scene_clusters"],
        "successes": result["successes"],
        "primary_contrast": result["primary_contrast"],
        "decision": result["decision"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
