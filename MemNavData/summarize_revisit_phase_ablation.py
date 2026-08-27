#!/usr/bin/env python3
"""Audit known-Revisit direct memory against the hard geometry router.

This is an internal architecture ablation on the already-consumed 20-scene
pool.  Both arms must replay the same deterministic Goal-A trace; the only
intended difference on Goal B is whether SIFT/RANSAC may veto memory use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from MemNavData.summarize_expanded_navdp_router_eval import (
    arm_summary,
    load_arm,
    paired_summary,
    require,
)


ARMS = ("geometry_router", "known_revisit_direct")
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_RESAMPLES = 100_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_cluster_interval(
    deltas_by_scene: dict[str, list[float]],
    *,
    seed: int = BOOTSTRAP_SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float]:
    require(bool(deltas_by_scene), "cannot bootstrap an empty scene set")
    require(resamples > 0, "bootstrap resamples must be positive")
    scenes = sorted(deltas_by_scene)
    values = np.asarray([deltas_by_scene[scene] for scene in scenes], dtype=float)
    require(
        values.ndim == 2 and values.shape[1] > 0,
        "each scene must have the same non-empty episode count",
    )
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, resamples, 10_000):
        count = min(10_000, resamples - start)
        indices = rng.integers(0, len(scenes), size=(count, len(scenes)))
        chunks.append(values[indices].mean(axis=(1, 2)))
    low, high = np.quantile(np.concatenate(chunks), [0.025, 0.975])
    return [float(low), float(high)]


def architecture_decision(*, gains: int, losses: int) -> dict[str, Any]:
    """Frozen consumed-pool decision; it never authorizes a paper claim."""

    if gains > 0 and losses == 0:
        branch = "advance_known_revisit_direct_to_fresh_confirmation"
    elif losses > gains or losses >= 2:
        branch = "retain_geometry_as_required_safety_expert"
    else:
        branch = "inconclusive_build_cluster_geometry_ablation"
    return {
        "branch": branch,
        "known_revisit_direct_preferred": (
            branch == "advance_known_revisit_direct_to_fresh_confirmation"
        ),
        "authorize_fresh_nonblind_confirmation": (
            branch == "advance_known_revisit_direct_to_fresh_confirmation"
        ),
        "authorize_blind_eval": False,
        "authorize_paper_claim": False,
    }


def validate_arm_summary(root: Path, arm: str) -> None:
    path = root / "summary.json"
    require(path.is_file(), f"missing summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "episodes": 2,
        "max_steps": 500,
        "exec_horizon": 8,
        "deterministic_plan_seeds": True,
        "trajectory_selector": "server",
        "server_backend": "hybrid_pose",
        "hybrid_route": (
            "memory_geometry" if arm == "geometry_router" else "phase"
        ),
        "leg1_mode": (
            "policy" if arm == "geometry_router" else "shared_trace"
        ),
        "write_leg1_trace": arm == "geometry_router",
    }
    if arm == "geometry_router":
        expected.update(
            {
                "router_visual_floor": 0.88,
                "router_min_matches": 20,
                "router_min_inliers": 12,
                "router_min_inlier_ratio": 0.5,
                "router_confirm_plans": 2,
                "router_verify_top_k": 8,
            }
        )
    for field, wanted in expected.items():
        require(
            summary.get(field) == wanted,
            f"{root}: summary {field} differs from the frozen ablation",
        )


def summarize(manifest_path: Path, run_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenes = manifest["selection"]["selected_scenes"]
    require(len(scenes) == 20, "ablation requires exactly 20 scene clusters")
    episodes = {
        scene: [record["episode"] for record in manifest["episodes"][scene]]
        for scene in scenes
    }
    require(
        all(values == ["episode_0000", "episode_0001"] for values in episodes.values()),
        "ablation requires the frozen two episodes per scene",
    )
    expected_keys = {
        (scene, episode) for scene in scenes for episode in episodes[scene]
    }
    rows = {arm: {} for arm in ARMS}
    for index, scene in enumerate(scenes):
        scene_root = run_root / "scenes" / f"{index:02d}_{scene}"
        for arm in ARMS:
            validate_arm_summary(scene_root / arm, arm)
            rows[arm].update(load_arm(scene_root, arm, scene))
    for arm in ARMS:
        require(set(rows[arm]) == expected_keys, f"{arm} rows differ from manifest")

    paired = paired_summary(
        "geometry_router",
        "known_revisit_direct",
        rows["geometry_router"],
        rows["known_revisit_direct"],
        expected_keys,
    )
    gains = int(paired["outcomes"]["right_only_joint_success"])
    losses = int(paired["outcomes"]["left_only_joint_success"])
    deltas_by_scene = {
        scene: [
            float(rows["known_revisit_direct"][(scene, episode)]["joint"])
            - float(rows["geometry_router"][(scene, episode)]["joint"])
            for episode in episodes[scene]
        ]
        for scene in scenes
    }
    paired["scene_cluster_bootstrap"] = {
        "seed": BOOTSTRAP_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "risk_difference_95": scene_cluster_interval(deltas_by_scene),
    }
    return {
        "scope": (
            "consumed 20-scene known-Revisit architecture ablation; "
            "not unseen confirmation"
        ),
        "question": (
            "Given the benchmark-declared Revisit phase, is RANSAC a useful "
            "safety expert or an over-conservative activation veto?"
        ),
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected_keys),
            "training_scene_overlap": sorted(
                set(scenes) & set(manifest["training_scenes"])
            ),
            "manifest_sha256": sha256_file(manifest_path),
            "source_inputs_sha256": sha256_file(run_root / "source_inputs.sha256"),
            "shared_trace_and_seed_contract": True,
        },
        "arms": {
            arm: arm_summary([rows[arm][key] for key in sorted(expected_keys)])
            for arm in ARMS
        },
        "paired_direct_minus_geometry": paired,
        "decision": architecture_decision(gains=gains, losses=losses),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    return args


def main() -> None:
    args = parse_args()
    report = summarize(args.manifest, args.run_root)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
