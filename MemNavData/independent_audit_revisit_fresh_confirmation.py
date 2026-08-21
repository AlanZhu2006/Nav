#!/usr/bin/env python3
"""Independent raw-CSV audit of the fresh Revisit confirmation report.

This intentionally imports no project summarizer code.  It recomputes paired
counts, exact McNemar p-values, and scene-cluster intervals from the frozen
manifest and raw rollout receipts, then checks the formal report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ARMS = ("geometry_router", "known_revisit_direct", "native")
CONTRASTS = {
    "direct_minus_geometry": ("geometry_router", "known_revisit_direct"),
    "direct_minus_native": ("native", "known_revisit_direct"),
    "geometry_minus_native": ("native", "geometry_router"),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    require(normalized in {"true", "false", "1", "0", "1.0", "0.0"},
            f"non-boolean receipt: {value!r}")
    return normalized in {"true", "1", "1.0"}


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = min(gains, losses)
    mass = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * mass / (2 ** discordant))


def load_raw(manifest: dict[str, Any], run_root: Path) -> dict[str, dict]:
    scenes = manifest["scenes"]
    per_scene = int(manifest["episodes_per_scene"])
    base_seed = int(manifest["evaluation"]["base_seed"])
    permutations = list(itertools.permutations(ARMS))
    rows: dict[str, dict] = {arm: {} for arm in ARMS}

    for scene_index, scene in enumerate(scenes):
        scene_root = run_root / "scenes" / f"{scene_index:02d}_{scene}"
        contract = json.loads((scene_root / "scene_contract.json").read_text())
        require(contract["scene"] == scene, f"contract scene mismatch: {scene}")
        require(contract["scene_index"] == scene_index,
                f"contract index mismatch: {scene}")
        require(contract["arm_order"] == list(permutations[scene_index % 6]),
                f"arm order mismatch: {scene}")

        expected_episodes = [
            row["episode"] for row in manifest["episodes"][scene]
        ]
        require(expected_episodes == [
            f"episode_{index:04d}" for index in range(per_scene)
        ], f"episode order mismatch: {scene}")
        trace_hashes = {
            episode: sha256(
                scene_root / "trace_source" / f"{episode}_leg1_trace.json")
            for episode in expected_episodes
        }

        for arm in ARMS:
            metric_path = scene_root / arm / "metric.csv"
            with metric_path.open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
            require([row["episode"] for row in metrics] == expected_episodes,
                    f"raw metric identity/order mismatch: {scene}/{arm}")
            for episode_index, metric in enumerate(metrics):
                episode = metric["episode"]
                key = (scene, episode)
                require(key not in rows[arm], f"duplicate raw row: {arm}/{key}")
                require(int(metric["seed"]) == base_seed + episode_index,
                        f"seed mismatch: {arm}/{key}")
                require(truth(metric["deterministic_plan_seeds"]),
                        f"deterministic seeding disabled: {arm}/{key}")
                require(metric["leg1_goal_source"] == "own",
                        f"Goal-A source changed: {arm}/{key}")
                require(metric["leg1_goal_source_episode"] == episode,
                        f"Goal-A episode changed: {arm}/{key}")
                require(metric["leg1_trace_sha256"] == trace_hashes[episode],
                        f"trace SHA mismatch: {arm}/{key}")

                expected_backend = "navdp" if arm == "native" else "hybrid_pose"
                expected_route = (
                    "memory_geometry" if arm == "geometry_router" else "phase"
                )
                require(metric["server_backend"] == expected_backend,
                        f"backend mismatch: {arm}/{key}")
                require(metric["hybrid_route"] == expected_route,
                        f"route mismatch: {arm}/{key}")
                require(metric["revisit_adapter"] == "legacy_metric",
                        f"adapter receipt mismatch: {arm}/{key}")
                require(metric["retrieval_override"] == "off",
                        f"oracle retrieval enabled: {arm}/{key}")

                plans_path = scene_root / arm / f"{episode}_plans.json"
                plans = json.loads(plans_path.read_text())
                require(plans.get("leg1_trace_sha256") == trace_hashes[episode],
                        f"plan trace SHA mismatch: {arm}/{key}")
                for plan in plans.get("legA", []) + plans.get("legB", []):
                    requested = plan.get("requested_diffusion_seed")
                    echoed = plan.get("diffusion_seed")
                    require(requested is not None and echoed is not None,
                            f"missing plan seed: {arm}/{key}")
                    require(int(requested) == int(echoed),
                            f"plan seed echo mismatch: {arm}/{key}")

                reached_a = truth(metric["reached_A"])
                reached_b = truth(metric["reached_B"])
                rows[arm][key] = {
                    "reached_a": reached_a,
                    "reached_b": reached_b,
                    "joint": reached_a and reached_b,
                    "seed": int(metric["seed"]),
                    "geo_a": float(metric["geo_A"]),
                    "geo_b": float(metric["geo_B"]),
                    "steps_a": int(metric["steps_A"]),
                    "spl_a": float(metric["spl_A"]),
                    "path_a": float(metric["len_A"]),
                    "final_dist_a": float(metric["final_dist_A"]),
                }
    return rows


def verify_pairing(rows: dict[str, dict]) -> None:
    keys = set(rows[ARMS[0]])
    require(len(keys) == 160, "raw result does not contain 160 episode keys")
    require(all(set(rows[arm]) == keys for arm in ARMS),
            "arm result keys differ")
    for key in sorted(keys):
        reference = rows[ARMS[0]][key]
        for arm in ARMS[1:]:
            candidate = rows[arm][key]
            for field in ("reached_a", "seed", "steps_a"):
                require(candidate[field] == reference[field],
                        f"shared A {field} mismatch: {arm}/{key}")
            for field in ("geo_a", "geo_b", "spl_a", "path_a", "final_dist_a"):
                require(math.isclose(candidate[field], reference[field],
                                     rel_tol=0.0, abs_tol=1e-9),
                        f"shared A {field} mismatch: {arm}/{key}")


def arm_counts(arm_rows: dict) -> dict[str, int]:
    ordered = list(arm_rows.values())
    novel = sum(row["reached_a"] for row in ordered)
    conditional = [row for row in ordered if row["reached_a"]]
    return {
        "novel_successes": novel,
        "conditional_eligible": len(conditional),
        "revisit_successes": sum(row["reached_b"] for row in conditional),
        "joint_successes": sum(row["joint"] for row in ordered),
    }


def paired(left: dict, right: dict, *, conditional: bool) -> dict[str, Any]:
    gains = losses = both = neither = 0
    eligible = 0
    by_scene: dict[str, list[int]] = {}
    for key in sorted(left):
        if conditional and not left[key]["reached_a"]:
            continue
        eligible += 1
        target = "reached_b" if conditional else "joint"
        lval = bool(left[key][target])
        rval = bool(right[key][target])
        delta = int(rval) - int(lval)
        by_scene.setdefault(key[0], []).append(delta)
        if lval and rval:
            both += 1
        elif rval:
            gains += 1
        elif lval:
            losses += 1
        else:
            neither += 1
    return {
        "eligible": eligible,
        "both": both,
        "gains": gains,
        "losses": losses,
        "neither": neither,
        "risk_difference": (gains - losses) / eligible,
        "mcnemar": exact_mcnemar(gains, losses),
        "by_scene": by_scene,
    }


def cluster_interval(
    scenes: list[str], result: dict[str, Any], *, seed: int, resamples: int
) -> list[float]:
    numerators = np.asarray([
        sum(result["by_scene"].get(scene, [])) for scene in scenes
    ], dtype=float)
    denominators = np.asarray([
        len(result["by_scene"].get(scene, [])) for scene in scenes
    ], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(scenes), size=(resamples, len(scenes)))
    sampled_den = denominators[indices].sum(axis=1)
    samples = numerators[indices].sum(axis=1)[sampled_den > 0] / sampled_den[
        sampled_den > 0
    ]
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def compare_report(
    manifest: dict[str, Any], rows: dict[str, dict], report: dict[str, Any]
) -> dict[str, Any]:
    counts = {arm: arm_counts(rows[arm]) for arm in ARMS}
    for arm, observed in counts.items():
        formal = report["arms"][arm]
        require(formal["novel"]["successes"] == observed["novel_successes"],
                f"report novel count mismatch: {arm}")
        require(formal["revisit_given_novel_success"]["eligible"]
                == observed["conditional_eligible"],
                f"report conditional denominator mismatch: {arm}")
        require(formal["revisit_given_novel_success"]["successes"]
                == observed["revisit_successes"],
                f"report Revisit count mismatch: {arm}")
        require(formal["joint"]["successes"] == observed["joint_successes"],
                f"report joint count mismatch: {arm}")

    analysis = manifest["analysis"]
    scenes = manifest["scenes"]
    base_seed = int(analysis["cluster_bootstrap_seed"])
    resamples = int(analysis["cluster_bootstrap_resamples"])
    contrasts = {}
    for offset, (name, (left_name, right_name)) in enumerate(CONTRASTS.items()):
        joint = paired(rows[left_name], rows[right_name], conditional=False)
        conditional = paired(rows[left_name], rows[right_name], conditional=True)
        joint_ci = cluster_interval(
            scenes, joint, seed=base_seed + offset * 2, resamples=resamples)
        conditional_ci = cluster_interval(
            scenes, conditional, seed=base_seed + offset * 2 + 1,
            resamples=resamples)
        formal_joint = report["contrasts"][name]["joint"]
        formal_conditional = report["contrasts"][name]["conditional_b"]
        require(math.isclose(
            formal_joint["joint_sr_delta_right_minus_left"],
            joint["risk_difference"], rel_tol=0.0, abs_tol=1e-15),
            f"joint risk difference mismatch: {name}")
        require(math.isclose(
            formal_joint["mcnemar_exact_two_sided_p"], joint["mcnemar"],
            rel_tol=0.0, abs_tol=1e-15), f"joint p-value mismatch: {name}")
        require(np.allclose(
            formal_joint["scene_cluster_bootstrap_risk_difference_95"],
            joint_ci, rtol=0.0, atol=1e-15), f"joint CI mismatch: {name}")
        require(math.isclose(
            formal_conditional["risk_difference_right_minus_left"],
            conditional["risk_difference"], rel_tol=0.0, abs_tol=1e-15),
            f"conditional risk difference mismatch: {name}")
        require(math.isclose(
            formal_conditional["mcnemar_exact_two_sided_p"],
            conditional["mcnemar"], rel_tol=0.0, abs_tol=1e-15),
            f"conditional p-value mismatch: {name}")
        require(np.allclose(
            formal_conditional["scene_cluster_bootstrap_risk_difference_95"],
            conditional_ci, rtol=0.0, atol=1e-15),
            f"conditional CI mismatch: {name}")
        contrasts[name] = {
            "joint": {**{k: v for k, v in joint.items() if k != "by_scene"},
                      "scene_cluster_bootstrap_95": joint_ci},
            "conditional_b": {
                **{k: v for k, v in conditional.items() if k != "by_scene"},
                "scene_cluster_bootstrap_95": conditional_ci,
            },
        }
    return {"arms": counts, "contrasts": contrasts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite: {args.out}")
    manifest = json.loads(args.manifest.read_text())
    report = json.loads(args.report.read_text())
    require(report["audit"]["status"] == "ok", "formal report audit is not ok")
    require(report["audit"]["manifest_sha256"] == sha256(args.manifest),
            "formal report manifest SHA mismatch")
    rows = load_raw(manifest, args.run_root)
    verify_pairing(rows)
    recomputed = compare_report(manifest, rows, report)
    output = {
        "audit": "ok",
        "independent_of_project_summarizer_imports": True,
        "manifest_sha256": sha256(args.manifest),
        "report_sha256": sha256(args.report),
        "raw_episode_keys": 160,
        **recomputed,
        "formal_decision": report["decision"],
    }
    args.out.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
