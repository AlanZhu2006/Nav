#!/usr/bin/env python3
"""Independent raw-file verifier for the Novel causal-control result.

This file intentionally does not import the evaluator or summarizer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ARMS = (
    "native", "raw_factual_history", "raw_deranged_history",
    "raw_randomized_bearing",
)
CONTRASTS = (
    ("raw_factual_history", "raw_randomized_bearing"),
    ("raw_factual_history", "raw_deranged_history"),
    ("raw_factual_history", "native"),
    ("raw_deranged_history", "native"),
    ("raw_randomized_bearing", "native"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_mcnemar(gains: int, losses: int) -> float:
    n = gains + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(gains, losses) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def verify(run_root: Path, manifest_path: Path, summary_path: Path) -> dict:
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    summary = json.loads(summary_path.read_text())
    require(
        manifest.get("schema_version")
        == "novel_memory_direction_control_v1_20260816",
        "manifest schema changed",
    )
    require(tuple(manifest.get("arms") or ()) == ARMS, "manifest arm set changed")
    require(manifest.get("confirmation_claim_allowed") is False, "manifest claims confirmation")
    require(summary.get("control_manifest_sha256") == manifest_sha, "summary manifest SHA changed")
    require(summary.get("confirmation_claim_allowed") is False, "summary claims confirmation")
    expected = {
        (str(row["scene"]), str(row["episode"])): row
        for row in manifest["episodes"]
    }
    untouched = set(manifest["untouched_final_scenes_remain_unread"])
    require(not ({scene for scene, _ in expected} & untouched), "final scene leaked into control")

    raw_rows = []
    completion_paths = sorted((run_root / "evaluation").glob("*/completion.json"))
    require(len(completion_paths) == len(expected), "completion population differs")
    for completion_path in completion_paths:
        completion = json.loads(completion_path.read_text())
        receipt = completion_path.with_name("completion.json.sha256").read_text().split()
        require(
            len(receipt) == 2 and receipt[0] == sha256_file(completion_path),
            "completion receipt failed",
        )
        identity = (str(completion["scene"]), str(completion["episode"]))
        require(identity in expected, "unexpected result identity")
        directory = completion_path.parent
        contract = json.loads((directory / "episode_contract.json").read_text())
        require(contract["arm_order"] == expected[identity]["arm_order"], "arm order differs")
        require(contract.get("confirmation_claim_allowed") is False, "episode claims confirmation")

        outcomes = {}
        geodesics = {}
        path_lengths = {}
        final_euclidean = {}
        final_geodesics = {}
        steps = {}
        spl = {}
        takeover_plans = {}
        plan_counts = {}
        fallback_plans = {}
        query_ids = set()
        replays = {}
        query_rollouts = {}
        for arm in ARMS:
            with (directory / arm / "metric.csv").open(newline="") as handle:
                metrics = list(csv.DictReader(handle))
            require(len(metrics) == 1, f"{identity}/{arm}: wrong metric count")
            metric = metrics[0]
            require(metric["analysis_role"] == "novel", "non-Novel query entered control")
            require(metric["arm"] == arm, "raw metric arm changed")
            outcomes[arm] = int(metric["reached"])
            geodesics[arm] = float(metric["geodesic_m"])
            path_lengths[arm] = float(metric["path_len_m"])
            final_euclidean[arm] = float(metric["final_goal_dist_m"])
            final_geodesics[arm] = float(metric["final_goal_geodesic_m"])
            steps[arm] = int(metric["steps"])
            query_ids.add(metric["query_id"])
            plan_paths = list((directory / arm).glob(f"{identity[1]}_*_plans.json"))
            require(len(plan_paths) == 1, "wrong plan-ledger count")
            plan = json.loads(plan_paths[0].read_text())
            require(plan.get("analysis_role_not_forwarded") is True, "role leaked to runtime")
            audit = plan.get("novel_causal_control") or {}
            require(audit.get("arm") == arm, "plan audit arm changed")
            require(audit.get("manifest_sha256") == manifest_sha, "plan audit manifest changed")
            replays[arm] = plan["replay"]
            query_rollouts[arm] = plan["rollout_traces"]["query"]
            for plan_index, decision in enumerate(plan["query_leg"]):
                requested = decision.get("requested_diffusion_seed")
                echoed = decision.get("diffusion_seed")
                require(
                    requested is not None and echoed is not None
                    and int(requested) == int(echoed),
                    f"{arm}: diffusion seed differs at plan {plan_index}",
                )
            takeover_plans[arm] = int(metric["adapter_takeover_plans"])
            plan_counts[arm] = len(plan["query_leg"])
            fallback_plans[arm] = plan_counts[arm] - takeover_plans[arm]
            denominator = max(geodesics[arm], path_lengths[arm])
            spl[arm] = (
                outcomes[arm] * geodesics[arm] / denominator
                if denominator > 0.0 else float(outcomes[arm])
            )
            if arm == "raw_randomized_bearing":
                ledger = audit.get("randomized_bearing_ledger") or []
                require(len(ledger) == len(plan["query_leg"]), "random ledger count differs")
                require(all(
                    bool(item["factual_takeover"])
                    == bool(item["randomized_takeover"])
                    for item in ledger
                ), "randomization changed proposal availability")
        require(len(query_ids) == 1, "paired query identity changed")
        for arm in ARMS[1:]:
            if takeover_plans[arm] == 0:
                require(
                    query_rollouts[arm] == query_rollouts["native"],
                    f"{arm}: zero-takeover rollout is not exact fallback",
                )
        require(outcomes == completion["outcomes"], "completion outcome differs from raw metric")
        for name, recomputed, tolerance in (
            ("geodesic_m", geodesics, 1e-12),
            ("path_length_m", path_lengths, 1e-12),
            ("final_distance_m", final_euclidean, 1e-12),
            ("final_geodesic_m", final_geodesics, 1e-12),
            ("spl", spl, 1e-12),
        ):
            require(all(
                math.isclose(float(completion[name][arm]), recomputed[arm],
                             rel_tol=0.0, abs_tol=tolerance)
                for arm in ARMS
            ), f"completion {name} differs from raw metrics")
        require(steps == completion["steps"], "completion steps differ")
        require(takeover_plans == completion["takeover_plans"],
                "completion takeover count differs")
        require(plan_counts == completion["plan_count"],
                "completion plan count differs")
        require(fallback_plans == completion["fallback_plans"],
                "completion fallback count differs")
        factual_fifo = {
            replays[arm].get("factual_fifo_decision_sha256") for arm in ARMS
        }
        require(len(factual_fifo) == 1 and None not in factual_fifo, "factual FIFO differs")
        require(replays["raw_deranged_history"].get("sidecar_is_deranged") is True,
                "deranged replay was not marked")
        require(replays["raw_factual_history"].get("sidecar_is_deranged") is False,
                "factual replay was marked deranged")
        require(
            replays["raw_factual_history"]["sidecar_memory_sha256"]
            == replays["raw_randomized_bearing"]["sidecar_memory_sha256"],
            "factual/randomized sidecars differ",
        )
        require(
            replays["raw_factual_history"]["sidecar_memory_sha256"]
            != replays["raw_deranged_history"]["sidecar_memory_sha256"],
            "deranged sidecar content did not change",
        )
        raw_rows.append({
            "scene": identity[0], "episode": identity[1],
            "query_id": next(iter(query_ids)), "outcomes": outcomes,
            "spl": spl, "path_length_m": path_lengths, "steps": steps,
            "final_geodesic_m": final_geodesics,
            "takeover_plans": takeover_plans,
            "fallback_plans": fallback_plans,
            "plan_count": plan_counts,
        })

    require(
        {(row["scene"], row["episode"]) for row in raw_rows} == set(expected),
        "raw result identity set differs from manifest",
    )
    recomputed_arms = {}
    for arm in ARMS:
        successes = sum(row["outcomes"][arm] for row in raw_rows)
        reported = summary["arm_metrics"][arm]
        require(successes == int(reported["successes"]), f"{arm}: success count differs")
        require(len(raw_rows) == int(reported["episodes"]), f"{arm}: denominator differs")
        mean_spl = sum(row["spl"][arm] for row in raw_rows) / len(raw_rows)
        require(math.isclose(mean_spl, float(reported["mean_spl"]), abs_tol=1e-15),
                f"{arm}: mean SPL differs")
        mean_final_geodesic = sum(
            row["final_geodesic_m"][arm] for row in raw_rows
        ) / len(raw_rows)
        require(
            math.isclose(
                mean_final_geodesic,
                float(reported["mean_final_geodesic_m"]),
                abs_tol=1e-15,
            ),
            f"{arm}: mean final geodesic differs",
        )
        require(sum(row["takeover_plans"][arm] for row in raw_rows)
                == int(reported["takeover_plans"]),
                f"{arm}: takeover count differs")
        require(sum(row["fallback_plans"][arm] for row in raw_rows)
                == int(reported["fallback_plans"]),
                f"{arm}: fallback count differs")
        recomputed_arms[arm] = {"successes": successes, "episodes": len(raw_rows)}

    recomputed_contrasts = {}
    for arm_a, arm_b in CONTRASTS:
        gains = sum(
            row["outcomes"][arm_a] == 1 and row["outcomes"][arm_b] == 0
            for row in raw_rows
        )
        losses = sum(
            row["outcomes"][arm_a] == 0 and row["outcomes"][arm_b] == 1
            for row in raw_rows
        )
        key = f"{arm_a}-minus-{arm_b}"
        reported = summary["contrasts"][key]
        require(gains == int(reported["paired_gains"]), f"{key}: gains differ")
        require(losses == int(reported["paired_losses"]), f"{key}: losses differ")
        p_value = exact_mcnemar(gains, losses)
        require(math.isclose(p_value, float(reported["exact_mcnemar_p"]), abs_tol=1e-15),
                f"{key}: McNemar differs")
        risk_difference = sum(
            row["outcomes"][arm_a] - row["outcomes"][arm_b]
            for row in raw_rows
        ) / len(raw_rows)
        require(math.isclose(risk_difference, float(reported["risk_difference"]), abs_tol=1e-15),
                f"{key}: risk difference differs")
        recomputed_contrasts[key] = {
            "paired_gains": gains, "paired_losses": losses,
            "risk_difference": risk_difference, "exact_mcnemar_p": p_value,
        }

    return {
        "schema_version": "independent_novel_memory_direction_verification_v1_20260816",
        "verified": True,
        "evaluation_stage": "consumed_development_mechanism_only",
        "confirmation_claim_allowed": False,
        "control_manifest_sha256": manifest_sha,
        "summary_sha256": sha256_file(summary_path),
        "population": {
            "episodes": len(raw_rows),
            "scenes": len({row["scene"] for row in raw_rows}),
        },
        "untouched_final_scene_intersection": [],
        "recomputed_arm_counts": recomputed_arms,
        "recomputed_contrasts": recomputed_contrasts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.run_root, args.manifest, args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
