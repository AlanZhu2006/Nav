#!/usr/bin/env python3
"""Fail-closed paired summary for geometry certificate vs CDEC cascade."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

from MemNavData.build_revisit_fresh_manifest import sha256_file
from MemNavData.summarize_certified_relocalization_closed_loop import (
    cluster_interval,
    conditional_paired,
)
from MemNavData.summarize_expanded_navdp_router_eval import (
    arm_summary,
    load_arm,
    paired_summary,
    require,
    truth,
)


SCHEMA_VERSION = "cdec_consumed_closed_loop_summary_v1_20260813"
ARMS = ("geometry_certificate", "cdec_cascade")
ORDERS = (
    ("geometry_certificate", "cdec_cascade"),
    ("cdec_cascade", "geometry_certificate"),
)
EXPECTED_ARTIFACT_SHA = (
    "eea77098531c6e5865516d49540374edca7bb87a419613ef40d56cc2c85add31"
)


def _int(value: Any, field: str) -> int:
    require(value not in (None, ""), f"missing integer field {field}")
    parsed = float(value)
    require(math.isfinite(parsed) and parsed.is_integer(), f"bad integer {field}")
    return int(parsed)


def read_metrics(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing metrics: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_arm_summary(path: Path, arm: str) -> dict[str, Any]:
    require(path.is_file(), f"missing summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "episodes": 8,
        "server_backend": "hybrid_pose",
        "hybrid_route": "certified_relocalization",
        "revisit_controller": "navdp_mixed",
        "revisit_adapter": "verified_bearing_v1",
        "revisit_adapter_fixed_radius_m": 2.5,
        "leg1_mode": "shared_trace",
        "write_leg1_trace": False,
        "deterministic_plan_seeds": True,
        "trajectory_selector": "server",
        "trajectory_selector_scope": "all",
        "max_steps": 500,
        "exec_horizon": 8,
        "graph_subgoal_spacing_m": 0.0,
        "graph_subgoal_arrival_m": 0.6,
        "certified_cdec_rescue": (
            "on" if arm == "cdec_cascade" else "off"
        ),
        "expected_cdec_artifact_sha256": (
            EXPECTED_ARTIFACT_SHA if arm == "cdec_cascade" else None
        ),
    }
    for field, wanted in expected.items():
        require(
            summary.get(field) == wanted,
            f"{path}: {field}={summary.get(field)!r}, expected {wanted!r}",
        )
    status = summary.get("cdec_server_status")
    require(isinstance(status, Mapping), f"{path}: CDEC server status missing")
    required_status = {
        "enabled": True,
        "artifact_sha256": EXPECTED_ARTIFACT_SHA,
        "deployment_approved": False,
        "authority": "rank_frozen_causal_shortlist_only",
        "activation_authority": "independent_atomic_pnp_certificate",
    }
    for field, wanted in required_status.items():
        require(status.get(field) == wanted, f"{path}: CDEC status {field} changed")
    require(
        _int(
            summary.get("certified_cdec_uncached_runtime_failure_count"),
            "CDEC runtime failures",
        )
        == 0,
        f"{path}: CDEC runtime failure",
    )
    if arm == "geometry_certificate":
        for field in (
            "certified_cdec_requested_plan_count",
            "certified_cdec_learned_selected_plan_count",
            "certified_cdec_uncached_invocation_count",
        ):
            require(_int(summary.get(field), field) == 0, f"{path}: {field} != 0")
    return summary


def cdec_neutral_payload(value: Any) -> Any:
    """Remove only timing and explicit arm-label diagnostics."""
    if isinstance(value, dict):
        ignored = {
            "router_ranking_mode",
            "certified_relocalization_learned_rescue_requested",
            "certified_relocalization_learned_proposal",
        }
        neutral = {}
        for key, item in value.items():
            if key in ignored or key.endswith("_ms"):
                continue
            if key == "certified_relocalization_proposal_attempts":
                # The CDEC arm may record a rejected second certificate (or a
                # same-anchor reuse) while still executing the exact native
                # fallback.  Preserve the geometry attempt, which must remain
                # identical across arms, but remove the explicitly learned
                # diagnostic attempt from the no-treatment comparison.
                if isinstance(item, list) and item:
                    first = item[0]
                    if (isinstance(first, Mapping)
                            and first.get("source") == "geometry"):
                        item = [first]
            neutral[key] = cdec_neutral_payload(item)
        return neutral
    if isinstance(value, list):
        return [cdec_neutral_payload(item) for item in value]
    return value


def first_attempt(plan: Mapping[str, Any]) -> Mapping[str, Any] | None:
    attempts = plan.get("certified_relocalization_proposal_attempts")
    if attempts is None:
        return None
    require(isinstance(attempts, list) and attempts, "invalid proposal attempts")
    first = attempts[0]
    require(isinstance(first, Mapping), "geometry proposal attempt is malformed")
    require(first.get("source") == "geometry", "first proposal is not geometry")
    return first


def learned_takeover(plan: Mapping[str, Any]) -> bool:
    return (
        plan.get("certified_relocalization_selected_proposal_source")
        == "learned_on_geometry_reject"
        and plan.get("certified_relocalization_accepted") is True
        and plan.get("revisit_adapter_takeover") is True
    )


def audit_episode(
    *,
    scene: str,
    episode: str,
    geometry_root: Path,
    cdec_root: Path,
    geometry_metric: Mapping[str, str],
    cdec_metric: Mapping[str, str],
    trace_sha: str,
) -> dict[str, Any]:
    geometry_path = geometry_root / f"{episode}_plans.json"
    cdec_path = cdec_root / f"{episode}_plans.json"
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    cdec = json.loads(cdec_path.read_text(encoding="utf-8"))
    for payload, name in ((geometry, "geometry"), (cdec, "cdec")):
        require(payload.get("leg1_trace_sha256") == trace_sha,
                f"{scene}/{episode}: {name} trace SHA changed")
    require(geometry.get("legA") == cdec.get("legA"),
            f"{scene}/{episode}: shared Goal-A plan trace differs")
    require(
        geometry.get("legA_memory_trace") == cdec.get("legA_memory_trace"),
        f"{scene}/{episode}: causal Goal-A memory differs",
    )
    plans_g = geometry.get("legB")
    plans_c = cdec.get("legB")
    require(isinstance(plans_g, list) and isinstance(plans_c, list),
            f"{scene}/{episode}: B plans missing")
    reached_a = truth(geometry_metric["reached_A"])
    require(reached_a == truth(cdec_metric["reached_A"]),
            f"{scene}/{episode}: shared A outcome differs")
    for field in (
        "seed", "leg1_trace_sha256", "reached_A", "spl_A", "geo_A",
        "len_A", "final_dist_A", "steps_A", "termination_reason_A",
        "blocked_steps_A",
    ):
        require(geometry_metric.get(field) == cdec_metric.get(field),
                f"{scene}/{episode}: shared A metric {field} differs")
    require(geometry_metric.get("leg1_trace_sha256") == trace_sha,
            f"{scene}/{episode}: metric trace SHA changed")
    if not reached_a:
        require(not plans_g and not plans_c,
                f"{scene}/{episode}: B ran after A failure")
        return {
            "scene": scene,
            "episode": episode,
            "reached_a": False,
            "learned_invoked": False,
            "learned_takeover": False,
            "geometry_accepted": False,
            "no_treatment_exact": True,
        }
    require(plans_g and plans_c, f"{scene}/{episode}: A success lacks B plans")
    require(all(
        plan.get("certified_relocalization_learned_rescue_requested") is False
        for plan in plans_g
    ), f"{scene}/{episode}: baseline requested learned rescue")
    require(all(
        plan.get("certified_relocalization_learned_rescue_requested") is True
        for plan in plans_c
    ), f"{scene}/{episode}: CDEC arm omitted learned request")

    first_g = plans_g[0]
    first_c = plans_c[0]
    for field in (
        "step",
        "requested_diffusion_seed",
        "diffusion_seed",
        "frame_idx",
        "goal_start_frame",
        "candidate_ceiling",
        "router_candidate_order_dino",
        "router_candidate_order_used",
    ):
        require(first_g.get(field) == first_c.get(field),
                f"{scene}/{episode}: pre-proposal {field} differs")
    attempt_g = first_attempt(first_g)
    attempt_c = first_attempt(first_c)
    require(attempt_g == attempt_c,
            f"{scene}/{episode}: geometry proposal/certificate differs")

    learned = first_c.get("certified_relocalization_learned_proposal")
    geometry_accepted = bool(
        attempt_c is not None and attempt_c.get("accepted") is True)
    takeover = learned_takeover(first_c)
    invoked = False
    if isinstance(learned, Mapping):
        status = learned.get("status")
        require(learned.get("activation_authorized") is False,
                f"{scene}/{episode}: learned ranker acquired activation authority")
        invoked = status not in {
            None, "not_requested", "not_evaluated_geometry_accepted"
        }
        require(status != "runtime_exception_fail_closed",
                f"{scene}/{episode}: learned runtime failure")
        if geometry_accepted:
            require(status == "not_evaluated_geometry_accepted",
                    f"{scene}/{episode}: learned ran before geometry pass")
    else:
        require(not geometry_accepted and attempt_c is None,
                f"{scene}/{episode}: missing learned diagnostic")

    if takeover:
        require(not geometry_accepted and invoked,
                f"{scene}/{episode}: takeover did not follow geometry reject")
        attempts = first_c["certified_relocalization_proposal_attempts"]
        require(len(attempts) == 2,
                f"{scene}/{episode}: learned takeover lacks second certificate")
        second = attempts[1]
        require(second.get("source") == "learned_on_geometry_reject"
                and second.get("accepted") is True,
                f"{scene}/{episode}: learned takeover lacks accepted certificate")

    no_treatment = not takeover
    if no_treatment:
        require(cdec_neutral_payload(plans_g) == cdec_neutral_payload(plans_c),
                f"{scene}/{episode}: no-treatment causal plans changed")
        for field in (
            "reached_B", "spl_B", "spl_B_with_terminal", "geo_B",
            "steps_B", "steps_B_diagnostic", "steps_B_at_reach", "len_B",
            "len_B_at_reach", "final_dist_B", "termination_reason_B",
            "blocked_steps_B", "blocked_step_rate_B",
            "terminal_final_goal_dist_m",
        ):
            require(geometry_metric.get(field) == cdec_metric.get(field),
                    f"{scene}/{episode}: no-treatment {field} changed")

    return {
        "scene": scene,
        "episode": episode,
        "reached_a": True,
        "geometry_accepted": geometry_accepted,
        "learned_invoked": invoked,
        "learned_takeover": takeover,
        "no_treatment_exact": no_treatment,
        "geometry_reached_b": truth(geometry_metric["reached_B"]),
        "cdec_reached_b": truth(cdec_metric["reached_B"]),
    }


def promotion_decision(
    *,
    gains: list[dict[str, Any]],
    losses: list[dict[str, Any]],
    mcnemar_p: float,
    cluster_interval_95: list[float],
    all_audits_pass: bool,
) -> dict[str, Any]:
    gain_scenes = {row["scene"] for row in gains}
    checks = {
        "gain_in_at_least_two_scene_clusters": len(gain_scenes) >= 2,
        "zero_paired_losses": len(losses) == 0,
        "exact_mcnemar_below_0_05": mcnemar_p < 0.05,
        "cluster_interval_lower_above_zero": cluster_interval_95[0] > 0.0,
        "all_causal_and_safety_audits_pass": all_audits_pass,
        "every_gain_has_learned_certified_takeover": all(
            row.get("learned_takeover") is True for row in gains
        ),
    }
    passed = all(checks.values())
    return {
        "pass": passed,
        "checks": checks,
        "branch": (
            "eligible_for_frozen_one_shot_system_confirmation"
            if passed else "do_not_promote_cdec"
        ),
        "authorize_retuning_on_consumed_pool": False,
        "authorize_blind_opening_without_explicit_user_approval": False,
    }


def summarize(manifest_path: Path, trace_receipt_path: Path,
              run_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_receipt = json.loads(trace_receipt_path.read_text(encoding="utf-8"))
    require(manifest["audit"]["status"] == "ok", "manifest audit failed")
    require(manifest["data_role_guards"]["blind_allowed"] is False,
            "manifest permits blind data")
    scenes = list(manifest["scenes"])
    episode_ids = {
        scene: [row["episode"] for row in manifest["episodes"][scene]]
        for scene in scenes
    }
    expected = {
        (scene, episode)
        for scene in scenes for episode in episode_ids[scene]
    }
    require(len(scenes) == 20 and len(expected) == 160,
            "comparison requires 20 scenes / 160 episodes")
    require(all(len(value) == 8 for value in episode_ids.values()),
            "each scene must contain eight episodes")
    require(trace_receipt.get("episode_target_or_outcome_read") is False,
            "trace receipt read outcome data")
    require(trace_receipt.get("manifest_sha256") == sha256_file(manifest_path),
            "trace/manifest binding changed")

    rows = {arm: {} for arm in ARMS}
    episode_audits: list[dict[str, Any]] = []
    arm_orders = {}
    for index, scene in enumerate(scenes):
        scene_root = run_root / "scenes" / f"{index:02d}_{scene}"
        contract = json.loads(
            (scene_root / "scene_contract.json").read_text(encoding="utf-8")
        )
        expected_order = list(ORDERS[index % 2])
        require(contract.get("schema_version") ==
                "cdec_consumed_closed_loop_scene_v1_20260813",
                f"scene contract changed: {scene}")
        require(contract.get("scene") == scene
                and contract.get("scene_index") == index,
                f"scene identity changed: {scene}")
        require(contract.get("manifest_sha256") == sha256_file(manifest_path),
                f"manifest SHA changed: {scene}")
        require(contract.get("trace_receipt_sha256") ==
                sha256_file(trace_receipt_path),
                f"trace receipt changed: {scene}")
        require(contract.get("cdec_artifact_sha256") == EXPECTED_ARTIFACT_SHA,
                f"artifact changed: {scene}")
        require(contract.get("arm_order") == expected_order,
                f"arm order changed: {scene}")
        require(contract.get("stagnation_graph") == "off",
                f"graph confound enabled: {scene}")
        arm_orders[scene] = expected_order

        metrics_by_arm = {}
        for arm in ARMS:
            arm_root = scene_root / arm
            validate_arm_summary(arm_root / "summary.json", arm)
            metrics = read_metrics(arm_root / "metric.csv")
            require(len(metrics) == 8, f"{scene}/{arm}: row count changed")
            require([row["episode"] for row in metrics] == episode_ids[scene],
                    f"{scene}/{arm}: episode order changed")
            metrics_by_arm[arm] = {row["episode"]: row for row in metrics}
            rows[arm].update(load_arm(scene_root, arm, scene))

        scene_trace = trace_receipt["scenes"].get(scene)
        require(isinstance(scene_trace, Mapping), f"missing trace scene: {scene}")
        for episode in episode_ids[scene]:
            trace_sha = scene_trace["episodes"].get(episode)
            require(isinstance(trace_sha, str) and len(trace_sha) == 64,
                    f"missing trace SHA: {scene}/{episode}")
            episode_audits.append(audit_episode(
                scene=scene,
                episode=episode,
                geometry_root=scene_root / "geometry_certificate",
                cdec_root=scene_root / "cdec_cascade",
                geometry_metric=metrics_by_arm["geometry_certificate"][episode],
                cdec_metric=metrics_by_arm["cdec_cascade"][episode],
                trace_sha=trace_sha,
            ))

    for arm in ARMS:
        require(set(rows[arm]) == expected, f"{arm}: result universe changed")
    paired = paired_summary(
        "geometry_certificate", "cdec_cascade",
        rows["geometry_certificate"], rows["cdec_cascade"], expected,
    )
    conditional = conditional_paired(
        "geometry_certificate", "cdec_cascade",
        rows["geometry_certificate"], rows["cdec_cascade"], expected,
    )
    analysis = manifest["analysis"]
    seed = int(analysis["cluster_bootstrap_seed"])
    resamples = int(analysis["cluster_bootstrap_resamples"])
    paired["scene_cluster_bootstrap_risk_difference_95"] = cluster_interval(
        scenes, episode_ids, rows["geometry_certificate"], rows["cdec_cascade"],
        conditional=False, seed=seed + 401, resamples=resamples,
    )
    conditional["scene_cluster_bootstrap_risk_difference_95"] = cluster_interval(
        scenes, episode_ids, rows["geometry_certificate"], rows["cdec_cascade"],
        conditional=True, seed=seed + 402, resamples=resamples,
    )

    gains = [
        row for row in episode_audits
        if row.get("cdec_reached_b") and not row.get("geometry_reached_b")
    ]
    losses = [
        row for row in episode_audits
        if row.get("geometry_reached_b") and not row.get("cdec_reached_b")
    ]
    require(len(gains) == paired["outcomes"]["right_only_joint_success"],
            "gain audit differs from paired result")
    require(len(losses) == paired["outcomes"]["left_only_joint_success"],
            "loss audit differs from paired result")
    require(all(row.get("learned_takeover") for row in gains + losses),
            "paired effect occurred without learned certified takeover")
    decision = promotion_decision(
        gains=gains,
        losses=losses,
        mcnemar_p=float(paired["mcnemar_exact_two_sided_p"]),
        cluster_interval_95=paired[
            "scene_cluster_bootstrap_risk_difference_95"],
        all_audits_pass=True,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "consumed 20-scene/160-episode paired closed loop; not fresh or "
            "paper-final confirmation"
        ),
        "audit": {
            "status": "ok",
            "scenes": len(scenes),
            "episodes": len(expected),
            "manifest_sha256": sha256_file(manifest_path),
            "trace_receipt_sha256": sha256_file(trace_receipt_path),
            "artifact_sha256": EXPECTED_ARTIFACT_SHA,
            "same_process_per_scene": True,
            "shared_goal_a_trace": True,
            "exact_goal_a_memory": True,
            "geometry_proposal_decision_paired": True,
            "stagnation_graph_disabled": True,
            "development_read": False,
            "blind_read": False,
        },
        "arm_order_by_scene": arm_orders,
        "arms": {
            arm: arm_summary([rows[arm][key] for key in sorted(expected)])
            for arm in ARMS
        },
        "contrasts": {"joint": paired, "conditional_b": conditional},
        "learned_runtime": {
            "a_success_episodes": sum(row["reached_a"] for row in episode_audits),
            "invoked_episodes": sum(row["learned_invoked"] for row in episode_audits),
            "takeover_episodes": sum(row["learned_takeover"] for row in episode_audits),
            "geometry_accepted_episodes": sum(
                row["geometry_accepted"] for row in episode_audits
            ),
            "no_treatment_exact_episodes": sum(
                row["no_treatment_exact"] for row in episode_audits
            ),
            "gain_episodes": [
                {"scene": row["scene"], "episode": row["episode"]}
                for row in gains
            ],
            "loss_episodes": [
                {"scene": row["scene"], "episode": row["episode"]}
                for row in losses
            ],
        },
        "decision": decision,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    return args


def main() -> None:
    args = parse_args()
    report = summarize(
        args.manifest, args.trace_receipt, args.run_root
    )
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "arms": report["arms"],
        "contrasts": report["contrasts"],
        "learned_runtime": report["learned_runtime"],
        "decision": report["decision"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
