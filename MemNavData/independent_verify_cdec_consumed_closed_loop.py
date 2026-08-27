#!/usr/bin/env python3
"""Independent raw-output verifier for the consumed CDEC closed loop.

This module intentionally imports neither the production summarizer nor its
statistical helpers.  It reconstructs the paired outcome, clustered interval,
strict promotion decision, and the critical causal intervention audit directly
from the frozen CSV/JSON episode outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "independent_cdec_consumed_closed_loop_v1_20260813"
ARMS = ("geometry_certificate", "cdec_cascade")
ORDERS = (
    ("geometry_certificate", "cdec_cascade"),
    ("cdec_cascade", "geometry_certificate"),
)
EXPECTED_ARTIFACT_SHA256 = (
    "eea77098531c6e5865516d49540374edca7bb87a419613ef40d56cc2c85add31")
BOOTSTRAP_CHUNK = 10_000


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return True
    if normalized in {"false", "0", "0.0", "", "none"}:
        return False
    raise RuntimeError(f"invalid boolean: {value!r}")


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = int(gains) + int(losses)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(int(gains), int(losses)) + 1))
    return float(min(1.0, 2.0 * tail / (2 ** discordant)))


def cluster_interval(
    scenes: list[str], episodes: Mapping[str, list[str]],
    geometry: Mapping[tuple[str, str], Mapping[str, Any]],
    cdec: Mapping[tuple[str, str], Mapping[str, Any]],
    *, conditional: bool, seed: int, resamples: int,
) -> list[float]:
    numerators = []
    denominators = []
    for scene in scenes:
        numerator = 0.0
        denominator = 0
        for episode in episodes[scene]:
            key = (scene, episode)
            if conditional and not geometry[key]["reached_a"]:
                continue
            denominator += 1
            target = "reached_b" if conditional else "joint"
            numerator += float(cdec[key][target]) - float(geometry[key][target])
        numerators.append(numerator)
        denominators.append(denominator)
    nums = np.asarray(numerators, dtype=np.float64)
    dens = np.asarray(denominators, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    chunks = []
    for start in range(0, int(resamples), BOOTSTRAP_CHUNK):
        count = min(BOOTSTRAP_CHUNK, int(resamples) - start)
        indices = rng.integers(0, len(scenes), size=(count, len(scenes)))
        sampled_den = dens[indices].sum(axis=1)
        valid = sampled_den > 0
        chunks.append(nums[indices].sum(axis=1)[valid] / sampled_den[valid])
    values = np.concatenate(chunks)
    require(values.size > 0, "cluster bootstrap has no valid sample")
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


def _neutral(value: Any) -> Any:
    if isinstance(value, dict):
        ignored = {
            "router_ranking_mode",
            "certified_relocalization_learned_rescue_requested",
            "certified_relocalization_learned_proposal",
        }
        output = {}
        for key, item in value.items():
            if key in ignored or key.endswith("_ms"):
                continue
            if key == "certified_relocalization_proposal_attempts":
                if (isinstance(item, list) and item
                        and isinstance(item[0], Mapping)
                        and item[0].get("source") == "geometry"):
                    item = [item[0]]
            output[key] = _neutral(item)
        return output
    if isinstance(value, list):
        return [_neutral(item) for item in value]
    return value


def _first_geometry_attempt(plan: Mapping[str, Any]) -> Mapping[str, Any] | None:
    attempts = plan.get("certified_relocalization_proposal_attempts")
    if attempts is None:
        return None
    require(isinstance(attempts, list) and attempts,
            "proposal attempts are malformed")
    require(isinstance(attempts[0], Mapping)
            and attempts[0].get("source") == "geometry",
            "first certificate attempt is not geometry")
    return attempts[0]


def _takeover(plan: Mapping[str, Any]) -> bool:
    return (
        plan.get("certified_relocalization_selected_proposal_source")
        == "learned_on_geometry_reject"
        and plan.get("certified_relocalization_accepted") is True
        and plan.get("revisit_adapter_takeover") is True)


def _read_metric(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"missing metric: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required = {
        "episode", "seed", "leg1_trace_sha256", "reached_A", "reached_B",
        "spl_A", "geo_A", "geo_B", "len_A", "len_B", "final_dist_A",
        "steps_A", "steps_B", "terminal_final_goal_dist_m",
    }
    require(reader.fieldnames is not None
            and not (required - set(reader.fieldnames)),
            f"metric schema changed: {path}")
    return rows


def _validate_summary(path: Path, arm: str) -> None:
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "episodes": 8,
        "server_backend": "hybrid_pose",
        "hybrid_route": "certified_relocalization",
        "revisit_controller": "navdp_mixed",
        "revisit_adapter": "verified_bearing_v1",
        "revisit_adapter_fixed_radius_m": 2.5,
        "leg1_mode": "shared_trace",
        "deterministic_plan_seeds": True,
        "trajectory_selector": "server",
        "trajectory_selector_scope": "all",
        "max_steps": 500,
        "exec_horizon": 8,
        "graph_subgoal_spacing_m": 0.0,
        "certified_cdec_rescue": "on" if arm == "cdec_cascade" else "off",
    }
    for key, wanted in expected.items():
        require(summary.get(key) == wanted,
                f"{path}: summary contract changed at {key}")
    status = summary.get("cdec_server_status")
    require(isinstance(status, Mapping)
            and status.get("enabled") is True
            and status.get("artifact_sha256") == EXPECTED_ARTIFACT_SHA256
            and status.get("deployment_approved") is False
            and status.get("authority") == "rank_frozen_causal_shortlist_only"
            and status.get("activation_authority")
            == "independent_atomic_pnp_certificate",
            f"{path}: learned runtime contract changed")
    require(int(summary.get("certified_cdec_uncached_runtime_failure_count", -1)) == 0,
            f"{path}: CDEC runtime failure")
    if arm == "geometry_certificate":
        for field in (
            "certified_cdec_requested_plan_count",
            "certified_cdec_learned_selected_plan_count",
            "certified_cdec_uncached_invocation_count",
        ):
            require(int(summary.get(field, -1)) == 0,
                    f"{path}: baseline invoked learned CDEC")


def _episode_audit(
    *, scene: str, episode: str, trace_sha: str,
    geometry_path: Path, cdec_path: Path,
    geometry_metric: Mapping[str, str], cdec_metric: Mapping[str, str],
) -> dict[str, Any]:
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    cdec = json.loads(cdec_path.read_text(encoding="utf-8"))
    require(geometry.get("leg1_trace_sha256") == trace_sha
            and cdec.get("leg1_trace_sha256") == trace_sha,
            f"{scene}/{episode}: trace SHA changed")
    require(geometry.get("legA") == cdec.get("legA"),
            f"{scene}/{episode}: Goal-A plans differ")
    require(geometry.get("legA_memory_trace") == cdec.get("legA_memory_trace"),
            f"{scene}/{episode}: Goal-A memory differs")
    for field in (
        "seed", "leg1_trace_sha256", "reached_A", "spl_A", "geo_A",
        "len_A", "final_dist_A", "steps_A", "termination_reason_A",
        "blocked_steps_A",
    ):
        require(geometry_metric.get(field) == cdec_metric.get(field),
                f"{scene}/{episode}: Goal-A metric differs at {field}")
    require(geometry_metric.get("leg1_trace_sha256") == trace_sha,
            f"{scene}/{episode}: metric trace SHA differs")

    reached_a = truth(geometry_metric["reached_A"])
    plans_g = geometry.get("legB")
    plans_c = cdec.get("legB")
    require(isinstance(plans_g, list) and isinstance(plans_c, list),
            f"{scene}/{episode}: missing B plans")
    if not reached_a:
        require(not plans_g and not plans_c,
                f"{scene}/{episode}: B ran after A failure")
        return {
            "reached_a": False, "reached_b_geometry": False,
            "reached_b_cdec": False, "joint_geometry": False,
            "joint_cdec": False, "learned_invoked": False,
            "learned_takeover": False, "geometry_accepted": False,
            "no_treatment_exact": True,
        }
    require(plans_g and plans_c, f"{scene}/{episode}: A success lacks B plans")
    require(all(
        plan.get("certified_relocalization_learned_rescue_requested") is False
        for plan in plans_g), f"{scene}/{episode}: baseline requested learned")
    require(all(
        plan.get("certified_relocalization_learned_rescue_requested") is True
        for plan in plans_c), f"{scene}/{episode}: CDEC request missing")
    for plan in plans_g + plans_c:
        requested = plan.get("requested_diffusion_seed")
        echoed = plan.get("diffusion_seed")
        require(requested is not None and int(requested) == int(echoed),
                f"{scene}/{episode}: diffusion seed contract changed")

    first_g, first_c = plans_g[0], plans_c[0]
    for field in (
        "step", "requested_diffusion_seed", "diffusion_seed", "frame_idx",
        "goal_start_frame", "candidate_ceiling",
        "router_candidate_order_dino", "router_candidate_order_used",
    ):
        require(first_g.get(field) == first_c.get(field),
                f"{scene}/{episode}: pre-proposal field changed: {field}")
    attempt_g = _first_geometry_attempt(first_g)
    attempt_c = _first_geometry_attempt(first_c)
    require(attempt_g == attempt_c,
            f"{scene}/{episode}: first geometry certificate differs")
    geometry_accepted = bool(
        attempt_c is not None and attempt_c.get("accepted") is True)
    learned = first_c.get("certified_relocalization_learned_proposal")
    invoked = False
    if isinstance(learned, Mapping):
        require(learned.get("activation_authorized") is False,
                f"{scene}/{episode}: model gained activation authority")
        status = learned.get("status")
        require(status != "runtime_exception_fail_closed",
                f"{scene}/{episode}: learned runtime failure")
        invoked = status not in {
            None, "not_requested", "not_evaluated_geometry_accepted"}
        if geometry_accepted:
            require(status == "not_evaluated_geometry_accepted",
                    f"{scene}/{episode}: learned ran before geometry pass")
    else:
        require(attempt_c is None and not geometry_accepted,
                f"{scene}/{episode}: missing learned diagnostic")

    takeover = _takeover(first_c)
    if takeover:
        require(not geometry_accepted and invoked,
                f"{scene}/{episode}: takeover order changed")
        attempts = first_c.get("certified_relocalization_proposal_attempts")
        require(isinstance(attempts, list) and len(attempts) == 2
                and attempts[1].get("source") == "learned_on_geometry_reject"
                and attempts[1].get("accepted") is True,
                f"{scene}/{episode}: takeover lacks second certificate")
    else:
        require(_neutral(plans_g) == _neutral(plans_c),
                f"{scene}/{episode}: no-treatment plan trace changed")
        for field in (
            "reached_B", "spl_B", "spl_B_with_terminal", "geo_B",
            "steps_B", "steps_B_diagnostic", "steps_B_at_reach", "len_B",
            "len_B_at_reach", "final_dist_B", "termination_reason_B",
            "blocked_steps_B", "blocked_step_rate_B",
            "terminal_final_goal_dist_m",
        ):
            require(geometry_metric.get(field) == cdec_metric.get(field),
                    f"{scene}/{episode}: no-treatment metric changed: {field}")

    reached_g = truth(geometry_metric["reached_B"])
    reached_c = truth(cdec_metric["reached_B"])
    if reached_g != reached_c:
        require(takeover, f"{scene}/{episode}: effect without learned takeover")
    return {
        "reached_a": True,
        "reached_b_geometry": reached_g,
        "reached_b_cdec": reached_c,
        "joint_geometry": reached_g,
        "joint_cdec": reached_c,
        "learned_invoked": invoked,
        "learned_takeover": takeover,
        "geometry_accepted": geometry_accepted,
        "no_treatment_exact": not takeover,
    }


def _verify_equal(expected: Any, observed: Any, path: str) -> None:
    if isinstance(expected, Mapping):
        require(isinstance(observed, Mapping), f"official {path} is not an object")
        for key, value in expected.items():
            require(key in observed, f"official report lacks {path}.{key}")
            _verify_equal(value, observed[key], f"{path}.{key}")
    elif isinstance(expected, list):
        require(isinstance(observed, list) and len(expected) == len(observed),
                f"official list differs at {path}")
        for index, value in enumerate(expected):
            _verify_equal(value, observed[index], f"{path}[{index}]")
    elif isinstance(expected, float):
        require(isinstance(observed, (int, float))
                and math.isclose(expected, float(observed), rel_tol=0.0, abs_tol=1e-15),
                f"official float differs at {path}: {observed} != {expected}")
    else:
        require(observed == expected,
                f"official value differs at {path}: {observed!r} != {expected!r}")


def verify(
    *, manifest_path: Path, trace_receipt_path: Path,
    run_root: Path, official_report_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(trace_receipt_path.read_text(encoding="utf-8"))
    official = json.loads(official_report_path.read_text(encoding="utf-8"))
    require(manifest.get("audit", {}).get("status") == "ok",
            "manifest audit failed")
    require(manifest.get("data_role_guards", {}).get("blind_allowed") is False,
            "manifest permits blind access")
    require(receipt.get("manifest_sha256") == sha256_file(manifest_path)
            and receipt.get("trace_payload_decoded") is False
            and receipt.get("episode_target_or_outcome_fields_accessed") is False
            and receipt.get("development_read") is False
            and receipt.get("blind_read") is False,
            "trace receipt scope changed")
    scenes = list(map(str, manifest.get("scenes", [])))
    require(len(scenes) == 20 and len(set(scenes)) == 20,
            "scene universe changed")
    episodes = {
        scene: [str(row["episode"]) for row in manifest["episodes"][scene]]
        for scene in scenes}
    require(all(len(value) == 8 for value in episodes.values()),
            "episode universe changed")

    records: dict[tuple[str, str], dict[str, Any]] = {}
    arm_success = {
        arm: {"novel": 0, "joint": 0, "conditional": 0, "eligible": 0}
        for arm in ARMS}
    for index, scene in enumerate(scenes):
        scene_root = run_root / "scenes" / f"{index:02d}_{scene}"
        contract = json.loads((scene_root / "scene_contract.json").read_text())
        require(contract.get("schema_version")
                == "cdec_consumed_closed_loop_scene_v1_20260813"
                and contract.get("scene") == scene
                and contract.get("scene_index") == index
                and contract.get("arm_order") == list(ORDERS[index % 2])
                and contract.get("cdec_artifact_sha256")
                == EXPECTED_ARTIFACT_SHA256
                and contract.get("stagnation_graph") == "off",
                f"{scene}: scene contract changed")
        metrics = {}
        for arm in ARMS:
            arm_root = scene_root / arm
            _validate_summary(arm_root / "summary.json", arm)
            rows = _read_metric(arm_root / "metric.csv")
            require([row["episode"] for row in rows] == episodes[scene],
                    f"{scene}/{arm}: metric identity changed")
            metrics[arm] = {row["episode"]: row for row in rows}
        trace_scene = receipt.get("scenes", {}).get(scene)
        require(isinstance(trace_scene, Mapping), f"{scene}: trace receipt missing")
        for episode in episodes[scene]:
            trace_sha = trace_scene.get("episodes", {}).get(episode)
            require(isinstance(trace_sha, str) and len(trace_sha) == 64,
                    f"{scene}/{episode}: trace hash missing")
            audit = _episode_audit(
                scene=scene, episode=episode, trace_sha=trace_sha,
                geometry_path=(scene_root / "geometry_certificate"
                               / f"{episode}_plans.json"),
                cdec_path=(scene_root / "cdec_cascade"
                           / f"{episode}_plans.json"),
                geometry_metric=metrics["geometry_certificate"][episode],
                cdec_metric=metrics["cdec_cascade"][episode],
            )
            records[(scene, episode)] = {
                "scene": scene, "episode": episode, **audit}
            for arm, b_key, j_key in (
                ("geometry_certificate", "reached_b_geometry", "joint_geometry"),
                ("cdec_cascade", "reached_b_cdec", "joint_cdec"),
            ):
                arm_success[arm]["novel"] += int(audit["reached_a"])
                arm_success[arm]["joint"] += int(audit[j_key])
                if audit["reached_a"]:
                    arm_success[arm]["eligible"] += 1
                    arm_success[arm]["conditional"] += int(audit[b_key])

    ordered = [records[key] for key in sorted(records)]
    require(len(ordered) == 160, "raw episode count changed")
    gains = [row for row in ordered
             if row["joint_cdec"] and not row["joint_geometry"]]
    losses = [row for row in ordered
              if row["joint_geometry"] and not row["joint_cdec"]]
    both = sum(row["joint_geometry"] and row["joint_cdec"] for row in ordered)
    neither = len(ordered) - both - len(gains) - len(losses)
    eligible = [row for row in ordered if row["reached_a"]]
    cboth = sum(row["reached_b_geometry"] and row["reached_b_cdec"]
                for row in eligible)
    cneither = len(eligible) - cboth - len(gains) - len(losses)
    geometry_rows = {
        (row["scene"], row["episode"]): {
            "reached_a": row["reached_a"],
            "reached_b": row["reached_b_geometry"],
            "joint": row["joint_geometry"],
        } for row in ordered}
    cdec_rows = {
        (row["scene"], row["episode"]): {
            "reached_a": row["reached_a"],
            "reached_b": row["reached_b_cdec"],
            "joint": row["joint_cdec"],
        } for row in ordered}
    analysis = manifest["analysis"]
    joint_ci = cluster_interval(
        scenes, episodes, geometry_rows, cdec_rows, conditional=False,
        seed=int(analysis["cluster_bootstrap_seed"]) + 401,
        resamples=int(analysis["cluster_bootstrap_resamples"]))
    conditional_ci = cluster_interval(
        scenes, episodes, geometry_rows, cdec_rows, conditional=True,
        seed=int(analysis["cluster_bootstrap_seed"]) + 402,
        resamples=int(analysis["cluster_bootstrap_resamples"]))
    pvalue = exact_mcnemar(len(gains), len(losses))
    gain_ids = [{"scene": row["scene"], "episode": row["episode"]}
                for row in gains]
    loss_ids = [{"scene": row["scene"], "episode": row["episode"]}
                for row in losses]
    checks = {
        "gain_in_at_least_two_scene_clusters": (
            len({row["scene"] for row in gains}) >= 2),
        "zero_paired_losses": not losses,
        "exact_mcnemar_below_0_05": pvalue < 0.05,
        "cluster_interval_lower_above_zero": joint_ci[0] > 0.0,
        "all_causal_and_safety_audits_pass": True,
        "every_gain_has_learned_certified_takeover": all(
            row["learned_takeover"] for row in gains),
    }
    decision = {
        "pass": all(checks.values()),
        "checks": checks,
        "branch": (
            "eligible_for_frozen_one_shot_system_confirmation"
            if all(checks.values()) else "do_not_promote_cdec"),
        "authorize_retuning_on_consumed_pool": False,
        "authorize_blind_opening_without_explicit_user_approval": False,
    }
    reconstructed = {
        "arms": {
            arm: {
                "episodes": 160,
                "novel": {"successes": values["novel"]},
                "joint": {"successes": values["joint"]},
                "revisit_given_novel_success": {
                    "eligible": values["eligible"],
                    "successes": values["conditional"],
                },
            } for arm, values in arm_success.items()
        },
        "contrasts": {
            "joint": {
                "outcomes": {
                    "both_joint_success": int(both),
                    "left_only_joint_success": len(losses),
                    "right_only_joint_success": len(gains),
                    "neither_joint_success": int(neither),
                },
                "joint_sr_delta_right_minus_left": (
                    (len(gains) - len(losses)) / 160),
                "mcnemar_exact_two_sided_p": pvalue,
                "scene_cluster_bootstrap_risk_difference_95": joint_ci,
            },
            "conditional_b": {
                "eligible_shared_novel_success": len(eligible),
                "outcomes": {
                    "both_revisit_success": int(cboth),
                    "left_only_revisit_success": len(losses),
                    "right_only_revisit_success": len(gains),
                    "neither_revisit_success": int(cneither),
                },
                "risk_difference_right_minus_left": (
                    (len(gains) - len(losses)) / len(eligible)
                    if eligible else None),
                "mcnemar_exact_two_sided_p": pvalue,
                "gains": gain_ids,
                "losses": loss_ids,
                "scene_cluster_bootstrap_risk_difference_95": conditional_ci,
            },
        },
        "learned_runtime": {
            "a_success_episodes": len(eligible),
            "invoked_episodes": sum(row["learned_invoked"] for row in ordered),
            "takeover_episodes": sum(row["learned_takeover"] for row in ordered),
            "geometry_accepted_episodes": sum(
                row["geometry_accepted"] for row in ordered),
            "no_treatment_exact_episodes": sum(
                row["no_treatment_exact"] for row in ordered),
            "gain_episodes": gain_ids,
            "loss_episodes": loss_ids,
        },
        "decision": decision,
    }
    _verify_equal(reconstructed, official, "report")
    require(official.get("audit", {}).get("development_read") is False
            and official.get("audit", {}).get("blind_read") is False
            and official.get("audit", {}).get("stagnation_graph_disabled") is True,
            "official scope changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "verified": True,
        "scope": {
            "independent_of_primary_summarizer": True,
            "consumed_pool_only": True,
            "development_read": False,
            "blind_read": False,
            "paper_final_confirmation": False,
        },
        "inputs": {
            "manifest_sha256": sha256_file(manifest_path),
            "trace_receipt_sha256": sha256_file(trace_receipt_path),
            "official_report_sha256": sha256_file(official_report_path),
        },
        "reconstructed": reconstructed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    return args


def main() -> None:
    args = parse_args()
    result = verify(
        manifest_path=args.manifest,
        trace_receipt_path=args.trace_receipt,
        run_root=args.run_root,
        official_report_path=args.official_report,
    )
    atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
