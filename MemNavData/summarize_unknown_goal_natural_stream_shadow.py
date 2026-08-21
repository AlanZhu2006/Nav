#!/usr/bin/env python3
"""Validate non-intervention and evidence completeness of stream shadow traces."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


LEGS = ("legA", "legB", "legC")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def summarize(run_root: Path, buffer_root: Path | None = None,
              allow_censored_legs: bool = False) -> dict[str, object]:
    paths = sorted(run_root.glob("episode_*_plans.json"))
    require(bool(paths), f"no plan traces under {run_root}")
    leg_counts = {leg: 0 for leg in LEGS}
    total_plans = 0
    candidate_plans = 0
    full_verification_plans = 0
    trials_total = 0
    max_pool = 0
    max_trials = 0
    active = 0
    takeovers = 0
    malformed_trials = 0
    missing_trace_frames = 0
    missing_buffer_images = 0
    censored_legs = {leg: 0 for leg in LEGS}
    plan_hashes = {}
    for episode_number, path in enumerate(paths, start=1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(isinstance(payload, dict), f"{path}: plan root must be object")
        plan_hashes[str(path.resolve())] = sha256(path)
        rollout_traces = payload.get("rollout_traces")
        memory_traces = payload.get("memory_traces")
        require(isinstance(rollout_traces, dict),
                f"{path}: rollout_traces absent")
        require(isinstance(memory_traces, dict),
                f"{path}: memory_traces absent")
        memory_by_frame = {}
        for leg in LEGS:
            memory = memory_traces.get(leg)
            require(isinstance(memory, list)
                    and (bool(memory) or allow_censored_legs),
                    f"{path}: {leg} memory trace absent/empty")
            if not memory:
                censored_legs[leg] += 1
            for item in memory:
                require(isinstance(item, dict),
                        f"{path}: malformed {leg} memory trace")
                frame_idx = item.get("frame_idx")
                require(isinstance(frame_idx, int) and frame_idx not in memory_by_frame,
                        f"{path}: duplicate/invalid memory frame")
                require(all(finite_number(item.get(key))
                            for key in ("x", "z", "yaw")),
                        f"{path}: non-finite memory pose")
                memory_by_frame[frame_idx] = item
        for leg in LEGS:
            plans = payload.get(leg)
            require(isinstance(plans, list)
                    and (bool(plans) or allow_censored_legs),
                    f"{path}: {leg} absent/empty")
            rollout = rollout_traces.get(leg)
            require(isinstance(rollout, list)
                    and (bool(rollout) or allow_censored_legs),
                    f"{path}: {leg} rollout trace absent/empty")
            require(bool(plans) == bool(rollout) == bool(memory_traces.get(leg)),
                    f"{path}: {leg} censoring fields disagree")
            rollout_by_step = {}
            for item in rollout:
                require(isinstance(item, dict),
                        f"{path}: malformed {leg} rollout trace")
                step = item.get("step")
                require(isinstance(step, int) and step not in rollout_by_step,
                        f"{path}: duplicate/invalid rollout step")
                require(all(finite_number(item.get(key))
                            for key in ("x", "y", "z", "yaw")),
                        f"{path}: non-finite rollout pose")
                digest = item.get("jpg_sha256")
                require(isinstance(digest, str) and len(digest) == 64,
                        f"{path}: rollout image hash absent")
                rollout_by_step[step] = item
            leg_counts[leg] += len(plans)
            for plan in plans:
                require(isinstance(plan, dict), f"{path}: malformed {leg} plan")
                total_plans += 1
                require(plan.get("router_active") is False,
                        f"{path}: router intervened in {leg}")
                active += int(plan.get("router_active") is True)
                takeover = plan.get("revisit_adapter_takeover")
                require(takeover is False,
                        f"{path}: adapter takeover was not explicitly false")
                takeovers += int(takeover is True)
                pool = plan.get("router_candidate_pool_size")
                considered = plan.get("router_candidates_considered")
                trials = plan.get("router_candidate_trials")
                require(isinstance(pool, int) and 0 <= pool <= 8,
                        f"{path}: invalid candidate pool {pool!r}")
                require(isinstance(considered, int) and 0 <= considered <= 8,
                        f"{path}: invalid considered count {considered!r}")
                require(isinstance(trials, list) and len(trials) == considered,
                        f"{path}: candidate trial count mismatch")
                plan_step = plan.get("step")
                current_frame = plan.get("frame_idx")
                if (not isinstance(plan_step, int)
                        or plan_step not in rollout_by_step
                        or not isinstance(current_frame, int)
                        or current_frame not in memory_by_frame):
                    missing_trace_frames += 1
                if buffer_root is not None and isinstance(current_frame, int):
                    image = buffer_root / f"ep_{episode_number:04d}" / f"{current_frame}.jpg"
                    if not image.is_file():
                        missing_buffer_images += 1
                max_pool = max(max_pool, pool)
                max_trials = max(max_trials, considered)
                if pool:
                    candidate_plans += 1
                    require(considered == pool,
                            f"{path}: shadow did not verify full candidate pool")
                    full_verification_plans += 1
                for trial in trials:
                    valid = (
                        isinstance(trial, dict)
                        and isinstance(trial.get("anchor"), int)
                        and finite_number(trial.get("score"))
                        and isinstance(trial.get("matches"), int)
                        and isinstance(trial.get("inliers"), int)
                        and finite_number(trial.get("inlier_ratio"))
                    )
                    if not valid:
                        malformed_trials += 1
                    anchor = trial.get("anchor") if isinstance(trial, dict) else None
                    if (not isinstance(anchor, int)
                            or anchor not in memory_by_frame
                            or (isinstance(current_frame, int)
                                and anchor >= current_frame)):
                        missing_trace_frames += 1
                    if buffer_root is not None and isinstance(anchor, int):
                        image = buffer_root / f"ep_{episode_number:04d}" / f"{anchor}.jpg"
                        if not image.is_file():
                            missing_buffer_images += 1
                    trials_total += 1
    require(malformed_trials == 0, "one or more geometry trials are malformed")
    require(missing_trace_frames == 0,
            "query/candidate frames do not map to saved natural traces")
    require(missing_buffer_images == 0,
            "query/candidate frames do not map to saved RGB buffers")
    report = {
        "schema_version": "unknown_goal_natural_stream_shadow_smoke_v1",
        "status": "complete",
        "scope": "collection-contract smoke only; no SR or method claim",
        "episodes": len(paths),
        "plan_files_sha256": plan_hashes,
        "leg_plan_counts": leg_counts,
        "plans_total": total_plans,
        "candidate_plans": candidate_plans,
        "full_verification_plans": full_verification_plans,
        "geometry_trials": trials_total,
        "memory_support_evidence_observed": bool(
            candidate_plans > 0 and trials_total > 0),
        "max_candidate_pool": max_pool,
        "max_trials_per_plan": max_trials,
        "router_active_plans": active,
        "adapter_takeover_plans": takeovers,
        "malformed_trials": malformed_trials,
        "missing_trace_frames": missing_trace_frames,
        "missing_buffer_images": missing_buffer_images,
        "allow_censored_legs": bool(allow_censored_legs),
        "censored_episode_legs": censored_legs,
        "contract_pass": bool(
            (allow_censored_legs or all(count > 0 for count in leg_counts.values()))
            and total_plans > 0
            and candidate_plans == full_verification_plans
            and active == 0
            and takeovers == 0
            and malformed_trials == 0
            and missing_trace_frames == 0
            and missing_buffer_images == 0
        ),
    }
    require(report["contract_pass"], "natural-stream smoke contract failed")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--buffer-root", type=Path)
    parser.add_argument("--allow-censored-legs", action="store_true")
    args = parser.parse_args()
    report = summarize(
        args.run_root,
        args.buffer_root,
        allow_censored_legs=args.allow_censored_legs,
    )
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
