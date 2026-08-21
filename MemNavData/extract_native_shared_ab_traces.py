#!/usr/bin/env python3
"""Seal successful native 3-leg A/B rollouts as shared factual prefixes.

The source native arm had no MemNav observer.  Replaying its already frozen
RGB/pose stream into memory is therefore explicitly observation-only: it
cannot alter A/B control, while every later C arm receives the same causal
history.  A's exact endpoint is the first factual B observation; B's exact
endpoint is the first factual C observation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from deterministic_eval_protocol import (
    LEG1_TRACE_SCHEMA_VERSION,
    NATIVE_OBSERVATION_REPLAY_CONTRACT,
    bytes_sha256,
    file_sha256,
    write_leg1_trace,
)


SCHEMA_VERSION = "native_shared_ab_extraction_v1_20260813"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def metric_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def float_or_none(value: str) -> float | None:
    return None if value in ("", None) else float(value)


def trace_payload(
    *,
    scene: str,
    episode: str,
    episode_seed: int,
    goal: bytes,
    poses: list[dict],
    plans: list[dict],
    reached: bool,
    path_len: float,
    final_distance: float,
    endpoint_pose: dict,
    leg_name: str,
    endpoint_source: str,
    checkpoint_sha256: str,
    source_metric_sha256: str,
    source_plans_sha256: str,
) -> dict:
    require(bool(poses), f"{leg_name} has no factual observations")
    require(len(poses) == int(poses[-1]["step"]) + 1, f"{leg_name} poses are sparse")
    return {
        "schema_version": LEG1_TRACE_SCHEMA_VERSION,
        "episode": episode,
        "episode_seed": int(episode_seed),
        "goal_sha256": bytes_sha256(goal),
        "goal_source_episode": episode,
        "source_scene": scene,
        "source_backend": "navdp",
        "source_hybrid_route": "phase",
        "source_control_contract": NATIVE_OBSERVATION_REPLAY_CONTRACT,
        "source_memory_observer_present": False,
        "source_navdp_checkpoint_sha256": checkpoint_sha256,
        # Required by the historical trace schema.  For a native source these
        # fields bind the destination replay server, not factual A/B control.
        "source_retrieval_candidate_min_gap": 16,
        "source_graph_subgoal_spacing_m": 0.0,
        "source_graph_subgoal_arrival_m": 0.60,
        "legacy_memory_fields_bind_replay_destination": True,
        "source_leg": leg_name,
        "source_metric_sha256": source_metric_sha256,
        "source_plans_sha256": source_plans_sha256,
        "endpoint_source": endpoint_source,
        "reached": bool(reached),
        "path_len": float(path_len),
        "path_len_at_reach": float(path_len) if reached else None,
        "step_at_reach": len(poses) if reached else None,
        "steps": len(poses),
        "termination_reason": "success" if reached else "source_failure",
        "blocked_step_count": 0,
        "final_goal_dist_m": float(final_distance),
        "end_position": [
            float(endpoint_pose[axis]) for axis in ("x", "y", "z")
        ],
        "end_yaw": float(endpoint_pose["yaw"]),
        "poses": poses,
        "plans": plans,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--native-eval-root", type=Path, required=True)
    parser.add_argument("--navdp-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene-ids", default="")
    parser.add_argument("--episode-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(not args.out.exists(), f"output already exists: {args.out}")
    checkpoint_sha = file_sha256(args.navdp_checkpoint)
    scenes_wanted = {
        item.strip() for item in args.scene_ids.split(",") if item.strip()
    }
    episodes_wanted = {
        item.strip() for item in args.episode_ids.split(",") if item.strip()
    }
    args.out.mkdir(parents=True)
    accepted = []
    excluded = []
    for scene_dir in sorted(args.native_eval_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        if scenes_wanted and scene not in scenes_wanted:
            continue
        metric_path = scene_dir / "metric.csv"
        require(metric_path.is_file(), f"missing native metric for {scene}")
        metric_sha = file_sha256(metric_path)
        for row in metric_rows(metric_path):
            episode = row["episode"]
            if episodes_wanted and episode not in episodes_wanted:
                continue
            reason = None
            if row.get("multigoal_contract_ok") != "1":
                reason = "strict_v4_contract_failed"
            elif row.get("server_backend") != "navdp":
                reason = "source_not_native_navdp"
            elif row.get("reached_A") != "1" or row.get("reached_B") != "1":
                reason = "outside_conditional_AB_denominator"
            if reason is not None:
                excluded.append({"scene": scene, "episode": episode, "reason": reason})
                continue

            source_episode = args.generation_root / scene / episode
            metadata_path = source_episode / "meta/gen_meta.json"
            require(metadata_path.is_file(), f"missing source metadata {scene}/{episode}")
            metadata = json.loads(metadata_path.read_text())
            require(
                metadata.get("gen_protocol") == "multileg_v4_role_paired_20260812",
                "source episode is not strict-v4",
            )
            switch_a = int(metadata["switches"][0])
            rgb_root = source_episode / "videos/chunk-000/observation.images.rgb"
            goal_a_path = rgb_root / f"{switch_a - 1}.jpg"
            goal_b_path = source_episode / "goal_1.jpg"
            require(goal_a_path.is_file() and goal_b_path.is_file(), "source goals missing")
            plans_path = scene_dir / f"{episode}_plans.json"
            require(plans_path.is_file(), f"missing source plans {scene}/{episode}")
            plans_sha = file_sha256(plans_path)
            source_plans = json.loads(plans_path.read_text())
            rollouts = source_plans["rollout_traces"]
            require(bool(rollouts["legA"]), "source A trace is empty")
            require(bool(rollouts["legB"]), "source B trace is empty")
            require(
                bool(rollouts["legC"]),
                "successful source B lacks the first factual C observation",
            )
            require(
                len(rollouts["legA"]) == int(row["steps_A"]),
                "A step/trace count mismatch",
            )
            require(
                len(rollouts["legB"]) == int(row["steps_B"]),
                "B step/trace count mismatch",
            )
            episode_seed = int(row["seed"])
            destination = args.out / scene
            destination.mkdir(exist_ok=True)
            payload_a = trace_payload(
                scene=scene,
                episode=episode,
                episode_seed=episode_seed,
                goal=goal_a_path.read_bytes(),
                poses=rollouts["legA"],
                plans=source_plans["legA"],
                reached=True,
                path_len=float(row["len_A"]),
                final_distance=float(row["final_dist_A"]),
                endpoint_pose=rollouts["legB"][0],
                leg_name="A",
                endpoint_source="first_factual_online_B_observation",
                checkpoint_sha256=checkpoint_sha,
                source_metric_sha256=metric_sha,
                source_plans_sha256=plans_sha,
            )
            payload_b = trace_payload(
                scene=scene,
                episode=episode,
                episode_seed=episode_seed,
                goal=goal_b_path.read_bytes(),
                poses=rollouts["legB"],
                plans=source_plans["legB"],
                reached=True,
                path_len=float(row["len_B"]),
                final_distance=float(row["final_dist_B"]),
                endpoint_pose=rollouts["legC"][0],
                leg_name="B",
                endpoint_source="first_factual_online_C_observation",
                checkpoint_sha256=checkpoint_sha,
                source_metric_sha256=metric_sha,
                source_plans_sha256=plans_sha,
            )
            trace_a_path = destination / f"{episode}_leg1_trace.json"
            trace_b_path = destination / f"{episode}_legB_trace.json"
            trace_a_sha = write_leg1_trace(trace_a_path, payload_a)
            trace_b_sha = write_leg1_trace(trace_b_path, payload_b)
            accepted.append({
                "scene": scene,
                "episode": episode,
                "episode_seed": episode_seed,
                "source_metadata_sha256": file_sha256(metadata_path),
                "source_metric_sha256": metric_sha,
                "source_plans_sha256": plans_sha,
                "online_A_trace_sha256": trace_a_sha,
                "online_B_trace_sha256": trace_b_sha,
            })

    require(bool(accepted), "no successful native A/B prefixes were extracted")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "conditional_population": "native strict-v4 episodes with A and B success",
        "source_memory_observer_present": False,
        "replay_is_observation_only": True,
        "source_navdp_checkpoint": str(args.navdp_checkpoint.resolve()),
        "source_navdp_checkpoint_sha256": checkpoint_sha,
        "generation_root": str(args.generation_root.resolve()),
        "native_eval_root": str(args.native_eval_root.resolve()),
        "accepted": accepted,
        "excluded": excluded,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(args.out),
        "accepted": len(accepted),
        "scenes": len({row["scene"] for row in accepted}),
        "manifest_sha256": file_sha256(manifest_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
