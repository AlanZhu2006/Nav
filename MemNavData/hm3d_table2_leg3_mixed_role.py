"""Pure contracts for the HM3D continual Table-2 Leg-3 experiment.

The experiment conditions on one already sealed, successful actual-mono
``A -> B`` prefix.  It then constructs one unsupported Novel query and one
historically supported Revisit query from the same physical endpoint.  The
runtime never receives the analysis role.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_SCHEMA = "hm3d_table2_leg3_mixed_role_protocol_v1_20260829"
FRAGMENT_SCHEMA = "hm3d_table2_leg3_mixed_role_fragment_v1_20260829"
POPULATION_SCHEMA = "hm3d_table2_leg3_mixed_role_population_v1_20260829"
VERIFICATION_SCHEMA = (
    "hm3d_table2_leg3_mixed_role_construction_verification_v1_20260829"
)
PREFIX_RECEIPT_SCHEMA = "hm3d_table2_actual_mono_ab_prefix_v1_20260829"
STRATA = ("front", "side", "rear")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(payload.get("schema_version") == PROTOCOL_SCHEMA,
            "Table-2 protocol schema changed")
    require(payload.get("leg3_query_outcomes_read_before_freeze") is False,
            "Table-2 protocol was frozen after Leg-3 outcomes")
    prefix = payload["shared_prefix"]
    require(prefix["sequence"] == ["A_novel", "B_novel"],
            "Table-2 A/B prefix sequence changed")
    require(prefix["A_controller"] == prefix["B_controller"]
            == "frozen_navdp_native_monocular_sidecar",
            "Table-2 factual prefix controller changed")
    query = payload["leg3_queries"]
    require(query["roles"] == ["novel", "revisit"],
            "Table-2 Leg-3 roles changed")
    require(query["runtime_role_visibility"] == "none",
            "Table-2 runtime role became visible")
    require(float(query["novel_max_combined_AB_covis_exclusive"]) == 0.10,
            "Table-2 Novel support ceiling changed")
    require(float(query["revisit_min_combined_AB_covis_inclusive"]) == 0.55,
            "Table-2 Revisit support floor changed")
    runtime = payload["runtime"]
    require(runtime["arms"] == ["mono_native", "mono_cec"],
            "Table-2 treatment arms changed")
    require(runtime["same_AB_prefix_replayed_before_every_arm"] is True,
            "Table-2 lost exact prefix pairing")
    require(runtime["maximum_steps"] == 600
            and runtime["execution_horizon"] == 8
            and math.isclose(float(runtime["success_radius_m"]), 1.0),
            "Table-2 navigation budget changed")
    guards = payload["guards"]
    require(guards["no_training"] is True
            and guards["no_metric_depth_for_control_or_CEC"] is True
            and guards["construction_before_policy_rollout"] is True
            and guards["independent_construction_verifier_required"] is True,
            "Table-2 guards changed")
    return payload


def stable_u32(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def stratum_order(population_index: int, scene: str,
                  episode: str) -> tuple[str, ...]:
    """Return a deterministic balanced-first direction search order."""

    require(population_index >= 0, "population index must be non-negative")
    preferred = population_index % len(STRATA)
    tail = list(STRATA)
    tail.remove(STRATA[preferred])
    if stable_u32("hm3d_table2_leg3_tail", scene, episode) % 2:
        tail.reverse()
    return (STRATA[preferred], *tail)


def _shift_step(row: dict[str, Any], offset: int) -> dict[str, Any]:
    shifted = deepcopy(row)
    shifted["step"] = int(row["step"]) + int(offset)
    return shifted


def compose_actual_ab_trace(trace_a: dict[str, Any], trace_b: dict[str, Any],
                            *, episode: str) -> dict[str, Any]:
    """Compose two factual native traces without inventing observations.

    Only the dense step coordinate is shifted.  Every RGB hash, pose, plan
    payload and controller receipt remains byte-derived from the original
    rollout that produced it.
    """

    require(trace_a.get("reached") is True and trace_b.get("reached") is True,
            "A/B prefix requires two successful factual traces")
    require(int(trace_a["steps"]) == len(trace_a["poses"])
            and int(trace_b["steps"]) == len(trace_b["poses"]),
            "A/B trace pose count changed")
    require(trace_a["source_scene"] == trace_b["source_scene"],
            "A/B traces belong to different scenes")
    require(int(trace_a["episode_seed"]) == int(trace_b["episode_seed"]),
            "A/B traces use different episode seeds")
    offset = len(trace_a["poses"])
    poses = [deepcopy(row) for row in trace_a["poses"]] + [
        _shift_step(row, offset) for row in trace_b["poses"]
    ]
    plans = [deepcopy(row) for row in trace_a["plans"]] + [
        _shift_step(row, offset) for row in trace_b["plans"]
    ]
    require([int(row["step"]) for row in poses] == list(range(len(poses))),
            "composed A/B observations are not dense")
    plan_steps = [int(row["step"]) for row in plans]
    require(plan_steps == sorted(set(plan_steps)),
            "composed A/B decision steps are not strictly increasing")
    require(all(row.get("navdp_depth_source") == "monocular_sidecar"
                and row.get("metric_depth_sensor_consumed") is False
                for row in plans),
            "composed A/B prefix is not fully monocular")

    payload = deepcopy(trace_b)
    payload.update({
        "episode": str(episode),
        "goal_source_episode": str(episode),
        "steps": len(poses),
        "poses": poses,
        "plans": plans,
        "path_len": float(trace_a["path_len"]) + float(trace_b["path_len"]),
        "path_len_at_reach": (
            float(trace_a["path_len"]) + float(
                trace_b.get("path_len_at_reach")
                if trace_b.get("path_len_at_reach") is not None
                else trace_b["path_len"]
            )
        ),
        "step_at_reach": len(poses) - 1,
        "prefix_semantics": "exact_actual_mono_A_then_B_observation_concat",
        "prefix_A_steps": offset,
        "prefix_B_steps": len(trace_b["poses"]),
    })
    return payload


def novel_query(row: dict[str, Any]) -> dict[str, Any]:
    queries = [
        query for pair in row["pairs"] for query in pair["queries"]
        if query["analysis_role"] == "novel"
    ]
    require(len(queries) == 1, "history has no unique Novel query")
    return queries[0]


def power(rows: Iterable[dict[str, Any]], *, target_histories: int,
          target_scenes: int, minimum_per_stratum: int) -> dict[str, Any]:
    rows = list(rows)
    counts = Counter(
        str(novel_query(row)["assigned_direction_stratum"]) for row in rows
    )
    for name in STRATA:
        counts.setdefault(name, 0)
    scenes = {str(row["scene"]) for row in rows}
    target_met = (
        len(rows) >= int(target_histories)
        and len(scenes) >= int(target_scenes)
        and all(counts[name] >= int(minimum_per_stratum) for name in STRATA)
    )
    return {
        "histories": len(rows),
        "scene_clusters": len(scenes),
        "direction_strata": {name: counts[name] for name in STRATA},
        "target_histories": int(target_histories),
        "target_scene_clusters": int(target_scenes),
        "minimum_histories_per_direction_stratum": int(minimum_per_stratum),
        "target_met": bool(target_met),
    }


__all__ = [
    "FRAGMENT_SCHEMA",
    "POPULATION_SCHEMA",
    "PREFIX_RECEIPT_SCHEMA",
    "PROTOCOL_SCHEMA",
    "STRATA",
    "VERIFICATION_SCHEMA",
    "compose_actual_ab_trace",
    "load_protocol",
    "novel_query",
    "power",
    "require",
    "sha256_file",
    "stable_u32",
    "stratum_order",
]
