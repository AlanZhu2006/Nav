"""Fail-closed contract for the HM3D actual-mono length benchmark."""

from __future__ import annotations

import math
from collections import Counter


SCHEMA_VERSION = "hm3d_table3_length_role_pair_v1_20260830"
ROLES = frozenset({"novel", "revisit"})
RUNTIME_VISIBLE_QUERY_FIELDS = frozenset({
    "query_id", "goal_rgb", "goal_rgb_sha256", "goal_depth",
    "goal_depth_sha256", "floor_position", "yaw_rad",
})


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def finite(value, name: str) -> float:
    require(not isinstance(value, bool), f"{name} must be numeric")
    result = float(value)
    require(math.isfinite(result), f"{name} must be finite")
    return result


def sha(value, name: str) -> str:
    require(isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value),
            f"{name} must be a lowercase SHA256")
    return value


def runtime_query(query: dict) -> dict:
    validate_query(query)
    return {field: query[field] for field in RUNTIME_VISIBLE_QUERY_FIELDS}


def validate_query(query: dict) -> dict:
    require(isinstance(query, dict), "query must be an object")
    require(query.get("analysis_role") in ROLES, "query role changed")
    require(isinstance(query.get("query_id"), str) and query["query_id"],
            "query identity is invalid")
    for field in ("goal_rgb", "goal_depth"):
        require(isinstance(query.get(field), str) and query[field],
                f"{field} is invalid")
    sha(query.get("goal_rgb_sha256"), "goal_rgb_sha256")
    sha(query.get("goal_depth_sha256"), "goal_depth_sha256")
    position = query.get("floor_position")
    require(isinstance(position, list) and len(position) == 3,
            "query floor position is invalid")
    for index, value in enumerate(position):
        finite(value, f"floor_position[{index}]")
    finite(query.get("yaw_rad"), "yaw_rad")
    require(finite(query.get("geodesic_from_a_end_m"), "geodesic") > 0,
            "query geodesic must be positive")
    finite(query.get("initial_path_bearing_rad"), "initial bearing")
    support = finite(query.get("max_online_a_covis"), "co-visibility")
    require(0.0 <= support <= 1.0, "co-visibility is outside [0,1]")
    return query


def angle_separation_degrees(first: float, second: float) -> float:
    return abs(math.degrees(
        (float(first) - float(second) + math.pi) % (2 * math.pi) - math.pi
    ))


def in_bin(distance: float, spec: dict) -> bool:
    lower = float(spec["lower_inclusive"])
    upper = float(spec["upper"])
    return distance >= lower and (
        distance <= upper if spec.get("upper_inclusive") else distance < upper
    )


def validate_episode(episode: dict, contract: dict) -> dict:
    require(episode.get("schema_version") == SCHEMA_VERSION,
            "episode schema changed")
    bins = {row["name"]: row for row in contract["bins_m"]}
    bin_name = episode.get("bin_name")
    require(bin_name in bins, "episode length bin changed")
    sha(episode.get("candidate_identity_sha256"), "candidate identity")
    sha(episode.get("online_a_receipt_sha256"), "online receipt")
    sha(episode.get("online_a_trace_sha256"), "online trace")
    require(episode.get("runtime_geometry")
            == "content_addressed_pinned_navmesh",
            "Table-III runtime geometry changed")
    require(isinstance(episode.get("runtime_navmesh"), str)
            and episode["runtime_navmesh"],
            "Table-III runtime navmesh path is missing")
    sha(episode.get("runtime_navmesh_sha256"), "runtime navmesh")
    require(int(episode.get("online_a_steps", 0)) >= 40,
            "online history is too short")
    endpoint = episode.get("online_a_endpoint")
    require(isinstance(endpoint, dict)
            and isinstance(endpoint.get("floor_position"), list)
            and len(endpoint["floor_position"]) == 3,
            "online endpoint is invalid")
    pairs = episode.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == 1,
            "Table-III episode must contain one role pair")
    pair = pairs[0]
    queries = pair.get("queries")
    require(isinstance(queries, list) and len(queries) == 2,
            "Table-III pair must contain two queries")
    require(Counter(q.get("analysis_role") for q in queries)
            == Counter({"novel": 1, "revisit": 1}),
            "Table-III pair roles changed")
    by_role = {}
    for query in queries:
        validate_query(query)
        by_role[query["analysis_role"]] = query
        distance = float(query["geodesic_from_a_end_m"])
        require(in_bin(distance, bins[bin_name]),
                "query escaped its frozen length bin")
        support = float(query["max_online_a_covis"])
        if query["analysis_role"] == "novel":
            require(support < float(contract["novel_max_covis_exclusive"]),
                    "Novel query has historical support")
        else:
            require(support >= float(contract["revisit_min_covis_inclusive"]),
                    "Revisit query lacks historical support")
    distance_error = abs(
        float(by_role["novel"]["geodesic_from_a_end_m"])
        - float(by_role["revisit"]["geodesic_from_a_end_m"])
    )
    require(distance_error <= float(contract["maximum_role_distance_mismatch_m"]),
            "role distances are not matched")
    require(abs(float(pair["role_distance_error_m"]) - distance_error) <= 1e-6,
            "declared role distance error changed")
    separation = angle_separation_degrees(
        by_role["novel"]["initial_path_bearing_rad"],
        by_role["revisit"]["initial_path_bearing_rad"],
    )
    require(separation >= float(contract["minimum_initial_bearing_separation_deg"]),
            "role bearings are insufficiently separated")
    require(abs(float(pair["role_initial_path_bearing_separation_deg"])
                - separation) <= 1e-6,
            "declared role bearing separation changed")
    return episode


def validate_manifest(manifest: dict) -> dict:
    require(manifest.get("schema_version") == SCHEMA_VERSION,
            "Table-III manifest schema changed")
    contract = manifest.get("contract")
    require(isinstance(contract, dict), "Table-III contract is missing")
    require(contract.get("online_history")
            == "actual_frozen_navdp_goal_a_causal_rgb_trace",
            "Table-III online-history contract changed")
    require(contract.get("query_execution")
            == "independent_reset_and_exact_online_a_replay",
            "Table-III query execution changed")
    require(contract.get("runtime_role_visibility") == "none",
            "runtime role became visible")
    episodes = manifest.get("episodes")
    require(isinstance(episodes, list) and episodes,
            "Table-III manifest has no episodes")
    identities = set()
    for episode in episodes:
        validate_episode(episode, contract)
        identity = (episode["scene"], episode["episode"])
        require(identity not in identities, "duplicate Table-III episode")
        identities.add(identity)
    return manifest


__all__ = [
    "RUNTIME_VISIBLE_QUERY_FIELDS", "SCHEMA_VERSION", "angle_separation_degrees",
    "in_bin", "runtime_query", "validate_episode", "validate_manifest",
    "validate_query",
]
