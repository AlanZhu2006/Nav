"""Fail-closed contract for matched Novel/Revisit queries after online Goal A.

The benchmark stores the causal role for stratified analysis, but a runtime arm
must receive only the RGB goal image.  Every pair starts from the same frozen
online-A endpoint and differs only in whether the goal view has audited visual
support in that online history.
"""

from __future__ import annotations

import math
from collections import Counter


SCHEMA_VERSION = "shared_online_role_pair_v1_20260814"
ROLES = frozenset({"novel", "revisit"})
RUNTIME_VISIBLE_QUERY_FIELDS = frozenset({
    "query_id",
    "goal_rgb",
    "goal_rgb_sha256",
    "goal_depth",
    "goal_depth_sha256",
    "floor_position",
    "yaw_rad",
})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value, name: str) -> float:
    _require(not isinstance(value, bool), f"{name} must be numeric")
    converted = float(value)
    _require(math.isfinite(converted), f"{name} must be finite")
    return converted


def _sha256(value, name: str) -> str:
    _require(isinstance(value, str), f"{name} must be a string")
    _require(
        len(value) == 64 and all(char in "0123456789abcdef" for char in value),
        f"{name} must be a lowercase SHA256",
    )
    return value


def runtime_query(query: dict) -> dict:
    """Return the only query fields an evaluation arm may receive."""
    validate_query(query)
    return {field: query[field] for field in RUNTIME_VISIBLE_QUERY_FIELDS}


def validate_query(query: dict) -> dict:
    _require(isinstance(query, dict), "query must be an object")
    role = query.get("analysis_role")
    _require(role in ROLES, "query has an unsupported analysis role")
    query_id = query.get("query_id")
    _require(isinstance(query_id, str) and query_id, "query_id is invalid")
    for field in ("goal_rgb", "goal_depth"):
        _require(
            isinstance(query.get(field), str) and query[field],
            f"{field} is invalid",
        )
    _sha256(query.get("goal_rgb_sha256"), "goal_rgb_sha256")
    _sha256(query.get("goal_depth_sha256"), "goal_depth_sha256")
    position = query.get("floor_position")
    _require(
        isinstance(position, list) and len(position) == 3,
        "floor_position must have three coordinates",
    )
    for index, value in enumerate(position):
        _finite(value, f"floor_position[{index}]")
    _finite(query.get("yaw_rad"), "yaw_rad")
    distance = _finite(query.get("geodesic_from_a_end_m"), "geodesic")
    _require(distance > 0.0, "query geodesic must be positive")
    _finite(query.get("initial_path_bearing_rad"), "initial_path_bearing_rad")
    maximum_covis = _finite(
        query.get("max_online_a_covis"), "max_online_a_covis"
    )
    _require(0.0 <= maximum_covis <= 1.0, "co-visibility is outside [0,1]")
    return query


def validate_episode(payload: dict, contract: dict) -> dict:
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        "episode schema changed",
    )
    for field in ("scene", "episode"):
        _require(
            isinstance(payload.get(field), str) and payload[field],
            f"{field} is invalid",
        )
    _sha256(payload.get("online_a_receipt_sha256"), "online_a_receipt_sha256")
    _sha256(payload.get("online_a_trace_sha256"), "online_a_trace_sha256")
    endpoint = payload.get("online_a_endpoint")
    _require(
        isinstance(endpoint, dict)
        and isinstance(endpoint.get("floor_position"), list)
        and len(endpoint["floor_position"]) == 3,
        "online-A endpoint is invalid",
    )
    for index, value in enumerate(endpoint["floor_position"]):
        _finite(value, f"online_a_endpoint.floor_position[{index}]")
    _finite(endpoint.get("yaw_rad"), "online_a_endpoint.yaw_rad")

    pairs = payload.get("pairs")
    _require(isinstance(pairs, list) and pairs, "episode must contain pairs")
    pair_ids = set()
    query_ids = set()
    maximum_distance_error = _finite(
        contract.get("maximum_role_distance_error_m"),
        "maximum_role_distance_error_m",
    )
    maximum_bearing_error = _finite(
        contract.get("maximum_role_initial_path_bearing_error_deg"),
        "maximum_role_initial_path_bearing_error_deg",
    )
    minimum_distance = _finite(
        contract.get("minimum_query_geodesic_m"),
        "minimum_query_geodesic_m",
    )
    novel_ceiling = _finite(
        contract.get("novel_max_online_a_covis_exclusive"),
        "novel_max_online_a_covis_exclusive",
    )
    revisit_floor = _finite(
        contract.get("revisit_min_online_a_covis_inclusive"),
        "revisit_min_online_a_covis_inclusive",
    )
    revisit_ceiling = _finite(
        contract.get("revisit_max_online_a_covis_inclusive"),
        "revisit_max_online_a_covis_inclusive",
    )
    revisit_ceiling_is_exclusive = bool(
        contract.get("revisit_max_online_a_covis_is_exclusive", False)
    )
    _require(
        0.0 <= novel_ceiling < revisit_floor <= revisit_ceiling <= 1.0,
        "Novel/Revisit support bands overlap or are invalid",
    )

    for pair in pairs:
        _require(isinstance(pair, dict), "pair must be an object")
        pair_id = pair.get("pair_id")
        _require(
            isinstance(pair_id, str) and pair_id and pair_id not in pair_ids,
            "pair_id is invalid or duplicated",
        )
        pair_ids.add(pair_id)
        queries = pair.get("queries")
        _require(
            isinstance(queries, list) and len(queries) == 2,
            "each pair must contain exactly two queries",
        )
        counts = Counter(query.get("analysis_role") for query in queries)
        _require(
            counts == Counter({"novel": 1, "revisit": 1}),
            "pair must contain one Novel and one Revisit query",
        )
        for query in queries:
            validate_query(query)
            _require(
                query["query_id"] not in query_ids,
                "query_id is duplicated",
            )
            query_ids.add(query["query_id"])
            _require(
                float(query["geodesic_from_a_end_m"]) >= minimum_distance,
                "query is below the minimum geodesic",
            )
            if query["analysis_role"] == "novel":
                _require(
                    float(query["max_online_a_covis"]) < novel_ceiling,
                    "Novel query has online-A visual support",
                )
            else:
                revisit_support = float(query["max_online_a_covis"])
                upper_ok = (
                    revisit_support < revisit_ceiling
                    if revisit_ceiling_is_exclusive
                    else revisit_support <= revisit_ceiling
                )
                _require(
                    revisit_floor <= revisit_support and upper_ok,
                    "Revisit query is outside the frozen support band",
                )
        distances = {
            query["analysis_role"]: float(query["geodesic_from_a_end_m"])
            for query in queries
        }
        observed_error = abs(distances["novel"] - distances["revisit"])
        _require(
            observed_error <= maximum_distance_error + 1e-9,
            "paired Novel/Revisit geodesics are not matched",
        )
        declared_error = _finite(
            pair.get("role_distance_error_m"), "role_distance_error_m"
        )
        _require(
            abs(declared_error - observed_error) <= 1e-6,
            "declared role distance error is inconsistent",
        )
        bearings = {
            query["analysis_role"]: float(query["initial_path_bearing_rad"])
            for query in queries
        }
        observed_bearing_error = abs(math.degrees(
            (bearings["novel"] - bearings["revisit"] + math.pi)
            % (2.0 * math.pi)
            - math.pi
        ))
        _require(
            observed_bearing_error <= maximum_bearing_error + 1e-9,
            "paired Novel/Revisit initial path bearings are not matched",
        )
        declared_bearing_error = _finite(
            pair.get("role_initial_path_bearing_error_deg"),
            "role_initial_path_bearing_error_deg",
        )
        _require(
            abs(declared_bearing_error - observed_bearing_error) <= 1e-6,
            "declared role bearing error is inconsistent",
        )
    return payload


def validate_manifest(payload: dict) -> dict:
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        "manifest schema changed",
    )
    contract = payload.get("contract")
    _require(isinstance(contract, dict), "manifest contract is missing")
    _require(
        contract.get("online_history")
        == "frozen_native_navdp_goal_a_rgb_depth_pose_trace",
        "online-history contract changed",
    )
    _require(
        contract.get("query_execution")
        == "independent_reset_and_exact_online_a_replay",
        "query execution contract changed",
    )
    _require(
        contract.get("runtime_role_visibility") == "none",
        "runtime must not receive the analysis role",
    )
    episodes = payload.get("episodes")
    _require(isinstance(episodes, list) and episodes, "manifest has no episodes")
    identities = set()
    for episode in episodes:
        validate_episode(episode, contract)
        identity = (episode["scene"], episode["episode"])
        _require(identity not in identities, "episode identity is duplicated")
        identities.add(identity)
    return payload


__all__ = [
    "RUNTIME_VISIBLE_QUERY_FIELDS",
    "ROLES",
    "SCHEMA_VERSION",
    "runtime_query",
    "validate_episode",
    "validate_manifest",
    "validate_query",
]
