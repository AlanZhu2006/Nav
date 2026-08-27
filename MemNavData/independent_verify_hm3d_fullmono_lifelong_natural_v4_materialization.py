#!/usr/bin/env python3
"""Independent raw-asset verifier for Natural-B v4 materialization."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = (
    "hm3d_fullmono_lifelong_natural_v4_materialization_"
    "independent_verification_v1_20260827"
)
PROTOCOL_SCHEMA = "hm3d_fullmono_lifelong_direct_natural_power_v4_20260827"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar(path: Path) -> None:
    receipt = path.with_name(path.name + ".sha256")
    require(receipt.is_file(), f"missing sidecar: {receipt}")
    fields = receipt.read_text().split()
    require(len(fields) == 2 and fields[0] == sha256(path)
            and fields[1] == path.name, f"sidecar mismatch: {path}")


def close(first: Any, second: Any, name: str) -> None:
    require(math.isclose(float(first), float(second), rel_tol=0.0,
                         abs_tol=1e-6), f"numeric mismatch: {name}")


def verify_file_ledger(root: Path, ledger: Path) -> int:
    count = 0
    for line in ledger.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        path = root / relative
        require(path.is_file() and sha256(path) == digest,
                f"file ledger mismatch: {relative}")
        count += 1
    require(count > 0, "file ledger is empty")
    return count


def planar_separation(groups: dict[tuple[str, str], list[list[float]]]) -> int:
    comparisons = 0
    for identity, positions in groups.items():
        for index, first in enumerate(positions):
            for second in positions[:index]:
                distance = math.hypot(
                    float(first[0]) - float(second[0]),
                    float(first[2]) - float(second[2]),
                )
                require(distance >= 2.0 - 1e-9,
                        f"candidate separation below 2 m: {identity}")
                comparisons += 1
    return comparisons


def load_audit_ledger(audit_root: Path, protocol: dict[str, Any]) -> dict:
    contract = protocol["sealed_natural_b_audit"]
    summary = audit_root / contract["summary"]
    verification = audit_root / contract["independent_verification"]
    require(sha256(summary) == contract["summary_sha256"],
            "Natural-B audit summary changed")
    require(sha256(verification)
            == contract["independent_verification_sha256"],
            "Natural-B audit verification changed")
    verified = json.loads(verification.read_text())
    require(verified.get("verified") is True
            and verified.get("reference_gate_met") is True,
            "Natural-B source audit is not independently verified")
    fragment_root = audit_root / contract["scene_fragments"]
    paths = sorted(fragment_root.glob("*/natural_b_audit.json"))
    require(len(paths) == int(contract["expected_scene_fragments"]),
            "Natural-B audit fragment count changed")
    ledger = {}
    recipient_scenes = set()
    for path in paths:
        sidecar(path)
        payload = json.loads(path.read_text())
        require(payload.get("query_policy_outcomes_read") is False
                and payload.get("navigation_outcomes_read") is False,
                "source audit read navigation outcomes")
        scene = str(payload["scene"])
        for recipient in payload["recipients"]:
            episode = str(recipient["episode"])
            for candidate in recipient["candidates"]:
                identity = (scene, str(candidate["candidate_identity"]))
                require(identity not in ledger, "duplicate audit candidate")
                ledger[identity] = {
                    "recipient_episode": episode,
                    "candidate": candidate,
                    "fragment_sha256": sha256(path),
                }
                recipient_scenes.add((scene, episode))
    require(len(ledger) == int(contract["expected_candidate_histories"]),
            "Natural-B audit candidate ledger changed")
    require(len(recipient_scenes)
            == int(contract["expected_recipient_histories"]),
            "Natural-B audit recipient ledger changed")
    return ledger


def verify_materialization(
    *, run_root: Path, audit_root: Path, protocol_path: Path
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "v4 protocol schema changed")
    require(audit_root.resolve()
            == Path(protocol["sealed_natural_b_audit"]["run_root"]).resolve(),
            "v4 audit root differs from protocol")
    audit = load_audit_ledger(audit_root, protocol)

    population_root = run_root / "ab_population"
    population_path = population_root / "population_receipt.json"
    sidecar(population_path)
    file_count = verify_file_ledger(
        population_root, population_root / "BENCHMARK_FILES.sha256"
    )
    require((population_root / "SEALED").is_file(),
            "materialized population is not sealed")
    population = json.loads(population_path.read_text())
    manifest_path = population_root / "role_pairs/manifest.json"
    require(sha256(manifest_path) == population["benchmark_manifest_sha256"],
            "population manifest binding changed")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["contract"]["runtime_role_visibility"] == "none",
            "runtime role visibility changed")
    require(population["query_policy_outcomes_read"] is False
            and population["navigation_outcome_selection"] is False,
            "population selection read navigation outcomes")

    materialized = {}
    positions: dict[tuple[str, str], list[list[float]]] = collections.defaultdict(list)
    strata = collections.Counter()
    asset_hashes = 0
    for item in manifest["episodes"]:
        scene, episode = str(item["scene"]), str(item["episode"])
        identity = (scene, episode)
        require(identity not in materialized, "duplicate materialized identity")
        require(identity in audit, "materialized identity absent from audit")
        source = audit[identity]
        construction = item["lifelong_construction"]
        candidate = source["candidate"]
        require(construction["recipient_episode"]
                == source["recipient_episode"],
                "source recipient identity changed")
        require(construction["candidate_identity"] == episode
                and int(construction["candidate_slot"])
                == int(candidate["candidate_slot"]),
                "candidate slot identity changed")
        require(construction["audit_fragment_sha256"]
                == source["fragment_sha256"],
                "candidate audit fragment binding changed")
        require(construction["query_policy_outcomes_read"] is False
                and construction["navigation_outcomes_read"] is False,
                "materialization read navigation outcomes")
        episode_root = population_root / "role_pairs" / scene / episode
        sidecar_path = episode_root / "role_pairs.json"
        require(sha256(sidecar_path) == item["role_pairs_sha256"],
                "role-pair sidecar changed")
        side_payload = json.loads(sidecar_path.read_text())
        require(side_payload["scene"] == scene
                and side_payload["episode"] == episode,
                "role-pair identity changed")
        queries = [
            query for pair in item["pairs"] for query in pair["queries"]
        ]
        novel = [q for q in queries if q["analysis_role"] == "novel"]
        revisit = [q for q in queries if q["analysis_role"] == "revisit"]
        require(len(novel) == 1 and len(revisit) == 1,
                "role-pair query cardinality changed")
        novel, revisit = novel[0], revisit[0]
        for query in (novel, revisit):
            for path_key, hash_key in (
                ("goal_rgb", "goal_rgb_sha256"),
                ("goal_depth", "goal_depth_sha256"),
            ):
                asset = episode_root / query[path_key]
                require(asset.is_file() and sha256(asset) == query[hash_key],
                        "query asset hash changed")
                asset_hashes += 1
        require(float(novel["max_online_a_covis"]) < 0.10,
                "materialized Novel has online-A support")
        require(0.55 <= float(revisit["max_online_a_covis"]) <= 0.90,
                "materialized Revisit left standard support band")
        close(novel["max_online_a_covis"], candidate["max_online_a_covis"],
              "Novel covis")
        close(novel["geodesic_from_a_end_m"],
              candidate["query_geodesic_m"], "A-to-B distance")
        close(novel["initial_path_bearing_rad"],
              candidate["initial_path_bearing_rad"], "Novel bearing")
        close(construction["B_to_C_geodesic_m"],
              candidate["paired_revisit_separation_m"], "B-to-C distance")
        require(construction["assigned_direction_stratum"]
                == candidate["assigned_direction_stratum"],
                "candidate direction stratum changed")
        floor = [float(value) for value in novel["floor_position"]]
        require(len(floor) == 3, "Novel floor position changed")
        for index in range(3):
            close(floor[index], construction["goal_floor_position"][index],
                  f"Novel floor position {index}")
        positions[(scene, source["recipient_episode"])].append(floor)
        strata[str(candidate["assigned_direction_stratum"])] += 1
        materialized[identity] = item

    require(set(materialized) == set(audit),
            "materialized population does not equal audit candidate ledger")
    comparisons = planar_separation(positions)
    scenes = {scene for scene, _episode in materialized}
    require(len(materialized) == 99 and len(positions) == 61
            and len(scenes) == 35,
            "v4 exact materialization totals changed")
    require(population["constructible_AB_C_histories"] == 99
            and population["constructible_scene_clusters"] == 35
            and population["construction_target_met"] is True
            and population["factual_B_authorized"] is True,
            "v4 population gate differs from independent recount")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "scope": "independent query-asset materialization audit only",
        "candidate_histories": len(materialized),
        "recipient_histories": len(positions),
        "scene_clusters": len(scenes),
        "direction_strata": dict(sorted(strata.items())),
        "query_asset_hashes_verified": asset_hashes,
        "population_file_ledger_entries_verified": file_count,
        "pairwise_candidate_separations_recomputed": comparisons,
        "minimum_candidate_planar_separation_m": 2.0,
        "audit_candidate_ledger_exactly_reproduced": True,
        "runtime_role_visibility": "none",
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False,
        "factual_B_gate_verified": True,
        "factual_B_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--audit-run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify_materialization(
        run_root=args.run_root,
        audit_root=args.audit_run_root,
        protocol_path=args.protocol,
    )
    require(not args.out.exists(), "verification output already exists")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
