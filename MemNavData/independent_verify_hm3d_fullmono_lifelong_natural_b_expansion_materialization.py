#!/usr/bin/env python3
"""Independently verify the v5 expansion materialization before factual B.

This verifier intentionally does not import the materializer or population
finalizer.  It recounts the sealed audit ledger and checks every copied query,
candidate identity, source receipt, and separation predicate from raw files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_materialization_"
    "verification_v1_20260830"
)
PROTOCOL_SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_execution_v5_20260830"
)
AUDIT_SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_audit_v1_20260828"
)
MATERIALIZATION_SCHEMA = (
    "hm3d_fullmono_lifelong_natural_b_expansion_scene_v1_20260830"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{path}: JSON root is not an object")
    return payload


def verify_sidecar(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"{path}: SHA sidecar missing")
    fields = sidecar.read_text().split()
    digest = sha256(path)
    require(len(fields) == 2 and fields[0] == digest
            and fields[1] == path.name, f"{path}: SHA sidecar changed")
    return digest


def contained(path: Path, root: Path, message: str) -> Path:
    resolved, base = path.resolve(), root.resolve()
    require(resolved == base or base in resolved.parents, message)
    return resolved


def verify_file_ledger(root: Path, ledger_name: str,
                       excluded: set[str]) -> int:
    ledger = root / ledger_name
    require(ledger.is_file(), f"{ledger}: file ledger missing")
    seen: set[Path] = set()
    for line in ledger.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2, f"{ledger}: malformed row")
        path = contained(root / fields[1].strip(), root,
                         "materialization ledger escaped root")
        require(path.is_file() and path not in seen,
                f"{ledger}: missing or duplicate path")
        require(sha256(path) == fields[0], f"{path}: ledger digest changed")
        seen.add(path)
    actual = {
        path.resolve() for path in root.rglob("*")
        if path.is_file() and path.name not in excluded
    }
    require(seen == actual, f"{ledger}: coverage changed")
    return len(seen)


def planar_distance(first: list[float], second: list[float]) -> float:
    require(len(first) == len(second) == 3,
            "candidate floor position must have three coordinates")
    return math.hypot(
        float(first[0]) - float(second[0]),
        float(first[2]) - float(second[2]),
    )


def audit_ledger(protocol: dict[str, Any]) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    dict[tuple[str, str], list[list[float]]],
    dict[str, str],
]:
    contract = protocol["sealed_natural_b_expansion_audit"]
    audit_root = Path(contract["run_root"])
    summary_path = audit_root / contract["summary"]
    verification_path = audit_root / contract["independent_verification"]
    require(sha256(summary_path) == contract["summary_sha256"],
            "expansion summary changed")
    require(sha256(verification_path)
            == contract["independent_verification_sha256"],
            "expansion independent verification changed")
    summary = read_json(summary_path)
    verification = read_json(verification_path)
    require(verification.get("verified") is True
            and verification.get("navigation_outcomes_read") is False
            and verification.get("query_policy_outcomes_read") is False,
            "expansion audit is not independently result-blind")
    require(int(summary["scene_fragments"])
            == int(contract["expected_scene_fragments"])
            and int(summary["expansion_candidate_histories"])
            == int(contract["expected_candidate_histories"])
            and int(summary["expansion_constructible_recipients"])
            == int(contract["expected_recipient_histories"])
            and int(summary["expansion_scene_clusters"])
            == int(contract["expected_scene_clusters"]),
            "expansion summary counts changed")

    paths = sorted((audit_root / contract["scene_fragments"]).glob(
        "*/natural_b_expansion_audit.json"
    ))
    require(len(paths) == int(contract["expected_scene_fragments"]),
            "expansion fragment count changed")
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    originals: dict[tuple[str, str], list[list[float]]] = {}
    fragment_hashes: dict[str, str] = {}
    recipient_keys: set[tuple[str, str]] = set()
    scenes: set[str] = set()
    for path in paths:
        digest = verify_sidecar(path)
        payload = read_json(path)
        require(payload.get("schema_version") == AUDIT_SCHEMA,
                "expansion audit fragment schema changed")
        require(payload.get("query_policy_outcomes_read") is False
                and payload.get("navigation_outcomes_read") is False
                and payload.get("evaluation_authorized") is False,
                "expansion fragment crossed the outcome boundary")
        scene = str(payload["scene"])
        fragment_hashes[scene] = digest
        for recipient in payload["recipients"]:
            episode = str(recipient["episode"])
            recipient_key = (scene, episode)
            original = [
                [float(value) for value in row["goal_floor_position"]]
                for row in recipient["original_candidates"]
            ]
            require(recipient_key not in originals,
                    "duplicate expansion recipient")
            originals[recipient_key] = original
            if recipient["candidates"]:
                recipient_keys.add(recipient_key)
                scenes.add(scene)
            for row in recipient["candidates"]:
                identity = str(row["candidate_identity"])
                key = (scene, identity)
                require(key not in candidates,
                        "duplicate expansion candidate identity")
                require(identity == f"{episode}__natural_b_{int(row['candidate_slot']):02d}",
                        "expansion candidate slot/identity changed")
                candidates[key] = {
                    "recipient_episode": episode,
                    "candidate_slot": int(row["candidate_slot"]),
                    "goal_floor_position": [
                        float(value) for value in row["goal_floor_position"]
                    ],
                    "max_online_a_covis": float(row["max_online_a_covis"]),
                    "assigned_direction_stratum": str(
                        row["assigned_direction_stratum"]
                    ),
                    "fragment_sha256": digest,
                }
    require(len(candidates) == int(contract["expected_candidate_histories"])
            and len(recipient_keys)
            == int(contract["expected_recipient_histories"])
            and len(scenes) == int(contract["expected_scene_clusters"]),
            "raw expansion ledger counts changed")
    return candidates, originals, fragment_hashes


def verify(*, run_root: Path, protocol_path: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "v5 expansion execution protocol schema changed")
    protocol_sha = sha256(protocol_path)
    candidates, originals, fragment_hashes = audit_ledger(protocol)
    contract = protocol["sealed_natural_b_expansion_audit"]

    construction_root = run_root / "construct_ab/scenes"
    completions = sorted(construction_root.glob("*/completion.json"))
    require(len(completions) == int(contract["expected_scene_fragments"]),
            "materialization completion count changed")
    completed_candidates = completed_recipients = 0
    completion_by_scene: dict[str, dict[str, Any]] = {}
    for path in completions:
        verify_sidecar(path)
        payload = read_json(path)
        require(payload.get("schema_version") == MATERIALIZATION_SCHEMA
                and payload.get("status") == "complete",
                "materialization completion schema changed")
        require(payload["protocol_sha256"] == protocol_sha,
                "materialization used another protocol")
        require(payload.get("audit_candidate_count_reproduced") is True
                and payload.get("candidate_positions_serialized") is True
                and payload.get(
                    "separation_against_original_and_expansion_recomputed"
                ) is True,
                "materialization did not reproduce the expansion audit")
        require(payload.get("query_policy_outcomes_read") is False
                and payload.get("navigation_outcomes_read") is False,
                "materialization read a navigation outcome")
        scene = str(payload["scene"])
        require(payload["audit_fragment_sha256"] == fragment_hashes[scene],
                "materialization references another audit fragment")
        completion_by_scene[scene] = payload
        completed_candidates += int(payload["constructible_AB_C_histories"])
        completed_recipients += int(payload["materialized_recipient_histories"])
    require(completed_candidates == int(contract["expected_candidate_histories"])
            and completed_recipients
            == int(contract["expected_recipient_histories"]),
            "materialization completion totals changed")

    ab_root = run_root / "ab_population"
    require((ab_root / "SEALED").is_file(), "A/B population is not sealed")
    receipt_path = ab_root / "population_receipt.json"
    receipt_sha = verify_sidecar(receipt_path)
    receipt = read_json(receipt_path)
    manifest_path = ab_root / "role_pairs/manifest.json"
    manifest_sha = verify_sidecar(manifest_path)
    manifest = read_json(manifest_path)
    episodes = manifest.get("episodes")
    require(isinstance(episodes, list)
            and len(episodes) == int(contract["expected_candidate_histories"]),
            "materialized expansion manifest count changed")
    require(receipt["benchmark_manifest_sha256"] == manifest_sha
            and int(receipt["constructible_AB_C_histories"])
            == int(contract["expected_candidate_histories"])
            and int(receipt["constructible_scene_clusters"])
            == int(contract["expected_scene_clusters"])
            and receipt["construction_target_met"] is True
            and receipt["factual_B_authorized"] is True
            and receipt["query_policy_outcomes_read"] is False
            and receipt["navigation_outcome_selection"] is False,
            "materialization receipt did not authorize exact result-blind B")

    materialized: dict[tuple[str, str], dict[str, Any]] = {}
    by_recipient: dict[tuple[str, str], list[list[float]]] = defaultdict(list)
    for item in episodes:
        scene, episode = str(item["scene"]), str(item["episode"])
        key = (scene, episode)
        require(key in candidates and key not in materialized,
                "materialized identity differs from expansion audit")
        expected = candidates[key]
        construction = item["lifelong_construction"]
        require(construction["recipient_episode"]
                == expected["recipient_episode"]
                and int(construction["candidate_slot"])
                == expected["candidate_slot"]
                and construction["candidate_identity"] == episode
                and construction["candidate_source"]
                == "sealed_natural_B_expansion_audit"
                and construction["audit_fragment_sha256"]
                == expected["fragment_sha256"],
                "materialized expansion provenance changed")
        floor = [float(value)
                 for value in construction["goal_floor_position"]]
        require(all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-9)
                    for a, b in zip(floor, expected["goal_floor_position"])),
                "materialized expansion floor position changed")
        require(math.isclose(
            float(construction["B_max_recipient_A_covis"]),
            expected["max_online_a_covis"], rel_tol=0.0, abs_tol=1e-9,
        ) and construction["assigned_direction_stratum"]
                == expected["assigned_direction_stratum"],
                "materialized expansion metadata changed")
        require(construction["query_policy_outcomes_read"] is False
                and construction["navigation_outcomes_read"] is False,
                "materialized candidate crossed the outcome boundary")

        sidecar_path = ab_root / "role_pairs" / scene / episode / "role_pairs.json"
        require(sha256(sidecar_path) == item["role_pairs_sha256"],
                "materialized role-pair sidecar changed")
        sidecar = read_json(sidecar_path)
        queries = [query for pair in sidecar["pairs"]
                   for query in pair["queries"]]
        novel = [row for row in queries if row["analysis_role"] == "novel"]
        revisit = [row for row in queries if row["analysis_role"] == "revisit"]
        require(len(novel) == len(revisit) == 1,
                "materialized role pair changed")
        require(float(novel[0]["max_online_a_covis"]) < 0.10
                and 0.55 <= float(revisit[0]["max_online_a_covis"]) <= 0.90,
                "materialized role support contract changed")
        for query in queries:
            for field in ("goal_rgb", "goal_depth"):
                asset = contained(
                    sidecar_path.parent / query[field],
                    sidecar_path.parent,
                    "materialized goal asset escaped its episode",
                )
                require(asset.is_file()
                        and sha256(asset) == query[f"{field}_sha256"],
                        "materialized goal asset changed")
        materialized[key] = construction
        recipient_key = (scene, expected["recipient_episode"])
        by_recipient[recipient_key].append(floor)

    require(set(materialized) == set(candidates),
            "materialized expansion ledger is not exact")
    minimum = float(protocol["novel_b_construction"][
        "minimum_candidate_planar_separation_m"
    ])
    for recipient, new_positions in by_recipient.items():
        prior = list(originals[recipient])
        for current in new_positions:
            require(all(planar_distance(current, other) >= minimum - 1e-9
                        for other in prior),
                    "materialized expansion violates original/new separation")
            prior.append(current)
    ledger_entries = verify_file_ledger(
        ab_root,
        "BENCHMARK_FILES.sha256",
        {"BENCHMARK_FILES.sha256", "SEALED"},
    )
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "scope": "result-blind expansion materialization before factual B",
        "protocol_sha256": protocol_sha,
        "audit_summary_sha256": contract["summary_sha256"],
        "audit_independent_verification_sha256": contract[
            "independent_verification_sha256"
        ],
        "scene_fragments": len(completions),
        "materialized_candidates": len(materialized),
        "materialized_recipients": len(by_recipient),
        "scene_clusters": len({scene for scene, _episode in materialized}),
        "AB_population_receipt_sha256": receipt_sha,
        "AB_manifest_sha256": manifest_sha,
        "AB_file_ledger_entries_verified": ledger_entries,
        "factual_B_gate_verified": True,
        "factual_B_executed": False,
        "query_policy_outcomes_read": False,
        "navigation_outcomes_read": False
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "materialization verification exists")
    result = verify(run_root=args.run_root, protocol_path=args.protocol)
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
