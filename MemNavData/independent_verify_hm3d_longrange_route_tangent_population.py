#!/usr/bin/env python3
"""Independently verify the fresh HM3D route-tangent population."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from MemNavData.hm3d_table3_length_contract import validate_manifest


VERIFY_SCHEMA = (
    "hm3d_longrange_route_tangent_population_verification_v1_20260903"
)
PROTOCOL_SCHEMA = (
    "hm3d_longrange_route_tangent_freeze_protocol_v2_20260903"
)
POPULATION_SCHEMA = (
    "hm3d_longrange_route_tangent_population_v2_20260903"
)
FRAGMENT_SCHEMA = "hm3d_table3_causal_survey_fragment_v1_20260830"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_ok(path: Path) -> None:
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing receipt for {path}")
    require(sidecar.read_text().split() == [sha256_file(path), path.name],
            f"invalid receipt for {path}")


def _query(payload: dict[str, Any], role: str) -> dict[str, Any]:
    rows = [row for pair in payload["pairs"] for row in pair["queries"]
            if row["analysis_role"] == role]
    require(len(rows) == 1, f"missing unique {role} query")
    return rows[0]


def verify(*, protocol_path: Path, population: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "freeze protocol schema changed")
    selection = protocol["selection"]
    sources = protocol["sources"]

    used_spec = sources["consumed_population"]
    used_root = Path(used_spec["root"])
    used_manifest_path = used_root / used_spec["manifest"]
    require(sha256_file(used_manifest_path) == used_spec["manifest_sha256"],
            "consumed manifest changed")
    used_manifest = json.loads(used_manifest_path.read_text())
    validate_manifest(used_manifest)
    used = {row["candidate_identity_sha256"]
            for row in used_manifest["episodes"]}

    eligible: list[dict[str, Any]] = []
    for kind in ("base", "expansion"):
        spec = sources[kind]
        plan_path = Path(spec["candidate_plan"])
        construction_protocol_path = Path(spec["construction_protocol"])
        require(sha256_file(plan_path) == spec["candidate_plan_sha256"],
                f"{kind} plan changed")
        require(sha256_file(construction_protocol_path)
                == spec["construction_protocol_sha256"],
                f"{kind} construction protocol changed")
        plan = json.loads(plan_path.read_text())
        construction_protocol = json.loads(
            construction_protocol_path.read_text())
        require(construction_protocol["source_candidate_plan"]["sha256"]
                == sha256_file(plan_path),
                f"{kind} protocol no longer binds its plan")
        for plan_index, candidate in enumerate(plan["episodes"]):
            completion_path = (
                Path(spec["construction_run_root"])
                / "construction_fragments" / f"{plan_index:03d}"
                / "completion.json"
            )
            _receipt_ok(completion_path)
            fragment = json.loads(completion_path.read_text())
            require(fragment.get("schema_version") == FRAGMENT_SCHEMA,
                    "construction fragment schema changed")
            require(fragment["candidate_identity_sha256"]
                    == candidate["candidate_identity_sha256"],
                    "construction candidate identity mismatch")
            require(fragment["source_candidate_plan_sha256"]
                    == sha256_file(plan_path),
                    "construction fragment references another plan")
            require(fragment["protocol_sha256"]
                    == sha256_file(construction_protocol_path),
                    "construction fragment references another protocol")
            if (fragment.get("constructed") is not True
                    or fragment.get("bin_name")
                    != selection["distance_bin"]
                    or fragment["candidate_identity_sha256"] in used):
                continue
            source_sidecar = (
                Path(fragment["role_pair_candidate"]) / "role_pairs.json")
            require(sha256_file(source_sidecar)
                    == fragment["role_pairs_sha256"],
                    "source role-pair candidate changed")
            payload = json.loads(source_sidecar.read_text())
            endpoint_y = float(
                payload["online_a_endpoint"]["floor_position"][1])
            revisit_y = float(_query(payload, "revisit")["floor_position"][1])
            vertical = abs(endpoint_y - revisit_y)
            if vertical <= float(
                    selection["revisit_vertical_error_max_m"]) + 1e-12:
                eligible.append({
                    "scene": str(fragment["scene"]),
                    "candidate_identity_sha256": str(
                        fragment["candidate_identity_sha256"]),
                    "revisit_vertical_error_m": vertical,
                })

    require(len(eligible) == int(selection[
        "expected_eligible_histories_before_scene_cap"]),
        "independent eligible history count changed")
    require(len({row["scene"] for row in eligible})
            == int(selection["expected_eligible_scene_clusters"]),
            "independent eligible scene count changed")
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_scene[row["scene"]].append(row)
    for rows in by_scene.values():
        rows.sort(key=lambda row: row["candidate_identity_sha256"])
    expected: list[dict[str, Any]] = []
    for rank in range(int(selection["maximum_histories_per_scene"])):
        for scene in sorted(by_scene):
            if rank < len(by_scene[scene]):
                expected.append(by_scene[scene][rank])

    receipt_path = population / "population_receipt.json"
    manifest_path = population / "role_pairs/manifest.json"
    _receipt_ok(receipt_path)
    _receipt_ok(manifest_path)
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    require(receipt.get("schema_version") == POPULATION_SCHEMA,
            "population receipt schema changed")
    require(receipt.get("protocol_sha256") == sha256_file(protocol_path),
            "population references another freeze protocol")
    require(receipt.get("benchmark_manifest_sha256")
            == sha256_file(manifest_path),
            "population manifest digest changed")
    require(receipt.get("query_policy_outcomes_read") is False
            and receipt.get("navigation_policy_outcomes_read") is False
            and receipt.get("formal_policy_evaluation_authorized") is False,
            "population crossed the result boundary before verification")
    validate_manifest(manifest)
    require(len(manifest["episodes"]) == len(expected)
            == int(selection["expected_selected_histories"]),
            "selected population size changed")
    require(selection["include_every_eligible_history"] is True
            and len(expected) == len(eligible),
            "v2 did not retain every eligible history")
    actual_identities = [
        row["candidate_identity_sha256"] for row in manifest["episodes"]
    ]
    expected_identities = [
        row["candidate_identity_sha256"] for row in expected
    ]
    require(actual_identities == expected_identities,
            "population is not the independently reproduced selection")
    require(not (set(actual_identities) & used),
            "fresh population overlaps a consumed candidate identity")

    scene_counts = Counter(row["scene"] for row in manifest["episodes"])
    require(len(scene_counts)
            == int(selection["expected_selected_scene_clusters"]),
            "selected scene coverage changed")
    require(max(scene_counts.values())
            <= int(selection["maximum_histories_per_scene"]),
            "selected population violates the scene cap")
    copied_ledger = []
    for expected_row, episode in zip(expected, manifest["episodes"]):
        require(episode.get("longrange_route_tangent_fresh") is True,
                "fresh-history marker missing")
        require(abs(float(episode["revisit_vertical_error_m"])
                    - expected_row["revisit_vertical_error_m"]) <= 1e-9,
                "stored vertical error changed")
        root = (population / "role_pairs" / episode["scene"]
                / episode["episode"])
        sidecar = root / "role_pairs.json"
        require(sidecar.is_file(), "copied role-pair sidecar is missing")
        require(sha256_file(sidecar) == episode["role_pairs_sha256"],
                "copied role-pair sidecar changed")
        payload = json.loads(sidecar.read_text())
        for query in payload["pairs"][0]["queries"]:
            for stem, digest_name in (
                    ("goal_rgb", "goal_rgb_sha256"),
                    ("goal_depth", "goal_depth_sha256")):
                asset = root / query[stem]
                require(asset.is_file()
                        and sha256_file(asset) == query[digest_name],
                        f"copied {stem} changed")
        source = Path(payload["online_a_episode"])
        require(sha256_file(source / "receipt.json")
                == payload["online_a_receipt_sha256"],
                "online history receipt changed")
        require(sha256_file(source / "online_a_trace.json")
                == payload["online_a_trace_sha256"],
                "online history trace changed")
        copied_ledger.append({
            "scene": episode["scene"],
            "episode": episode["episode"],
            "candidate_identity_sha256": episode[
                "candidate_identity_sha256"],
            "role_pairs_sha256": episode["role_pairs_sha256"],
        })

    return {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "formal_policy_evaluation_authorized": True,
        "protocol_sha256": sha256_file(protocol_path),
        "population_receipt_sha256": sha256_file(receipt_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "eligible_histories_before_scene_cap": len(eligible),
        "histories": len(actual_identities),
        "scene_clusters": len(scene_counts),
        "consumed_identity_overlap": 0,
        "distance_bin": selection["distance_bin"],
        "maximum_revisit_vertical_error_m": max(
            row["revisit_vertical_error_m"] for row in expected),
        "copied_ledger": copied_ledger,
        "query_policy_outcomes_read": False,
        "navigation_policy_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), f"refusing to overwrite {args.out}")
    result = verify(
        protocol_path=args.protocol.resolve(),
        population=args.population.resolve(),
    )
    args.out.write_text(json.dumps(
        result, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256_file(args.out)}  {args.out.name}\n"
    )
    print(json.dumps({
        "verified": True,
        "histories": result["histories"],
        "scene_clusters": result["scene_clusters"],
        "benchmark_manifest_sha256": result[
            "benchmark_manifest_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
