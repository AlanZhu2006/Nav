#!/usr/bin/env python3
"""Freeze unused same-floor HM3D histories for route-tangent confirmation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from MemNavData.finalize_hm3d_table3_causal_survey_merged_population import (
    read_source,
)
from MemNavData.hm3d_table3_length_contract import validate_manifest


PROTOCOL_SCHEMA = (
    "hm3d_longrange_route_tangent_freeze_protocol_v2_20260903"
)
POPULATION_SCHEMA = (
    "hm3d_longrange_route_tangent_population_v2_20260903"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    encoded = json.dumps(
        value, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    path.write_text(encoded)
    path.with_name(path.name + ".sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n"
    )


def _revisit_query(payload: dict[str, Any]) -> dict[str, Any]:
    pairs = payload.get("pairs")
    require(isinstance(pairs, list) and len(pairs) == 1,
            "candidate must contain exactly one role pair")
    queries = pairs[0].get("queries")
    require(isinstance(queries, list) and len(queries) == 2,
            "candidate role pair changed")
    revisit = [row for row in queries if row.get("analysis_role") == "revisit"]
    require(len(revisit) == 1, "candidate has no unique Revisit query")
    return revisit[0]


def eligible_rows(
    rows: list[dict[str, Any]],
    *,
    consumed_identities: set[str],
    distance_bin: str,
    maximum_vertical_error_m: float,
) -> list[dict[str, Any]]:
    """Select from construction evidence without reading policy outcomes."""

    eligible: list[dict[str, Any]] = []
    for fragment in rows:
        if (fragment.get("constructed") is not True
                or fragment.get("bin_name") != distance_bin
                or str(fragment.get("candidate_identity_sha256"))
                in consumed_identities):
            continue
        source = Path(fragment["role_pair_candidate"])
        sidecar = source / "role_pairs.json"
        require(sidecar.is_file(), "constructed role-pair sidecar is missing")
        require(sha256_file(sidecar) == fragment["role_pairs_sha256"],
                "constructed role-pair sidecar changed")
        payload = json.loads(sidecar.read_text())
        require(payload.get("candidate_identity_sha256")
                == fragment["candidate_identity_sha256"],
                "candidate identity changed after construction")
        revisit = _revisit_query(payload)
        endpoint = payload.get("online_a_endpoint", {}).get("floor_position")
        target = revisit.get("floor_position")
        require(isinstance(endpoint, list) and len(endpoint) == 3,
                "history endpoint is malformed")
        require(isinstance(target, list) and len(target) == 3,
                "Revisit target is malformed")
        vertical = abs(float(endpoint[1]) - float(target[1]))
        if vertical <= float(maximum_vertical_error_m) + 1e-12:
            eligible.append({
                **fragment,
                "role_pair_payload": payload,
                "revisit_vertical_error_m": vertical,
            })
    return eligible


def deterministic_scene_cap(
    rows: list[dict[str, Any]], *, maximum_per_scene: int,
) -> list[dict[str, Any]]:
    """Take one row per scene before any scene contributes its second row."""

    require(maximum_per_scene > 0, "maximum_per_scene must be positive")
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene"])].append(row)
    for scene_rows in by_scene.values():
        scene_rows.sort(key=lambda row: str(row["candidate_identity_sha256"]))
    selected: list[dict[str, Any]] = []
    for rank in range(maximum_per_scene):
        for scene in sorted(by_scene):
            if rank < len(by_scene[scene]):
                selected.append(by_scene[scene][rank])
    return selected


def freeze(*, protocol_path: Path, out: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "route-tangent freeze protocol changed")
    selection = protocol["selection"]
    evaluation = protocol["evaluation"]
    guards = protocol["guards"]
    require(selection["read_query_policy_outcomes"] is False
            and selection["read_navigation_policy_outcomes"] is False
            and selection[
                "exclude_every_candidate_identity_in_consumed_manifest"]
            is True,
            "selection crossed the result-blind boundary")
    require(selection["include_every_eligible_history"] is True,
            "v2 must retain every result-blind eligible history")
    require(guards["population_freezes_before_any_new_query_rollout"] is True
            and guards["no_threshold_or_radius_sweep_after_unseal"] is True,
            "population freeze guards changed")
    require(evaluation["distance_gate"] is False
            and evaluation["stuck_trigger"] is False
            and evaluation["post_accept_native_fallback"] is False
            and evaluation["post_accept_endpoint_fallback"] is False,
            "route-tangent method gained a forbidden branch")
    require(not out.exists(), f"output population already exists: {out}")

    sources = protocol["sources"]
    loaded: list[dict[str, Any]] = []
    source_ledgers: list[dict[str, Any]] = []
    base_contract = None
    for kind in ("base", "expansion"):
        spec = sources[kind]
        plan_path = Path(spec["candidate_plan"])
        construction_protocol = Path(spec["construction_protocol"])
        require(sha256_file(plan_path) == spec["candidate_plan_sha256"],
                f"{kind} candidate plan changed")
        require(sha256_file(construction_protocol)
                == spec["construction_protocol_sha256"],
                f"{kind} construction protocol changed")
        rows, ledger, _plan, source_contract = read_source(
            kind=kind,
            run_root=Path(spec["construction_run_root"]),
            plan_path=plan_path,
            protocol_path=construction_protocol,
        )
        loaded.extend(rows)
        source_ledgers.extend(ledger)
        if base_contract is None:
            base_contract = source_contract
        else:
            for field in ("history", "query_construction",
                          "length_definition", "runtime"):
                require(source_contract[field] == base_contract[field],
                        f"expansion changed frozen field {field}")

    expansion = sources["expansion"]
    expansion_verify_path = Path(expansion[
        "independent_plan_verification"])
    require(sha256_file(expansion_verify_path)
            == expansion["independent_plan_verification_sha256"],
            "expansion independent verification changed")
    expansion_verify = json.loads(expansion_verify_path.read_text())
    require(expansion_verify.get("verified") is True
            and expansion_verify.get("query_policy_outcomes_read") is False
            and expansion_verify.get("navigation_policy_outcomes_read")
            is False,
            "expansion plan is not result-blind and independently verified")

    consumed_spec = sources["consumed_population"]
    consumed_root = Path(consumed_spec["root"])
    consumed_manifest_path = consumed_root / consumed_spec["manifest"]
    require(sha256_file(consumed_manifest_path)
            == consumed_spec["manifest_sha256"],
            "consumed population manifest changed")
    consumed_manifest = json.loads(consumed_manifest_path.read_text())
    validate_manifest(consumed_manifest)
    consumed_receipt_path = (
        consumed_root / consumed_spec["population_receipt"])
    consumed_receipt = json.loads(consumed_receipt_path.read_text())
    require(consumed_receipt.get("schema_version")
            == consumed_spec["population_receipt_schema"]
            and consumed_receipt.get("benchmark_manifest_sha256")
            == consumed_spec["manifest_sha256"],
            "consumed population receipt changed")
    consumed_identities = {
        str(row["candidate_identity_sha256"])
        for row in consumed_manifest["episodes"]
    }
    require(len(consumed_identities) == len(consumed_manifest["episodes"]),
            "consumed identities are not unique")

    eligible = eligible_rows(
        loaded,
        consumed_identities=consumed_identities,
        distance_bin=str(selection["distance_bin"]),
        maximum_vertical_error_m=float(
            selection["revisit_vertical_error_max_m"]),
    )
    eligible_scenes = {str(row["scene"]) for row in eligible}
    require(len(eligible)
            == int(selection[
                "expected_eligible_histories_before_scene_cap"]),
            "fresh same-floor eligible history count changed")
    require(len(eligible_scenes)
            == int(selection["expected_eligible_scene_clusters"]),
            "fresh same-floor scene count changed")
    selected = deterministic_scene_cap(
        eligible,
        maximum_per_scene=int(selection["maximum_histories_per_scene"]),
    )
    require(len(selected) == int(selection["expected_selected_histories"]),
            "selected history count changed")
    require(len({str(row["scene"]) for row in selected})
            == int(selection["expected_selected_scene_clusters"]),
            "selected scene count changed")
    counts = Counter(str(row["scene"]) for row in selected)
    require(max(counts.values())
            <= int(selection["maximum_histories_per_scene"]),
            "scene cap was violated")
    require(len(selected) == len(eligible),
            "v2 unexpectedly discarded an eligible fresh history")
    require(max(counts.values()) == int(selection[
                "maximum_observed_histories_in_one_scene"]),
            "observed scene multiplicity changed")

    temporary = Path(tempfile.mkdtemp(
        prefix=out.name + ".partial.", dir=out.parent))
    try:
        role_root = temporary / "role_pairs"
        role_root.mkdir(parents=True)
        episodes: list[dict[str, Any]] = []
        selected_ledger: list[dict[str, Any]] = []
        for population_index, fragment in enumerate(selected):
            payload = dict(fragment["role_pair_payload"])
            source = Path(fragment["role_pair_candidate"])
            destination = role_root / payload["scene"] / payload["episode"]
            require(not destination.exists(), "duplicate selected episode")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
            copied_sidecar = destination / "role_pairs.json"
            copied = json.loads(copied_sidecar.read_text())
            copied.update({
                "population_index": population_index,
                "scene_index": int(fragment["source_scene_index"]),
                "population_source": fragment["source_kind"],
                "source_plan_index": int(fragment["source_plan_index"]),
                "longrange_route_tangent_fresh": True,
                "revisit_vertical_error_m": float(
                    fragment["revisit_vertical_error_m"]),
            })
            copied_sidecar.write_text(json.dumps(
                copied, indent=2, sort_keys=True, allow_nan=False,
            ) + "\n")
            copied["role_pairs_sha256"] = sha256_file(copied_sidecar)
            episodes.append(copied)
            selected_ledger.append({
                "population_index": population_index,
                "scene": str(fragment["scene"]),
                "episode": str(payload["episode"]),
                "candidate_identity_sha256": str(
                    fragment["candidate_identity_sha256"]),
                "source_kind": str(fragment["source_kind"]),
                "source_plan_index": int(fragment["source_plan_index"]),
                "source_role_pairs_sha256": str(
                    fragment["role_pairs_sha256"]),
                "copied_role_pairs_sha256": copied[
                    "role_pairs_sha256"],
                "revisit_vertical_error_m": float(
                    fragment["revisit_vertical_error_m"]),
            })

        contract = dict(consumed_manifest["contract"])
        contract.update({
            "fresh_history_selection": (
                "unused_candidate_identity_same_floor_20_to_30_m"),
            "revisit_vertical_error_max_m": float(
                selection["revisit_vertical_error_max_m"]),
            "maximum_selected_histories_per_scene": int(
                selection["maximum_histories_per_scene"]),
            "query_policy_outcomes_read": False,
        })
        manifest = {
            "schema_version": consumed_manifest["schema_version"],
            "scope": protocol["scope"],
            "contract": contract,
            "episodes": episodes,
        }
        validate_manifest(manifest)
        manifest_path = role_root / "manifest.json"
        atomic_json(manifest_path, manifest)
        receipt = {
            "schema_version": POPULATION_SCHEMA,
            "scope": protocol["scope"],
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": sha256_file(protocol_path),
            "source_candidate_fragments": len(loaded),
            "source_fragment_ledger_sha256": canonical_sha256(
                source_ledgers),
            "consumed_manifest_sha256": sha256_file(
                consumed_manifest_path),
            "consumed_identities": len(consumed_identities),
            "eligible_histories_before_scene_cap": len(eligible),
            "eligible_scene_clusters": len(eligible_scenes),
            "selected_histories": len(episodes),
            "selected_scene_clusters": len(counts),
            "selected_scene_counts": dict(sorted(counts.items())),
            "selected_ledger": selected_ledger,
            "selected_ledger_sha256": canonical_sha256(selected_ledger),
            "benchmark_manifest_sha256": sha256_file(manifest_path),
            "query_policy_outcomes_read": False,
            "navigation_policy_outcomes_read": False,
            "formal_policy_evaluation_authorized": False,
        }
        atomic_json(temporary / "population_receipt.json", receipt)
        os.replace(temporary, out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(
        protocol_path=args.protocol.resolve(), out=args.out.resolve())
    print(json.dumps({
        "selected_histories": result["selected_histories"],
        "selected_scene_clusters": result["selected_scene_clusters"],
        "benchmark_manifest_sha256": result[
            "benchmark_manifest_sha256"],
        "query_policy_outcomes_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
