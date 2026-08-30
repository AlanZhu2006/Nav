#!/usr/bin/env python3
"""Seal a powered Table-III population from base plus append-only expansion."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from hm3d_table3_length_contract import SCHEMA_VERSION, validate_manifest


PROTOCOL_SCHEMA = "hm3d_table3_causal_survey_protocol_v1_20260830"
EXPANSION_PLAN_SCHEMA = "hm3d_table3_causal_survey_expansion_plan_v1_20260831"
EXPANSION_VERIFY_SCHEMA = "hm3d_table3_causal_survey_expansion_plan_verification_v1_20260831"
FRAGMENT_SCHEMA = "hm3d_table3_causal_survey_fragment_v1_20260830"
POPULATION_SCHEMA = "hm3d_table3_causal_survey_merged_population_v1_20260831"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def verify_sidecar(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file() and sidecar.read_text().split() == [digest, path.name],
            f"invalid construction receipt: {path}")
    return digest


def select_powered(
    rows: list[dict[str, Any]], *, histories: int, scenes: int,
    maximum_per_scene: int,
) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_scene[str(row["scene"])].append(row)
    require(len(by_scene) >= scenes, "insufficient constructible scene clusters")
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        scene = str(row["scene"])
        if counts[scene] == 0:
            selected.append(row)
            counts[scene] += 1
            if len(counts) >= scenes:
                break
    for row in rows:
        scene = str(row["scene"])
        if row in selected or counts[scene] >= maximum_per_scene:
            continue
        selected.append(row)
        counts[scene] += 1
        if len(selected) == histories:
            break
    require(len(selected) == histories, "insufficient constructible histories")
    return selected


def read_source(
    *, kind: str, run_root: Path, plan_path: Path, protocol_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            f"{kind} protocol schema changed")
    require(protocol["source_candidate_plan"]["sha256"] == sha256(plan_path)
            and int(protocol["source_candidate_plan"]["candidate_count"])
            == len(plan["episodes"]),
            f"{kind} plan/protocol binding changed")
    plan_sha = sha256(plan_path)
    protocol_sha = sha256(protocol_path)
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    for plan_index, candidate in enumerate(plan["episodes"]):
        path = (run_root / "construction_fragments"
                / f"{plan_index:03d}" / "completion.json")
        digest = verify_sidecar(path)
        fragment = json.loads(path.read_text())
        require(fragment.get("schema_version") == FRAGMENT_SCHEMA
                and int(fragment["history_index"])
                == int(candidate["history_index"])
                and fragment["candidate_identity_sha256"]
                == candidate["candidate_identity_sha256"]
                and fragment["source_candidate_plan_sha256"] == plan_sha
                and fragment["protocol_sha256"] == protocol_sha
                and fragment.get("query_policy_outcomes_read") is False,
                f"{kind} fragment {plan_index} changed")
        if kind == "expansion":
            require(int(fragment.get("plan_index", -1)) == plan_index,
                    f"expansion plan index {plan_index} changed")
        identity = str(fragment["candidate_identity_sha256"])
        require(identity not in seen, f"duplicate {kind} candidate identity")
        seen.add(identity)
        row = dict(fragment, source_kind=kind, source_plan_index=plan_index,
                   source_scene_index=int(candidate["scene_index"]))
        rows.append(row)
        ledger.append({
            "source_kind": kind, "plan_index": plan_index,
            "history_index": int(candidate["history_index"]),
            "candidate_identity_sha256": identity,
            "path": str(path.resolve()), "sha256": digest,
        })
    return rows, ledger, plan, protocol


def finalize(
    *, base_run_root: Path, base_plan_path: Path, base_protocol_path: Path,
    expansion_run_root: Path, expansion_plan_path: Path,
    expansion_protocol_path: Path, expansion_verification_path: Path,
    out: Path,
) -> dict[str, Any]:
    require(not out.exists(), "merged causal-survey population exists")
    expansion_verification = json.loads(expansion_verification_path.read_text())
    require(expansion_verification.get("schema_version") == EXPANSION_VERIFY_SCHEMA
            and expansion_verification.get("verified") is True
            and expansion_verification["plan_sha256"] == sha256(expansion_plan_path)
            and expansion_verification["construction_protocol_sha256"]
            == sha256(expansion_protocol_path)
            and expansion_verification.get("base_candidates_deleted_or_replaced") is False
            and expansion_verification.get("query_policy_outcomes_read") is False
            and expansion_verification.get("navigation_policy_outcomes_read") is False
            and expansion_verification.get("query_policy_evaluation_authorized") is False,
            "expansion plan was not independently verified")

    base_rows, base_ledger, base_plan, base_protocol = read_source(
        kind="base", run_root=base_run_root, plan_path=base_plan_path,
        protocol_path=base_protocol_path,
    )
    expansion_rows, expansion_ledger, expansion_plan, expansion_protocol = read_source(
        kind="expansion", run_root=expansion_run_root,
        plan_path=expansion_plan_path, protocol_path=expansion_protocol_path,
    )
    require(expansion_plan.get("schema_version") == EXPANSION_PLAN_SCHEMA,
            "expansion plan schema changed")
    require(expansion_plan.get("base_candidates_deleted_or_replaced") is False
            and expansion_plan.get("query_policy_outcomes_read") is False
            and expansion_plan.get("navigation_policy_outcomes_read") is False,
            "expansion plan crossed the policy boundary")
    for field in ("history", "query_construction", "length_definition",
                  "runtime", "population_gate"):
        require(expansion_protocol[field] == base_protocol[field],
                f"expansion changed {field}")

    all_rows = base_rows + expansion_rows
    identities = [str(row["candidate_identity_sha256"]) for row in all_rows]
    require(len(identities) == len(set(identities)),
            "base/expansion candidate identity overlap")
    require(len(base_rows) == len(base_plan["episodes"]),
            "base candidates were deleted")

    gate = base_protocol["population_gate"]
    selected: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for spec in base_protocol["length_definition"]["bins_m"]:
        name = str(spec["name"])
        eligible = [row for row in all_rows
                    if row["bin_name"] == name and row["constructed"]]
        chosen = select_powered(
            eligible,
            histories=int(gate["minimum_histories_per_bin"]),
            scenes=int(gate["minimum_scene_clusters_per_bin"]),
            maximum_per_scene=int(
                gate["maximum_selected_histories_per_scene_per_bin"]),
        )
        selected.extend(chosen)
        diagnostics[name] = {
            "base_frozen_candidates": sum(row["bin_name"] == name
                                          for row in base_rows),
            "expansion_frozen_candidates": sum(row["bin_name"] == name
                                               for row in expansion_rows),
            "base_constructible_histories": sum(
                row["bin_name"] == name and row["constructed"]
                for row in base_rows),
            "expansion_constructible_histories": sum(
                row["bin_name"] == name and row["constructed"]
                for row in expansion_rows),
            "combined_constructible_histories": len(eligible),
            "combined_constructible_scene_clusters": len({
                row["scene"] for row in eligible
            }),
            "selected_histories": len(chosen),
            "selected_scene_clusters": len({row["scene"] for row in chosen}),
            "selected_from_base": sum(row["source_kind"] == "base"
                                      for row in chosen),
            "selected_from_expansion": sum(row["source_kind"] == "expansion"
                                           for row in chosen),
            "failure_reasons": dict(Counter(
                str(row.get("reason", "")) for row in all_rows
                if row["bin_name"] == name and not row["constructed"]
            )),
        }

    role_root = out / "role_pairs"
    role_root.mkdir(parents=True)
    episodes: list[dict[str, Any]] = []
    for population_index, fragment in enumerate(selected):
        source = Path(fragment["role_pair_candidate"])
        source_sidecar = source / "role_pairs.json"
        require(sha256(source_sidecar) == fragment["role_pairs_sha256"],
                "selected role-pair candidate changed")
        payload = json.loads(source_sidecar.read_text())
        require(payload["candidate_identity_sha256"]
                == fragment["candidate_identity_sha256"]
                and int(payload["history_index"]) == int(fragment["history_index"]),
                "selected role-pair identity changed")
        destination = role_root / payload["scene"] / payload["episode"]
        require(not destination.exists(), "duplicate selected episode identity")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        copied_sidecar = destination / "role_pairs.json"
        copied = json.loads(copied_sidecar.read_text())
        copied.update({
            "population_index": population_index,
            "scene_index": int(fragment["source_scene_index"]),
            "population_source": fragment["source_kind"],
            "source_plan_index": int(fragment["source_plan_index"]),
        })
        copied_sidecar.write_text(json.dumps(
            copied, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        copied["role_pairs_sha256"] = sha256(copied_sidecar)
        episodes.append(copied)

    query = base_protocol["query_construction"]
    contract = {
        "online_history": "controlled_causal_rgb_geodesic_survey",
        "query_execution": "independent_reset_and_exact_online_a_replay",
        "runtime_role_visibility": "none",
        "bins_m": base_protocol["length_definition"]["bins_m"],
        "novel_max_covis_exclusive": query[
            "novel_max_history_covis_exclusive"],
        "revisit_min_covis_inclusive": query[
            "revisit_min_history_covis_inclusive"],
        "maximum_role_distance_mismatch_m": query[
            "maximum_role_distance_mismatch_m"],
        "minimum_initial_bearing_separation_deg": query[
            "minimum_initial_bearing_separation_deg"],
        "minimum_histories_per_bin": int(gate["minimum_histories_per_bin"]),
        "minimum_scene_clusters_per_bin": int(
            gate["minimum_scene_clusters_per_bin"]),
        "maximum_selected_histories_per_scene_per_bin": int(
            gate["maximum_selected_histories_per_scene_per_bin"]),
        "query_policy_outcomes_read": False,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "HM3D append-only causal-survey length-stratified role-free evaluation",
        "contract": contract,
        "episodes": episodes,
    }
    validate_manifest(manifest)
    manifest_path = role_root / "manifest.json"
    manifest_path.write_text(json.dumps(
        manifest, indent=2, sort_keys=True, allow_nan=False
    ) + "\n")
    manifest_path.with_name("manifest.json.sha256").write_text(
        f"{sha256(manifest_path)}  manifest.json\n"
    )

    completion_ledger = base_ledger + expansion_ledger
    population = {
        "schema_version": POPULATION_SCHEMA,
        "scope": manifest["scope"],
        "history_source": "controlled_causal_rgb_geodesic_survey",
        "base_candidate_plan_sha256": sha256(base_plan_path),
        "base_protocol_sha256": sha256(base_protocol_path),
        "expansion_candidate_plan_sha256": sha256(expansion_plan_path),
        "expansion_protocol_sha256": sha256(expansion_protocol_path),
        "expansion_plan_verification_sha256": sha256(expansion_verification_path),
        "completion_fragments": completion_ledger,
        "completion_ledger_sha256": canonical_sha256(completion_ledger),
        "base_candidates_preserved": len(base_rows),
        "base_candidates_deleted_or_replaced": False,
        "benchmark_manifest_sha256": sha256(manifest_path),
        "histories": len(episodes),
        "queries": 2 * len(episodes),
        "selection_diagnostics": diagnostics,
        "query_policy_outcomes_read": False,
        "formal_policy_evaluation_authorized": False,
        "fallback_completion_allowed": False,
    }
    receipt = out / "population_receipt.json"
    receipt.write_text(json.dumps(population, indent=2, sort_keys=True) + "\n")
    receipt.with_name("population_receipt.json.sha256").write_text(
        f"{sha256(receipt)}  population_receipt.json\n"
    )
    return population


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-root", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--expansion-run-root", type=Path, required=True)
    parser.add_argument("--expansion-plan", type=Path, required=True)
    parser.add_argument("--expansion-protocol", type=Path, required=True)
    parser.add_argument("--expansion-plan-verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        base_run_root=args.base_run_root.resolve(),
        base_plan_path=args.base_plan.resolve(),
        base_protocol_path=args.base_protocol.resolve(),
        expansion_run_root=args.expansion_run_root.resolve(),
        expansion_plan_path=args.expansion_plan.resolve(),
        expansion_protocol_path=args.expansion_protocol.resolve(),
        expansion_verification_path=args.expansion_plan_verification.resolve(),
        out=args.out.resolve(),
    )
    print(json.dumps({
        "histories": result["histories"],
        "queries": result["queries"],
        "base_candidates_preserved": result["base_candidates_preserved"],
        "selection_diagnostics": result["selection_diagnostics"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
