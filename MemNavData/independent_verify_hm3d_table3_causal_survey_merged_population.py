#!/usr/bin/env python3
"""Independently authorize a base-plus-expansion causal-survey population."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_hm3d_table3_length_role_pairs import audit


PROTOCOL_SCHEMA = "hm3d_table3_causal_survey_protocol_v1_20260830"
EXPANSION_VERIFY_SCHEMA = "hm3d_table3_causal_survey_expansion_plan_verification_v1_20260831"
FRAGMENT_SCHEMA = "hm3d_table3_causal_survey_fragment_v1_20260830"
POPULATION_SCHEMA = "hm3d_table3_causal_survey_merged_population_v1_20260831"
VERIFY_SCHEMA = "hm3d_table3_causal_survey_population_verification_v2_20260831"


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


def sidecar_sha(path: Path) -> str:
    digest = sha256(path)
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file() and sidecar.read_text().split() == [digest, path.name],
            f"invalid sidecar for {path}")
    return digest


def read_source(
    *, kind: str, run_root: Path, plan_path: Path, protocol_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA
            and protocol["source_candidate_plan"]["sha256"] == sha256(plan_path)
            and int(protocol["source_candidate_plan"]["candidate_count"])
            == len(plan["episodes"]),
            f"{kind} plan/protocol binding changed")
    rows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    plan_sha, protocol_sha = sha256(plan_path), sha256(protocol_path)
    for plan_index, candidate in enumerate(plan["episodes"]):
        path = (run_root / "construction_fragments"
                / f"{plan_index:03d}" / "completion.json")
        digest = sidecar_sha(path)
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
        rows.append(dict(
            fragment, source_kind=kind, source_plan_index=plan_index,
            source_scene_index=int(candidate["scene_index"]),
        ))
        ledger.append({
            "source_kind": kind, "plan_index": plan_index,
            "history_index": int(candidate["history_index"]),
            "candidate_identity_sha256": candidate["candidate_identity_sha256"],
            "path": str(path.resolve()), "sha256": digest,
        })
    return rows, ledger, plan, protocol


def independent_select(
    rows: list[dict[str, Any]], *, histories: int, scenes: int,
    maximum_per_scene: int,
) -> list[dict[str, Any]]:
    available_scenes = {str(row["scene"]) for row in rows}
    require(len(available_scenes) >= scenes,
            "independent scene gate failed")
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row in rows:
        scene = str(row["scene"])
        if counts[scene] == 0:
            selected.append(row)
            counts[scene] = 1
            if len(counts) == scenes:
                break
    for row in rows:
        scene = str(row["scene"])
        if row in selected or counts[scene] >= maximum_per_scene:
            continue
        selected.append(row)
        counts[scene] += 1
        if len(selected) == histories:
            break
    require(len(selected) == histories,
            "independent history gate failed")
    return selected


def verify(
    *, population_root: Path, base_run_root: Path, base_plan_path: Path,
    base_protocol_path: Path, expansion_run_root: Path,
    expansion_plan_path: Path, expansion_protocol_path: Path,
    expansion_plan_verification_path: Path,
) -> dict[str, Any]:
    receipt_path = population_root / "population_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    require(receipt.get("schema_version") == POPULATION_SCHEMA
            and receipt.get("history_source")
            == "controlled_causal_rgb_geodesic_survey"
            and receipt.get("base_candidates_deleted_or_replaced") is False
            and receipt.get("query_policy_outcomes_read") is False
            and receipt.get("formal_policy_evaluation_authorized") is False
            and receipt.get("fallback_completion_allowed") is False,
            "merged population bypassed independent authorization")
    require(receipt["base_candidate_plan_sha256"] == sha256(base_plan_path)
            and receipt["base_protocol_sha256"] == sha256(base_protocol_path)
            and receipt["expansion_candidate_plan_sha256"]
            == sha256(expansion_plan_path)
            and receipt["expansion_protocol_sha256"]
            == sha256(expansion_protocol_path)
            and receipt["expansion_plan_verification_sha256"]
            == sha256(expansion_plan_verification_path),
            "merged population source binding changed")
    expansion_verify = json.loads(expansion_plan_verification_path.read_text())
    require(expansion_verify.get("schema_version") == EXPANSION_VERIFY_SCHEMA
            and expansion_verify.get("verified") is True
            and expansion_verify.get("query_policy_outcomes_read") is False
            and expansion_verify.get("navigation_policy_outcomes_read") is False,
            "expansion plan verifier no longer passes")

    base_rows, base_ledger, base_plan, base_protocol = read_source(
        kind="base", run_root=base_run_root, plan_path=base_plan_path,
        protocol_path=base_protocol_path,
    )
    expansion_rows, expansion_ledger, expansion_plan, expansion_protocol = read_source(
        kind="expansion", run_root=expansion_run_root,
        plan_path=expansion_plan_path, protocol_path=expansion_protocol_path,
    )
    for field in ("history", "query_construction", "length_definition",
                  "runtime", "population_gate"):
        require(base_protocol[field] == expansion_protocol[field],
                f"expanded source changed {field}")
    require(len(base_rows) == len(base_plan["episodes"])
            == int(receipt["base_candidates_preserved"]),
            "base source was not fully preserved")
    require(len(expansion_rows) == len(expansion_plan["episodes"]),
            "expansion source is incomplete")
    combined = base_rows + expansion_rows
    identities = [str(row["candidate_identity_sha256"]) for row in combined]
    require(len(identities) == len(set(identities)),
            "base and expansion identities overlap")

    ledger = base_ledger + expansion_ledger
    require(ledger == receipt["completion_fragments"]
            and canonical_sha256(ledger) == receipt["completion_ledger_sha256"],
            "completion ledger changed")
    gate = base_protocol["population_gate"]
    expected: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    for spec in base_protocol["length_definition"]["bins_m"]:
        name = str(spec["name"])
        eligible = [row for row in combined
                    if row["bin_name"] == name and row["constructed"]]
        chosen = independent_select(
            eligible,
            histories=int(gate["minimum_histories_per_bin"]),
            scenes=int(gate["minimum_scene_clusters_per_bin"]),
            maximum_per_scene=int(
                gate["maximum_selected_histories_per_scene_per_bin"]),
        )
        expected.extend(chosen)
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
                str(row.get("reason", "")) for row in combined
                if row["bin_name"] == name and not row["constructed"]
            )),
        }
    require(diagnostics == receipt["selection_diagnostics"],
            "population selection diagnostics changed")

    manifest_path = population_root / "role_pairs" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(sha256(manifest_path) == receipt["benchmark_manifest_sha256"],
            "benchmark manifest binding changed")
    require(len(expected) == len(manifest["episodes"])
            == int(receipt["histories"]) == 48
            and int(receipt["queries"]) == 96,
            "powered 48-history population changed")
    for population_index, (source, stored) in enumerate(
        zip(expected, manifest["episodes"])
    ):
        require(int(stored["population_index"]) == population_index
                and stored["candidate_identity_sha256"]
                == source["candidate_identity_sha256"]
                and int(stored["history_index"]) == int(source["history_index"])
                and stored["population_source"] == source["source_kind"]
                and int(stored["source_plan_index"])
                == int(source["source_plan_index"]),
                f"selected population row {population_index} changed")

    benchmark = audit(population_root / "role_pairs")
    require(benchmark["ok"] is True
            and benchmark["query_policy_outcomes_read"] is False
            and benchmark["online_history"]
            == "controlled_causal_rgb_geodesic_survey",
            "independent benchmark audit failed")
    require(benchmark["manifest_sha256"] == sha256(manifest_path),
            "benchmark audit hash changed")
    require(benchmark["histories_by_bin"] == {
        "0_to_20_m": 16, "20_to_30_m": 16, "30_to_50_m": 16,
    }, "independent 16/16/16 gate failed")

    return {
        "schema_version": VERIFY_SCHEMA,
        "verified": True,
        "population_receipt_sha256": sha256(receipt_path),
        "benchmark_manifest_sha256": benchmark["manifest_sha256"],
        "histories_by_bin": benchmark["histories_by_bin"],
        "scene_clusters_by_bin": benchmark["scene_clusters_by_bin"],
        "history_source": benchmark["online_history"],
        "base_candidates_preserved": len(base_rows),
        "expansion_candidates_appended": len(expansion_rows),
        "completion_ledger_sha256": canonical_sha256(ledger),
        "query_policy_outcomes_read": False,
        "formal_policy_evaluation_authorized": True,
        "fallback_completion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population-root", type=Path, required=True)
    parser.add_argument("--base-run-root", type=Path, required=True)
    parser.add_argument("--base-plan", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--expansion-run-root", type=Path, required=True)
    parser.add_argument("--expansion-plan", type=Path, required=True)
    parser.add_argument("--expansion-protocol", type=Path, required=True)
    parser.add_argument("--expansion-plan-verification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), "merged population verification exists")
    result = verify(
        population_root=args.population_root.resolve(),
        base_run_root=args.base_run_root.resolve(),
        base_plan_path=args.base_plan.resolve(),
        base_protocol_path=args.base_protocol.resolve(),
        expansion_run_root=args.expansion_run_root.resolve(),
        expansion_plan_path=args.expansion_plan.resolve(),
        expansion_protocol_path=args.expansion_protocol.resolve(),
        expansion_plan_verification_path=(
            args.expansion_plan_verification.resolve()),
    )
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.out.with_name(args.out.name + ".sha256").write_text(
        f"{sha256(args.out)}  {args.out.name}\n"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
