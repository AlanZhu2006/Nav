#!/usr/bin/env python3
"""Seal the result-blind cross-history A/B construction population."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from audit_shared_online_role_pairs import audit
from hm3d_fullmono_lifelong import load_protocol, require, sha256_file
from shared_online_role_pair_contract import validate_manifest


SCHEMA = "hm3d_fullmono_lifelong_ab_population_v1_20260824"


def write_manifest(root: Path, payload: dict) -> str:
    """Write a role-pair manifest without importing the Habitat builder.

    The finalizer is a renderer-free CPU seal.  Importing the scene builder
    pulled in Habitat's quaternion dependency even though all query assets and
    contracts had already been materialized.  Keep the seal dependency-light
    and validate the assembled manifest directly instead.
    """

    root.mkdir(parents=True, exist_ok=True)
    if payload["episodes"]:
        validate_manifest(payload)
    path = root / "manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    digest = sha256_file(path)
    (root / "manifest.json.sha256").write_text(
        digest + "  manifest.json\n"
    )
    return digest


def finalize(
    *,
    parent_root: Path,
    protocol_path: Path,
    construction_root: Path,
    out: Path,
) -> dict:
    protocol = load_protocol(protocol_path)
    parent_path = parent_root / protocol["parent"]["parent_manifest"]
    require(sha256_file(parent_path) == protocol["parent"]["parent_manifest_sha256"],
            "parent manifest changed")
    parent = json.loads(parent_path.read_text())
    require(not out.exists(), f"sealed population output exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=out.name + ".tmp.", dir=out.parent))
    fragments = []
    episodes = []
    identities = set()
    frozen_contract = None
    try:
        benchmark_root = temporary / "role_pairs"
        for index, scene_raw in enumerate(parent["scenes"]):
            scene = str(scene_raw)
            fragment = construction_root / f"{index:02d}_{scene}"
            completion_path = fragment / "completion.json"
            require(completion_path.is_file(), f"{scene}: construction incomplete")
            completion = json.loads(completion_path.read_text())
            require(completion.get("status") == "complete", f"{scene}: bad status")
            require(completion.get("query_policy_outcomes_read") is False,
                    f"{scene}: construction read policy outcomes")
            require(completion["protocol_sha256"] == sha256_file(protocol_path),
                    f"{scene}: protocol binding changed")
            source_manifest = fragment / "role_pairs/manifest.json"
            require(sha256_file(source_manifest)
                    == completion["role_pair_manifest_sha256"],
                    f"{scene}: fragment manifest changed")
            payload = json.loads(source_manifest.read_text())
            fragment_contract = payload.get("contract")
            require(isinstance(fragment_contract, dict),
                    f"{scene}: fragment contract missing")
            if frozen_contract is None:
                frozen_contract = fragment_contract
            else:
                require(fragment_contract == frozen_contract,
                        f"{scene}: fragment contract changed")
            for row in payload["episodes"]:
                identity = (str(row["scene"]), str(row["episode"]))
                require(identity not in identities, "duplicate recipient history")
                identities.add(identity)
                source_episode = fragment / "role_pairs" / identity[0] / identity[1]
                destination = benchmark_root / identity[0] / identity[1]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source_episode, destination)
                episodes.append(row)
            fragments.append({
                "scene_index": index,
                "scene": scene,
                "completion_sha256": sha256_file(completion_path),
                "materialized_A_histories": int(
                    completion["materialized_A_histories"]
                ),
                "constructible_AB_C_histories": int(
                    completion["constructible_AB_C_histories"]
                ),
            })
        require(frozen_contract is not None, "no frozen fragment contract")
        manifest = {
            "schema_version": "shared_online_role_pair_v1_20260814",
            "purpose": (
                "sealed result-blind actual-mono A plus unsupported "
                "cross-history Novel-B collection population"
            ),
            "source_online_root": str(
                (parent_root / "construction/scenes").resolve()
            ),
            "source_online_manifest_sha256": protocol["parent"][
                "fullmono_population_receipt_sha256"
            ],
            "construction_seed": int(protocol["novel_b_construction"].get(
                "construction_seed", 20260824
            )),
            # Preserve the exact contract already frozen into every result-
            # blind scene fragment rather than reconstructing it from a
            # renderer-side module at seal time.
            "contract": frozen_contract,
            "episodes": episodes,
        }
        manifest_sha = write_manifest(benchmark_root, manifest)
        audited = audit(benchmark_root) if episodes else {
            "ok": True, "episodes": 0, "scenes": 0,
            "manifest_sha256": manifest_sha,
        }
        scene_clusters = len({row["scene"] for row in episodes})
        power_gate = protocol.get("construction_power_gate")
        if power_gate is None:
            target_histories = 0
            target_scenes = 0
            target_met = True
        else:
            target_histories = int(power_gate["minimum_candidate_histories"])
            target_scenes = int(power_gate["minimum_scene_clusters"])
            target_met = (
                len(episodes) >= target_histories
                and scene_clusters >= target_scenes
            )
        receipt = {
            "schema_version": SCHEMA,
            "scope": protocol["scope"],
            "protocol_sha256": sha256_file(protocol_path),
            "parent_manifest_sha256": sha256_file(parent_path),
            "source_materialized_A_histories": sum(
                row["materialized_A_histories"] for row in fragments
            ),
            "constructible_AB_C_histories": len(episodes),
            "constructible_scene_clusters": scene_clusters,
            "construction_target_histories": target_histories,
            "construction_target_scene_clusters": target_scenes,
            "construction_target_met": target_met,
            "factual_B_authorized": target_met,
            "query_policy_outcomes_read": False,
            "navigation_outcome_selection": False,
            "benchmark_manifest_sha256": manifest_sha,
            "benchmark_audit": audited,
            "fragments": fragments,
        }
        receipt_path = temporary / "population_receipt.json"
        receipt_path.write_text(json.dumps(
            receipt, indent=2, sort_keys=True, allow_nan=False
        ) + "\n")
        (temporary / "population_receipt.json.sha256").write_text(
            sha256_file(receipt_path) + "  population_receipt.json\n"
        )
        files = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name not in {"BENCHMARK_FILES.sha256", "SEALED"}
        )
        with (temporary / "BENCHMARK_FILES.sha256").open("w") as handle:
            for path in files:
                handle.write(f"{sha256_file(path)}  {path.relative_to(temporary)}\n")
        (temporary / "SEALED").write_text("sealed before factual B navigation\n")
        temporary.replace(out)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--construction-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(
        parent_root=args.parent_root,
        protocol_path=args.protocol,
        construction_root=args.construction_root,
        out=args.out,
    )
    print(json.dumps({
        "histories": result["constructible_AB_C_histories"],
        "scenes": result["constructible_scene_clusters"],
        "query_policy_outcomes_read": result["query_policy_outcomes_read"],
        "out": str(args.out.resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
