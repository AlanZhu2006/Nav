#!/usr/bin/env python3
"""Extract a compact, auditable Final14 CEC mechanism ledger.

The formal authority run stores full per-step plan files.  This utility keeps
only the first (and asserted time-invariant) retrieval/witness decision for
each query, plus the closed-loop endpoint.  It does not select thresholds or
modify a navigation outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


SCHEMA = "final14_cec_mechanism_ledger_v1_20260830"
ROLES = ("novel", "revisit")
ARMS = ("mono_cec", "mono_unthresholded_witness")
SUPPORT_THRESHOLD = 0.5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing input: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def manifest_queries(manifest: Mapping[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for episode in manifest["episodes"]:
        scene = str(episode["scene"])
        episode_id = str(episode["episode"])
        for pair in episode["pairs"]:
            for query in pair["queries"]:
                role = str(query["analysis_role"])
                key = (scene, episode_id, role)
                require(key not in result, f"duplicate manifest query: {key}")
                curve = query.get("covis_curve")
                require(isinstance(curve, list) and curve,
                        f"missing covisibility curve: {key}")
                result[key] = {
                    "query_id": str(query["query_id"]),
                    "max_history_covis": float(query["max_online_a_covis"]),
                    "covis_curve": [float(value) for value in curve],
                }
    return result


def decision_row(plan: Mapping[str, Any], label: str) -> dict[str, Any]:
    rows = [
        row for row in plan["query_leg"]
        if row.get("router_candidate_order_dino") is not None
    ]
    require(bool(rows), f"no CEC decision rows: {label}")
    fields = (
        "router_candidate_order_dino",
        "router_candidate_order_used",
        "router_selected_anchor",
        "router_selected_candidate_dino_rank",
        "certified_relocalization_accepted",
        "certified_relocalization_ok",
        "certified_relocalization_reason",
        "certified_relocalization_certificate",
        "certified_relocalization_pnp",
    )
    reference = {field: rows[0].get(field) for field in fields}
    reference_text = json.dumps(reference, sort_keys=True, allow_nan=False)
    require(all(
        json.dumps({field: row.get(field) for field in fields},
                   sort_keys=True, allow_nan=False) == reference_text
        for row in rows
    ), f"CEC decision changed during query: {label}")
    return reference


def support_at(curve: list[float], anchor: int) -> float:
    require(0 <= anchor < len(curve),
            f"anchor {anchor} outside covisibility curve of length {len(curve)}")
    return float(curve[anchor])


def extract(evaluation_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = read_object(manifest_path)
    query_truth = manifest_queries(manifest)
    records: list[dict[str, Any]] = []
    source_files: list[dict[str, str]] = []

    history_dirs = sorted(path for path in evaluation_root.iterdir()
                          if path.is_dir())
    require(len(history_dirs) == 21,
            f"expected 21 Final14 histories, found {len(history_dirs)}")
    for history_dir in history_dirs:
        contract_path = history_dir / "episode_contract.json"
        contract = read_object(contract_path)
        require(contract.get("runtime_role_visibility") == "none",
                f"role label visible at runtime: {history_dir}")
        require(contract.get("sole_intervention") == "operational_authority_policy",
                f"authority contract changed: {history_dir}")
        scene = str(contract["scene"])
        episode_id = str(contract["episode"])
        history_index = int(contract["history_index"])
        source_files.append({
            "relative_path": str(contract_path.relative_to(evaluation_root)),
            "sha256": sha256_file(contract_path),
        })

        for role in ROLES:
            truth = query_truth[(scene, episode_id, role)]
            plans: dict[str, dict[str, Any]] = {}
            decisions: dict[str, dict[str, Any]] = {}
            plan_receipts: dict[str, dict[str, str]] = {}
            for arm in ARMS:
                paths = sorted((history_dir / arm).glob(f"*_{role}_plans.json"))
                require(len(paths) == 1,
                        f"expected one {arm}/{role} plan under {history_dir}")
                path = paths[0]
                plan = read_object(path)
                require(plan.get("analysis_role_not_forwarded") is True,
                        f"role forwarding audit failed: {path}")
                plans[arm] = plan
                decisions[arm] = decision_row(plan, str(path))
                plan_receipts[arm] = {
                    "relative_path": str(path.relative_to(evaluation_root)),
                    "sha256": sha256_file(path),
                }
                source_files.append(plan_receipts[arm])

            strict = decisions["mono_cec"]
            witness = decisions["mono_unthresholded_witness"]
            proposal_fields = (
                "router_candidate_order_dino",
                "router_candidate_order_used",
                "router_selected_anchor",
                "router_selected_candidate_dino_rank",
            )
            require(all(strict[field] == witness[field] for field in proposal_fields),
                    f"proposal/witness mismatch: {(scene, episode_id, role)}")

            dino_order = [int(value)
                          for value in strict["router_candidate_order_dino"]]
            require(len(dino_order) == 8 and len(set(dino_order)) == 8,
                    f"invalid top-8 proposal: {(scene, episode_id, role)}")
            selected_anchor = int(strict["router_selected_anchor"])
            curve = truth["covis_curve"]
            proposal_support = [support_at(curve, anchor)
                                for anchor in dino_order]
            selected_support = support_at(curve, selected_anchor)

            # The strict arm may stop at an early evidence check and therefore
            # legitimately omit a pose.  The unthresholded arm evaluates the
            # same selected proposal through the finite-PnP witness; use that
            # arm for witness diagnostics, not for proposal matching.
            pnp = witness["certified_relocalization_pnp"]
            require(isinstance(pnp, dict),
                    f"missing finite-PnP receipt: {(scene, episode_id, role)}")
            strict_accept = bool(strict["certified_relocalization_accepted"])
            witness_accept = bool(witness["certified_relocalization_accepted"])
            require(witness_accept == (pnp.get("pose9") is not None),
                    f"finite-PnP authority/pose mismatch: "
                    f"{(scene, episode_id, role)}")

            arm_results = {}
            for arm in ARMS:
                result = plans[arm]["query_result"]
                final_distance = float(result["final_goal_dist_m"])
                reached = bool(result["reached"])
                require(reached == (final_distance < 1.0),
                        f"success receipt mismatch: {(scene, episode_id, role, arm)}")
                arm_results[arm] = {
                    "success": reached,
                    "final_goal_distance_m": final_distance,
                }

            records.append({
                "history_index": history_index,
                "scene": scene,
                "episode": episode_id,
                "query_id": truth["query_id"],
                "analysis_role": role,
                "runtime_role_visible": False,
                "max_history_covis": truth["max_history_covis"],
                "support_threshold_for_diagnostic_only": SUPPORT_THRESHOLD,
                "dino_top1_anchor": dino_order[0],
                "dino_top1_covis": proposal_support[0],
                "dino_top1_supported": proposal_support[0] >= SUPPORT_THRESHOLD,
                "dino_top8_anchors": dino_order,
                "dino_top8_covis": proposal_support,
                "dino_top8_contains_supported_anchor": any(
                    value >= SUPPORT_THRESHOLD for value in proposal_support
                ),
                "geometry_selected_anchor": selected_anchor,
                "geometry_selected_dino_rank": int(
                    strict["router_selected_candidate_dino_rank"]
                ),
                "geometry_selected_covis": selected_support,
                "geometry_selected_supported": selected_support >= SUPPORT_THRESHOLD,
                "finite_pnp_witness_available": witness_accept,
                "strict_certificate_accept": strict_accept,
                "strict_certificate_reason": str(
                    strict["certified_relocalization_reason"]
                ),
                "pnp_status": str(pnp["status"]),
                "pnp_inliers": (int(pnp["inliers"])
                                if pnp.get("inliers") is not None else None),
                "pnp_query_inlier_coverage": (
                    float(pnp["query_inlier_coverage"])
                    if pnp.get("query_inlier_coverage") is not None else None
                ),
                "pnp_reference_inlier_coverage": (
                    float(pnp["reference_inlier_coverage"])
                    if pnp.get("reference_inlier_coverage") is not None else None
                ),
                "pnp_reprojection_rmse_px": (
                    float(pnp["reprojection_rmse_px"])
                    if pnp.get("reprojection_rmse_px") is not None else None
                ),
                "closed_loop": arm_results,
                "source_plans": plan_receipts,
            })

    records.sort(key=lambda row: (row["history_index"], row["analysis_role"]))
    require(len(records) == 42 and len(query_truth) == 42,
            "Final14 query universe is incomplete")
    source_files.sort(key=lambda row: row["relative_path"])
    source_digest = hashlib.sha256("".join(
        f"{row['sha256']}  {row['relative_path']}\n" for row in source_files
    ).encode()).hexdigest()
    return {
        "schema_version": SCHEMA,
        "status": "complete_compact_extraction",
        "scope": "consumed_final14_mechanism_attribution_not_fresh_confirmation",
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "authority_evaluation_root": str(evaluation_root),
        "support_threshold_for_diagnostic_only": SUPPORT_THRESHOLD,
        "support_threshold_selected_after_outcomes": False,
        "navigation_outcomes_used_for_threshold_selection": False,
        "history_count": len(history_dirs),
        "query_count": len(records),
        "source_file_count": len(source_files),
        "source_file_digest_sha256": source_digest,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, extract(args.evaluation_root, args.manifest))
    print(args.output)


if __name__ == "__main__":
    main()
