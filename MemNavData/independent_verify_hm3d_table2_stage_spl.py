#!/usr/bin/env python3
"""Independently recover Table-II Leg-1/Leg-2 SPL from sealed raw rows.

The conference meeting verifier seals the factual A/B success waterfall and
the paired Leg-3 result.  This additive post-seal verifier binds that receipt,
then recomputes the two missing factual-prefix SPL values from every raw
metric row.  It never selects a prefix and never reads a downstream query to
change the Table-II population.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "hm3d_table2_stage_spl_verification_v1_20260830"
MEETING_SCHEMA = "hm3d_table2_meeting_result_verification_v1_20260830"
PARENT_SCHEMA = "hm3d_fresh_fullmono_parent_manifest_v1_20260820"
GOAL_A_COMPLETION_SCHEMA = "hm3d_fullmono_goal_a_scene_v1_20260820"
FACTUAL_B_COMPLETION_SCHEMA = (
    "hm3d_fullmono_lifelong_b_collection_v1_20260824"
)
UNION_SCHEMA = "hm3d_fullmono_lifelong_population_union_v1_20260830"
SOURCE_VERIFIERS = {
    "original_v4": "independent_natural_v4_population_verification.json",
    "natural_b_expansion": (
        "independent_natural_b_expansion_population_verification.json"
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified(path: Path) -> tuple[dict[str, Any], str]:
    require(path.is_file(), f"missing input: {path}")
    sidecar = path.with_name(path.name + ".sha256")
    require(sidecar.is_file(), f"missing SHA sidecar: {sidecar}")
    digest = sha256(path)
    fields = sidecar.read_text().split()
    require(
        len(fields) == 2
        and fields[0] == digest
        and Path(fields[1]).name == path.name,
        f"invalid SHA sidecar: {sidecar}",
    )
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload, digest


def contained(path: Path, root: Path, message: str) -> Path:
    resolved, base = path.resolve(), root.resolve()
    require(resolved == base or base in resolved.parents, message)
    return resolved


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_nonnegative(value: str | float | int, label: str) -> float:
    parsed = float(value)
    require(math.isfinite(parsed) and parsed >= 0.0, f"invalid {label}")
    return parsed


def spl(reached: int, geodesic_m: float, path_m: float) -> float:
    require(reached in {0, 1}, "success flag is not binary")
    geodesic_m = finite_nonnegative(geodesic_m, "geodesic distance")
    path_m = finite_nonnegative(path_m, "path length")
    if not reached:
        return 0.0
    denominator = max(geodesic_m, path_m)
    return 1.0 if denominator == 0.0 else geodesic_m / denominator


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)


def parent_episode_map(parent: dict[str, Any]) -> dict[str, list[str]]:
    raw = parent["episodes"]
    if isinstance(raw, dict):
        result = {
            str(scene): [str(row["episode"]) for row in scene_rows]
            for scene, scene_rows in raw.items()
        }
    elif isinstance(raw, list):
        result: dict[str, list[str]] = {}
        for row in raw:
            result.setdefault(str(row["scene"]), []).append(
                str(row["episode"])
            )
    else:
        raise RuntimeError("Goal-A parent episode layout changed")
    require(sum(map(len, result.values())) == int(parent["episode_count"]),
            "Goal-A parent denominator changed")
    return result


def digest_receipts(receipts: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for identity, value in sorted(receipts):
        digest.update(f"{identity}\t{value}\n".encode())
    return digest.hexdigest()


def audit_goal_a(
    parent_manifest_path: Path,
    meeting: dict[str, Any],
) -> dict[str, Any]:
    parent = json.loads(parent_manifest_path.read_text())
    require(parent.get("schema_version") == PARENT_SCHEMA,
            "Goal-A parent manifest schema changed")
    require(sha256(parent_manifest_path)
            == meeting["receipts"]["parent_manifest_sha256"],
            "Goal-A parent manifest differs from meeting receipt")
    parent_root = parent_manifest_path.parent.parent
    parent_verification_path = (
        parent_root / "hm3d_fullmono_mixed_role_independent_verification.json"
    )
    parent_verification, parent_verification_sha = load_verified(
        parent_verification_path
    )
    require(parent_verification_sha
            == meeting["receipts"]["goal_A_independent_verification_sha256"],
            "Goal-A verifier differs from meeting receipt")
    expected = parent_episode_map(parent)
    scenes = [str(scene) for scene in parent["scenes"]]
    require(set(expected).issubset(scenes), "Goal-A scene map changed")

    spl_values: list[float] = []
    successes = 0
    raw_receipts: list[tuple[str, str]] = []
    for scene_index, scene in enumerate(scenes):
        wanted = expected.get(scene, [])
        if not wanted:
            continue
        scene_root = parent_root / "goal_a/scenes" / f"{scene_index:02d}_{scene}"
        completion_path = scene_root / "completion.json"
        completion, completion_sha = load_verified(completion_path)
        require(completion.get("schema_version") == GOAL_A_COMPLETION_SCHEMA
                and completion.get("status") == "complete",
                f"{scene}: Goal-A completion changed")
        records = completion.get("records")
        require(isinstance(records, list), f"{scene}: Goal-A records missing")
        by_episode = {str(record["episode"]): record for record in records}
        require(set(by_episode) == set(wanted),
                f"{scene}: Goal-A episode set changed")
        raw_receipts.append((f"A/{scene}/completion", completion_sha))
        for episode in wanted:
            record = by_episode[episode]
            trace_path = contained(
                Path(record["trace_path"]), scene_root,
                f"{scene}/{episode}: Goal-A trace escaped scene root",
            )
            trace_sha = sha256(trace_path)
            require(trace_sha == str(record["trace_sha256"]),
                    f"{scene}/{episode}: Goal-A trace changed")
            trace = json.loads(trace_path.read_text())
            require(str(trace.get("source_scene")) == scene
                    and str(trace.get("episode")) == episode,
                    f"{scene}/{episode}: Goal-A trace identity changed")
            metric_path = trace_path.parent / "metric.csv"
            metric_rows = rows(metric_path)
            require(len(metric_rows) == 1
                    and metric_rows[0].get("episode") == episode,
                    f"{scene}/{episode}: Goal-A metric row changed")
            row = metric_rows[0]
            reached = int(float(row["reached_A"]))
            require(reached == int(bool(trace["reached"]))
                    == int(record["reached_a"]),
                    f"{scene}/{episode}: Goal-A success mismatch")
            require(row.get("leg1_trace_sha256") == trace_sha,
                    f"{scene}/{episode}: Goal-A metric trace binding changed")
            geodesic = finite_nonnegative(row["geo_A"], "Goal-A geodesic")
            path = finite_nonnegative(row["len_A"], "Goal-A path")
            require(close(path, trace["path_len"]),
                    f"{scene}/{episode}: Goal-A path mismatch")
            value = spl(reached, geodesic, path)
            require(close(value, float(row["spl_A"])),
                    f"{scene}/{episode}: Goal-A SPL mismatch")
            successes += reached
            spl_values.append(value)
            raw_receipts.extend([
                (f"A/{scene}/{episode}/trace", trace_sha),
                (f"A/{scene}/{episode}/metric", sha256(metric_path)),
            ])

    attempts = len(spl_values)
    expected_stage = meeting["leg1_novel"]
    require(attempts == int(expected_stage["attempts"])
            and successes == int(expected_stage["successes"]),
            "Goal-A raw recount differs from meeting result")
    return {
        "denominator": expected_stage["denominator"],
        "attempts": attempts,
        "successes": successes,
        "sr": successes / attempts,
        "spl": sum(spl_values) / attempts,
        "raw_metric_rows": attempts,
        "raw_artifact_set_sha256": digest_receipts(raw_receipts),
    }


def audit_goal_b(
    source_union_root: Path,
    meeting: dict[str, Any],
) -> dict[str, Any]:
    union_path = source_union_root / "population/population.json"
    union, union_sha = load_verified(union_path)
    require(union.get("schema_version") == UNION_SCHEMA,
            "source population union schema changed")
    require(union_sha
            == meeting["receipts"]["source_population_union_sha256"],
            "source population union differs from meeting receipt")
    source_rows = union.get("source_populations")
    require(isinstance(source_rows, list)
            and {str(row["name"]) for row in source_rows}
            == set(SOURCE_VERIFIERS),
            "Table-II source populations changed")

    all_spl: list[float] = []
    all_receipts: list[tuple[str, str]] = []
    successes = 0
    breakdown: dict[str, dict[str, Any]] = {}
    expected_verifier_hashes = meeting["receipts"][
        "source_factual_B_verification_sha256"
    ]
    for source in source_rows:
        name = str(source["name"])
        root = Path(source["run_root"])
        verifier, verifier_sha = load_verified(root / SOURCE_VERIFIERS[name])
        require(verifier.get("verified") is True
                and verifier.get("query_navigation_outcomes_read") is False,
                f"{name}: factual-B verifier changed")
        require(verifier_sha == expected_verifier_hashes[name],
                f"{name}: factual-B verifier differs from meeting receipt")
        manifest_path = root / "ab_population/role_pairs/manifest.json"
        manifest, manifest_sha = load_verified(manifest_path)
        episodes = manifest.get("episodes")
        expected_attempts = int(verifier["factual_B_rollouts"])
        require(isinstance(episodes, list)
                and len(episodes) == expected_attempts,
                f"{name}: factual-B manifest denominator changed")
        source_spl: list[float] = []
        source_successes = 0
        all_receipts.append((f"B/{name}/manifest", manifest_sha))
        for index, item in enumerate(episodes):
            scene, episode = str(item["scene"]), str(item["episode"])
            label = f"{index:03d}_{scene}_{episode}"
            output = root / "factual_b" / label
            completion_path = output / "completion.json"
            completion, completion_sha = load_verified(completion_path)
            require(completion.get("schema_version")
                    == FACTUAL_B_COMPLETION_SCHEMA
                    and completion.get("status") == "complete"
                    and int(completion["history_index"]) == index
                    and str(completion["scene"]) == scene
                    and str(completion["episode"]) == episode,
                    f"{name}/{label}: factual-B completion changed")
            trace_path = contained(
                Path(completion["B_trace_path"]), output,
                f"{name}/{label}: factual-B trace escaped output",
            )
            trace_sha = sha256(trace_path)
            require(trace_sha == str(completion["B_trace_sha256"]),
                    f"{name}/{label}: factual-B trace changed")
            trace = json.loads(trace_path.read_text())
            metric_path = output / "result/metric.csv"
            metric_sha = sha256(metric_path)
            require(metric_sha == str(completion["result_metric_sha256"]),
                    f"{name}/{label}: factual-B metric changed")
            metric_rows = rows(metric_path)
            require(len(metric_rows) == 1
                    and metric_rows[0].get("analysis_role") == "novel"
                    and metric_rows[0].get("scene") == scene
                    and metric_rows[0].get("episode") == episode,
                    f"{name}/{label}: factual-B metric row changed")
            row = metric_rows[0]
            reached = int(row["reached"])
            require(reached == int(bool(completion["reached_B"]))
                    == int(bool(trace["reached"])),
                    f"{name}/{label}: factual-B success mismatch")
            geodesic = finite_nonnegative(
                row["geodesic_m"], "factual-B geodesic"
            )
            path = finite_nonnegative(row["path_len_m"], "factual-B path")
            require(close(path, completion["path_len_B_m"])
                    and close(path, trace["path_len"]),
                    f"{name}/{label}: factual-B path mismatch")
            require(close(float(row["final_goal_dist_m"]),
                          completion["final_goal_dist_B_m"]),
                    f"{name}/{label}: factual-B final distance mismatch")
            value = spl(reached, geodesic, path)
            source_successes += reached
            source_spl.append(value)
            all_receipts.extend([
                (f"B/{name}/{label}/completion", completion_sha),
                (f"B/{name}/{label}/trace", trace_sha),
                (f"B/{name}/{label}/metric", metric_sha),
            ])
        require(source_successes == int(verifier["factual_B_successes"]),
                f"{name}: factual-B success recount changed")
        successes += source_successes
        all_spl.extend(source_spl)
        breakdown[name] = {
            "attempts": len(source_spl),
            "successes": source_successes,
            "spl": sum(source_spl) / len(source_spl),
            "manifest_sha256": manifest_sha,
            "independent_verification_sha256": verifier_sha,
        }

    attempts = len(all_spl)
    expected_stage = meeting["leg2_novel"]
    require(attempts == int(expected_stage["attempts"])
            and successes == int(expected_stage["successes"]),
            "Goal-B raw recount differs from meeting result")
    return {
        "denominator": expected_stage["denominator"],
        "attempts": attempts,
        "successes": successes,
        "sr": successes / attempts,
        "spl": sum(all_spl) / attempts,
        "raw_metric_rows": attempts,
        "source_breakdown": breakdown,
        "raw_artifact_set_sha256": digest_receipts(all_receipts),
    }


def verify(
    *,
    meeting_verification_path: Path,
    parent_manifest_path: Path,
    source_union_root: Path,
) -> dict[str, Any]:
    meeting, meeting_sha = load_verified(meeting_verification_path)
    require(meeting.get("schema_version") == MEETING_SCHEMA
            and meeting.get("verified") is True,
            "sealed Table-II meeting result did not pass")
    goal_a = audit_goal_a(parent_manifest_path, meeting)
    goal_b = audit_goal_b(source_union_root, meeting)
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "scope": "post-seal factual Leg-1/Leg-2 SR and SPL recount",
        "meeting_verification_sha256": meeting_sha,
        "selection_or_policy_execution_performed": False,
        "downstream_query_outcomes_used_for_selection": False,
        "leg1_novel": goal_a,
        "leg2_novel": goal_b,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting-verification", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--source-union-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(
        meeting_verification_path=args.meeting_verification.resolve(),
        parent_manifest_path=args.parent_manifest.resolve(),
        source_union_root=args.source_union_root.resolve(),
    )
    write_exclusive(args.out.resolve(), payload)
    print(json.dumps({
        "verified": True,
        "leg1": payload["leg1_novel"],
        "leg2": payload["leg2_novel"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
