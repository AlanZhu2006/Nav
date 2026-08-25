#!/usr/bin/env python3
"""Audit the frozen 25-unit controller-portability repair set.

The repair population is selected only from file completeness and recorded
runtime-contract failures.  Navigation outcomes are never read while choosing
the units.  Existing complete arms remain read-only; repaired arms live under
a separate immutable root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCHEMA = "lifelong_nnr_controller_repair_manifest_v1_20260824"
SCOPES = ("all_prior", "initial_leg_only", "forced_reject_native")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def entry_key(row: dict) -> tuple[str, int, str, str, str]:
    return (
        str(row["controller"]),
        int(row["population_index"]),
        str(row["scene"]),
        str(row["episode"]),
        str(row["scope"]),
    )


def label(row: dict) -> str:
    return (
        f"{int(row['population_index']):03d}_"
        f"{row['scene']}_{row['episode']}"
    )


def result_files(run: Path, episode: str) -> tuple[Path, Path, Path]:
    result = run / "result"
    return (
        result / "metric.csv",
        result / f"{episode}_plans.json",
        result / "summary.json",
    )


def is_complete(run: Path, episode: str) -> bool:
    return all(path.is_file() for path in result_files(run, episode))


def failure_reason(run: Path, episode: str) -> str:
    if is_complete(run, episode):
        return "complete"
    log = run / "logs/evaluator.log"
    if not log.is_file():
        return "absent"
    text = log.read_text(encoding="utf-8", errors="replace")
    if "query C reopened a session within one goal" in text:
        return "session_reopen"
    if "shared trace rendered RGB mismatch" in text:
        return "rgb_mismatch"
    return "other_partial"


def validate_result(run: Path, expected: dict, population_row: dict) -> dict:
    metric_path, plans_path, summary_path = result_files(
        run, str(expected["episode"]))
    require(
        metric_path.is_file() and plans_path.is_file()
        and summary_path.is_file(),
        f"incomplete repaired result {run}",
    )
    with metric_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 1, f"{run}: expected one metric row")
    metric = rows[0]
    require(
        metric["scene"] == expected["scene"]
        and metric["episode"] == expected["episode"]
        and metric["history_scope"] == expected["scope"],
        f"{run}: repaired result identity changed",
    )
    require(
        metric["benchmark_sha256"] == population_row["benchmark_sha256"]
        and metric["online_B_trace_sha256"]
        == population_row["online_b_trace_sha256"],
        f"{run}: repaired result escaped the sealed population",
    )
    summary = load_json(summary_path)
    require(
        int(summary["episodes"]) == 1
        and summary["history_scope"] == expected["scope"],
        f"{run}: repaired summary mismatch",
    )
    plans = load_json(plans_path)
    require(
        plans["history_scope"] == expected["scope"]
        and plans["runtime_role_visible"] is False,
        f"{run}: repaired plan contract mismatch",
    )
    return {
        "run_root": str(run.resolve()),
        "metric_sha256": file_sha256(metric_path),
        "plans_sha256": file_sha256(plans_path),
        "summary_sha256": file_sha256(summary_path),
    }


def audit(
    manifest_path: Path,
    population_path: Path,
    repair_root: Path | None = None,
) -> dict:
    manifest = load_json(manifest_path)
    require(manifest.get("schema") == SCHEMA, "repair manifest schema changed")
    require(
        file_sha256(population_path) == manifest["population_sha256"],
        "support population hash changed",
    )
    population_root = population_path.parent
    require((population_root / "SEALED").is_file(), "population is not sealed")
    receipt = (population_root / "population.json.sha256").read_text().split()
    require(
        bool(receipt) and receipt[0] == manifest["population_sha256"],
        "population receipt disagrees with repair manifest",
    )
    population = load_json(population_path)
    selected = list(population["accepted"])
    require(
        len(selected) == int(manifest["population_size"]),
        "repair population size changed",
    )
    scopes = tuple(manifest["scopes"])
    require(scopes == SCOPES, "repair scopes changed")
    controllers = dict(manifest["original_controller_runs"])
    entries = list(manifest["entries"])
    require(
        len(entries) == int(manifest["expected_missing"]),
        "repair entry count changed",
    )
    manifest_keys = {entry_key(row) for row in entries}
    require(len(manifest_keys) == len(entries), "duplicate repair entry")

    for row in entries:
        index = int(row["population_index"])
        require(0 <= index < len(selected), "repair population index escaped")
        source = selected[index]
        require(
            row["scene"] == source["scene"]
            and row["episode"] == source["episode"],
            "repair identity differs from sealed population",
        )
        require(row["controller"] in controllers, "unknown repair controller")
        require(row["scope"] in scopes, "unknown repair scope")
        require(
            row["reason"] in {"absent", "session_reopen", "rgb_mismatch"},
            "unregistered repair reason",
        )

    observed_missing: dict[tuple[str, int, str, str, str], str] = {}
    original_complete = 0
    for controller, raw_root in sorted(controllers.items()):
        root = Path(raw_root)
        for index, source in enumerate(selected):
            identity = {
                "population_index": index,
                "scene": str(source["scene"]),
                "episode": str(source["episode"]),
            }
            for scope in scopes:
                row = {
                    "controller": controller,
                    **identity,
                    "scope": scope,
                }
                run = root / "evaluation" / label(row) / scope
                reason = failure_reason(run, row["episode"])
                if reason == "complete":
                    original_complete += 1
                else:
                    observed_missing[entry_key(row)] = reason
    require(
        original_complete == int(manifest["expected_original_complete"]),
        "original complete-arm count changed",
    )
    require(
        set(observed_missing) == manifest_keys,
        "observed incomplete set differs from frozen repair manifest",
    )
    for row in entries:
        require(
            observed_missing[entry_key(row)] == row["reason"],
            f"repair reason changed for {entry_key(row)}",
        )

    repaired = []
    if repair_root is not None:
        observed_repair_keys = set()
        for metric_path in repair_root.glob(
                "*/evaluation/*/*/result/metric.csv"):
            run = metric_path.parent.parent
            relative = run.relative_to(repair_root)
            require(
                len(relative.parts) == 4
                and relative.parts[1] == "evaluation",
                f"unexpected repair layout {run}",
            )
            controller, _evaluation, run_label, scope = relative.parts
            with metric_path.open(newline="") as handle:
                metric_rows = list(csv.DictReader(handle))
            require(len(metric_rows) == 1, f"{run}: bad repair metric")
            metric = metric_rows[0]
            prefix, scene, episode = run_label.split("_", 2)
            key = (
                controller, int(prefix), scene, episode,
                str(metric["history_scope"]),
            )
            require(scope == metric["history_scope"], f"{run}: scope drift")
            observed_repair_keys.add(key)
        require(
            observed_repair_keys == manifest_keys,
            "completed repair set differs from frozen manifest",
        )
        for row in entries:
            run = (
                repair_root / row["controller"] / "evaluation"
                / label(row) / row["scope"]
            )
            repaired.append({
                **{key: row[key] for key in (
                    "controller", "population_index", "scene", "episode",
                    "scope", "reason")},
                **validate_result(
                    run, row, selected[int(row["population_index"])]),
            })

    return {
        "schema": "lifelong_nnr_controller_repair_audit_v1_20260824",
        "verified": True,
        "phase": "post_repair" if repair_root is not None else "pre_repair",
        "manifest_sha256": file_sha256(manifest_path),
        "population_sha256": file_sha256(population_path),
        "controllers": sorted(controllers),
        "population_size": len(selected),
        "expected_total_arms": len(controllers) * len(scopes) * len(selected),
        "original_complete_arms": original_complete,
        "repair_arms": len(repaired),
        "missing_reasons": {
            reason: sum(row["reason"] == reason for row in entries)
            for reason in ("absent", "session_reopen", "rgb_mismatch")
        },
        "repairs": repaired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.manifest, args.population, args.repair_root)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        require(not args.out.exists(), "audit output already exists")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
