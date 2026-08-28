#!/usr/bin/env python3
"""Fail-closed, outcome-blind audit for HM3D factual-C repair attempt 2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "hm3d_lifelong_underpowered_collect_repair_attempt2_v1_20260828"
BASE_SCHEMA = "hm3d_lifelong_underpowered_collect_repair_v1_20260828"
POPULATION_SHA = (
    "ec11c0dbc43a4abe585330c1ce52a8c14ad1d4b1da6fd8397e1d15592707a6d5"
)
STARTUP_FAILURE = "unrecognized arguments: --reject-policy shared_native_exact"
QUERY_OUTPUTS = (
    "shared_c_population",
    "shared_c_evaluation",
    "shared_c_aggregate",
    "shared_c_independent_verification.json",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    require(isinstance(payload, dict), f"{path}: expected JSON object")
    return payload


def required_result_paths(root: Path, label: str, episode: str) -> list[Path]:
    item = root / "shared_c_collection" / label
    result = item / "result"
    return [
        result / "metric.csv",
        result / f"{episode}_shared_C_trace.json",
        result / f"{episode}_plans.json",
        result / "summary.json",
        item / "compute_identity.json",
        item / "result_inputs.sha256",
    ]


def verify_sha256_manifest(path: Path) -> None:
    require(path.is_file(), f"missing checksum manifest: {path}")
    rows = path.read_text().splitlines()
    require(bool(rows), f"empty checksum manifest: {path}")
    for row in rows:
        fields = row.split(maxsplit=1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                f"malformed checksum row in {path}")
        target = Path(fields[1].lstrip("*"))
        if not target.is_absolute():
            target = path.parent / target
        require(target.is_file(), f"manifest input disappeared: {target}")
        require(sha256_file(target) == fields[0],
                f"manifest input changed: {target}")


def _label(index: int, rows: list[dict[str, Any]]) -> str:
    row = rows[index]
    return f"{index:03d}_{row['scene']}_{row['episode']}"


def _fingerprints(
    run_root: Path, rows: list[dict[str, Any]], indices: list[int],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for index in indices:
        row = rows[index]
        label = _label(index, rows)
        paths = required_result_paths(run_root, label, str(row["episode"]))
        missing = [str(path) for path in paths if not path.is_file()]
        require(not missing, f"{label}: structurally incomplete: {missing}")
        result[label] = {
            str(path.relative_to(run_root)): sha256_file(path) for path in paths
        }
    return result


def _verify_startup_archive(
    *, run_root: Path, archive: Path, rows: list[dict[str, Any]],
    started: list[int], repair: list[int], phase: str,
) -> None:
    started_labels = {_label(index, rows) for index in started}
    repair_labels = {_label(index, rows) for index in repair}
    active_root = run_root / "shared_c_collection"
    if phase == "pre_archive":
        require(not archive.exists(), "attempt-2 incident archive already exists")
        for index in repair:
            active = active_root / _label(index, rows)
            if index in started:
                require(active.is_dir(), f"index {index}: startup partial missing")
                log = active / "logs/server_hub.log"
                require(log.is_file() and STARTUP_FAILURE in log.read_text(
                    errors="replace"), f"index {index}: startup failure changed")
                require(not (active / "logs/evaluator.log").exists(),
                        f"index {index}: evaluator unexpectedly started")
            else:
                require(not active.exists(),
                        f"index {index}: cancelled nonstarter produced output")
        return

    require(archive.is_dir(), "attempt-2 incident archive is missing")
    actual = {path.name for path in archive.iterdir() if path.is_dir()}
    require(actual == started_labels,
            "attempt-2 incident archive directory set changed")
    require(not ({path.name for path in archive.iterdir()} - started_labels),
            "unexpected file in attempt-2 incident archive")
    if phase != "ready_to_seal":
        for label in repair_labels:
            require(not (active_root / label).exists(),
                    f"{label}: active repair path exists before retry")
    for label in started_labels:
        log = archive / label / "logs/server_hub.log"
        require(log.is_file() and STARTUP_FAILURE in log.read_text(
            errors="replace"), f"{label}: archived startup failure changed")
        require(not (archive / label / "logs/evaluator.log").exists(),
                f"{label}: archived evaluator unexpectedly exists")


def _verify_item_runtime(
    *, root: Path, label: str, episode: str, expected_node: str,
) -> dict[str, str]:
    paths = required_result_paths(root, label, episode)
    missing = [str(path) for path in paths if not path.is_file()]
    require(not missing, f"{label}: output is structurally incomplete")
    identity = load_object(root / "shared_c_collection" / label /
                           "compute_identity.json")
    require(str(identity.get("host", "")).split(".", 1)[0] == expected_node,
            f"{label}: compute host changed")
    hub = identity.get("cec_hub", {})
    require(hub.get("cli_contract") == "legacy_shared_native_exact",
            f"{label}: legacy hub CLI contract was not sealed")
    require(hub.get("reject_policy") == "shared_native_exact",
            f"{label}: reject policy changed")
    verify_sha256_manifest(root / "shared_c_collection" / label /
                           "result_inputs.sha256")
    return {str(path.relative_to(root)): sha256_file(path) for path in paths}


def audit(
    *, protocol_path: Path, base_protocol_path: Path, run_root: Path,
    phase: str,
) -> dict[str, Any]:
    protocol = load_object(protocol_path)
    require(protocol.get("schema_version") == SCHEMA,
            "attempt-2 protocol schema changed")
    authority = protocol.get("source_authority", {})
    require(str(run_root.resolve()) == authority.get("run_root"),
            "run root differs from frozen attempt-2 authority")
    require(authority.get("population_sha256") == POPULATION_SHA,
            "population authority changed")
    require(sha256_file(base_protocol_path) ==
            authority.get("base_repair_protocol_sha256"),
            "base repair protocol hash changed")
    base = load_object(base_protocol_path)
    require(base.get("schema_version") == BASE_SCHEMA,
            "base repair protocol schema changed")
    freeze = protocol.get("freeze_boundary", {})
    for key in (
        "successful_factual_C_navigation_outcomes_read",
        "B2_navigation_outcomes_read",
        "attempt1_navigation_outcomes_exist",
        "repair_selection_uses_navigation_outcomes",
    ):
        require(freeze.get(key) is False, f"freeze boundary changed: {key}")

    population_path = run_root / str(authority["population_relative_path"])
    require(sha256_file(population_path) == POPULATION_SHA,
            "source population file changed")
    population = load_object(population_path)
    rows = population.get("accepted")
    require(isinstance(rows, list) and len(rows) == 22,
            "source population no longer has 22 histories")
    require(len({str(row["scene"]) for row in rows}) == 15,
            "source population no longer has 15 scene clusters")

    contract = protocol.get("repair_contract", {})
    repair = contract.get("all_repair_indices")
    started = contract.get("attempt1_partial_indices_to_archive")
    retained = contract.get("preserved_completed_indices")
    require(repair == [0, 1, 7, 9, 11, 13], "repair set changed")
    require(started == [0, 7, 11], "attempt-1 partial set changed")
    require(sorted(repair + retained) == list(range(22)),
            "repair/retained partition changed")
    items = protocol.get("repair_items")
    require(isinstance(items, list) and len(items) == 6,
            "repair item map changed")
    by_index = {int(item["index"]): item for item in items}
    require(sorted(by_index) == repair, "repair item indices changed")
    for index in repair:
        require(by_index[index].get("label") == _label(index, rows),
                f"index {index}: repair label changed")

    present = [name for name in QUERY_OUTPUTS if (run_root / name).exists()]
    require(not present, "downstream output exists before factual-C seal: " +
            ", ".join(present))
    baseline_path = Path(authority["completed_output_baseline_audit"])
    require(sha256_file(baseline_path) ==
            authority["completed_output_baseline_audit_sha256"],
            "completed-output baseline audit changed")
    baseline = load_object(baseline_path)
    require(baseline.get("completed_indices_verified") == retained,
            "baseline retained-index set changed")
    retained_fingerprints = _fingerprints(run_root, rows, retained)
    require(retained_fingerprints ==
            baseline.get("completed_required_file_sha256"),
            "one or more retained factual-C outputs changed")

    old_archive = Path(base["repair_contract"][
        "archive_failed_partial_outputs"])
    require(old_archive.is_dir(), "original six-partial archive disappeared")
    require({path.name for path in old_archive.iterdir() if path.is_dir()} ==
            {_label(index, rows) for index in repair},
            "original six-partial archive changed")
    archive = Path(contract["attempt1_archive_root"])
    if phase == "pre_archive":
        _verify_startup_archive(
            run_root=run_root, archive=archive, rows=rows, started=started,
            repair=repair, phase=phase,
        )
    else:
        _verify_startup_archive(
            run_root=run_root, archive=archive, rows=rows, started=started,
            repair=repair,
            phase="ready_to_seal" if phase == "ready_to_seal" else "post_archive",
        )

    runtime_fingerprints: dict[str, dict[str, str]] = {}
    runtime_health_contract: dict[str, Any] = {}
    if phase == "smoke_ready":
        smoke_root = Path(contract["collect_smoke_root"])
        index = int(contract["smoke_index"])
        label = _label(index, rows)
        runtime_fingerprints[label] = _verify_item_runtime(
            root=smoke_root, label=label,
            episode=str(rows[index]["episode"]),
            expected_node=str(contract["smoke_node"]),
        )
        health = load_object(smoke_root / "shared_c_collection" / label /
                             "hub_health.json")
        require(health.get("ok") is True, "smoke hub health failed")
        # The frozen HM3D hub predates the two explicit authority fields in
        # ``/healthz``.  Its exact reject semantics are already proved above
        # by the hash-bound compute identity (legacy_shared_native_exact) and
        # by the AST compatibility gate used before startup.  Do not require
        # fields that this immutable health schema never emitted.  We still
        # fail closed on the controller, initialization and reset state.  A
        # newer explicit schema, when present, must agree with the same
        # authority contract.
        require(health.get("schema") == "cec_controller_portability_hub_v2",
                "smoke hub health schema changed")
        require(health.get("controller") == "navdp",
                "smoke hub controller changed")
        require(health.get("initialized") is True and
                health.get("reset_required") is False,
                "smoke hub did not finish in a healthy initialized state")
        require(health.get("force_reject_native") is False,
                "smoke hub force-reject mode changed")
        explicit_policy = health.get("reject_policy")
        explicit_controller = health.get("reject_controller")
        if explicit_policy is not None or explicit_controller is not None:
            require(explicit_policy == "shared_native_exact" and
                    explicit_controller == "navdp",
                    "smoke hub explicit authority semantics changed")
            health_schema = "explicit_authority_fields"
        else:
            health_schema = "legacy_health_plus_identity_and_ast"
        runtime_health_contract[label] = {
            "schema": str(health["schema"]),
            "controller": str(health["controller"]),
            "initialized": True,
            "reset_required": False,
            "force_reject_native": False,
            "authority_receipt_mode": health_schema,
        }
    elif phase == "ready_to_seal":
        for index in repair:
            item = by_index[index]
            label = _label(index, rows)
            runtime_fingerprints[label] = _verify_item_runtime(
                root=run_root, label=label,
                episode=str(rows[index]["episode"]),
                expected_node=str(item["node"]),
            )

    return {
        "schema_version": (
            "hm3d_lifelong_underpowered_collect_repair_attempt2_audit_"
            "v2_20260829"
        ),
        "verified": True,
        "phase": phase,
        "source_population_sha256": POPULATION_SHA,
        "retained_indices_verified": retained,
        "repair_indices": repair,
        "attempt1_partial_indices": started,
        "retained_required_file_sha256": retained_fingerprints,
        "runtime_required_file_sha256": runtime_fingerprints,
        "runtime_health_contract": runtime_health_contract,
        "successful_factual_C_navigation_outcomes_read": False,
        "B2_navigation_outcomes_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("pre_archive", "post_archive", "smoke_ready", "ready_to_seal"),
        required=True,
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = audit(
        protocol_path=args.protocol, base_protocol_path=args.base_protocol,
        run_root=args.run_root, phase=args.phase,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
