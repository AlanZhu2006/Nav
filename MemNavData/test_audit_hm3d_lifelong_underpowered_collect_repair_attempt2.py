from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from audit_hm3d_lifelong_underpowered_collect_repair_attempt2 import (
    POPULATION_SHA,
    STARTUP_FAILURE,
    audit,
    required_result_paths,
    sha256_file,
)


REPAIR = [0, 1, 7, 9, 11, 13]
STARTED = [0, 7, 11]
RETAINED = [index for index in range(22) if index not in REPAIR]
NODES = {0: "gh005", 1: "gh001", 7: "ga005", 9: "ga003",
         11: "ga028", 13: "ga002"}


def _json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _rows() -> list[dict[str, str]]:
    return [
        {"scene": f"scene_{index % 15:02d}", "episode": f"episode_{index:04d}"}
        for index in range(22)
    ]


def _label(rows: list[dict[str, str]], index: int) -> str:
    row = rows[index]
    return f"{index:03d}_{row['scene']}_{row['episode']}"


def _complete(root: Path, rows: list[dict[str, str]], index: int,
              node: str = "retained") -> None:
    label = _label(rows, index)
    paths = required_result_paths(root, label, rows[index]["episode"])
    for path in paths[:4]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"opaque-{index}-{path.name}\n")
    identity = {
        "host": node,
        "cec_hub": {
            "cli_contract": "legacy_shared_native_exact",
            "reject_policy": "shared_native_exact",
        },
    }
    _json(paths[4], identity)
    manifest_rows = []
    for path in paths[:5]:
        manifest_rows.append(f"{sha256_file(path)}  {path}\n")
    paths[5].write_text("".join(manifest_rows))


@pytest.fixture
def authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    run = tmp_path / "run"
    rows = _rows()
    population = run / "population/population.json"
    _json(population, {"accepted": rows})
    monkeypatch.setattr(
        "audit_hm3d_lifelong_underpowered_collect_repair_attempt2.sha256_file",
        lambda path, original=sha256_file: (
            POPULATION_SHA if Path(path) == population else original(Path(path))
        ),
    )
    for index in RETAINED:
        _complete(run, rows, index)

    old_archive = run / "failed_attempts/original"
    for index in REPAIR:
        (old_archive / _label(rows, index)).mkdir(parents=True)
    for index in STARTED:
        active = run / "shared_c_collection" / _label(rows, index)
        (active / "logs").mkdir(parents=True)
        (active / "logs/server_hub.log").write_text(STARTUP_FAILURE + "\n")

    base = tmp_path / "base.json"
    _json(base, {
        "schema_version": "hm3d_lifelong_underpowered_collect_repair_v1_20260828",
        "repair_contract": {"archive_failed_partial_outputs": str(old_archive)},
    })
    baseline = tmp_path / "baseline.json"
    fingerprints = {}
    for index in RETAINED:
        label = _label(rows, index)
        fingerprints[label] = {
            str(path.relative_to(run)): sha256_file(path)
            for path in required_result_paths(
                run, label, rows[index]["episode"])
        }
    _json(baseline, {
        "completed_indices_verified": RETAINED,
        "completed_required_file_sha256": fingerprints,
    })

    archive = run / "failed_attempts/attempt1_startup"
    smoke = tmp_path / "smoke"
    protocol = tmp_path / "attempt2.json"
    items = [
        {"index": index, "label": _label(rows, index), "node": NODES[index],
         "partition": "h100_tandon" if index < 2 else "a100_tandon",
         "lane": index % 2}
        for index in REPAIR
    ]
    _json(protocol, {
        "schema_version": (
            "hm3d_lifelong_underpowered_collect_repair_attempt2_v1_20260828"),
        "source_authority": {
            "run_root": str(run.resolve()),
            "population_relative_path": "population/population.json",
            "population_sha256": POPULATION_SHA,
            "base_repair_protocol_sha256": sha256_file(base),
            "completed_output_baseline_audit": str(baseline),
            "completed_output_baseline_audit_sha256": sha256_file(baseline),
        },
        "freeze_boundary": {
            "successful_factual_C_navigation_outcomes_read": False,
            "B2_navigation_outcomes_read": False,
            "attempt1_navigation_outcomes_exist": False,
            "repair_selection_uses_navigation_outcomes": False,
        },
        "repair_contract": {
            "all_repair_indices": REPAIR,
            "attempt1_partial_indices_to_archive": STARTED,
            "preserved_completed_indices": RETAINED,
            "attempt1_archive_root": str(archive),
            "collect_smoke_root": str(smoke),
            "smoke_index": 0,
            "smoke_node": "gh005",
        },
        "repair_items": items,
    })
    return {
        "run": run, "rows": rows, "base": base, "baseline": baseline,
        "archive": archive, "smoke": smoke, "protocol": protocol,
    }


def _audit(authority: dict[str, object], phase: str) -> dict[str, object]:
    return audit(
        protocol_path=Path(authority["protocol"]),
        base_protocol_path=Path(authority["base"]),
        run_root=Path(authority["run"]), phase=phase,
    )


def _archive_startup_partials(authority: dict[str, object]) -> None:
    run = Path(authority["run"])
    archive = Path(authority["archive"])
    rows = authority["rows"]
    assert isinstance(rows, list)
    archive.mkdir(parents=True)
    for index in STARTED:
        source = run / "shared_c_collection" / _label(rows, index)
        source.rename(archive / source.name)


def test_pre_and_post_archive_are_outcome_blind(authority: dict[str, object]) -> None:
    assert _audit(authority, "pre_archive")["verified"] is True
    _archive_startup_partials(authority)
    payload = _audit(authority, "post_archive")
    assert payload["runtime_required_file_sha256"] == {}
    assert payload["successful_factual_C_navigation_outcomes_read"] is False


def test_smoke_requires_proved_legacy_contract(authority: dict[str, object]) -> None:
    _archive_startup_partials(authority)
    rows = authority["rows"]
    assert isinstance(rows, list)
    smoke = Path(authority["smoke"])
    _complete(smoke, rows, 0, "gh005.cluster")
    label = _label(rows, 0)
    _json(smoke / "shared_c_collection" / label / "hub_health.json", {
        "ok": True,
        "schema": "cec_controller_portability_hub_v2",
        "controller": "navdp",
        "initialized": True,
        "reset_required": False,
        "force_reject_native": False,
    })
    audit_payload = _audit(authority, "smoke_ready")
    assert audit_payload["verified"] is True
    assert audit_payload["runtime_health_contract"][label][
        "authority_receipt_mode"] == "legacy_health_plus_identity_and_ast"
    identity = smoke / "shared_c_collection" / label / "compute_identity.json"
    payload = json.loads(identity.read_text())
    payload["cec_hub"]["cli_contract"] = "explicit_cli"
    _json(identity, payload)
    with pytest.raises(RuntimeError, match="legacy hub CLI contract"):
        _audit(authority, "smoke_ready")


def test_smoke_accepts_consistent_explicit_health_authority(
    authority: dict[str, object],
) -> None:
    _archive_startup_partials(authority)
    rows = authority["rows"]
    assert isinstance(rows, list)
    smoke = Path(authority["smoke"])
    _complete(smoke, rows, 0, "gh005.cluster")
    label = _label(rows, 0)
    health = {
        "ok": True,
        "schema": "cec_controller_portability_hub_v2",
        "controller": "navdp",
        "initialized": True,
        "reset_required": False,
        "force_reject_native": False,
        "reject_policy": "shared_native_exact",
        "reject_controller": "navdp",
    }
    path = smoke / "shared_c_collection" / label / "hub_health.json"
    _json(path, health)
    payload = _audit(authority, "smoke_ready")
    assert payload["runtime_health_contract"][label][
        "authority_receipt_mode"] == "explicit_authority_fields"
    health["reject_controller"] = "vint"
    _json(path, health)
    with pytest.raises(RuntimeError, match="explicit authority semantics"):
        _audit(authority, "smoke_ready")


def test_ready_to_seal_checks_exact_nodes_and_retained_hashes(
    authority: dict[str, object],
) -> None:
    _archive_startup_partials(authority)
    run = Path(authority["run"])
    rows = authority["rows"]
    assert isinstance(rows, list)
    for index in REPAIR:
        _complete(run, rows, index, NODES[index])
    assert _audit(authority, "ready_to_seal")["verified"] is True
    retained = required_result_paths(
        run, _label(rows, 2), rows[2]["episode"])[0]
    retained.write_text("changed\n")
    with pytest.raises(RuntimeError, match="retained factual-C outputs changed"):
        _audit(authority, "ready_to_seal")
