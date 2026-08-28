import hashlib
import json
from pathlib import Path

import pytest

from MemNavData.audit_hm3d_lifelong_underpowered_amendment import audit


def write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    run = tmp_path / "run"
    population = run / "population" / "population.json"
    rows = [
        {"scene": "s0", "episode": "e0"},
        {"scene": "s1", "episode": "e1"},
    ]
    digest = write_json(population, {"accepted": rows})
    (population.parent / "SEALED").write_text("sealed\n")
    (population.parent / "population.json.sha256").write_text(
        f"{digest}  population.json\n")
    verification = {
        "verified": True,
        "population_sha256": digest,
        "supported_population": 2,
        "scene_clusters": 2,
        "strong_support_histories": 1,
        "query_navigation_outcomes_read": False,
        "factual_C_B2_C2_executed": False,
        "target_met": False,
    }
    write_json(run / "verification.json", verification)
    protocol = {
        "schema_version": (
            "hm3d_fullmono_lifelong_underpowered_amendment_v1_20260828"),
        "amends": {
            "original_powered_confirmation_claim_permanently_withheld": True,
        },
        "freeze_boundary": {
            "factual_C_outcomes_read": False,
            "B2_outcomes_read": False,
        },
        "source_population": {
            "run_root": str(run.resolve()),
            "relative_path": "population/population.json",
            "sha256": digest,
            "histories": 2,
            "scene_clusters": 2,
            "independent_population_verification": "verification.json",
        },
    }
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, protocol)
    return protocol_path, run


def test_clean_underpowered_population_passes(tmp_path: Path) -> None:
    protocol, run = fixture(tmp_path)
    result = audit(
        protocol_path=protocol, run_root=run, require_pristine=True)
    assert result["verified"] is True
    assert result["histories"] == 2
    assert result["original_powered_confirmation_claim_withheld"] is True


def test_existing_query_output_fails_closed(tmp_path: Path) -> None:
    protocol, run = fixture(tmp_path)
    (run / "shared_c_collection").mkdir()
    with pytest.raises(RuntimeError, match="query output already exists"):
        audit(protocol_path=protocol, run_root=run, require_pristine=True)


def test_verifier_outcome_access_fails_closed(tmp_path: Path) -> None:
    protocol, run = fixture(tmp_path)
    path = run / "verification.json"
    payload = json.loads(path.read_text())
    payload["query_navigation_outcomes_read"] = True
    write_json(path, payload)
    with pytest.raises(RuntimeError, match="query outcomes"):
        audit(protocol_path=protocol, run_root=run)
