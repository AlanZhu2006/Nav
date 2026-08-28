import hashlib
import json
from pathlib import Path

import pytest

import MemNavData.audit_hm3d_lifelong_underpowered_collect_repair as module


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def make_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    rows = [
        {"scene": f"s{i % 15}", "episode": f"e{i}"} for i in range(22)
    ]
    population = run / "population/population.json"
    write_json(population, {"accepted": rows})
    digest = hashlib.sha256(population.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "EXPECTED_POPULATION_SHA", digest)
    completed = [2, 3, 4, 5, 6, 8, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21]
    failed = [0, 1, 7, 9, 11, 13]
    archive = run / "failed_attempts/archive"
    items = []
    for index in failed:
        label = f"{index:03d}_s{index % 15}_e{index}"
        failure_class = ("invalid_empty_fifo_assumption" if index in {0, 13}
                         else "cross_node_RGB_replay_mismatch")
        items.append({
            "index": index, "label": label, "failure_class": failure_class,
        })
        partial = run / "shared_c_collection" / label
        (partial / "logs").mkdir(parents=True)
        message = ("NavDP replay queue length does not match frozen plan count"
                   if index in {0, 13}
                   else "shared Goal-B trace rendered RGB mismatch")
        (partial / "logs/evaluator.log").write_text(message + "\n")
    for index in completed:
        row = rows[index]
        label = f"{index:03d}_{row['scene']}_{row['episode']}"
        for path in module.required_result_paths(run, label, row["episode"]):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(path.name + "\n")
    protocol = tmp_path / "protocol.json"
    write_json(protocol, {
        "schema_version": module.SCHEMA,
        "source_authority": {
            "run_root": str(run.resolve()),
            "population_relative_path": "population/population.json",
            "population_sha256": digest,
        },
        "freeze_boundary": {
            "successful_factual_C_navigation_outcomes_read_before_repair": False,
            "B2_navigation_outcomes_read_before_repair": False,
            "repair_selection_uses_navigation_outcomes": False,
        },
        "incident": {
            "completed_indices": completed,
            "failed_indices": failed,
        },
        "repair_items": items,
        "repair_contract": {
            "archive_failed_partial_outputs": str(archive),
        },
    })
    return protocol, run, archive


def test_pre_and_post_archive(tmp_path: Path, monkeypatch) -> None:
    protocol, run, archive = make_fixture(tmp_path, monkeypatch)
    checked = module.audit(
        protocol_path=protocol, run_root=run, phase="pre_archive")
    assert checked["verified"] is True
    payload = json.loads(protocol.read_text())
    for item in payload["repair_items"]:
        source = run / "shared_c_collection" / item["label"]
        archive.mkdir(parents=True, exist_ok=True)
        source.rename(archive / item["label"])
    checked = module.audit(
        protocol_path=protocol, run_root=run, phase="post_archive")
    assert checked["repair_indices"] == [0, 1, 7, 9, 11, 13]


def test_downstream_output_fails_closed(tmp_path: Path, monkeypatch) -> None:
    protocol, run, _archive = make_fixture(tmp_path, monkeypatch)
    (run / "shared_c_population").mkdir()
    with pytest.raises(RuntimeError, match="downstream query output"):
        module.audit(protocol_path=protocol, run_root=run, phase="pre_archive")
