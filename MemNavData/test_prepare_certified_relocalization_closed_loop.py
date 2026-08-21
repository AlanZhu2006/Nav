import hashlib
import json

import pytest

from prepare_certified_relocalization_closed_loop import prepare


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_prepare_hashes_dependencies_and_writes_read_only_receipts(tmp_path):
    dependencies = {}
    for name in ("gatecurr600", "navdp_checkpoint", "lingbot_map_long"):
        path = tmp_path / name
        path.write_bytes((name + "\n").encode())
        dependencies[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
    scenes = [f"scene_{index:02d}" for index in range(20)]
    manifest = {
        "audit": {"status": "ok"},
        "data_role_guards": {"blind_allowed": False},
        "scenes": scenes,
        "episodes": {
            scene: [{"episode": f"episode_{index:04d}"}
                    for index in range(8)]
            for scene in scenes
        },
        "dependencies": dependencies,
    }
    manifest_path = tmp_path / "upstream.json"
    manifest_path.write_text(json.dumps(manifest))
    source_receipt = tmp_path / "SOURCE_BUNDLE.sha256"
    source_receipt.write_text("placeholder\n")
    run_root = tmp_path / "run"
    result = prepare(
        manifest_path, _sha(manifest_path),
        source_receipt, _sha(source_receipt), run_root)
    assert result["status"] == "prepared"
    receipt = json.loads((run_root / "dependency_receipt.json").read_text())
    assert receipt["dependencies"] == dependencies
    for name in (
        "data_manifest.json", "data_manifest.json.sha256",
        "dependency_receipt.json", "dependency_receipt.json.sha256",
        "source_bundle.sha256",
    ):
        assert (run_root / name).stat().st_mode & 0o222 == 0
    with pytest.raises(RuntimeError, match="already exists"):
        prepare(
            manifest_path, _sha(manifest_path),
            source_receipt, _sha(source_receipt), run_root)


def test_prepare_rejects_dependency_hash_change(tmp_path):
    paths = {}
    for name in ("gatecurr600", "navdp_checkpoint", "lingbot_map_long"):
        path = tmp_path / name
        path.write_text(name)
        paths[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
    paths["gatecurr600"]["sha256"] = "0" * 64
    scenes = [f"s{index}" for index in range(20)]
    manifest = {
        "audit": {"status": "ok"},
        "data_role_guards": {"blind_allowed": False},
        "scenes": scenes,
        "episodes": {scene: [{} for _ in range(8)] for scene in scenes},
        "dependencies": paths,
    }
    source = tmp_path / "manifest.json"
    source.write_text(json.dumps(manifest))
    receipt = tmp_path / "source.sha256"
    receipt.write_text("receipt")
    with pytest.raises(RuntimeError, match="dependency SHA changed"):
        prepare(source, _sha(source), receipt, _sha(receipt), tmp_path / "run")
