from __future__ import annotations

import hashlib
import json
from pathlib import Path

from MemNavData.freeze_mp3d_table1_fullmono_expanded_source_ledger import (
    identity_key,
    read_consumed_query_manifests,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "MemNavData/mp3d_table1_fullmono_source_expansion_protocol_20260829.json"
SOURCE_MANIFEST = (
    ROOT / ".diagnostics/paper_power_expansion_freeze_20260814_pre_result"
    / "paper_power_expansion_manifest.json"
)
RUNNER = ROOT / "MemNavData/run_hm3d_fullmono_server_scene.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_phase2_source_identity_is_complete_and_disjoint():
    protocol = json.loads(PROTOCOL.read_text())
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    source = protocol["expansion_source"]
    scenes = manifest["selection"]["selected_scenes"]
    assert sha256(SOURCE_MANIFEST) == source["manifest_sha256"]
    assert [row["scene_id"] for row in protocol["dataset"]["scenes"]] == scenes
    assert [row["rank"] for row in protocol["dataset"]["scenes"]] == list(range(16))
    assert set(scenes).isdisjoint({
        "s8pcmisQ38h", "e9zR4mvMWw7", "rqfALeAoiTq", "zsNo4HB9uLZ",
        "yqstnuAEVhm", "gxdoqLR6rwA", "pLe4wQe7qrG", "cV4RVeZvu5T",
        "mJXqzFtmKg4", "gTV8FGcVJC9", "oLBMNvg9in8", "uNb9QFRL6hY",
        "rPc6DW4iMge", "ac26ZMwG7aT", "dhjEzFoUFzH", "qoiz87JEwZ2",
        "b8cTxDM8gDG", "i5noydFURQK", "wc2JMjhGNzB", "gZ6f7yhEvPG",
    })
    for scene in scenes:
        assert [row["episode"] for row in manifest["episodes"][scene]] == source[
            "episode_ids"
        ]


def test_consumed_query_reader_resolves_episode_relative_images(tmp_path: Path):
    root = tmp_path / "benchmark"
    goal = root / "scene" / "episode" / "pair_00/novel/goal.jpg"
    goal.parent.mkdir(parents=True)
    goal.write_bytes(b"goal-bytes")
    digest = sha256(goal)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({
        "episodes": [{
            "scene": "scene",
            "episode": "episode",
            "pairs": [{"queries": [{
                "goal_rgb": "pair_00/novel/goal.jpg",
                "goal_rgb_sha256": digest,
                "floor_position": [1.0, 0.0, 2.0],
                "yaw_rad": 0.3,
            }]}],
        }],
    }))
    rows, receipts = read_consumed_query_manifests([{
        "path": str(manifest), "sha256": sha256(manifest),
    }])
    assert len(rows["scene"]) == 1
    assert identity_key(rows["scene"][0]) == (
        digest, 1.0, 0.0, 2.0, 0.3,
    )
    assert receipts[0]["query_identities"] == 1


def test_runtime_has_explicit_mp3d_collection_mode():
    text = RUNNER.read_text()
    assert '"${MODE}" == mp3d_collect' in text
    assert "collect_mp3d_table1_fullmono_goal_a.py" in text
    assert 'PYTHONPATH="${MEMNAV_PYTHONPATH_VALUE}"' in text
    assert 'PYTHONPATH="${NAVDP_PYTHONPATH_VALUE}"' in text
