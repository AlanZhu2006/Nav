#!/usr/bin/env python3

import json
import tempfile
from pathlib import Path

from MemNavData.audit_hm3d_heldout_scene_selection import verify_selection


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "MemNavData/hm3d_consumed_scene_audit_20260816.json"
MEMBERS = (ROOT / ".diagnostics/datasets/goat-bench/hm3d_val_receipts/"
           "hm3d-val-habitat-v0.2.members.txt")


def test_frozen_selection_recomputes_from_sources_and_archive() -> None:
    receipt = verify_selection(AUDIT, MEMBERS, ROOT)
    assert receipt["verified"] is True
    assert receipt["archive_scene_count"] == 100
    assert receipt["consumed_scene_count"] == 36
    assert receipt["unconsumed_scene_count"] == 64
    assert len(receipt["selected_scenes"]) == 10
    assert receipt["selected_overlap_with_consumed"] == []


def test_selection_audit_rejects_a_consumed_scene() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "audit.json"
        payload = json.loads(AUDIT.read_text(encoding="utf-8"))
        payload["selected_scenes"][0] = {
            "index": 0, "directory": "00800-TEEsavR23oF",
            "scene_id": "TEEsavR23oF"}
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            verify_selection(path, MEMBERS, ROOT)
        except RuntimeError as error:
            assert "selection differs" in str(error)
        else:
            raise AssertionError("consumed-scene selection was accepted")
