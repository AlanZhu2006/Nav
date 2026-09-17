"""Recover missing per-rollout pose indices from the original complete log.

No navigation, poses, success labels or existing file are changed. Every frame
index and RGB hash must agree with the recorded prefix and actual query trace.
"""
import argparse
import json
from pathlib import Path

from MemNavData.table2_mixed_local import load, dump, sha


def recover(root):
    log = root / "lingbot_pose_readout.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    summary = load(root / "summary.json")
    cursor, recovered = 0, []
    for record in summary["records"]:
        folder = Path(record["directory"])
        query = load(record["query_file"])
        prefix = load(Path(query["prefix_root"])/"online_a_trace.json")["poses"] if query["prefix_root"] else []
        actual = load(folder/"actual_trace.json")["poses"]
        hashes = [p["jpg_sha256"] for p in prefix + actual]
        segment = rows[cursor:cursor+len(hashes)]
        assert [r["frame_idx"] for r in segment] == list(range(len(hashes)))
        assert [r["image_sha256"] for r in segment] == hashes
        target = folder / "lingbot_frame_poses.json"
        if target.exists():
            assert load(target) == segment
        else:
            dump(target, segment)
        recovered.append(dict(directory=str(folder), source_row_offset=cursor,
                              frames=len(segment), output_sha256=sha(target)))
        cursor += len(hashes)
    assert cursor == len(rows), "Unassigned pose records remain"
    dump(root / "pose_index_recovery.json", dict(source_log_sha256=sha(log),
         source_summary_sha256=sha(root/"summary.json"), recovered=recovered,
         navigation_rerun=False, original_records_modified=False, script_sha256=sha(__file__)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    recover(parser.parse_args().root)
