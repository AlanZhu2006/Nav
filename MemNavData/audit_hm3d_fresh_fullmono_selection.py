#!/usr/bin/env python3
"""Verify the frozen 54-scene fresh HM3D reserve without reading outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


SCHEMA = "hm3d_fresh_fullmono_scene_selection_audit_v1_20260820"
PROTOCOL_SCHEMA = "hm3d_fresh_fullmono_mixed_role_protocol_v1_20260820"
DIRECTORY_RE = re.compile(r"^(?P<index>\d{5})-(?P<scene>[A-Za-z0-9]+)$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_scenes(member_list: Path) -> list[dict[str, Any]]:
    roots: dict[str, set[str]] = {}
    for raw in member_list.read_text(encoding="utf-8").splitlines():
        require(raw and not raw.startswith("/"), f"unsafe member {raw!r}")
        parts = raw.rstrip("/").split("/")
        require(len(parts) in {1, 2} and ".." not in parts,
                f"unexpected member {raw!r}")
        roots.setdefault(parts[0], set())
        if len(parts) == 2:
            roots[parts[0]].add(parts[1])
    rows = []
    for directory, files in roots.items():
        match = DIRECTORY_RE.fullmatch(directory)
        require(match is not None, f"bad scene directory {directory!r}")
        index = int(match.group("index"))
        scene = match.group("scene")
        require(files == {f"{scene}.basis.glb", f"{scene}.basis.navmesh"},
                f"asset membership changed for {scene}")
        rows.append({"archive_index": index, "directory": directory,
                     "scene_id": scene})
    rows.sort(key=lambda row: (row["archive_index"], row["directory"]))
    require(len(rows) == 100, "HM3D val archive must contain 100 scenes")
    require(len({row["archive_index"] for row in rows}) == 100,
            "duplicate archive index")
    require(len({row["scene_id"] for row in rows}) == 100,
            "duplicate scene identity")
    return rows


def verify(protocol_path: Path, member_list: Path, prior_audit_path: Path,
           heldout10_protocol_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA,
            "fresh protocol schema changed")
    dataset = protocol["dataset"]
    require(sha256(member_list) == dataset["archive_member_list_sha256"],
            "HM3D member list changed")
    require(sha256(prior_audit_path) == dataset["prior_consumed_audit_sha256"],
            "prior 36-scene audit changed")
    require(sha256(heldout10_protocol_path) ==
            dataset["prior_heldout10_protocol_sha256"],
            "prior heldout10 protocol changed")

    prior = json.loads(prior_audit_path.read_text(encoding="utf-8"))
    heldout = json.loads(heldout10_protocol_path.read_text(encoding="utf-8"))
    consumed36 = {str(value) for value in prior["consumed_scene_ids"]}
    consumed10 = {str(row["scene_id"]) for row in heldout["scenes"]}
    require(len(consumed36) == 36 and len(consumed10) == 10,
            "prior consumed populations changed")
    require(not (consumed36 & consumed10),
            "prior consumed populations unexpectedly overlap")
    consumed = consumed36 | consumed10
    require(len(consumed) == dataset["prior_consumed_scene_count"] == 46,
            "46-scene consumed union changed")

    archive = archive_scenes(member_list)
    archive_ids = {row["scene_id"] for row in archive}
    require(consumed <= archive_ids, "consumed identity absent from archive")
    fresh = [row for row in archive if row["scene_id"] not in consumed]
    require(len(fresh) == dataset["fresh_scene_count"] == 54,
            "fresh reserve size changed")
    expected = [
        {"rank": rank, "archive_index": row["archive_index"],
         "directory": row["directory"], "scene_id": row["scene_id"]}
        for rank, row in enumerate(fresh)
    ]
    require(expected == dataset["scenes"],
            "frozen fresh-scene order differs from archive subtraction")
    selection = protocol["population_selection"]
    require(selection["initial_scene_prefix"] == 30 and
            selection["extension_block_scenes"] == 6 and
            selection["maximum_scene_prefix"] == 54,
            "prefix expansion schedule changed")
    require(selection["native_raw_cec_query_outcomes_read"] is False,
            "selection is not query-outcome blind")
    require(protocol["query_outcomes_read_before_freeze"] is False,
            "protocol was not frozen before query outcomes")
    return {
        "schema_version": SCHEMA,
        "verified": True,
        "query_outcome_blind": True,
        "protocol_sha256": sha256(protocol_path),
        "member_list_sha256": sha256(member_list),
        "prior_consumed_audit_sha256": sha256(prior_audit_path),
        "prior_heldout10_protocol_sha256": sha256(heldout10_protocol_path),
        "archive_scene_count": len(archive),
        "prior_consumed_scene_count": len(consumed),
        "fresh_scene_count": len(fresh),
        "selected_overlap_with_consumed": [],
        "initial_scene_prefix": 30,
        "extension_prefixes": [36, 42, 48, 54],
        "fresh_scenes": expected,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--member-list", type=Path, required=True)
    parser.add_argument("--prior-audit", type=Path, required=True)
    parser.add_argument("--heldout10-protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    payload = verify(args.protocol.resolve(), args.member_list.resolve(),
                     args.prior_audit.resolve(),
                     args.heldout10_protocol.resolve())
    if args.out is not None:
        write_exclusive(args.out.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
