#!/usr/bin/env python3
"""Independently verify the outcome-blind HM3D held-out scene selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "hm3d_consumed_scene_audit_v1_20260816"
RECEIPT_SCHEMA = "hm3d_heldout_scene_selection_verification_v1_20260816"
DIRECTORY_RE = re.compile(r"^(?P<index>\d{5})-(?P<scene>[A-Za-z0-9]+)$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_ids_from_manifest(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    require(isinstance(episodes, list), f"manifest episodes missing: {path}")
    scene_ids: set[str] = set()
    for row in episodes:
        require(isinstance(row, dict), f"non-object episode row: {path}")
        value = row.get("scene_id")
        require(isinstance(value, str) and value,
                f"episode scene_id missing: {path}")
        scene_ids.add(value)
    require(scene_ids, f"manifest contains no scene identities: {path}")
    return scene_ids


def archive_directories(member_list: Path) -> list[dict[str, Any]]:
    lines = member_list.read_text(encoding="utf-8").splitlines()
    roots: dict[str, set[str]] = {}
    for raw in lines:
        require(raw and not raw.startswith("/"),
                f"invalid archive member: {raw!r}")
        parts = raw.rstrip("/").split("/")
        require(len(parts) in {1, 2} and ".." not in parts,
                f"unsafe/unexpected archive member: {raw}")
        root = parts[0]
        roots.setdefault(root, set())
        if len(parts) == 2:
            roots[root].add(parts[1])
    records: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    seen_scenes: set[str] = set()
    for directory, files in roots.items():
        match = DIRECTORY_RE.fullmatch(directory)
        require(match is not None, f"unexpected HM3D directory: {directory}")
        index = int(match.group("index"))
        scene = match.group("scene")
        require(index not in seen_indices and scene not in seen_scenes,
                f"duplicate HM3D index/scene: {directory}")
        expected = {f"{scene}.basis.glb", f"{scene}.basis.navmesh"}
        require(files == expected,
                f"asset members differ for {directory}: {sorted(files)}")
        seen_indices.add(index)
        seen_scenes.add(scene)
        records.append({"archive_index": index, "directory": directory,
                        "scene_id": scene})
    records.sort(key=lambda row: (row["archive_index"], row["directory"]))
    require(len(records) == 100, "HM3D val member list must contain 100 scenes")
    return records


def verify_selection(
    audit_path: Path,
    member_list: Path,
    repo_root: Path,
) -> dict[str, Any]:
    require(audit_path.is_file() and not audit_path.is_symlink(),
            "scene audit must be a physical file")
    require(member_list.is_file() and not member_list.is_symlink(),
            "archive member list must be a physical file")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("schema_version") == AUDIT_SCHEMA and
            audit.get("status") == "ok", "scene audit schema/status changed")
    member_sha = sha256_file(member_list)
    require(member_sha == audit.get("archive", {}).get("member_list_sha256"),
            "archive member-list hash changed")

    source_records: list[dict[str, Any]] = []
    consumed: set[str] = set()
    for source in audit.get("consumption_sources", []):
        relative = Path(str(source.get("path", "")))
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"unsafe consumption-source path: {relative}")
        path = repo_root / relative
        require(path.is_file() and not path.is_symlink(),
                f"missing physical consumption source: {path}")
        digest = sha256_file(path)
        require(digest == source.get("sha256"),
                f"consumption-source hash changed: {relative}")
        ids = scene_ids_from_manifest(path)
        consumed.update(ids)
        source_records.append({"path": relative.as_posix(),
                               "sha256": digest,
                               "unique_scene_count": len(ids)})
    require(len(source_records) == 5, "expected five prior HM3D manifests")
    expected_consumed = sorted(str(value)
                               for value in audit["consumed_scene_ids"])
    require(sorted(consumed) == expected_consumed and len(consumed) == 36,
            "recomputed consumed-scene union differs")

    archive = archive_directories(member_list)
    archive_ids = {row["scene_id"] for row in archive}
    require(consumed <= archive_ids,
            "a consumed HM3D scene is absent from the val archive")
    unconsumed = [row for row in archive if row["scene_id"] not in consumed]
    require(len(unconsumed) == 64, "unconsumed HM3D population is not 64")
    recomputed = [
        {"index": index, "directory": row["directory"],
         "scene_id": row["scene_id"]}
        for index, row in enumerate(unconsumed[:10])
    ]
    require(recomputed == audit.get("selected_scenes"),
            "deterministic first-ten selection differs")
    require(not (consumed & {row["scene_id"] for row in recomputed}),
            "selected scene overlaps prior outcomes")
    require(audit.get("outcome_fields_read_for_selection") is False,
            "outcome-blind selection guard changed")
    return {
        "schema_version": RECEIPT_SCHEMA,
        "verified": True,
        "audit_sha256": sha256_file(audit_path),
        "member_list_sha256": member_sha,
        "archive_scene_count": len(archive),
        "consumption_source_count": len(source_records),
        "consumption_sources": source_records,
        "consumed_scene_count": len(consumed),
        "unconsumed_scene_count": len(unconsumed),
        "selection_rule_recomputed": True,
        "selected_overlap_with_consumed": [],
        "outcome_fields_read_for_selection": False,
        "selected_scenes": recomputed,
    }


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--member-list", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify_selection(args.audit.resolve(), args.member_list.resolve(),
                               args.repo_root.resolve())
    if args.out is not None:
        write_exclusive(args.out.resolve(), payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
