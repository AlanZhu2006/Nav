"""Audit scene-level disjointness between NavDP training data and HM3D val.

Question: are the HM3D scenes in the released NavDP training corpus
(InternData-N1, ``vln_n1/traj_data/hm3d_*``) disjoint from the HM3D v0.2
val split (100 scenes) that every fresh/heldout/consumed pool in this
project is drawn from?

Scope and honesty boundary: this audits the *public* InternData-N1
release listing only. The frozen ``navdp_checkpoint.ckpt`` carries no
training manifest, so the claim this audit can support is
"the released training corpus draws its HM3D scenes exclusively from the
train split (numeric prefixes < 00800) and contains none of the 100 val
scene ids" — not a statement about undisclosed data the checkpoint may
additionally have seen. NavDP's paper corpus (arXiv 2505.08712) also
includes Matterport3D, Gibson, Replica, HSSD and 3D-Front; only the HM3D
axis is audited here.

Inputs are cached HuggingFace tree listings (JSON arrays with ``path``
entries, one per scene tarball) so the audit is reproducible offline:

    curl -sf "https://huggingface.co/api/datasets/InternRobotics/\
InternData-N1/tree/main/vln_n1/traj_data/<group>?recursive=false"

Usage:
    python MemNavData/audit_navdp_training_hm3d_disjointness_20260828.py \
        --listing hm3d_d435i=<hm3d_d435i_tree.json> \
        --listing hm3d_zed=<hm3d_zed_tree.json> \
        --val-members <hm3d-val-habitat-v0.2.members.txt> \
        --out <audit.json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SCENE_RE = re.compile(r"^(\d{5})-([A-Za-z0-9]{11})")
HM3D_TRAIN_PREFIX_EXCLUSIVE_MAX = 800


def parse_listing_scene_ids(listing_path: Path) -> dict[str, int]:
    """Map scene hash id -> numeric prefix for one cached HF tree JSON."""
    entries = json.loads(listing_path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"empty or non-list listing: {listing_path}")
    out: dict[str, int] = {}
    for entry in entries:
        name = str(entry["path"]).split("/")[-1]
        match = SCENE_RE.match(name)
        if match is None:
            raise ValueError(f"unrecognized scene entry name: {name!r}")
        out[match.group(2)] = int(match.group(1))
    return out


def parse_val_member_scene_ids(members_path: Path) -> dict[str, int]:
    """Map scene hash id -> numeric prefix from the archive member list."""
    out: dict[str, int] = {}
    for line in members_path.read_text().splitlines():
        match = SCENE_RE.match(line.strip())
        if match is not None:
            out.setdefault(match.group(2), int(match.group(1)))
    if not out:
        raise ValueError(f"no scene ids found in {members_path}")
    return out


def run_audit(listings: dict[str, Path], members_path: Path) -> dict:
    val_ids = parse_val_member_scene_ids(members_path)
    groups = {}
    union_train_ids: set[str] = set()
    for group, path in sorted(listings.items()):
        ids = parse_listing_scene_ids(path)
        union_train_ids |= set(ids)
        prefixes = sorted(ids.values())
        groups[group] = {
            "listing_file": str(path),
            "scene_count": len(ids),
            "prefix_min": prefixes[0],
            "prefix_max": prefixes[-1],
            "all_prefixes_in_train_range": (
                prefixes[-1] < HM3D_TRAIN_PREFIX_EXCLUSIVE_MAX),
            "val_intersection": sorted(set(ids) & set(val_ids)),
        }
    intersection = sorted(union_train_ids & set(val_ids))
    return {
        "schema": "navdp_training_hm3d_disjointness_audit_v1_20260828",
        "question": (
            "released InternData-N1 hm3d_* training scenes vs HM3D v0.2 "
            "val split scene ids"),
        "val_member_scene_count": len(val_ids),
        "training_groups": groups,
        "union_training_scene_count": len(union_train_ids),
        "val_intersection": intersection,
        "disjoint": not intersection and all(
            g["all_prefixes_in_train_range"] for g in groups.values()),
        "claim_boundary": (
            "public InternData-N1 release only; the frozen checkpoint "
            "carries no training manifest, so undisclosed additional "
            "training data cannot be excluded"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listing", action="append", required=True, metavar="GROUP=PATH",
        help="cached HF tree JSON for one hm3d_* group, as name=path")
    parser.add_argument("--val-members", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    listings = {}
    for spec in args.listing:
        group, _, path = spec.partition("=")
        if not path:
            parser.error(f"--listing must be GROUP=PATH, got {spec!r}")
        listings[group] = Path(path)

    report = run_audit(listings, args.val_members)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    verdict = "DISJOINT" if report["disjoint"] else "OVERLAP"
    print(f"{verdict}: {report['union_training_scene_count']} training "
          f"scenes vs {report['val_member_scene_count']} val scenes, "
          f"intersection={len(report['val_intersection'])} -> {args.out}")


if __name__ == "__main__":
    main()
