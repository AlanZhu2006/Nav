#!/usr/bin/env python3
"""Independently recount saved probe predictions, without model metric helpers."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    destination = root / "independent_verification.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((root / "manifest.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert digest(root / "manifest.json") == report["manifest_sha256"]
    assert not report["deployment_approved"] and report["navigation_SR"] is None
    assert set(manifest["train_scene_ids"]).isdisjoint(manifest["validation_scene_ids"])
    pairs = {pair["pair_id"]: pair for pair in manifest["pairs"]}
    assert len(pairs) == len(manifest["pairs"])
    grouped = defaultdict(list)
    identities = set()
    with (root / "predictions.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            identity = (row["split"], row["arm"], row["pair_id"])
            assert identity not in identities
            identities.add(identity)
            pair = pairs[row["pair_id"]]
            assert pair["scene"] == row["scene"] and pair["split"] == row["split"]
            for axis, index in (("forward", 0), ("lateral", 1)):
                assert math.isclose(float(row[f"target_{axis}_m"]), pair["target_xy_m"][index],
                                    rel_tol=1e-6, abs_tol=1e-6)
            grouped[(row["split"], row["arm"])].append(row)
    checks = {}
    for (split, arm), rows in grouped.items():
        expected = report["metrics"][split][arm]
        errors, angles = [], []
        eligible = hit_position = hit_angle = 0
        for row in rows:
            tx, ty = (float(row[f"target_{axis}_m"]) for axis in ("forward", "lateral"))
            direction_ok = math.hypot(tx, ty) >= .25
            eligible += direction_ok
            if row["predicted_forward_m"] == "" or row["predicted_lateral_m"] == "":
                continue
            px, py = (float(row[f"predicted_{axis}_m"]) for axis in ("forward", "lateral"))
            error = math.hypot(px - tx, py - ty)
            errors.append(error)
            hit_position += error <= .5
            if direction_ok and math.hypot(px, py) > 1e-6:
                delta = math.atan2(py, px) - math.atan2(ty, tx)
                angle = abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))
                angles.append(angle)
                hit_angle += angle <= 30
        assert len(rows) == expected["pairs"]
        assert len(errors) == expected["valid_predictions"]
        assert eligible == expected["anchor_local_direction_eligible"]
        assert hit_position == expected["position_within_05m"]
        assert hit_angle == expected["anchor_local_direction_within_30deg"]
        assert len(angles) == expected["anchor_local_direction_valid"]
        assert np.allclose(np.mean(errors), expected["position_error_m"]["mean"], atol=1e-8)
        assert np.allclose(np.median(errors), expected["position_error_m"]["median"], atol=1e-8)
        checks[f"{split}/{arm}"] = {"pairs": len(rows), "valid_predictions": len(errors),
                                    "mean_position_error_m": float(np.mean(errors))}
    result = {"verified": True, "groups": checks, "navigation_SR": None,
              "sources": {filename: digest(root / filename) for filename in
                          ("manifest.json", "report.json", "predictions.csv", "probe.pt")}}
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verified": True, "groups": len(checks), "output": str(destination)}))


if __name__ == "__main__":
    main()
